# Copyright 2024 RecML authors <recommendations-ml@google.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DLRM experiment."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import dataclasses
from typing import Generic, Literal, TypeVar

import fiddle as fdl
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
import jaxtyping as jt
import numpy as np
import optax
from recml import mlrx
from recml.layers.linen import sparsecore
import tensorflow as tf


@dataclasses.dataclass
class Feature:
  name: str


FeatureT = TypeVar('FeatureT', bound=Feature)


@dataclasses.dataclass
class DenseFeature(Feature):
  """Dense feature."""


@dataclasses.dataclass
class SparseFeature(Feature):
  """Sparse feature."""

  vocab_size: int
  embedding_dim: int
  max_sequence_length: int | None = None
  combiner: Literal['mean', 'sum', 'sqrtn'] = 'mean'
  sparsity: float = 0.8


@dataclasses.dataclass
class FeatureSet(Generic[FeatureT]):
  """A collection of features."""

  features: Sequence[FeatureT]

  def __post_init__(self):
    feature_names = [f.name for f in self.features]
    if len(feature_names) != len(set(feature_names)):
      raise ValueError(
          f'Feature names must be unique. Got names: {feature_names}.'
      )

  def dense_features(self) -> FeatureSet[DenseFeature]:
    return FeatureSet([f for f in self if isinstance(f, DenseFeature)])

  def sparse_features(self) -> FeatureSet[SparseFeature]:
    return FeatureSet([f for f in self if isinstance(f, SparseFeature)])

  def __iter__(self) -> Iterator[FeatureT]:
    return iter(self.features)

  def __or__(self, other: FeatureSet[Feature]) -> FeatureSet[Feature]:
    return FeatureSet([*self.features, *other.features])


class DLRMModel(nn.Module):
  """DLRM DCN v2 model."""

  features: FeatureSet
  embedding_optimizer: embedding_spec.OptimizerSpec
  bottom_mlp_dims: Sequence[int]
  top_mlp_dims: Sequence[int]
  dcn_layers: int
  dcn_inner_dim: int

  # We need to track the embedder on the Flax module to ensure it is not
  # re-created on cloning. It is not possible to create an embedder inside
  # setup() because it is called lazily at compile time. The embedder needs
  # to be created before `model.init` so we can use it to create a preprocessor.
  # A simpler pattern that works is passing `embedder` directly to the module.
  _embedder: sparsecore.SparsecoreEmbedder | None = None

  @property
  def embedder(self) -> sparsecore.SparsecoreEmbedder:
    if self._embedder is not None:
      return self._embedder

    embedder = sparsecore.SparsecoreEmbedder(
        specs={
            f.name: sparsecore.EmbeddingSpec(
                input_dim=f.vocab_size,
                embedding_dim=f.embedding_dim,
                max_sequence_length=f.max_sequence_length,
                combiner=f.combiner,
            )
            for f in self.features.sparse_features()
        },
        optimizer=self.embedding_optimizer,
    )
    object.__setattr__(self, '_embedder', embedder)
    return embedder

  def bottom_mlp(self, inputs: Mapping[str, jt.Array]) -> jt.Array:
    x = jnp.concatenate(
        [inputs[f.name] for f in self.features.dense_features()], axis=-1
    )

    for dim in self.bottom_mlp_dims:
      x = nn.Dense(dim)(x)
      x = nn.relu(x)
    return x

  def top_mlp(self, x: jt.Array) -> jt.Array:
    for dim in self.top_mlp_dims[:-1]:
      x = nn.Dense(dim)(x)
      x = nn.relu(x)

    x = nn.Dense(self.top_mlp_dims[-1])(x)
    return x

  def dcn(self, x0: jt.Array) -> jt.Array:
    xl = x0
    input_dim = x0.shape[-1]

    for i in range(self.dcn_layers):
      u_kernel = self.param(
          f'u_kernel_{i}',
          nn.initializers.xavier_normal(),
          (input_dim, self.dcn_inner_dim),
      )
      v_kernel = self.param(
          f'v_kernel_{i}',
          nn.initializers.xavier_normal(),
          (self.dcn_inner_dim, input_dim),
      )
      bias = self.param(f'bias_{i}', nn.initializers.zeros, (input_dim,))

      u = jnp.matmul(xl, u_kernel)
      v = jnp.matmul(u, v_kernel)
      v += bias

      xl = x0 * v + xl

    return xl

  @nn.compact
  def __call__(
      self, inputs: Mapping[str, jt.Array], training: bool = False
  ) -> jt.Array:
    dense_embeddings = self.bottom_mlp(inputs)
    sparse_embeddings = self.embedder.make_sparsecore_module()(inputs)
    sparse_embeddings = jax.tree.flatten(sparse_embeddings)[0]
    concatenated_embeddings = jnp.concatenate(
        (dense_embeddings, *sparse_embeddings), axis=-1
    )
    interaction_outputs = self.dcn(concatenated_embeddings)
    predictions = self.top_mlp(interaction_outputs)
    predictions = jnp.reshape(predictions, (-1,))
    return predictions


class CriteoFactory(mlrx.Factory[tf.data.Dataset]):
  """Data loader for dummy Criteo data optimized for Jax training."""

  features: FeatureSet
  global_batch_size: int
  use_cached_data: bool = False

  def make(self) -> tf.data.Dataset:
    data = {}
    batch_size = self.global_batch_size // jax.process_count()

    for f in self.features.dense_features():
      feature = np.random.normal(0.0, 1.0, size=(batch_size, 1))
      data[f.name] = feature.astype(np.float32)

    for f in self.features.sparse_features():
      non_zero_mask = (
          np.random.normal(size=(batch_size, f.embedding_dim)) > f.sparsity
      )
      sparse_feature = np.random.randint(
          low=0,
          high=f.vocab_size,
          size=(batch_size, f.embedding_dim),
      )
      sparse_feature = np.where(
          non_zero_mask, sparse_feature, np.zeros_like(sparse_feature)
      )
      data[f.name] = tf.constant(sparse_feature, dtype=tf.int64)

    label = np.random.randint(0, 2, size=(batch_size,))

    dataset = tf.data.Dataset.from_tensors((data, label))
    dataset = dataset.take(1).repeat()
    dataset = dataset.prefetch(buffer_size=2048)
    options = tf.data.Options()
    options.deterministic = False
    options.threading.private_threadpool_size = 96
    dataset = dataset.with_options(options)
    return dataset


@dataclasses.dataclass
class PredictionTask(mlrx.JaxTask):
  """Prediction task."""

  train_data: CriteoFactory
  eval_data: CriteoFactory
  model: DLRMModel
  optimizer: mlrx.Factory[optax.GradientTransformation]

  def create_datasets(
      self,
  ) -> tuple[mlrx.TFDatasetIterator, mlrx.TFDatasetIterator]:
    global_batch_size = self.train_data.global_batch_size
    train_iter = mlrx.TFDatasetIterator(
        dataset=self.train_data.make(),
        postprocessor=self.model.embedder.make_preprocessor(global_batch_size),
    )
    eval_iter = mlrx.TFDatasetIterator(
        dataset=self.eval_data.make(),
        postprocessor=self.model.embedder.make_preprocessor(global_batch_size),
    )
    return train_iter, eval_iter

  def create_state(self, batch: jt.PyTree, rng: jt.Array) -> mlrx.JaxState:
    inputs, _ = batch
    params = self.model.init(rng, inputs)
    optimizer = self.optimizer.make()
    return mlrx.JaxState.create(params=params, tx=optimizer)

  def train_step(
      self, batch: jt.PyTree, state: mlrx.JaxState, rng: jt.Array
  ) -> tuple[mlrx.JaxState, Mapping[str, mlrx.Metric]]:
    inputs, label = batch

    def _loss_fn(params: jt.PyTree) -> tuple[jt.Scalar, jt.Array]:
      logits = self.model.apply(params, inputs, training=True)
      loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, label), axis=0)
      return loss, logits

    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True, allow_int=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.update(grads=grads)

    metrics = {
        'loss': mlrx.metrics.scalar(loss),
        'accuracy': mlrx.metrics.binary_accuracy(label, logits, threshold=0.0),
        'auc': mlrx.metrics.aucpr(label, logits, from_logits=True),
        'aucroc': mlrx.metrics.aucroc(label, logits, from_logits=True),
        'label/mean': mlrx.metrics.mean(label),
        'prediction/mean': mlrx.metrics.mean(jax.nn.sigmoid(logits)),
    }
    return state, metrics

  def eval_step(
      self, batch: jt.PyTree, state: mlrx.JaxState
  ) -> Mapping[str, mlrx.Metric]:
    inputs, label = batch
    logits = self.model.apply(state.params, inputs, training=False)
    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, label), axis=0)

    metrics = {
        'loss': mlrx.metrics.mean(loss),
        'accuracy': mlrx.metrics.binary_accuracy(label, logits, threshold=0.0),
        'auc': mlrx.metrics.aucpr(label, logits, from_logits=True),
        'aucroc': mlrx.metrics.aucroc(label, logits, from_logits=True),
        'label/mean': mlrx.metrics.mean(label),
        'prediction/mean': mlrx.metrics.mean(jax.nn.sigmoid(logits)),
    }
    return metrics


def features() -> fdl.Config[FeatureSet]:
  """Creates a feature collection for the DLRM model."""
  table_sizes = [
      (40000000, 3),
      (39060, 2),
      (17295, 1),
      (7424, 2),
      (20265, 6),
      (3, 1),
      (7122, 1),
      (1543, 1),
      (63, 1),
      (40000000, 7),
      (3067956, 3),
      (405282, 8),
      (10, 1),
      (2209, 6),
      (11938, 9),
      (155, 5),
      (4, 1),
      (976, 1),
      (14, 1),
      (40000000, 12),
      (40000000, 100),
      (40000000, 27),
      (590152, 10),
      (12973, 3),
      (108, 1),
      (36, 1),
  ]
  return fdl.Config(
      FeatureSet,
      features=[
          fdl.Config(DenseFeature, name=f'float-feature-{i}') for i in range(13)
      ]
      + [
          fdl.Config(
              SparseFeature,
              vocab_size=vocab_size,
              embedding_dim=embedding_dim,
              name=f'categorical-feature-{i}',
          )
          for i, (vocab_size, embedding_dim) in enumerate(table_sizes)
      ],
  )


def experiment() -> fdl.Config[mlrx.Experiment]:
  """DLRM experiment."""

  feature_set = features()

  task = fdl.Config(
      PredictionTask,
      train_data=fdl.Config(
          CriteoFactory,
          features=feature_set,
          global_batch_size=131_072,
      ),
      eval_data=fdl.Config(
          CriteoFactory,
          features=feature_set,
          global_batch_size=131_072,
          use_cached_data=True,
      ),
      model=fdl.Config(
          DLRMModel,
          features=feature_set,
          embedding_optimizer=fdl.Config(
              embedding_spec.AdagradOptimizerSpec,
              learning_rate=0.01,
          ),
          bottom_mlp_dims=[512, 256, 128],
          top_mlp_dims=[1024, 1024, 512, 256, 1],
          dcn_layers=3,
          dcn_inner_dim=512,
      ),
      optimizer=fdl.Config(
          mlrx.AdagradFactory,
          learning_rate=0.01,
          # Sparsecore embedding parameters are optimized in the backward pass.
          freeze_mask=rf'.*{sparsecore.EMBEDDING_PARAM_NAME}.*',
      ),
  )
  trainer = fdl.Config(
      mlrx.JaxTrainer,
      partitioner=fdl.Config(mlrx.DataParallelPartitioner),
      train_steps=1_000,
      steps_per_eval=100,
      steps_per_loop=100,
  )
  return fdl.Config(mlrx.Experiment, task=task, trainer=trainer)

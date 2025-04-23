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
"""Sparsecore embedding layers."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import functools
from typing import Any, Literal, Mapping, TypeVar

from etils import epy
from flax import linen as nn
from flax import typing
import jax
import jax.numpy as jnp
import numpy as np
from recml.core.ops import embedding_ops
import tensorflow as tf

with epy.lazy_imports():
  # pylint: disable=g-import-not-at-top
  from jax_tpu_embedding.sparsecore.lib.flax import embed
  from jax_tpu_embedding.sparsecore.lib.nn import embedding
  from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
  from jax_tpu_embedding.sparsecore.utils import utils
  # pylint: enable=g-import-not-at-top

A = TypeVar('A')
CSR_INPUTS_KEY = 'csr_inputs'
EMBEDDING_PARAM_NAME = embed.EMBEDDING_PARAM_NAME


@dataclasses.dataclass
class EmbeddingSpec:
  """Sparsecore embedding spec.

  Attributes:
    input_dim: The cardinality of the input feature or size of its vocabulary.
    embedding_dim: The length of each embedding vector.
    max_sequence_length: An optional maximum sequence length. If set, the looked
      up embeddings will not be aggregated over the sequence dimension.
      Otherwise the embeddings will be aggregated over the sequence dimension
      using the `combiner`. Defaults to None.
    combiner: The combiner to use to aggregate the embeddings over the sequence
      dimension. This is ignored when `max_sequence_length` is set. Allowed
      values are 'sum', 'mean', and 'sqrtn'. Defaults to 'mean'.
    initializer: The initializer to use for the embedding table. Defaults to
      truncated_normal(stddev=1 / sqrt(embedding_dim)) if not set.
    optimizer: An optional custom optimizer to use for the embedding table.
    weight_name: An optional weight feature name to use for performing a
      weighted aggregation on the output of the embedding lookup. Defaults to
      None.
  """

  input_dim: int
  embedding_dim: int
  max_sequence_length: int | None = None
  combiner: Literal['sum', 'mean', 'sqrtn'] = 'mean'
  initializer: jax.nn.initializers.Initializer | None = None
  optimizer: embedding_spec.OptimizerSpec | None = None
  weight_name: str | None = None

  def __post_init__(self):
    if self.max_sequence_length is not None and self.weight_name is not None:
      raise ValueError(
          '`max_sequence_length` and `weight_name` cannot both be set. Weighted'
          ' aggregation can only be performed when the embeddings are'
          ' aggregated over the sequence dimension.'
      )

  def __hash__(self):
    return id(self)


@dataclasses.dataclass
class SparsecoreEmbedder:
  """Sparsecore embedder.

  Attributes:
    specs: A mapping from feature name to embedding specs.
    optimizer: The default optimizer to use for the embedding variables.
    sharding_strategy: The sharding strategy to use for the embedding table.
      Defaults to 'MOD' sharding. See the sparsecore documentation for more
      details.
    num_sc_per_device: The number of sparsecores per Jax device. By default, a
      fixed mapping is used to determine this based on device 0. This may fail
      on newer TPU architectures if the mapping is not updated of if device 0 is
      not a TPU device with a sparsecore.
    static_buffer_size_multiplier: The multiplier to use for the static buffer
      size. Defaults to 256.

  Example usage:
  ```python
  class DLRMModel(nn.Module):
    # The embedder must be a property of the Flax model and cannot be created
    # inside setup().
    embedder: sparsecore.SparsecoreEmbedder
    ...

    def setup(self):
      self.sparsecore_module = self.embedder.make_sparsecore_module()
      ...

    def __call__(self, inputs: Mapping[str, jax.Array]) -> jax.Array:
      embedding_activations = self.sparsecore_module(inputs)
      ...

  # Instantiate the model and the embedder.
  model = DLRMModel(embedder=embedder)

  # Create the eager preprocessor.
  preprocessor = model.embedder.make_preprocessor(global_batch_size)

  # Fetch and preprocess the inputs on CPU.
  inputs = ...

  # Post-process the sparse features into CSR format on CPU.
  processed_inputs = preprocessor(inputs)

  # Shard the inputs and put them on device.
  sharded_inputs = ...

  # Initialize and call the model on TPU inside JIT as usual.
  vars = model.init(jax.random.key(0), sharded_inputs)
  embedding_activations = model.apply(vars, sharded_inputs)
  ```
  """

  specs: Mapping[str, EmbeddingSpec]
  optimizer: embedding_spec.OptimizerSpec
  sharding_strategy: str = 'MOD'
  num_sc_per_device: int = dataclasses.field(
      default_factory=utils.num_sparsecores_per_device
  )
  static_buffer_size_multiplier: int = 256

  def __post_init__(self):
    self._feature_specs = None
    self._global_batch_size = None

  def _init_feature_specs(
      self, batch_size: int
  ) -> Mapping[str, embedding_spec.FeatureSpec]:
    """Returns the feature specs for sparsecore embedding lookup."""
    if self._feature_specs is not None:
      return self._feature_specs

    feature_specs = {}
    shared_tables = {}
    for name, spec in self.specs.items():
      if spec in shared_tables:
        table_spec = shared_tables[spec]
      else:
        table_spec = embedding_spec.TableSpec(
            vocabulary_size=spec.input_dim,
            embedding_dim=spec.embedding_dim,
            initializer=(
                spec.initializer
                or jax.nn.initializers.truncated_normal(
                    stddev=1.0 / jnp.sqrt(spec.embedding_dim)
                )
            ),
            optimizer=spec.optimizer or self.optimizer,
            combiner=spec.combiner,
            name=f'{name}_table',
        )
        shared_tables[spec] = table_spec

      if spec.max_sequence_length is not None:
        batch_dim = batch_size * spec.max_sequence_length
      else:
        batch_dim = batch_size

      feature_specs[name] = embedding_spec.FeatureSpec(
          name=name,
          table_spec=table_spec,
          input_shape=(batch_dim, 1),
          output_shape=(batch_dim, spec.embedding_dim),
      )

    embedding.auto_stack_tables(
        feature_specs,
        jax.device_count(),
        self.num_sc_per_device,
        stack_to_max_ids_per_partition=lambda n, bs: bs,
        stack_to_max_unique_ids_per_partition=lambda n, bs: bs,
    )
    embedding.prepare_feature_specs_for_training(
        feature_specs,
        jax.device_count(),
        self.num_sc_per_device,
    )
    self._feature_specs = feature_specs
    self._global_batch_size = batch_size
    return feature_specs

  def make_preprocessor(self, batch_size: int) -> Callable[..., Any]:
    """Returns a preprocessor for sparsecore embedding lookup."""
    feature_specs = self._init_feature_specs(batch_size)
    weights_names = {
        name: spec.weight_name
        for name, spec in self.specs.items()
        if spec.weight_name is not None
    }

    def _to_np(x: Any) -> np.ndarray:
      if isinstance(x, np.ndarray):
        return x
      if isinstance(x, (tf.SparseTensor, tf.RaggedTensor)):
        raise NotImplementedError(
            'Sparsecore embedding layer does not support sparse or'
            ' raggedtensors.'
        )
      if isinstance(x, tf.Tensor):
        return x.numpy()
      if isinstance(x, jax.Array):
        return jax.device_get(x)
      return np.array(x)

    def _preprocessor(inputs):
      if isinstance(inputs, tuple):
        inputs, *rem = inputs
      else:
        rem = None

      sparse_features = set()
      features = {}
      weights = {}
      for key in feature_specs:
        features[key] = _to_np(inputs[key])
        sparse_features.add(key)
        if key in weights_names:
          weights[key] = _to_np(inputs[weights_names[key]])
          sparse_features.add(weights_names[key])
        else:
          weights[key] = np.ones_like(features[key])

      csr_inputs = embed.EmbeddingLookups(
          *embedding.preprocess_sparse_dense_matmul_input(
              features=features,
              features_weights=weights,
              feature_specs=feature_specs,
              local_device_count=jax.local_device_count(),
              global_device_count=jax.device_count(),
              num_sc_per_device=self.num_sc_per_device,
              sharding_strategy=self.sharding_strategy,
              static_buffer_size_multiplier=self.static_buffer_size_multiplier,
              allow_id_dropping=False,
          )[:-1]
      )
      processed_inputs = {
          k: v for k, v in inputs.items() if k not in sparse_features
      }
      processed_inputs[CSR_INPUTS_KEY] = csr_inputs

      if rem is not None:
        processed_inputs = (processed_inputs, *rem)
      return processed_inputs

    return _preprocessor

  def make_sparsecore_module(self, **kwargs) -> _SparsecoreEmbed:
    """Returns the sparsecore embedding layer."""
    if self._feature_specs is None or self._global_batch_size is None:
      raise ValueError(
          'The feature specs are not initialized. Make sure to call'
          ' `make_preprocessor` before calling `sparsecore_layer`.'
      )

    def _key(k: str | tuple[str, str]) -> str:
      return k[0] if isinstance(k, tuple) else k

    return _SparsecoreEmbed(
        feature_specs=self._feature_specs,
        global_batch_size=self._global_batch_size,
        sharding_axis=0,
        sharding_strategy=self.sharding_strategy,
        num_sc_per_device=self.num_sc_per_device,
        **kwargs,
    )


class _SparsecoreEmbed(nn.Module):
  """Sparsecore embedding layer."""

  feature_specs: embedding.Nested[embedding_spec.FeatureSpec]
  global_batch_size: int
  sharding_axis: str | int
  sharding_strategy: str
  num_sc_per_device: int

  @property
  def abstract_mesh(self) -> jax.sharding.AbstractMesh:
    abstract_mesh = jax.sharding.get_abstract_mesh()
    if not abstract_mesh.shape_tuple:
      raise ValueError(
          'No abstract mesh shape was set with `jax.sharding.use_mesh`. Make'
          ' sure to set the mesh when calling the sparsecore module.'
      )
    return abstract_mesh

  @property
  def sharding_axis_name(self) -> str:
    if isinstance(self.sharding_axis, int):
      return self.abstract_mesh.axis_names[self.sharding_axis]
    return self.sharding_axis

  @property
  def num_shards(self) -> int:
    return self.abstract_mesh.shape[self.sharding_axis_name]

  def setup(self):
    initializer = functools.partial(
        embedding.init_embedding_variables,
        table_specs=embedding.get_table_specs(self.feature_specs),
        global_sharding=jax.sharding.NamedSharding(
            self.abstract_mesh,
            jax.sharding.PartitionSpec(self.sharding_axis_name, None),
        ),
        num_sparsecore_per_device=self.num_sc_per_device,
        # We need to by-pass the mesh check to use the abstract mesh.
        bypass_mesh_check=True,
    )
    self.embedding_table = self.param(
        name=EMBEDDING_PARAM_NAME,
        init_fn=_with_sparsecore_layout(
            initializer, (self.sharding_axis_name,), self.abstract_mesh
        ),
    )

  def __call__(
      self, inputs: Mapping[str, jax.Array]
  ) -> embedding.Nested[jax.Array]:
    """Computes the embedding activations.

    Args:
      inputs: A mapping from feature name to the feature values. The values must
        have been preprocessed by the preprocessor returned by
        `make_preprocessor`.

    Returns:
      The activations structure with the same structure as specs.
    """
    activations = embedding_ops.sparsecore_lookup(
        embedding_ops.SparsecoreParams(
            feature_specs=self.feature_specs,
            abstract_mesh=self.abstract_mesh,
            data_axes=(self.sharding_axis_name,),
            embedding_axes=(self.sharding_axis_name, None),
            sharding_strategy=self.sharding_strategy,
        ),
        self.embedding_table,
        inputs[CSR_INPUTS_KEY],
    )

    # Reshape the activations if the batch size is not the same as the global
    # batch size.
    def _maybe_reshape_activation(activation: jax.Array) -> jax.Array:
      if activation.shape[0] != self.global_batch_size:
        return jnp.reshape(
            activation,
            (
                self.global_batch_size,
                activation.shape[0] // self.global_batch_size,
                activation.shape[1],
            ),
        )
      return activation

    return jax.tree.map(_maybe_reshape_activation, activations)


class _WithSparseCoreLayout(nn.Partitioned[A]):

  def get_sharding(self, _):
    assert self.mesh is not None
    return embed.Layout(
        embed.DeviceLocalLayout(major_to_minor=(0, 1), _tiling=((8,),)),
        jax.sharding.NamedSharding(self.mesh, self.get_partition_spec()),
    )


def _with_sparsecore_layout(
    fn: Callable[..., Any],
    names: typing.LogicalNames,
    abstract_mesh: jax.sharding.AbstractMesh,
):
  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    return _WithSparseCoreLayout(fn(*args, **kwargs), names, mesh=abstract_mesh)  # pytype: disable=wrong-arg-types

  return wrapper

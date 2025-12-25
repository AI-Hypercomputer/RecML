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
  from jax_tpu_embedding.sparsecore.lib.nn import table_stacking
  from jax_tpu_embedding.sparsecore.utils import utils
  # pylint: enable=g-import-not-at-top

A = TypeVar('A')
CSR_INPUTS_KEY = 'csr_inputs'
EMBEDDING_PARAM_NAME = 'sc_embedding_variables'
OptimizerSpec = Any


def _num_sparsecores_per_device() -> int:
  """Returns the number of sparsecores per tensorcore device."""
  return utils.num_sparsecores_per_device()


# TODO(aahil): This should be common between Keras, Flax, NNX.
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
  optimizer: OptimizerSpec | None = None
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
class SparsecoreConfig:
  """Sparsecore embedding configuration.

  Attributes:
    specs: A mapping from feature name to embedding specs.
    optimizer: The default optimizer to use for the embedding variables.
    sharding_axis: The axis to use for sharding the embedding table. Can be
      either an integer mesh axis index or a string mesh axis name. Defaults to
      axis 0.
    sharding_strategy: The sharding strategy to use for the embedding table.
      Defaults to 'MOD' sharding. See the sparsecore documentation for more
      details.
    allow_id_dropping: Whether to allow dropping of IDs that do not fit within
      the XLA buffers allocated for each partition. Defaults to False.
    num_sc_per_device: The number of sparsecores per Jax device. By default, a
      fixed mapping is used to determine this based on device 0. This may fail
      on newer TPU architectures if the mapping is not updated of if device 0 is
      not a TPU device with a sparsecore.
    max_ids_per_partition_fn: A function that accepts the name of the table and
      its inputs size and returns the maximum number of IDs to process per
      partition. Defaults to the size of the inputs.
    max_unique_ids_per_partition_fn: A function that accepts the name of the
      table and its inputs size and returns the maximum number of unique IDs to
      on each partition. Defaults to the size of the inputs.
    local_device_count: The number of Jax devices in the local process. Defaults
      to `jax.local_device_count`.
    global_device_count: The number of Jax devices in the global process.
      Defaults to `jax.device_count`.
    num_sc_per_device: The number of sparsecores per Jax device. If not set,
      tries to fetch it from a fixed mapping.

  Example usage:
  ```python
  class DLRMModel(nn.Module):
    # The config must be a property of the Flax model and cannot be created
    # inside setup().
    sparsecore_config: sparsecore.SparsecoreConfig
    ...

    def setup(self):
      self.sparsecore_module = SparsecoreEmbed(self.sparsecore_config)
      ...

    def __call__(self, inputs: Mapping[str, jax.Array]) -> jax.Array:
      embedding_activations = self.sparsecore_module(inputs)
      ...

  # Instantiate the model and the embedder.
  model = DLRMModel(sparsecore_config, ...)

  # Create the eager preprocessor.
  preprocessor = SparsecorePreprocessor(sparsecore_config, global_batch_size)

  # Fetch and preprocess the inputs on CPU.
  inputs = ...

  # Post-process the sparse features into CSR format on CPU.
  processed_inputs = preprocessor(inputs)

  # Shard the inputs and put them on device.
  sharded_inputs = ...

  # Initialize and call the model on TPU inside JIT as usual.
  vars = model.init(jax.random.key(0), ...)
  embedding_activations = model.apply(vars, sharded_inputs)
  ```
  """

  specs: Mapping[str, EmbeddingSpec]
  optimizer: OptimizerSpec
  sharding_axis: str | int = 0
  sharding_strategy: str = 'MOD'
  allow_id_dropping: bool = False

  # TODO(aahil): Come up with better defaults / heuristics here.
  max_ids_per_partition_fn: Callable[[str, int], int] = dataclasses.field(
      default=lambda n, bs: bs
  )
  max_unique_ids_per_partition_fn: Callable[[str, int], int] = (
      dataclasses.field(default=lambda n, bs: bs)
  )

  # Optional device information.
  local_device_count: int = dataclasses.field(
      default_factory=jax.local_device_count
  )
  global_device_count: int = dataclasses.field(default_factory=jax.device_count)
  num_sc_per_device: int = dataclasses.field(
      default_factory=_num_sparsecores_per_device
  )

  _feature_specs: Mapping[str, embedding_ops.FeatureSpec] | None = (
      dataclasses.field(init=False, default=None)
  )
  _global_batch_size: int | None = dataclasses.field(init=False, default=None)

  @property
  def feature_specs(self) -> Mapping[str, embedding_ops.FeatureSpec]:
    """Returns the feature specs for sparsecore embedding lookup."""
    if self._feature_specs is None:
      raise ValueError(
          'The feature specs are not initialized. Make sure to call'
          ' `init_feature_specs` before accessing the'
          ' feature specs.'
      )
    return self._feature_specs

  @property
  def global_batch_size(self) -> int:
    """Returns the global batch size for sparsecore embedding lookup."""
    if self._global_batch_size is None:
      raise ValueError(
          'The global batch size is not initialized. Make sure to call'
          ' `init_feature_specs` before accessing the'
          ' global batch size.'
      )
    return self._global_batch_size

  def init_feature_specs(self, batch_size: int):
    """Creates the feature specs for sparsecore embedding lookup."""
    if self._feature_specs is not None and self._global_batch_size is not None:
      if batch_size != self._global_batch_size:
        raise ValueError(
            'The batch size is already initialized to'
            f' {self._global_batch_size}. It cannot be changed to'
            f' {batch_size}.'
        )
      return

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
        self.global_device_count,
        self.num_sc_per_device,
        stack_to_max_ids_per_partition=self.max_ids_per_partition_fn,
        stack_to_max_unique_ids_per_partition=self.max_unique_ids_per_partition_fn,
    )
    embedding.prepare_feature_specs_for_training(
        feature_specs,
        self.global_device_count,
        self.num_sc_per_device,
    )
    self._feature_specs = feature_specs
    self._global_batch_size = batch_size


@dataclasses.dataclass
class SparsecorePreprocessor:
  """Preprocessor for sparsecore embedding lookup.

  Attributes:
    sparsecore_config: The sparsecore config used to create the tables.
    global_batch_size: The global batch size across all devices to partition the
      inputs across.
    _batch_number: The batch number for preprocessing, incremented on each call.
  """

  sparsecore_config: SparsecoreConfig
  global_batch_size: int
  _batch_number: int = dataclasses.field(init=False, default=0)

  def __post_init__(self):
    self.sparsecore_config.init_feature_specs(self.global_batch_size)

  def __call__(
      self, inputs: Mapping[str, Any] | tuple[Mapping[str, Any], ...]
  ) -> Mapping[str, Any] | tuple[Mapping[str, Any], ...]:
    """Returns a preprocessor for sparsecore embedding lookup."""

    def _to_np(x: Any) -> np.ndarray:
      if isinstance(x, np.ndarray):
        return x
      if isinstance(x, (tf.SparseTensor, tf.RaggedTensor)):
        raise NotImplementedError(
            'Sparsecore embedding layer does not support sparse or'
            ' ragged tensors yet.'
        )
      if isinstance(x, tf.Tensor):
        return x.numpy()  # pylint: disable=attribute-error
      if isinstance(x, jax.Array):
        return jax.device_get(x)
      return np.array(x)

    if isinstance(inputs, tuple):
      rem = inputs[1:]
      inputs = inputs[0]
    else:
      rem = None

    sparse_features = set()
    features = {}
    weights = {}
    for key in self.sparsecore_config.feature_specs:
      features[key] = _to_np(inputs[key])
      sparse_features.add(key)
      if self.sparsecore_config.specs[key].weight_name is not None:
        weights[key] = _to_np(
            inputs[self.sparsecore_config.specs[key].weight_name]
        )
        sparse_features.add(self.sparsecore_config.specs[key].weight_name)
      else:
        weights[key] = np.ones_like(features[key])

      if self.sparsecore_config.specs[key].max_sequence_length is not None:
        features[key] = np.reshape(features[key], (-1, 1))
        if weights[key] is not None:
          weights[key] = np.reshape(weights[key], (-1, 1))

    self._batch_number += 1
    csr_inputs, _ = embedding.preprocess_sparse_dense_matmul_input(
        features=features,
        features_weights=weights,
        feature_specs=self.sparsecore_config.feature_specs,
        local_device_count=self.sparsecore_config.local_device_count,
        global_device_count=self.sparsecore_config.global_device_count,
        num_sc_per_device=self.sparsecore_config.num_sc_per_device,
        sharding_strategy=self.sparsecore_config.sharding_strategy,
        allow_id_dropping=self.sparsecore_config.allow_id_dropping,
        batch_number=self._batch_number,
    )

    processed_inputs = {
        k: v for k, v in inputs.items() if k not in sparse_features
    }
    processed_inputs[CSR_INPUTS_KEY] = csr_inputs

    if rem is not None:
      processed_inputs = (processed_inputs, *rem)
    return processed_inputs


class SparsecoreEmbed(nn.Module):
  """Sparsecore embedding layer.

  Attributes:
    sparsecore_config: A sparsecore config specifying how to create the tables.
    mesh: The mesh to use for the embedding layer. If not provided, the global
      mesh set by `jax.set_mesh` will be used. If neither is set, an
      error will be raised.
  """

  sparsecore_config: SparsecoreConfig
  mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None

  def get_mesh(self) -> jax.sharding.Mesh | jax.sharding.AbstractMesh:
    if self.mesh is not None:
      return self.mesh
    abstract_mesh = jax.sharding.get_abstract_mesh()
    if not abstract_mesh.shape_tuple:
      raise ValueError(
          'No abstract mesh shape was set with `jax.set_mesh`. Make'
          ' sure to set the mesh when calling the sparsecore module.'
      )
    return abstract_mesh

  def get_sharding_axis(
      self, mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh
  ) -> str:
    if isinstance(self.sparsecore_config.sharding_axis, int):
      return mesh.axis_names[self.sparsecore_config.sharding_axis]
    return self.sparsecore_config.sharding_axis

  def setup(self):
    mesh = self.get_mesh()
    sharding_axis_name = self.get_sharding_axis(mesh)

    initializer = functools.partial(
        embedding.init_embedding_variables,
        table_specs=embedding.get_table_specs(
            self.sparsecore_config.feature_specs
        ),
        global_sharding=jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(sharding_axis_name, None)
        ),
        num_sparsecore_per_device=self.sparsecore_config.num_sc_per_device,
        # We need to by-pass the mesh check to allow using an abstract mesh.
        bypass_mesh_check=isinstance(mesh, jax.sharding.AbstractMesh),
    )
    self.embedding_table = self.param(
        name=EMBEDDING_PARAM_NAME,
        init_fn=embed.with_sparsecore_layout(
            initializer, (sharding_axis_name,), mesh  # type: ignore
        ),
    )

  def __call__(
      self, inputs: Mapping[str, jax.Array]
  ) -> embedding_ops.Nested[jax.Array]:
    """Computes the embedding activations.

    Args:
      inputs: A mapping from feature name to the feature values. The values must
        have been preprocessed by the preprocessor returned by
        `make_preprocessor`.

    Returns:
      The activations structure with the same structure as specs.
    """
    mesh = self.get_mesh()
    sharding_axis_name = self.get_sharding_axis(mesh)
    activations = embedding_ops.sparsecore_lookup(
        embedding_ops.SparsecoreParams(
            feature_specs=self.sparsecore_config.feature_specs,
            mesh=mesh,
            data_axes=(sharding_axis_name,),
            embedding_axes=(sharding_axis_name, None),
            sharding_strategy=self.sparsecore_config.sharding_strategy,
        ),
        self.embedding_table,
        inputs[CSR_INPUTS_KEY],
    )

    # Reshape the activations if the batch size is not the same as the global
    # batch size.
    def _maybe_reshape_activation(activation: jax.Array) -> jax.Array:
      if activation.shape[0] != self.sparsecore_config.global_batch_size:
        return jnp.reshape(
            activation,
            (
                self.sparsecore_config.global_batch_size,
                activation.shape[0] // self.sparsecore_config.global_batch_size,
                activation.shape[1],
            ),
        )
      return activation

    return jax.tree.map(_maybe_reshape_activation, activations)


def gather_table(
    sparsecore_config: SparsecoreConfig,
    sc_params: Mapping[str, embedding.EmbeddingVariables],
    name: str,
) -> jax.Array:
  """Gathers a table from a stacked table on device.

  Args:
    sparsecore_config: The sparsecore config used to create the tables.
    sc_params: A mapping from table name to the embedding variables. This must
      consist of the Flax variables corresponding to the sparsecore module
      returned by `SparsecoreEmbed`.
    name: The name of the table to gather.

  Returns:
    The unstacked and unsharded embedding table on device.
  """

  embedding_specs = embedding.create_proto_from_feature_specs(
      feature_specs=sparsecore_config.feature_specs,
      global_device_count=sparsecore_config.global_device_count,
      num_sparsecore_per_device=sparsecore_config.num_sc_per_device,
  )

  for stacked_table_spec in embedding_specs.stacked_table_specs:
    table_specs = {ts.table_name: ts for ts in stacked_table_spec.table_specs}
    if f'{name}_table' not in table_specs:
      continue

    table_spec = table_specs[f'{name}_table']

    num_sparsecores = stacked_table_spec.num_sparsecores
    stacked_embedding_dim = stacked_table_spec.stack_embedding_dim
    row_offset = table_spec.row_offset_in_shard
    chunk_size = table_spec.padded_vocab_size // num_sparsecores
    rotation = table_spec.shard_rotation
    vocab_size = table_spec.vocab_size
    embedding_dim = table_spec.embedding_dim

    stacked_table = sc_params[stacked_table_spec.stack_name]
    stacked_table_3d = jnp.reshape(
        stacked_table.table, (num_sparsecores, -1, stacked_embedding_dim)
    )

    shards = stacked_table_3d[:, row_offset : row_offset + chunk_size, :]

    # Undo the shard rotation (note '-' for reverse direction)
    shards = jnp.roll(shards, -rotation, axis=0)

    # Undo the mod sharding
    un_mod_shard = shards.transpose((1, 0, 2))

    # Remove the first dimension
    table = un_mod_shard.reshape(-1, stacked_embedding_dim)

    # Remove paddings
    table = table[:vocab_size, :embedding_dim]

    return table

  raise ValueError(
      f'Table {name} not found in feature specs'
      f' {sparsecore_config.feature_specs}.'
  )


def fetch_tables(
    sparsecore_config: SparsecoreConfig,
    sc_params: Mapping[str, embedding.EmbeddingVariables],
    as_tf_variables: bool = True,
    donate: bool = True,
) -> Mapping[str, jax.Array] | Mapping[str, tf.Variable]:
  """Unstacks and unshards the stacked tables and fetches them to the host.

  Args:
    sparsecore_config: The sparsecore config used to create the tables.
    sc_params: A mapping from table name to the embedding variables. This must
      consist of the Flax variables corresponding to the sparsecore module
      returned by `make_sparsecore_module`.
    as_tf_variables: Whether to return the tables as TF variables. Defaults to
      False.
    donate: Whether to donate the stacked tables, i.e. remove them from device
      HBM, to save memory. Defaults to True.

  Returns:
    A mapping from table name to the unstacked and unsharded embedding table
    on the host.
  """
  tables = table_stacking.unstack_and_unshard_stacked_tables(
      stacked_tables={k: v.table for k, v in sc_params.items()},
      embedding_specs=embedding.create_proto_from_feature_specs(
          feature_specs=sparsecore_config.feature_specs,
          global_device_count=sparsecore_config.global_device_count,
          num_sparsecore_per_device=sparsecore_config.num_sc_per_device,
      ),
      donate=donate,
  )
  tables = {k.removesuffix('_table'): v for k, v in tables.items()}
  if as_tf_variables:
    tables = {
        k: tf.Variable(tf.convert_to_tensor(v)) for k, v in tables.items()
    }
  return tables


def cpu_lookup(
    sparsecore_config: SparsecoreConfig,
    tables: Mapping[str, tf.Variable],
    inputs: Mapping[str, tf.Tensor | tf.SparseTensor | tf.RaggedTensor],
) -> Mapping[str, tf.Tensor]:
  """Performs embedding lookups on the host.

  Args:
    sparsecore_config: The sparsecore config used to create the tables.
    tables: A mapping of the embedding tables on the host. This must in the same
      format as the output of `fetch_tables`.
    inputs: A mapping of the input features on the host. This must be in the
      same format as the input to the preprocessor created by
      `make_preprocessor`.

  Returns:
    A mapping of the embedding activations on the host. This has the same
    structure as the output of the sparsecore module.
  """

  activations = {}
  for name, spec in sparsecore_config.specs.items():
    feature = inputs[name]
    weight = inputs[spec.weight_name] if spec.weight_name is not None else None
    if isinstance(feature, tf.Tensor):
      activation = tf.nn.embedding_lookup(tables[name], feature)

      if spec.max_sequence_length is None:
        activation = _reduce(activation, weight, spec.combiner)

      activations[name] = activation
    else:
      raise NotImplementedError(
          'Sparsecore embedding layer does not support sparse or ragged'
          ' tensors yet.'
      )

  return activations


def _reduce(
    inputs: tf.Tensor | tf.RaggedTensor,
    weights: tf.Tensor | tf.RaggedTensor | None = None,
    combiner: Literal['sum', 'mean', 'sqrtn'] = 'mean',
) -> tf.Tensor:
  """Performs a weighted reduction across the penultimate dimension of a tensor.

  Args:
    inputs: A dense or ragged tensor of shape [D_1, ..., D_N] to reduce.
    weights: Optional weights to apply to the reduction. If given, the
      dimensions of the weights must be [D_1, ..., D_N][:`axis` + 1]. Note that
      the if the inputs have ragged dimensions, the weights must have the same
      ragged dimensions.
    combiner: The combiner to use for the reduction. Can be one of ['sum',
      'mean', 'sqrtn'].

  Returns:
    The reduced inputs of rank N - 1.
  """
  if weights is not None:
    weights = tf.expand_dims(tf.cast(weights, inputs.dtype), axis=-1)
    inputs = inputs * weights

  out = tf.reduce_sum(inputs, axis=-2)
  if combiner == 'mean':
    weight_sum = tf.reduce_sum(weights, axis=-2)
    out = tf.math.divide_no_nan(out, weight_sum)
  elif combiner == 'sqrtn':
    weight_sum = tf.math.sqrt(tf.reduce_sum(weights**2, axis=-2))
    out = tf.math.divide_no_nan(out, weight_sum)
  else:
    raise ValueError("`combiner` must be one of ['mean', 'sqrtn', 'sum'].")

  return out

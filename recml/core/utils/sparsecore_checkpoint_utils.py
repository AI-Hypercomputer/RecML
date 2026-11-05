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
"""Checkpoint utilities for the SparseCore embedding tables of Keras models.

`jax_tpu_embedding` stacks, pads and mod-shards the SparseCore embedding tables
of `keras_rs.layers.DistributedEmbedding` (and their optimizer slots) for the
total number of SparseCores of the TPU topology (`global_device_count *
num_sc_per_device`), so both the shape and the row layout of a stacked variable
change across topologies (e.g. `glp=4x4` train vs. `glp=2x2` eval). This module
saves the topology of the stacked variables with the checkpoint, and reshards
them across topologies, so that checkpoints can be restored on another topology
than the one they were saved on.

SparseCore support:
- Supported: models with and without SparseCore-sharded embedding tables
  (tables placed on SparseCore and stacked by `jax_tpu_embedding`, see
  `has_sparsecore_tables`).
  - SparseCore-sharded tables are resharded when restored on another TPU
    topology than the one they were saved on (e.g. `glp=4x4` train vs.
    `glp=2x2` eval). Checkpoints with sparsecore-sharded tables but saved
    without `sparsecore_topology` can only be restored on the saving topology,
    and raise `ValueError` on another one.
  - Models without SparseCore-sharded tables, including embedding tables kept
    in HBM on TPUs with SparseCores, are saved and restored by Orbax as usual:
    they do not write `sparsecore_topology` on save and skip
    `prepare_sparsecore_restore` on restore.
- Not supported: restoring a checkpoint of a model with SparseCore-sharded
  tables into a model without them, or vice versa, since stacked SparseCore
  tables are not interchangeable with HBM `Embedding` variables. For example, a
  side eval on a TPU without SparseCores (e.g. `vlp`) cannot restore the
  checkpoints of a training job with SparseCore-sharded tables (e.g. on `glp`).
  To transfer only the shared dense weights between such V3 checkpoints, use
  `keras_utils.restore_partial_checkpoint` with the dense trainable variables.

TODO(b/576848327): Remove this module (and its `jax_tpu_embedding` dependency)
once keras_rs `DistributedEmbedding` supports resharding SparseCore tables and
optimizer slots across TPU topologies.
"""

from collections.abc import Mapping, Sequence
import copy
import dataclasses
import math
import os
from typing import Any, NamedTuple, cast

from absl import logging
from etils import epath
import jax
from jax.experimental import layout as jax_layout
from jax_tpu_embedding.sparsecore.lib.nn import embedding as jte_embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.nn import table_stacking as jte_table_stacking
import keras
import numpy as np
import orbax.checkpoint as ocp

# Checkpoint item keys matching `keras_utils` (kept private here to avoid a
# circular dependency with `keras_utils`).
_STATE_CHECKPOINT_KEY = "state"
_TRAINABLE_VARIABLES_KEY = "trainable_variables"
_SHARED_VARIABLE_MAP_KEY = "shared_variable_map"
# Values of `keras_utils.CheckpointVersion` (a `StrEnum`) supported here.
_CHECKPOINT_VERSION_V2 = "v2"
_CHECKPOINT_VERSION_V3 = "v3"

# Optional checkpoint item key storing the source SparseCore topology
# (`num_shards` and `num_sc_per_device` per stacked table; see
# `sparsecore_topology`). Saved only for models with SparseCore tables and read
# on restore to unshard/unstack and reshard stacked tables when restoring on a
# different TPU topology.
# TODO(b/576848327): Remove once `keras_rs` `DistributedEmbedding` natively
# supports cross-topology resharding.
SPARSECORE_TOPOLOGY_KEY = "sparsecore_topology"


def settings_in_stack(
    table_stack: Sequence[embedding_spec.TableSpec],
    num_shards: int,
    num_sc_per_device: int,
) -> Mapping[str, embedding_spec.TableSettingInStack]:
  """Returns each table's `TableSettingInStack` for a SparseCore topology.

  Recomputes the row offsets and shard rotations that
  `jte_table_stacking.stack_tables` assigns to the tables of a stack for
  `num_shards` SparseCores, keeping the table order of the existing stack.
  `keras_rs.layers.DistributedEmbedding` stacks tables with the default
  `rotation=None`, which `stack_tables` resolves to `num_sc_per_device`.

  Args:
    table_stack: The stacked `TableSpec`s of one stack (for any topology).
    num_shards: The total number of SparseCores of the topology.
    num_sc_per_device: The number of SparseCores per chip of the topology.

  Returns:
    Table name to its setting in the stack for the topology.
  """
  ordered = sorted(
      table_stack, key=lambda spec: spec.setting_in_stack.row_offset_in_shard
  )
  first_setting = ordered[0].setting_in_stack
  _, table_to_padded_vocab_size = (
      jte_table_stacking.round_up_dim_and_vocab_size(
          {spec.name: spec for spec in ordered}, num_shards
      )
  )
  return jte_table_stacking.compute_table_to_setting_in_stack(
      stack_name=first_setting.stack_name,
      table_names=[spec.name for spec in ordered],
      padded_embedding_dim=first_setting.padded_embedding_dim,
      table_to_padded_vocab_size=table_to_padded_vocab_size,
      num_shards=num_shards,
      rotation=num_sc_per_device,
  )


def reshard_stacked_array(
    stacked_array: jax.Array,
    table_stack: Sequence[embedding_spec.TableSpec],
    *,
    src_num_shards: int,
    src_num_sc_per_device: int,
    dst_num_shards: int,
    dst_num_sc_per_device: int,
    out_sharding: jax.sharding.Sharding | jax_layout.Format,
) -> jax.Array:
  """Reshards a stacked SparseCore table or slot tensor across topologies.

  Unstacks `stacked_array` (laid out for the source topology) into its logical
  tables with `jte_table_stacking.unshard_and_unstack_tables`, then restacks
  them for the destination topology of `table_stack` with
  `jte_table_stacking.stack_and_shard_tables`, all in one `jax.jit`
  computation. XLA shards the intermediate tables (whose vocab sizes need not
  be divisible by the device count), so they stay distributed across the
  devices of all hosts, and every process must call this in lockstep.

  `stacked_array` is consumed to reduce the peak memory usage: it is donated
  when its per-device shards have the size of the result's (otherwise XLA can't
  reuse its buffers), and deleted after the computation otherwise.

  Args:
    stacked_array: The `[stack_vocab_size, stack_embedding_dim]` stacked tensor
      as saved for the source topology, in SparseCore shard order.
    table_stack: The stacked `TableSpec`s of the stack for the destination
      topology.
    src_num_shards: The total number of SparseCores of the source topology.
    src_num_sc_per_device: The number of SparseCores per chip of the source
      topology.
    dst_num_shards: The total number of SparseCores of the destination topology.
    dst_num_sc_per_device: The number of SparseCores per chip of the destination
      topology.
    out_sharding: The sharding of the result, or its `jax_layout.Format` (e.g.
      the SparseCore `backend_layout` of `keras_rs`, so that the result already
      has the device layout of the SparseCore variables).

  Returns:
    The stacked tensor laid out for the destination topology.

  Raises:
    ValueError: If the stacking layout of `table_stack` differs from the one
      recomputed for the destination topology, e.g. because the
      jax_tpu_embedding stacking scheme changed.
  """
  # TODO(b/576848327): Replace with keras_rs's cross-topology resharding once
  # supported.
  dst_settings = settings_in_stack(
      table_stack, dst_num_shards, dst_num_sc_per_device
  )
  for spec in table_stack:
    if spec.setting_in_stack != dst_settings[spec.name]:
      raise ValueError(
          f"Unexpected SparseCore stacking layout for table '{spec.name}':"
          f" {spec.setting_in_stack} != {dst_settings[spec.name]}. Cannot"
          " reshard the checkpoint across TPU topologies."
      )
  src_settings = settings_in_stack(
      table_stack, src_num_shards, src_num_sc_per_device
  )
  stacked_spec = table_stack[0].stacked_table_spec
  stack_name = stacked_spec.stack_name
  src_table_stack = []
  for spec in table_stack:
    src_spec = copy.copy(spec)
    src_spec.setting_in_stack = src_settings[spec.name]
    src_table_stack.append(src_spec)

  def _reshard(stacked: jax.Array) -> jax.Array:
    tables = jte_table_stacking.unshard_and_unstack_tables(
        src_table_stack, {stack_name: stacked}, src_num_shards
    )
    restacked = jte_table_stacking.stack_and_shard_tables(
        list(table_stack), tables, dst_num_shards, pad_value=0.0
    )[stack_name]
    restacked = cast(jax.Array, restacked)
    return restacked.reshape(-1, stacked.shape[-1]).astype(stacked.dtype)

  dst_sharding = (
      out_sharding.sharding
      if isinstance(out_sharding, jax_layout.Format)
      else out_sharding
  )
  src_shape = tuple(stacked_array.shape)
  dst_shape = (stacked_spec.stack_vocab_size, src_shape[-1])
  # XLA only reuses donated buffers of the size of the output buffers, and
  # warns and keeps the others alive.
  is_jax_array = isinstance(stacked_array, jax.Array)
  can_donate = (
      is_jax_array
      and isinstance(dst_sharding, jax.sharding.Sharding)
      and math.prod(stacked_array.sharding.shard_shape(src_shape))
      == math.prod(dst_sharding.shard_shape(dst_shape))
  )
  resharded = jax.jit(
      _reshard,
      out_shardings=out_sharding,
      donate_argnums=0 if can_donate else (),
  )(stacked_array)
  if is_jax_array and not stacked_array.is_deleted():
    stacked_array.delete()
  return resharded


def row_sharding(
    sharding: jax.sharding.NamedSharding, num_rows: int
) -> jax.sharding.NamedSharding:
  """Returns a sharding of the rows of an array over the devices of `sharding`.

  `sharding` must shard the rows over all its devices. If `num_rows` is not
  divisible by the device count, the rows are split into
  `gcd(num_rows, device_count)` shards, each replicated on consecutive devices,
  instead of replicating the whole array on every device.

  Args:
    sharding: The sharding of the rows over all its devices.
    num_rows: The number of rows of the array.

  Returns:
    `sharding` if `num_rows` is divisible by its device count, else a sharding
    of the rows over a `("rows", "replicas")` mesh of the same devices.
  """
  devices = sharding.mesh.devices.reshape(-1)
  num_row_shards = math.gcd(num_rows, devices.size)
  if num_row_shards == devices.size:
    return sharding
  mesh = jax.sharding.Mesh(
      devices.reshape(num_row_shards, -1), ("rows", "replicas")
  )
  return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("rows"))


class SparseCoreLayerState(NamedTuple):
  """The SparseCore state of a `keras_rs.layers.DistributedEmbedding` layer.

  Attributes:
    config: The SparseCore lookup config of the layer (with `feature_specs`,
      `mesh` and `num_sc_per_device`).
    layout: The SparseCore `TensorLayout` of the stacked variables.
    table_and_slot_variables: The stacked table and slot variables by stack
      name.
  """

  config: Any
  layout: keras.distribution.TensorLayout
  table_and_slot_variables: Mapping[str, Any]


def sparsecore_layer_states(model: keras.Model) -> list[SparseCoreLayerState]:
  """Returns the SparseCore state of the SparseCore layers of `model`.

  Matches the JAX `keras_rs.layers.DistributedEmbedding` layers with tables on
  SparseCore, including the one wrapped by `SparseCoreEmbeddings`. Only these
  layers set `_sparsecore_built`, so models on other backends have none.

  Args:
    model: The model being saved or restored. Must be built, along with all of
      its layers.

  Returns:
    The SparseCore state of each built SparseCore layer.

  Raises:
    ValueError: If `model` or any of its layers is not built, or if a built
      SparseCore layer is missing its SparseCore state.
  """
  states = []
  # pylint: disable-next=protected-access
  for layer in model._flatten_layers(include_self=True):
    if not getattr(layer, "built", False) or not getattr(
        layer, "_sparsecore_built", True
    ):
      raise ValueError(
          "All layers of `model` must be built before inspecting SparseCore"
          f" state, but layer '{layer.name}' ({type(layer).__name__}) is not"
          " built."
      )
    if not getattr(layer, "_sparsecore_built", False):
      continue
    config = getattr(layer, "_config", None)
    layout = getattr(layer, "_sparsecore_layout", None)
    table_and_slot_variables = getattr(layer, "_table_and_slot_variables", None)
    if config is None or layout is None or table_and_slot_variables is None:
      missing = [
          name
          for name, value in (
              ("_config", config),
              ("_sparsecore_layout", layout),
              ("_table_and_slot_variables", table_and_slot_variables),
          )
          if value is None
      ]
      raise ValueError(
          f"SparseCore layer '{layer.name}' has `_sparsecore_built=True` but"
          f" is missing {missing}."
      )
    states.append(
        SparseCoreLayerState(config, layout, table_and_slot_variables)
    )
  return states


def has_sparsecore_tables(model: keras.Model) -> bool:
  """Returns whether `model` has embedding tables on SparseCore.

  Args:
    model: The model being saved or restored.

  Raises:
    ValueError: See `sparsecore_layer_states`.
  """
  return bool(sparsecore_layer_states(model))


def sparsecore_topology(model: keras.Model) -> dict[str, dict[str, int]]:
  """Returns the SparseCore topology of each stacked table of `model`.

  Saved with the checkpoint so that a restore on another TPU topology knows
  how the stacked tables (and their optimizer slots, which share the layout of
  their table) were sharded.

  Args:
    model: The model being saved.

  Returns:
    Stacked table variable path to `{"num_shards": ..., "num_sc_per_device":
    ...}`. Empty for models without SparseCore tables.

  Raises:
    ValueError: If two stacked tables have the same variable path (e.g. two
      SparseCore layers with the same name and stack names), or see
      `sparsecore_layer_states`.
  """
  topology = {}
  for state in sparsecore_layer_states(model):
    num_sc_per_device = int(state.config.num_sc_per_device)
    num_shards = int(state.config.mesh.devices.size) * num_sc_per_device
    for emb_vars in state.table_and_slot_variables.values():
      # V2 checkpoints store the variables in a list, so only this check
      # catches duplicate paths, whose topologies would overwrite each other.
      if emb_vars.table.path in topology:
        raise ValueError(
            "Duplicate SparseCore table variable path"
            f" '{emb_vars.table.path}'. Give the SparseCore layers unique"
            " names."
        )
      topology[emb_vars.table.path] = {
          "num_shards": num_shards,
          "num_sc_per_device": num_sc_per_device,
      }
  return topology


def sparsecore_topology_save_args(model: keras.Model) -> dict[str, Any]:
  """Returns the Orbax save args of `sparsecore_topology`, if any.

  Only models with SparseCore tables get the item, so the checkpoints of other
  models are unchanged.

  Args:
    model: The model being saved.

  Returns:
    `{SPARSECORE_TOPOLOGY_KEY: ocp.args.JsonSave(...)}` to merge into the Orbax
    `Composite` save args, or an empty dict for models without SparseCore
    tables.
  """
  topology = sparsecore_topology(model)
  if not topology:
    return {}
  return {
      SPARSECORE_TOPOLOGY_KEY: ocp.args.JsonSave(
          {SPARSECORE_TOPOLOGY_KEY: topology}
      )
  }


def saved_sparsecore_topology(
    checkpoint_path: str,
) -> dict[str, dict[str, int]] | None:
  """Returns the saved `sparsecore_topology`, or None if it wasn't saved.

  Args:
    checkpoint_path: Path to the checkpoint directory of one step.

  Returns:
    The `sparsecore_topology` saved with the checkpoint, or None if the
    checkpoint has no `SPARSECORE_TOPOLOGY_KEY` item (e.g. it was saved for a
    model without SparseCore tables).
  """
  topology_path = epath.Path(checkpoint_path) / SPARSECORE_TOPOLOGY_KEY
  if not topology_path.exists():
    return None
  checkpointer = ocp.Checkpointer(ocp.handlers.JsonCheckpointHandler())
  try:
    return checkpointer.restore(
        os.fspath(topology_path), args=ocp.args.JsonRestore()
    )[SPARSECORE_TOPOLOGY_KEY]
  finally:
    checkpointer.close()


def saved_trainable_metadata(
    model: keras.Model,
    checkpoint_path: str,
    version: str,
) -> tuple[dict[str, Any], dict[str, str]]:
  """Returns the checkpoint metadata of the model's trainable variables and aliases.

  Matches the saved trainable variables like the restore does: by path for V3
  checkpoints (following the aliases of shared variables), and by order for V2
  checkpoints. The checkpoint version must be passed explicitly (e.g. from the
  checkpoint manager or `keras_utils` version detection); it is never guessed
  from the structure of the saved metadata.

  Args:
    model: The model being restored.
    checkpoint_path: The resolved checkpoint directory path of one step.
    version: The `keras_utils.CheckpointVersion` of the checkpoint (a `StrEnum`,
      so its "v2"/"v3" string values are accepted too): `V3` for trainable
      variables keyed by path (saved by `KerasOrbaxCheckpointManagerV3`), or
      `V2` for trainable variables in a list (saved by
      `KerasOrbaxCheckpointManagerV2`). V1 checkpoints are not supported.

  Returns:
    A tuple of:
    - Variable path to its Orbax `ArrayMetadata`.
    - The V3 `shared_variable_map` (alias variable path to the canonical path it
      is stored under), or `{}` for V2 checkpoints or if none was saved.

  Raises:
    ValueError: If `version` is not V2 or V3, if the checkpoint metadata can't
      be read, if the checkpoint has no saved trainable variables, if the saved
      trainable variables don't have the layout of the checkpoint version, or
      if a V2 checkpoint has another number of trainable variables than the
      model. Without the metadata, the SparseCore tables can't be resharded,
      and restoring them as is would fail with a shape mismatch or silently
      load rows sharded for another topology.
  """
  if version not in (_CHECKPOINT_VERSION_V2, _CHECKPOINT_VERSION_V3):
    raise ValueError(
        f"Unsupported checkpoint version '{version}' for restoring the"
        " SparseCore tables across TPU topologies. Expected"
        f" '{_CHECKPOINT_VERSION_V2}' or '{_CHECKPOINT_VERSION_V3}'."
    )
  try:
    saved_trainable_meta, shared_var_map = _read_saved_trainable_meta(
        checkpoint_path, version
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    raise ValueError(
        f"Failed to read the metadata of the checkpoint in {checkpoint_path} to"
        " restore the SparseCore tables across TPU topologies."
    ) from e

  if saved_trainable_meta is None:
    raise ValueError(
        f"The checkpoint in {checkpoint_path} has no saved trainable variables"
        f" (no '{_STATE_CHECKPOINT_KEY}/{_TRAINABLE_VARIABLES_KEY}' item), so"
        " its SparseCore tables can't be restored across TPU topologies."
    )
  if version == _CHECKPOINT_VERSION_V3:
    return (
        _match_v3_trainable_metadata(
            model, saved_trainable_meta, shared_var_map, checkpoint_path
        ),
        shared_var_map,
    )
  else:
    return (
        _match_v2_trainable_metadata(
            model, saved_trainable_meta, checkpoint_path
        ),
        {},
    )


def _read_saved_trainable_meta(
    checkpoint_path: str, version: str
) -> tuple[Any | None, dict[str, str]]:
  """Reads the saved trainable variable metadata and shared variable map.

  Args:
    checkpoint_path: The resolved checkpoint directory path of one step.
    version: The `keras_utils.CheckpointVersion` of the checkpoint (V2 or V3).
      The shared variable map is only read for V3 checkpoints.

  Returns:
    A tuple of:
    - The metadata PyTree of the `trainable_variables` of the Orbax state item
      (a dict for V3 checkpoints, a list for V2 ones), or None if the
      checkpoint has no state item or no saved trainable variables.
    - The `shared_variable_map` of a V3 checkpoint (alias variable path to the
      canonical path it is stored under), or {} for V2 checkpoints, if it
      wasn't saved, or if there are no saved trainable variables.
  """
  state_dir = epath.Path(checkpoint_path) / _STATE_CHECKPOINT_KEY
  if not state_dir.exists():
    return None, {}
  saved_state_metadata = ocp.handlers.PyTreeCheckpointHandler().metadata(
      state_dir
  )
  saved_trainable_meta = saved_state_metadata.get(_TRAINABLE_VARIABLES_KEY)
  if saved_trainable_meta is None or version != _CHECKPOINT_VERSION_V3:
    return saved_trainable_meta, {}

  map_path = epath.Path(checkpoint_path) / _SHARED_VARIABLE_MAP_KEY
  if not map_path.exists():
    return saved_trainable_meta, {}
  checkpointer = ocp.Checkpointer(ocp.handlers.JsonCheckpointHandler())
  try:
    shared_var_map = checkpointer.restore(
        os.fspath(map_path), args=ocp.args.JsonRestore()
    )[_SHARED_VARIABLE_MAP_KEY]
  finally:
    checkpointer.close()
  return saved_trainable_meta, shared_var_map


def _match_v3_trainable_metadata(
    model: keras.Model,
    saved_trainable_meta: Any,
    shared_var_map: Mapping[str, str],
    checkpoint_path: str,
) -> dict[str, Any]:
  """Matches the saved trainable variables of a V3 checkpoint by path.

  Variables missing from the checkpoint are skipped. Shared variables are
  looked up under the canonical path they are stored under.

  Args:
    model: The model being restored.
    saved_trainable_meta: The saved trainable variable metadata, keyed by path.
    shared_var_map: Alias variable path to its canonical stored path.
    checkpoint_path: The checkpoint directory path, for error messages.

  Returns:
    Variable path to its Orbax `ArrayMetadata`.

  Raises:
    ValueError: If `saved_trainable_meta` is not keyed by path.
  """
  if not isinstance(saved_trainable_meta, Mapping):
    raise ValueError(
        f"Expected the V3 checkpoint in {checkpoint_path} to save its"
        " trainable variables keyed by path, but got"
        f" {type(saved_trainable_meta).__name__}."
    )
  meta_by_path = {}
  for var in model.trainable_variables:
    stored_path = var.path
    if stored_path not in saved_trainable_meta:
      stored_path = shared_var_map.get(var.path)
    if stored_path is not None and stored_path in saved_trainable_meta:
      meta_by_path[var.path] = saved_trainable_meta[stored_path]
  return meta_by_path


def _match_v2_trainable_metadata(
    model: keras.Model,
    saved_trainable_meta: Any,
    checkpoint_path: str,
) -> dict[str, Any]:
  """Matches the saved trainable variables of a V2 checkpoint by order.

  Args:
    model: The model being restored.
    saved_trainable_meta: The saved trainable variable metadata, in the order of
      `model.trainable_variables` when saved.
    checkpoint_path: The checkpoint directory path, for error messages.

  Returns:
    Variable path to its Orbax `ArrayMetadata`.

  Raises:
    ValueError: If `saved_trainable_meta` is not a list, or has another number
      of variables than the model.
  """
  if isinstance(saved_trainable_meta, Mapping) or not isinstance(
      saved_trainable_meta, Sequence
  ):
    raise ValueError(
        f"Expected the V2 checkpoint in {checkpoint_path} to save its"
        " trainable variables in a list, but got"
        f" {type(saved_trainable_meta).__name__}."
    )
  trainable_variables = model.trainable_variables
  if len(saved_trainable_meta) != len(trainable_variables):
    raise ValueError(
        f"The V2 checkpoint in {checkpoint_path} has"
        f" {len(saved_trainable_meta)} trainable variables, but the model has"
        f" {len(trainable_variables)}. V2 checkpoints are restored by variable"
        " order, so the model must have the same trainable variables."
    )
  return {
      var.path: meta
      for var, meta in zip(trainable_variables, saved_trainable_meta)
  }


class _RestoreValuePlaceholder:
  """Stands in for `keras.Variable._value` to restore a differently shaped value.

  Used by `prepare_sparsecore_restore` for SparseCore variables saved for
  another topology. The restore paths of `keras_utils` build the Orbax restore
  target from `var.value.shape`, `.dtype` and `.sharding`, so these attributes
  make Orbax restore the saved shape. They then call `var.value.delete()` to
  free each variable before restoring, which frees the replaced value. Until
  then, `abort_sparsecore_restore` can put the replaced value back.
  """

  __slots__ = ("shape", "dtype", "sharding", "original")

  def __init__(
      self,
      shape: tuple[int, ...],
      dtype: Any,
      sharding: jax.sharding.Sharding,
      original: jax.Array | None,
  ):
    self.shape = shape
    self.dtype = dtype
    self.sharding = sharding
    self.original = original

  def delete(self) -> None:
    if self.original is not None:
      self.original.delete()
      self.original = None


@dataclasses.dataclass(frozen=True)
class StackRestore:
  """How to restore one stacked SparseCore table and its optimizer slots.

  Attributes:
    variables: The stacked table variable followed by its slot variables.
    saved_shapes: The shape of each variable in the checkpoint.
    table_stack: The (destination) `TableSpec`s of the tables in the stack.
    layout: The SparseCore `TensorLayout` of the variables.
    src_num_shards: The number of SparseCores the checkpoint was saved with.
    src_num_sc_per_device: The SparseCores per chip it was saved with.
    dst_num_shards: The number of SparseCores of the current topology.
    dst_num_sc_per_device: The SparseCores per chip of the current topology.
  """

  variables: Sequence[keras.Variable]
  saved_shapes: Sequence[tuple[int, ...]]
  table_stack: Sequence[embedding_spec.TableSpec]
  layout: keras.distribution.TensorLayout
  src_num_shards: int
  src_num_sc_per_device: int
  dst_num_shards: int
  dst_num_sc_per_device: int

  @property
  def needs_reshard(self) -> bool:
    """Whether the checkpoint was saved for another SparseCore topology."""
    return (self.src_num_shards, self.src_num_sc_per_device) != (
        self.dst_num_shards,
        self.dst_num_sc_per_device,
    )


def _stack_shape(
    table_stack: Sequence[embedding_spec.TableSpec], num_shards: int
) -> tuple[int, int]:
  """Returns the shape of the stacked `table_stack` for `num_shards`."""
  _, table_to_padded_vocab_size = (
      jte_table_stacking.round_up_dim_and_vocab_size(
          {spec.name: spec for spec in table_stack}, num_shards
      )
  )
  return (
      sum(table_to_padded_vocab_size.values()),
      table_stack[0].setting_in_stack.padded_embedding_dim,
  )


def _plan_stack_restore(
    *,
    stack_name: str,
    feature_specs: Any,
    table_stacks: dict[str, Sequence[embedding_spec.TableSpec]],
    variables: Sequence[keras.Variable],
    saved_shapes: Sequence[tuple[int, ...]],
    layout: keras.distribution.TensorLayout,
    src_topology: Mapping[str, int] | None,
    dst_num_shards: int,
    dst_num_sc_per_device: int,
) -> StackRestore | None:
  """Validates saved stack shapes and returns a `StackRestore` if resharding is needed."""
  if src_topology is None:
    shapes = [tuple(var.shape) for var in variables]
    # If the topology wasn't saved, the stack can only be restored if its
    # shapes match the current ones.
    if list(saved_shapes) != shapes:
      raise ValueError(
          f"The SparseCore stack '{stack_name}' was saved with shapes"
          f" {list(saved_shapes)} but without its SparseCore topology, so"
          f" it can't be resharded for the current shapes {shapes}."
          " Restore it on the TPU topology it was saved on, and save it"
          " again with `KerasOrbaxCheckpointManagerV2/V3`."
      )
    logging.warning(
        "The SparseCore stack '%s' was saved without its SparseCore"
        " topology. Restoring it as is, assuming it was saved on the"
        " current one (%d SparseCores, %d per chip); if not, the restored"
        " SparseCore tables are wrong.",
        stack_name,
        dst_num_shards,
        dst_num_sc_per_device,
    )
    return None

  src_num_shards = int(src_topology["num_shards"])
  src_num_sc_per_device = int(src_topology["num_sc_per_device"])
  if (src_num_shards, src_num_sc_per_device) == (
      dst_num_shards,
      dst_num_sc_per_device,
  ):
    # Same topology: leave the stack untouched so Orbax restores it directly
    # (and raises ValueError if its shape changed).
    return None

  if not table_stacks:
    table_stacks.update(
        jte_table_stacking.get_table_stacks(
            jte_embedding.get_table_specs(feature_specs)
        )
    )
  table_stack = table_stacks[stack_name]
  # The stack can only be resharded if it holds the same tables, i.e. if its
  # saved shapes match the current tables stacked for `src_num_shards`.
  expected_shape = _stack_shape(table_stack, src_num_shards)
  if any(shape != expected_shape for shape in saved_shapes):
    raise ValueError(
        f"The SparseCore stack '{stack_name}' was saved with shapes"
        f" {list(saved_shapes)} for {src_num_shards} SparseCores, but its"
        f" current tables stacked for them have shape {expected_shape}. The"
        " tables of the stack changed since the checkpoint was saved."
    )
  return StackRestore(
      variables=variables,
      saved_shapes=saved_shapes,
      table_stack=table_stack,
      layout=layout,
      src_num_shards=src_num_shards,
      src_num_sc_per_device=src_num_sc_per_device,
      dst_num_shards=dst_num_shards,
      dst_num_sc_per_device=dst_num_sc_per_device,
  )


def plan_sparsecore_restore(
    model: keras.Model,
    trainable_metadata: Mapping[str, Any],
    saved_topology: Mapping[str, Mapping[str, int]] | None,
    shared_var_map: Mapping[str, str] | None = None,
) -> list[StackRestore]:
  """Returns the SparseCore stacks of `model` that must be resharded.

  Stacks saved for the current topology are skipped early so Orbax restores them
  as is without inspecting `TableSpec` stacking math. Stacks of checkpoints
  saved without their topology are restored as is if their shapes match.

  Args:
    model: The model being restored.
    trainable_metadata: Path of each trainable variable of `model` to its Orbax
      `ArrayMetadata` in the checkpoint.
    saved_topology: The `sparsecore_topology` saved with the checkpoint (see
      `saved_sparsecore_topology`), or None if it wasn't saved.
    shared_var_map: Optional V3 `shared_variable_map` mapping alias variable
      paths to the canonical stored paths used as keys in `saved_topology`.

  Raises:
    ValueError: If a stack saved with another shape has no saved topology, if
      the checkpoint has a saved topology but not for a stack, or if a saved
      stack doesn't hold the current tables of the stack.
  """
  sc_layer_states = sparsecore_layer_states(model)
  if not sc_layer_states or not trainable_metadata:
    return []
  if not model._jax_state_synced:  # pylint: disable=protected-access
    model.jax_state_sync()

  plans = []
  for state in sc_layer_states:
    dst_num_sc_per_device = state.config.num_sc_per_device
    dst_num_shards = state.config.mesh.devices.size * dst_num_sc_per_device
    table_stacks: dict[str, Sequence[embedding_spec.TableSpec]] = {}
    for stack_name, emb_vars in state.table_and_slot_variables.items():
      stored_meta = trainable_metadata.get(emb_vars.table.path)
      if stored_meta is None:
        continue
      src_topology = None
      if saved_topology is not None:
        src_topology = saved_topology.get(emb_vars.table.path)
        if src_topology is None and shared_var_map is not None:
          # `sparsecore_topology` is keyed only by the canonical `var.path` at
          # save time; resolve alias paths of shared SparseCore layers in V3
          # checkpoints through `shared_variable_map`.
          canonical_path = shared_var_map.get(emb_vars.table.path)
          if canonical_path is not None:
            src_topology = saved_topology.get(canonical_path)
        if src_topology is None:
          # E.g. the variable path changed since the checkpoint was saved.
          raise ValueError(
              "The checkpoint has no saved SparseCore topology for"
              f" {emb_vars.table.path}, only for"
              f" {sorted(saved_topology)}."
          )
      slot_vars = [
          v for v in keras.tree.flatten(emb_vars.slot) if v is not None
      ]
      variables = [emb_vars.table, *slot_vars]
      # Slots have the shape of their table.
      saved_shapes = [
          tuple(trainable_metadata.get(var.path, stored_meta).shape)
          for var in variables
      ]
      plan = _plan_stack_restore(
          stack_name=stack_name,
          feature_specs=state.config.feature_specs,
          table_stacks=table_stacks,
          variables=variables,
          saved_shapes=saved_shapes,
          layout=state.layout,
          src_topology=src_topology,
          dst_num_shards=dst_num_shards,
          dst_num_sc_per_device=dst_num_sc_per_device,
      )
      if plan is not None:
        plans.append(plan)
  return plans


def prepare_sparsecore_restore(
    model: keras.Model,
    checkpoint_path: str,
    version: str,
) -> list[StackRestore]:
  """Prepares the SparseCore tables of `model`, if any, for a cross-topology restore.

  The restore paths of `keras_utils` restore each variable into
  `jax.ShapeDtypeStruct(var.value.shape, ...)`, so restoring a checkpoint saved
  on another topology as is would fail with a shape mismatch, or load rows
  sharded for the wrong topology when the shapes happen to match.

  Call this before building the Orbax restore target and
  `finish_sparsecore_restore` after the restore. This first compares the
  checkpoint's saved SparseCore topology (see `sparsecore_topology`) with the
  current one and returns immediately when they match, without reading the
  checkpoint's state metadata. Otherwise, it reads the trainable variable
  metadata and, for each stacked variable saved with another shape, replaces its
  value with a placeholder of the saved shape so that Orbax restores the raw
  saved tensor. Stacks saved for the current topology are not touched.

  The current values are only freed when the restore paths delete the restored
  variables. If the restore fails before that, `abort_sparsecore_restore`
  undoes this; after that, the model must be rebuilt, like for any failed
  restore.

  Args:
    model: The model being restored.
    checkpoint_path: The resolved checkpoint directory path of one step.
    version: The `keras_utils.CheckpointVersion` of the checkpoint (V2 or V3),
      see `saved_trainable_metadata`. Only used if the checkpoint's state
      metadata must be read.

  Returns:
    The stacks to reshard, to pass to `finish_sparsecore_restore`.

  Raises:
    ValueError: See `saved_trainable_metadata` and `plan_sparsecore_restore`.
  """
  if not has_sparsecore_tables(model):
    return []
  saved_topology = saved_sparsecore_topology(checkpoint_path)
  if saved_topology is not None and saved_topology == sparsecore_topology(
      model
  ):
    return []
  trainable_metadata, shared_var_map = saved_trainable_metadata(
      model, checkpoint_path, version=version
  )
  if not trainable_metadata:
    return []
  plans = plan_sparsecore_restore(
      model, trainable_metadata, saved_topology, shared_var_map=shared_var_map
  )
  for plan in plans:
    sc_sharding = plan.layout.backend_layout.sharding
    for var, saved_shape in zip(plan.variables, plan.saved_shapes):
      # Restore the raw saved value instead of laying it out for the current
      # topology.
      var._layout = None  # pylint: disable=protected-access
      if saved_shape != tuple(var.shape):
        dtype = var.value.dtype
        sharding = row_sharding(sc_sharding, saved_shape[0])
        num_devices = sc_sharding.mesh.devices.size
        num_row_shards = sharding.mesh.devices.shape[0]
        if num_row_shards < num_devices:
          # TODO(b/577354074): Shard the saved rows unevenly instead, so that
          # every device holds 1/num_devices of the stack.
          logging.warning(
              "Restoring SparseCore variable %s with saved shape %s in %d row"
              " shards, each replicated on %d of the %d devices (%.2f GiB per"
              " device), since its rows are not divisible by the device count.",
              var.path,
              saved_shape,
              num_row_shards,
              num_devices // num_row_shards,
              num_devices,
              math.prod(saved_shape)
              * np.dtype(dtype).itemsize
              / num_row_shards
              / 2**30,
          )
        var._value = _RestoreValuePlaceholder(  # pylint: disable=protected-access
            shape=saved_shape,
            dtype=dtype,
            sharding=sharding,
            original=var.value,
        )
  return plans


def abort_sparsecore_restore(plans: Sequence[StackRestore]) -> None:
  """Undoes `prepare_sparsecore_restore` if the restore fails early.

  Only valid before the restore paths delete the restored variables.

  Args:
    plans: The result of `prepare_sparsecore_restore`.
  """
  for plan in plans:
    for var in plan.variables:
      value = var._value  # pylint: disable=protected-access
      if isinstance(value, _RestoreValuePlaceholder):
        var._value = value.original  # pylint: disable=protected-access
      var._layout = plan.layout  # pylint: disable=protected-access


def finish_sparsecore_restore(plans: Sequence[StackRestore]) -> None:
  """Reshards the restored SparseCore stacks for the current topology.

  Args:
    plans: The result of `prepare_sparsecore_restore`, after the restore.
  """
  for plan in plans:
    if plan.needs_reshard:
      # TODO(b/577354074): Validate the cross-topology restore on multi-host
      # TPU topologies (e.g. 16x16 train -> 2x4 eval and back).
      logging.info(
          "Resharding SparseCore stack %s (saved shape %s) from %d SparseCores"
          " (%d per chip) to %d (%d per chip) across %d processes.",
          plan.table_stack[0].stacked_table_spec.stack_name,
          plan.saved_shapes[0],
          plan.src_num_shards,
          plan.src_num_sc_per_device,
          plan.dst_num_shards,
          plan.dst_num_sc_per_device,
          jax.process_count(),
      )
    for var in plan.variables:
      var._layout = plan.layout  # pylint: disable=protected-access
      restored = var._value  # pylint: disable=protected-access
      if plan.needs_reshard or tuple(restored.shape) != tuple(var.shape):
        restored = reshard_stacked_array(
            restored,
            plan.table_stack,
            src_num_shards=plan.src_num_shards,
            src_num_sc_per_device=plan.src_num_sc_per_device,
            dst_num_shards=plan.dst_num_shards,
            dst_num_sc_per_device=plan.dst_num_sc_per_device,
            # The SparseCore `Format` (with its TPU tiling), so that
            # `var.assign` doesn't copy the resharded stack to relayout it.
            out_sharding=plan.layout.backend_layout,
        )
      var.assign(restored)

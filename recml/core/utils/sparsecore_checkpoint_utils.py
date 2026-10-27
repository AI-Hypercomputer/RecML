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

TODO(b/576848327): Remove this module (and its `jax_tpu_embedding` dependency)
once keras_rs `DistributedEmbedding` supports resharding SparseCore tables and
optimizer slots across TPU topologies.
"""

from collections.abc import Mapping, Sequence
import copy
import dataclasses
import math
import os
from typing import Any, cast

from absl import logging
from etils import epath
import jax
from jax_tpu_embedding.sparsecore.lib.nn import embedding as jte_embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.nn import table_stacking as jte_table_stacking
import keras
import numpy as np
import orbax.checkpoint as ocp


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
    out_sharding: jax.sharding.Sharding,
) -> jax.Array:
  """Reshards a stacked SparseCore table or slot tensor across topologies.

  Unstacks `stacked_array` (laid out for the source topology) into its logical
  tables with `jte_table_stacking.unshard_and_unstack_tables`, then restacks
  them for the destination topology of `table_stack` with
  `jte_table_stacking.stack_and_shard_tables`, all in one `jax.jit`
  computation. XLA shards the intermediate tables (whose vocab sizes need not
  be divisible by the device count), so they stay distributed across the
  devices of all hosts, and every process must call this in lockstep.

  `stacked_array` is donated to reduce the peak memory usage.

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
    out_sharding: The sharding of the result.

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
  src_table_stack = []
  for spec in table_stack:
    src_spec = copy.copy(spec)
    src_spec.setting_in_stack = src_settings[spec.name]
    src_table_stack.append(src_spec)
  stack_name = table_stack[0].stacked_table_spec.stack_name

  def _reshard(stacked: jax.Array) -> jax.Array:
    tables = jte_table_stacking.unshard_and_unstack_tables(
        src_table_stack, {stack_name: stacked}, src_num_shards
    )
    restacked = jte_table_stacking.stack_and_shard_tables(
        list(table_stack), tables, dst_num_shards, pad_value=0.0
    )[stack_name]
    restacked = cast(jax.Array, restacked)
    return restacked.reshape(-1, stacked.shape[-1]).astype(stacked.dtype)

  return jax.jit(_reshard, out_shardings=out_sharding, donate_argnums=0)(
      stacked_array
  )


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
  """
  devices = sharding.mesh.devices.reshape(-1)
  num_row_shards = math.gcd(num_rows, devices.size)
  if num_row_shards == devices.size:
    return sharding
  mesh = jax.sharding.Mesh(
      devices.reshape(num_row_shards, -1), ("rows", "replicas")
  )
  return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("rows"))


# How the SparseCore stacked tables were sharded (see `sparsecore_topology`).
SPARSECORE_TOPOLOGY_KEY = "sparsecore_topology"


def sparsecore_layer_states(
    model: keras.Model,
) -> list[tuple[Any, Any, Mapping[str, Any]]]:
  """Returns the SparseCore state of the SparseCore layers of `model`.

  Matches the JAX `keras_rs.layers.DistributedEmbedding` layers with tables on
  SparseCore, including the one wrapped by `SparseCoreEmbeddings`.

  Args:
    model: The model being saved or restored.

  Returns:
    For each layer, its SparseCore lookup config, SparseCore `TensorLayout` and
    stacked table and slot variables by stack name.
  """
  states = []
  for layer in model._flatten_layers(include_self=True):  # pylint: disable=protected-access
    if not getattr(layer, "built", False) or not getattr(
        layer, "_sparsecore_built", False
    ):
      continue
    state = (
        getattr(layer, "_config", None),
        getattr(layer, "_sparsecore_layout", None),
        getattr(layer, "_table_and_slot_variables", None),
    )
    if all(x is not None for x in state):
      states.append(state)
  return states


def has_sparsecore_tables(model: keras.Model) -> bool:
  """Returns whether `model` has embedding tables on SparseCore."""
  return keras.backend.backend() == "jax" and bool(
      sparsecore_layer_states(model)
  )


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
  """
  if keras.backend.backend() != "jax":
    return {}
  topology = {}
  for config, _, table_vars in sparsecore_layer_states(model):
    num_sc_per_device = int(config.num_sc_per_device)
    num_shards = int(config.mesh.devices.size) * num_sc_per_device
    for emb_vars in table_vars.values():
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


class _RestoreValuePlaceholder:
  """Stands in for `keras.Variable._value` to restore a differently shaped value.

  Used by `prepare_sparsecore_restore` for SparseCore variables saved for
  another topology. The restore paths of `keras_utils` build the Orbax restore
  target from `var.value.shape`, `.dtype` and `.sharding`, so these attributes
  make Orbax restore the saved shape. They then call `var.value.delete()` to
  free each variable before restoring; the placeholder holds no buffer, so
  `delete` is a no-op.
  """

  __slots__ = ("shape", "dtype", "sharding")

  def __init__(
      self,
      shape: tuple[int, ...],
      dtype: Any,
      sharding: jax.sharding.Sharding,
  ):
    self.shape = shape
    self.dtype = dtype
    self.sharding = sharding

  def delete(self) -> None:
    pass


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
  layout: Any
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


def plan_sparsecore_restore(
    model: keras.Model,
    checkpoint_path: str,
    saved_trainable_metadata: Mapping[str, Any],
) -> list[StackRestore]:
  """Returns the SparseCore stacks of `model` that must be resharded.

  Stacks saved for the current topology are left out: Orbax restores them as
  is. Stacks of checkpoints saved without their topology are restored as is if
  their shapes match.

  Args:
    model: The model being restored.
    checkpoint_path: Path to the checkpoint directory of the restored step.
    saved_trainable_metadata: Path of each trainable variable of `model` to its
      Orbax `ArrayMetadata` in the checkpoint.

  Raises:
    ValueError: If a stack saved with another shape has no saved topology.
  """
  sc_layer_states = sparsecore_layer_states(model)
  if not sc_layer_states or not saved_trainable_metadata:
    return []
  if not model._jax_state_synced:  # pylint: disable=protected-access
    model.jax_state_sync()
  saved_topology = saved_sparsecore_topology(checkpoint_path) or {}

  plans = []
  for config, sc_layout, table_vars in sc_layer_states:
    dst_num_sc_per_device = config.num_sc_per_device
    dst_num_shards = config.mesh.devices.size * dst_num_sc_per_device
    table_stacks = jte_table_stacking.get_table_stacks(
        jte_embedding.get_table_specs(config.feature_specs)
    )
    for stack_name, table_stack in table_stacks.items():
      emb_vars = table_vars[stack_name]
      stored_meta = saved_trainable_metadata.get(emb_vars.table.path)
      if stored_meta is None:
        continue
      variables = [emb_vars.table, *keras.tree.flatten(emb_vars.slot)]
      # Slots have the shape of their table.
      saved_shapes = [
          tuple(saved_trainable_metadata.get(var.path, stored_meta).shape)
          for var in variables
      ]
      shapes_match = all(
          shape == tuple(var.shape)
          for var, shape in zip(variables, saved_shapes)
      )
      src_topology = saved_topology.get(emb_vars.table.path)
      if src_topology is None:
        # Checkpoints saved before the topology was persisted.
        if not shapes_match:
          raise ValueError(
              f"The SparseCore stack '{stack_name}' was saved with shapes"
              f" {saved_shapes} but without its SparseCore topology, so it"
              " can't be resharded for the current shapes"
              f" {[tuple(var.shape) for var in variables]}. Restore it on the"
              " TPU topology it was saved on, and save it again with"
              " `KerasOrbaxCheckpointManagerV2/V3`."
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
        continue
      plan = StackRestore(
          variables=variables,
          saved_shapes=saved_shapes,
          table_stack=table_stack,
          layout=sc_layout,
          src_num_shards=int(src_topology["num_shards"]),
          src_num_sc_per_device=int(src_topology["num_sc_per_device"]),
          dst_num_shards=dst_num_shards,
          dst_num_sc_per_device=dst_num_sc_per_device,
      )
      if plan.needs_reshard or not shapes_match:
        plans.append(plan)
  return plans


def prepare_sparsecore_restore(
    model: keras.Model,
    checkpoint_path: str,
    saved_trainable_metadata: Mapping[str, Any],
) -> list[StackRestore]:
  """Prepares the SparseCore tables of `model` for a cross-topology restore.

  The restore paths of `keras_utils` restore each variable into
  `jax.ShapeDtypeStruct(var.value.shape, ...)`, so restoring a checkpoint saved
  on another topology as is would fail with a shape mismatch, or load rows
  sharded for the wrong topology when the shapes happen to match.

  Call this before building the Orbax restore target and
  `finish_sparsecore_restore` after the restore. This reads the source
  topology saved with the checkpoint (see `sparsecore_topology`) and, for each
  stacked variable saved with another shape, replaces its value with a
  placeholder of the saved shape, so that Orbax restores the raw saved tensor.
  Stacks saved for the current topology are not touched.

  If the restore fails in between, the model must be rebuilt: like the restore
  paths themselves, this frees the current buffers of the restored variables.

  Args:
    model: The model being restored.
    checkpoint_path: Path to the checkpoint directory of the restored step.
    saved_trainable_metadata: Path of each trainable variable of `model` to its
      Orbax `ArrayMetadata` in the checkpoint.

  Returns:
    The stacks to reshard, to pass to `finish_sparsecore_restore`.

  Raises:
    ValueError: If a stack saved with another shape has no saved topology.
  """
  plans = plan_sparsecore_restore(
      model, checkpoint_path, saved_trainable_metadata
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
        var.value.delete()
        var._value = _RestoreValuePlaceholder(  # pylint: disable=protected-access
            shape=saved_shape,
            dtype=dtype,
            sharding=sharding,
        )
  return plans


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
            out_sharding=plan.layout.backend_layout.sharding,
        )
      var.assign(restored)

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
reshards such variables across topologies, so that checkpoints can be restored
on another topology than the one they were saved on.

TODO(b/576848327): Remove this module (and its `jax_tpu_embedding` dependency)
once keras_rs `DistributedEmbedding` supports resharding SparseCore tables and
optimizer slots across TPU topologies.
"""

from collections.abc import Mapping, Sequence
import copy
import math
from typing import cast

import jax
from jax.experimental import layout as jax_layout
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.nn import table_stacking as jte_table_stacking


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

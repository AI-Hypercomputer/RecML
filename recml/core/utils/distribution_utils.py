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
"""Utilities for inspecting the active JAX/Keras distribution mesh."""

from collections.abc import Callable
import math
from typing import Any
import weakref

import jax
import jax.numpy as jnp
import keras
from keras.src.backend.common import variables as keras_variables

# Conventional name of the mesh axis used for model/tensor parallelism. Any
# other axis of the mesh is assumed to shard the batch.
MODEL_AXIS_NAME = 'model'

AxisSpec = str | tuple[str, ...] | None


def get_abstract_mesh() -> jax.sharding.AbstractMesh | None:
  """Returns the active JAX abstract mesh, if one can be determined.

  Prefers the mesh installed by an enclosing `jax.sharding.use_mesh` context.
  Falls back to deriving an abstract mesh from the active Keras distribution's
  device mesh.

  Returns:
    The active abstract mesh, or None if there is no mesh in scope.
  """
  if (global_abstract_mesh := jax.sharding.get_abstract_mesh()).shape_tuple:
    return global_abstract_mesh
  distribution = keras.distribution.distribution()
  if distribution is None:
    return None
  device_mesh = getattr(distribution, 'device_mesh', None)
  if device_mesh is None:
    return None
  return jax.sharding.AbstractMesh(
      axis_sizes=tuple(device_mesh.shape),
      axis_names=tuple(device_mesh.axis_names),
      # `axis_types` must have one entry per axis. Passing a bare `AxisType`
      # raises `ValueError` on any mesh with more than one axis.
      axis_types=tuple(
          jax.sharding.AxisType.Auto for _ in device_mesh.axis_names
      ),
  )


def get_batch_dim_name() -> AxisSpec:
  """Returns the `batch_dim_name` of the active Keras distribution, if any.

  Distributions that shard the batch over several mesh axes (e.g. hybrid FSDP
  over `('replica', 'fsdp')`) report a tuple here rather than a single name.

  Returns:
    The batch dimension name(s), or None if no distribution is active or the
    distribution does not declare one.
  """
  distribution = keras.distribution.distribution()
  if distribution is None:
    return None
  return getattr(distribution, 'batch_dim_name', None)


def normalize_axis(axis: AxisSpec | list[str]) -> AxisSpec:
  """Normalizes an axis specification to a bare name, a tuple, or None.

  A single-element sequence is collapsed to the bare axis name so that it can
  be used interchangeably with a scalar axis in `jax.sharding.PartitionSpec`
  and `jax.lax.all_gather`.

  Args:
    axis: An axis name, a sequence of axis names, or None.

  Returns:
    None for an empty or missing spec, a bare name for a single axis, or a
    tuple of names otherwise.
  """
  if not isinstance(axis, (tuple, list)):
    return axis
  if not axis:
    return None
  return axis[0] if len(axis) == 1 else tuple(axis)


def resolve_batch_axis(
    abstract_mesh: jax.sharding.AbstractMesh | None,
) -> AxisSpec:
  """Resolves the mesh axis or axes over which the batch is sharded.

  The active Keras distribution is authoritative: if it declares a
  `batch_dim_name`, that is used verbatim. Otherwise every mesh axis other
  than `MODEL_AXIS_NAME` is assumed to shard the batch, which covers both
  plain data parallelism and hybrid FSDP meshes.

  Args:
    abstract_mesh: The mesh to resolve against, typically from
      `get_abstract_mesh`.

  Returns:
    The batch axis name, a tuple of names if the batch is sharded over several
    axes, or None if the batch axis cannot be determined.
  """
  batch_dim_name = get_batch_dim_name()
  if batch_dim_name is not None:
    return normalize_axis(batch_dim_name)
  if abstract_mesh is None or not abstract_mesh.axis_names:
    return None
  batch_axes = tuple(
      axis for axis in abstract_mesh.axis_names if axis != MODEL_AXIS_NAME
  )
  if not batch_axes:
    # A mesh consisting only of the model axis still has to place the batch
    # somewhere; fall back to the full set of axis names.
    batch_axes = tuple(abstract_mesh.axis_names)
  return normalize_axis(batch_axes)


def fuse_table_grads_enabled() -> bool:
  """Returns whether the active distribution fuses table gradients.

  A distribution opts in by setting a truthy `fuse_table_grads` attribute.
  Layers that read a large table with a batch (embedding lookups, tied output
  projections) then run that read through `fused_table_grad_call`.

  Returns:
    True if a distribution is active and its `fuse_table_grads` is truthy.
  """
  return bool(
      getattr(keras.distribution.distribution(), 'fuse_table_grads', False)
  )


class FusedTableGradDataParallel(keras.distribution.DataParallel):
  """`DataParallel` that fuses gradients across reads of each replicated table.

  Under plain `DataParallel`, XLA partitions every embedding lookup's gradient
  (a scatter-add into a replicated table) as a local scatter plus an
  all-reduce, and it seeds the next scatter of the same table with that
  all-reduce's result. A table read by N lookups then pays a chain of up to N
  table-sized all-reduces per step. With this distribution, every read of a
  table runs under `jax.shard_map` on a shared per-device copy of the table.
  Each device adds up its reads' table gradients locally, and one all-reduce
  per table sums the result across devices. The forward computation is
  unchanged; only the order in which the table gradient is summed changes.
  """

  fuse_table_grads = True


# Shared per-device copies of replicated tables. Maps id(table) to a weak
# reference to the table, the table's value in the trace that read it last, and
# the copies of that value. See `_shared_table_copies`.
_shared_table_copies_cache: dict[
    int, tuple[weakref.ref[keras.Variable], jax.Array, jax.Array]
] = {}


def _shared_table_copies(
    table: keras.Variable, table_value: jax.Array, num_copies: int
) -> jax.Array:
  """Returns `num_copies` copies of `table_value` stacked on dim 0.

  Every call on `table` with the same traced `table_value` gets the same
  copies. JAX then adds up the table gradients of all those calls on each
  device before it reaches the transpose of the tile, which sums the copies
  across devices once.

  The copies are a tracer, so all calls on a table must run at the same trace
  level. If the first call runs inside a nested transformation such as
  `jax.checkpoint` and a later one outside it, JAX raises an escaped-tracer
  error.

  Args:
    table: The table variable.
    table_value: The table's value in the current trace.
    num_copies: The number of devices that share the table.

  Returns:
    An array of shape `(num_copies * table.shape[0], *table.shape[1:])`.
  """
  key = id(table)
  cached = _shared_table_copies_cache.get(key)
  if cached is not None and cached[0]() is table and cached[1] is table_value:
    return cached[2]
  copies = jnp.tile(table_value, (num_copies,) + (1,) * (table_value.ndim - 1))
  _shared_table_copies_cache[key] = (
      # Drops the entry, and the tracers in it, when the table goes away.
      weakref.ref(table, lambda _: _shared_table_copies_cache.pop(key, None)),
      table_value,
      copies,
  )
  return copies


def fused_table_grad_call(
    fn: Callable[..., Any],
    table: keras.Variable,
    *batch_args: Any,
) -> Any:
  """Calls `fn(table, *batch_args)` on each device and fuses table gradients.

  The table must be replicated. Each device gets its own copy, and all calls on
  the table in the same trace share that copy. So each device adds up the table
  gradients of all calls, and one all-reduce per table sums them across
  devices. All calls on a table must run at the same trace level (see
  `_shared_table_copies`).

  Under a Keras autocast scope, `fn` gets the table in the scope's dtype, but
  the gradient is still all-reduced in the variable dtype.

  Args:
    fn: A function of the full table and the local batch shards. Each output row
      must depend only on the same row of the batch args.
    table: A replicated table variable.
    *batch_args: Arrays whose dim 0 is the global batch.

  Returns:
    The output of `fn`, sharded on dim 0 over the batch axes.

  Raises:
    NotImplementedError: If the active distribution shards the table.
  """
  # TODO(jinchi): Support sharded (FSDP) tables.
  if any(keras.distribution.distribution().get_variable_layout(table).axes):
    raise NotImplementedError(
        'fused_table_grad_call supports only replicated tables.'
    )
  abstract_mesh = get_abstract_mesh()
  batch_axis = resolve_batch_axis(abstract_mesh)

  # Read the table in its variable dtype and cast inside `fn`, so the gradient
  # is all-reduced in that dtype rather than in bf16.
  autocast_scope = keras_variables.get_autocast_scope()
  with keras_variables.AutocastScope(None):
    table_value = keras.ops.convert_to_tensor(table)

  if abstract_mesh is not None and isinstance(table_value, jax.core.Tracer):
    mesh = abstract_mesh
    batch_axes = batch_axis if isinstance(batch_axis, tuple) else (batch_axis,)
    axis_sizes = dict(zip(abstract_mesh.axis_names, abstract_mesh.axis_sizes))
    table_value = _shared_table_copies(
        table, table_value, math.prod(axis_sizes[axis] for axis in batch_axes)
    )
    table_spec = jax.P(batch_axis)
  else:
    # Eager `shard_map` on an `AbstractMesh` rejects inputs without a
    # `NamedSharding`, so eager calls use the concrete mesh and one table.
    mesh = keras.distribution.distribution().device_mesh.backend_mesh
    table_spec = jax.P()

  def local_fn(full_table, *local_batch_args):
    if autocast_scope is not None:
      full_table = autocast_scope.maybe_cast(full_table)
    return fn(full_table, *local_batch_args)

  # The transpose of the tile sums the copies' gradients across devices.
  return jax.shard_map(
      local_fn,
      mesh=mesh,
      in_specs=(table_spec,) + (jax.P(batch_axis),) * len(batch_args),
      out_specs=jax.P(batch_axis),
      check_vma=False,
  )(table_value, *batch_args)

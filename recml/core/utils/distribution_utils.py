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

import jax
import keras

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

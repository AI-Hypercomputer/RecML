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
"""Utilities for partitioning."""

import abc
from collections.abc import Callable, Mapping, Sequence
import math
from typing import Any, ContextManager

import flax.linen as nn
import jax
import numpy as np


PyTree = Any
State = Any
CreateStateFn = Callable[[PyTree], State]
InitFn = Callable[[PyTree, jax.Array], State]
StepFn = Callable[[PyTree, State], Any]


class Partitioner(abc.ABC):
  """An abstract class defining partitioning logic for data and computation."""

  @abc.abstractmethod
  def shard_inputs(self, inputs: Any) -> PyTree:
    """Shards the input batches and put them on the device."""

  @abc.abstractmethod
  def partition_init(
      self, init_fn: CreateStateFn, *, abstract_batch: PyTree | None = None
  ) -> CreateStateFn:
    """Shards the initialization function."""

  @abc.abstractmethod
  def partition_step(self, fn: StepFn, *, training: bool = False) -> StepFn:
    """Shards the training and evaluation steps."""


class NullPartitioner(Partitioner):
  """A null partitioner."""

  def shard_inputs(self, inputs: PyTree) -> PyTree:
    return inputs

  def partition_init(
      self, init_fn: CreateStateFn, *, abstract_batch: PyTree | None = None
  ) -> CreateStateFn:
    return init_fn

  def partition_step(self, fn: StepFn, *, training: bool = False) -> StepFn:
    return fn


class DataParallelPartitioner(Partitioner):
  """Data parallel partitioner."""

  def __init__(self, data_axis: str = "batch"):
    self.mesh = jax.make_mesh((jax.device_count(),), (data_axis,))
    self.data_sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec(data_axis)
    )
    self.state_sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec()
    )

  def shard_inputs(self, inputs: PyTree) -> PyTree:
    local_devices = self.mesh.local_devices
    local_device_count = len(local_devices)
    device_count = len(self.mesh.devices)

    def _shard(x: np.ndarray) -> jax.Array:
      per_proc_batch_size = x.shape[0]
      per_replica_batch_size = per_proc_batch_size // local_device_count
      if per_proc_batch_size % local_device_count != 0:
        raise ValueError(
            "The per process batch size must be divisible by the number of"
            " local devices. Got per process batch size:"
            f" {per_proc_batch_size} and local device count:"
            f" {local_device_count}."
        )

      per_device_arrays = np.split(x, local_device_count, axis=0)
      device_buffers = [
          jax.device_put(arr, device)
          for arr, device in zip(per_device_arrays, local_devices)
      ]

      global_batch_size = per_replica_batch_size * device_count
      return jax.make_array_from_single_device_arrays(
          (global_batch_size,) + x.shape[1:], self.data_sharding, device_buffers
      )

    return jax.tree.map(_shard, inputs)

  def partition_init(
      self, init_fn: CreateStateFn, *, abstract_batch: PyTree | None = None
  ) -> CreateStateFn:
    with jax.set_mesh(self.mesh):
      if abstract_batch is not None:
        abstract_state = jax.eval_shape(init_fn, abstract_batch)
        specs = nn.get_partition_spec(abstract_state)
        self.state_sharding = jax.tree.map(
            lambda x: jax.sharding.NamedSharding(self.mesh, x), specs
        )
      init_fn = jax.jit(init_fn, out_shardings=self.state_sharding)

    def _wrapped_init(batch: PyTree) -> State:
      with jax.set_mesh(self.mesh):
        state = init_fn(batch)
        state = _maybe_unbox_state(state)
        return state

    return _wrapped_init

  def partition_step(self, fn: StepFn, *, training: bool = False) -> StepFn:
    jit_kws = {}
    if training:
      jit_kws["out_shardings"] = (self.state_sharding, None)
      jit_kws["donate_argnums"] = (1,)

    with jax.set_mesh(self.mesh):
      step_fn = jax.jit(
          fn,
          in_shardings=(self.data_sharding, self.state_sharding),
          **jit_kws,
      )

    def _wrapped_step(batch: PyTree, state: State) -> Any:
      with jax.set_mesh(self.mesh):
        return step_fn(batch, state)

    return _wrapped_step


class ModelParallelPartitioner(Partitioner):
  """Model parallel partitioner.

  This only works with multi-controller Jax, i.e. communications along the ICI
  for TPUs. For scaling beyond a single TPU slice this needs to be extended to
  support Megascale XLA or single-controller Pathways. Consider using T5X, Pax,
  MaxText externally or Gemax internally for these use cases.

  By default, all axes of the input are used for data parallelism. This results
  in fully-sharded data-parallelism for ND topologies or data-parallelism for 1D
  topologies. The range of axes can be configured using the `dp_axes` argument,
  i.e. axes[:dp_axes] will be used for data parallelism.

  IMPORTANT: `shard_inputs` operates on a per process batch. This means that the
  input batch size on CPU must already be the per process batch size,
  i.e. global batch size // jax.process_count(). It is the responsibility of the
  CPU input pipeline to ensure that inputs are different across processes.
  """

  def __init__(
      self,
      axes: Sequence[tuple[str, int]] = (("batch", -1),),
      dp_axes: int | None = None,
      rules: Mapping[str, str] | None = None,
      aot_compile: bool = False,
      options: jax.stages.CompilerOptions | None = None,
      devices: Sequence[jax.Device] | None = None,
  ):
    if not axes:
      raise ValueError("At least one axis must be specified in `axes`.")
    if dp_axes == 0:
      raise ValueError(
          "Data parallelism axes range must be positive or negative."
      )

    devices = devices if devices is not None else jax.devices()
    axis_names = [axis for axis, _ in axes]
    axis_sizes = [dim for _, dim in axes]
    if any(dim <= 0 for dim in axis_sizes[1:]):
      raise ValueError(
          "All dimensions except the first in the axes must be positive"
          f" integers. Got axes: {axes}."
      )
    if axis_sizes[0] == -1:
      axis_sizes[0] = len(devices) // math.prod(axis_sizes[1:])

    self.mesh = jax.make_mesh(axis_sizes, axis_names, devices=devices)
    self.rules = rules
    self.aot_compile = aot_compile
    self.options = options

    dp_axis_names, dp_axis_sizes = zip(*axes[:dp_axes])
    num_processes = jax.process_count()
    if math.prod(dp_axis_sizes) % num_processes != 0:
      raise ValueError(
          "The data parallel dimensions in the mesh must be divisible by the"
          " number of processes as we assume data parallelism across"
          f" processes. Got process count: {num_processes} and data"
          f" parallelism dimensions: {dp_axis_sizes} for axes: {axes} and mesh"
          f" devices: {self.mesh.devices}."
      )

    self.data_sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec(dp_axis_names)
    )
    self.state_sharding = None
    self.abstract_batch = None
    self.abstract_state = None

  @property
  def mesh_context_manager(
      self,
  ) -> Callable[[jax.sharding.Mesh], ContextManager[None]]:
    return jax.set_mesh

  def shard_inputs(self, inputs: PyTree) -> PyTree:
    def _shard(x: np.ndarray) -> jax.Array:
      return jax.make_array_from_process_local_data(self.data_sharding, x)

    return jax.tree.map(_shard, inputs)

  def partition_init(
      self, init_fn: CreateStateFn, *, abstract_batch: PyTree | None = None
  ) -> CreateStateFn:
    if abstract_batch is None:
      raise ValueError(
          "An `abstract_batch` is required for partitioning `init_fn` with a"
          " model parallel partitioner."
      )

    with self.mesh_context_manager(self.mesh):
      abstract_state = jax.eval_shape(init_fn, abstract_batch)
      specs = nn.get_partition_spec(abstract_state)

      if self.rules is not None:
        specs = nn.logical_to_mesh(specs, self.rules)

      state_sharding = jax.tree.map(
          lambda x: jax.sharding.NamedSharding(self.mesh, x), specs
      )
      compiled_init_fn = jax.jit(init_fn, out_shardings=state_sharding)

    def _init(batch: PyTree) -> State:
      with self.mesh_context_manager(self.mesh):
        state = compiled_init_fn(batch)
        state = _maybe_unbox_state(state)
      return state

    self.abstract_batch = abstract_batch
    self.abstract_state = abstract_state
    self.state_sharding = state_sharding
    return _init

  def partition_step(self, fn: StepFn, *, training: bool = False) -> StepFn:
    jit_kws = {}
    if training:
      jit_kws["out_shardings"] = (self.state_sharding, None)
      jit_kws["donate_argnums"] = (1,)
    else:
      jit_kws["out_shardings"] = None

    with self.mesh_context_manager(self.mesh):
      step_fn = jax.jit(
          fn,
          in_shardings=(self.data_sharding, self.state_sharding),
          compiler_options=(self.options if not self.aot_compile else None),
          **jit_kws,
      )
    if self.aot_compile:
      if self.abstract_batch is None or self.abstract_state is None:
        raise ValueError(
            "An `abstract_batch` and `abstract_state` must be set on the model"
            " parallel partitioner when `aot_compile` is set to True in order"
            " to compile the step. Make sure you call"
            " `partitioner.partition_init(...)` first."
        )

      step_fn = step_fn.lower(self.abstract_batch, self.abstract_state).compile(
          self.options
      )

    def _step(batch: PyTree, state: State) -> Any:
      with self.mesh_context_manager(self.mesh):
        return step_fn(batch, state)

    return _step


def _maybe_unbox_state(x: Any) -> Any:
  def _maybe_unbox(x: Any) -> Any:
    if isinstance(x, nn.Partitioned):
      return x.unbox()
    return x

  return jax.tree.map(
      _maybe_unbox,
      x,
      is_leaf=lambda k: isinstance(k, nn.Partitioned),
  )

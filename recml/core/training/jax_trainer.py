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
"""Jax task and trainer."""

import abc
from collections.abc import Callable, Iterator, Mapping, Sequence
import functools
import math
import os
import pprint
from typing import Any, Generic, Protocol, Self, TypeVar

from absl import logging
from clu import data as clu_data
from clu import periodic_actions
import clu.metrics as clu_metrics
from flax import struct
import flax.linen as nn
import jax
import jax.numpy as jnp
import keras
import numpy as np
import optax
import orbax.checkpoint as ocp
from recml.core.data import iterator as iterator_lib
from recml.core.metrics import base_metrics
from recml.core.metrics import tools as metrics_tools
from recml.core.training import core
from recml.core.training import partitioning
import tensorflow as tf


# pylint: disable=logging-fstring-interpolation, bad-whitespace

StateT = TypeVar("StateT")
MetricsT = TypeVar("MetricsT", bound=Mapping[str, clu_metrics.Metric])
MetaT = TypeVar("MetaT")
PyTree = Any


class State(Protocol):
  """State interface."""

  @property
  def step(self) -> int | jax.Array:
    """Returns the current step."""

  @property
  def opt_state(self) -> optax.OptState:
    """Returns the optimizer state."""


class JaxState(struct.PyTreeNode, Generic[MetaT]):
  """A training state for a Jax model created using Flax / Haiku.

  Attributes:
    step: A counter of the current step of the job. It starts at zero and it is
      incremented by 1 on a call to `state.update(...)`. This should be a Jax
      array and not a Python integer.
    params: A pytree of trainable variables that will be updated by `tx` and
      used in `apply`.
    tx: An optax gradient transformation that will be used to update the
      parameters contained in `params` on a call to `state.update(...)`.
    opt_state: The optimizer state for `tx`. This is usually created by calling
      `tx.init(params)`.
    _apply: An optional function that can be used to apply the forward pass of
      the model. For Flax models this is usually set to `model.apply` while for
      Haiku models this is usually set to `transform.apply`.
    _model: An optional reference to a stateless Flax model for convenience.
    mutable: A pytree of mutable variables that are used by `apply`.
    meta: Arbitrary metadata that is recorded on the state. This can be useful
      for tracking additional references in the state.
  """

  step: jax.Array
  params: PyTree = struct.field(pytree_node=True)
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState = struct.field(pytree_node=True)
  mutable: PyTree = struct.field(pytree_node=True, default_factory=dict)
  meta: MetaT = struct.field(pytree_node=False, default_factory=dict)
  _apply: Callable[..., Any] | None = struct.field(
      pytree_node=False, default_factory=None
  )
  _model: nn.Module | None = struct.field(pytree_node=False, default=None)

  @property
  def model(self) -> nn.Module:
    """Returns a reference to the model used to create the state."""
    if self._model is None:
      raise ValueError("No Flax `model` is set on the state.")
    return self._model

  def apply(self, *args, **kwargs) -> Any:
    """Applies the forward pass of the model."""
    if self._apply is None:
      raise ValueError("No `apply` function is set on the state.")
    return self._apply(*args, **kwargs)

  @classmethod
  def create(
      cls,
      *,
      apply: Callable[..., Any] | None = None,
      model: nn.Module | None = None,
      params: PyTree,
      tx: optax.GradientTransformation,
      **kwargs,
  ) -> Self:
    """Creates a new instance from a Jax model / apply fn and Optax optimizer.

    Args:
      apply: A function that can be used to apply the forward pass of the model.
        For Flax models this is usually set to `model.apply`. This cannot be set
        along with `model`.
      model: A reference to a stateless Flax model. This cannot be set along
        with `apply`. When set the `apply` attribute of the state will be set to
        `model.apply`.
      params: A pytree of trainable variables that will be updated by `tx` and
        used in `apply`.
      tx: An optax gradient transformation that will be used to update the
        parameters contained in `params` on a call to `state.update(...)`.
      **kwargs: Other updates to set on the new state.

    Returns:
      An new instance of the state.
    """
    if apply is not None and model is not None:
      raise ValueError("Only one of `apply` or `model` can be provided.")
    elif model is not None:
      apply = model.apply

    return cls(
        step=jnp.zeros([], dtype=jnp.int32),
        params=params,
        tx=tx,
        opt_state=tx.init(params),
        _apply=apply,
        _model=model,
        **kwargs,
    )

  def update(self, *, grads: PyTree, **kwargs) -> Self:
    """Applies gradients to the parameters."""
    updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
    new_params = optax.apply_updates(self.params, updates)
    return self.replace(
        step=self.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        **kwargs,
    )


class KerasState(struct.PyTreeNode):
  """A training state for a Jax model created using Keras.

  Attributes:
    step: A counter of the current step of the job. It starts at zero and it is
      incremented by 1 on a call to `state.update(...)`.
    call_fn: A function that can be used to apply the forward pass of the model.
      Usually this is set to `model.stateless_call`.
    tvars: A sequence of trainable variables that will be updated by `tx` and
      used in `call_fn`.
    ntvars: A sequence of non-trainable variables that are optionally updated by
      the user in or after `call_fn`.
    tx: An optax gradient transformation that will be used to update the
      parameters on a call to `state.update(...)`.
    opt_state: The optimizer state for `tx`.
    tvars_paths: The relative paths of the trainable variables based on the
      names of the scope they were created in. These are used to create a
      dictionary of parameters during optimization, which allows using masked
      Optax transformations.

  Example usage:
    >>> model = keras.Sequential([
    ...     keras.layers.Dense(256),
    ...     keras.layers.Dense(256),
    ...     keras.layers.Dense(1),
    ... ])
    >>> tx = optax.sgd(0.1)
    >>> state = KerasState.create(model, tx)

    >>> x = jax.random.uniform(jax.random.key(0), (128, 512), jnp.float32)
    >>> y = jax.random.uniform(jax.random.key(1), (128,), jnp.float32)

    >>> def loss_fn(tvars: Sequence[jax.Array]) -> jax.Array:
    ...   y_pred, ntvars = state.call_fn(tvars, state.ntvars, x)
    ...   return jnp.mean(optax.l2_loss(y_pred, y)), ntvars

    >>> grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    >>> (loss, ntvars), grads = grad_fn(state.tvars)
    >>> state = state.update(grads=grads, ntvars=ntvars)
  """

  step: int | jax.Array
  model: keras.Model = struct.field(pytree_node=False)
  tvars: Sequence[jax.Array] = struct.field(pytree_node=True)
  ntvars: Sequence[jax.Array] = struct.field(pytree_node=True)
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState = struct.field(pytree_node=True)
  tvars_paths: Sequence[str] = struct.field(pytree_node=False)

  @classmethod
  def create(
      cls, *, model: keras.Model, tx: optax.GradientTransformation
  ) -> Self:
    """Creates a new instance from a Keras model and Optax optimizer.

    Args:
      model: A Keras model with all variables built.
      tx: An optax gradient transformation.

    Returns:
      An instantiated instance.
    """

    if not model.built:
      raise ValueError(
          "The model has not been built. The model's variables must be built"
          " before they are detached from it."
      )

    trainable_variables = [v.value for v in model.trainable_variables]
    non_trainable_variables = [v.value for v in model.non_trainable_variables]
    trainable_variable_paths = [str(v.path) for v in model.trainable_variables]
    opt_state = tx.init(
        dict(zip(trainable_variable_paths, trainable_variables))
    )

    return cls(
        step=jnp.zeros([], dtype=jnp.int32),
        model=model,
        tvars=trainable_variables,
        ntvars=non_trainable_variables,
        tx=tx,
        opt_state=opt_state,
        tvars_paths=trainable_variable_paths,
    )

  def update(self, *, grads: PyTree, **kwargs) -> Self:
    """Applies gradients to the parameters.

    Uses the gradients to update the state of the optimizer and the trainable
    variables.

    Args:
      grads: The gradients to update trainable variables `state.tvars`.
      **kwargs: Other updates to set on the new state.

    Returns:
      A new instance with the updates applied.
    """
    grads_ = dict(zip(self.tvars_paths, grads))
    tvars_ = dict(zip(self.tvars_paths, self.tvars))
    updates, new_opt_state = self.tx.update(grads_, self.opt_state, tvars_)
    new_tvars_ = optax.apply_updates(tvars_, updates)
    new_tvars = [new_tvars_[path] for path in self.tvars_paths]
    return self.replace(
        step=self.step + 1,
        tvars=new_tvars,
        opt_state=new_opt_state,
        **kwargs,
    )

  def sync_model_variables(self):
    """Syncs the model variables with the variables tracked on the state.

    NOTE: This is a slow operation and should never be used during training.

    Raises:
      ValueError: If called in a tracing context.
    """
    if core.in_tracing_context():
      raise ValueError(
          "The model variables cannot be synced in a tracing context."
      )
    for var, value in zip(self.model.trainable_variables, self.tvars):
      var.assign(value)
    for var, value in zip(self.model.non_trainable_variables, self.ntvars):
      var.assign(value)


class JaxTask(abc.ABC, Generic[StateT]):
  """A base task interface for modeling."""

  @abc.abstractmethod
  def create_datasets(self) -> core.DatasetT:
    """Creates training and evaluation datasets.

    Returns:`
      One of the following:
        1) A `tf.data.Dataset` or `Iterator` instance that will be
           used for training.
        2) A tuple of `tf.data.Dataset` or `Iterator` instances where
           the first element is the training dataset and the second element is
           the evaluation dataset.
        3) A tuple of `tf.data.Dataset` or `Iterator` instances where
           the first element is the training dataset and the second element is a
           dictionary of evaluation datasets keyed by name.
    """

  @abc.abstractmethod
  def create_state(self, batch: PyTree, rng: jax.Array) -> StateT:
    """Creates the training state.

    Args:
      batch: A pytree of arrays making up a dummy batch for state
        initialization.
      rng: A prng key that is passed from the trainer to control randomness
        during variable initialization.

    Returns:
      The state to use for training.
    """

  @abc.abstractmethod
  def train_step(
      self, batch: PyTree, state: StateT, rng: jax.Array
  ) -> tuple[StateT, Mapping[str, clu_metrics.Metric]]:
    """Updates the training state and accumulates metrics.

    Args:
      batch: A pytree of arrays sampled from the training dataset.
      state: The training state created by `create_state`.
      rng: A prng key that is passed from the trainer to control randomness
        during training such as dropout.

    Returns:
      A tuple[state, metrics] where the state is the updated training state
      after the step, and metrics is a mapping from string metric name to a CLU
      metric. The name of the metric instance in the returned map will
      correspond to the reported scalar on TensorBoard.
    """

  @abc.abstractmethod
  def eval_step(
      self, batch: PyTree, state: StateT
  ) -> Mapping[str, clu_metrics.Metric]:
    """Performs evaluation and accumulates evaluation metrics.

    Args:
      batch: A pytree of arrays sampled from an eval iterator.
      state: The training state created by `create_state`.

    Returns:
      A mapping from string metric name to a CLU metric. The name of the metric
      instance in the returned map will correspond to the reported scalar on
      TensorBoard.
    """

  def restore_state(self, state: StateT) -> StateT:
    """Restores the initialized state from a checkpoint.

    By default this function defaults to returning the state as is.

    Args:
      state: The training state to restore.

    Returns:
      The restored state.
    """
    return state

  def export_model(self, state: StateT, model_dir: str):
    """Exports the model and relevant assets from the state.

    Args:
      state: The training state after training has completed.
      model_dir: The model directory passed to the trainer.
    """


# TODO(aahil): Converge with Jax CTL in this trainer.
class JaxTrainer(core.Trainer[JaxTask]):
  """A lightweight Jax trainer that uses Orbax and CLU."""

  def __init__(
      self,
      *,
      partitioner: partitioning.Partitioner | None = None,
      model_dir: str | None = None,
      train_steps: int = 0,
      steps_per_loop: int = 1_000,
      steps_per_eval: int | None = None,
      checkpoint_interval: int | None = None,
      max_checkpoints_to_keep: int = 5,
      continuous_eval_timeout: int = 30,
      rng_seed: int = core.DEFAULT_RNG_SEED,
      rng_impl: str | None = None,
      enable_checkpointing: bool = True,
  ):
    """Initializes the instance.

    Args:
      partitioner: A `Partitioner` instance that determines how parallelism and
        compilation are handled. By default no parallelism or partitioning is
        performed. Note that Jax does not perform well in eager mode so using at
        least a `DataParallelPartitioner` is recommended as it tends to be
        non-restrictive for arbitrary workloads.
      model_dir: The model directory to use for writing checkpoints, logs,
        exporting models, etc. Defaults to '/tmp'.
      train_steps: The total number of training steps to perform during
        training.
      steps_per_loop: The number of steps to perform in every training loop. By
        default this is set to 1000 which means that a training loop will take
        1000 steps before checkpointing, logging, or optionally evaluating. A
        larger `steps_per_loop` is preferred to keep the accelerators idle for
        less time. Note that checkpoints can be saved more frequently than this
        when required by setting `checkpoint_interval`. Furthermore, per-step
        scalars can be logged during training without decreasing this value by
        returning `ScalarMetric` instances in the `train_step`.
      steps_per_eval: The number of steps to perform on every evaluation. By
        default this is set to None which means that evaluation will exhaust the
        entire dataset.
      checkpoint_interval: The number of training steps after which a checkpoint
        is saved. By default this is set to None which means that checkpoints
        will be saved once at the end of each loop.
      max_checkpoints_to_keep: The maximum number of checkpoints to keep on
        disk. By default this is set to 5.
      continuous_eval_timeout: The number of seconds to wait for a new
        checkpoint before timing out during continuous evaluation. When a
        timeout happens, the job will check for a marker file on disk and if it
        exists, it will terminate successfully. Defaults to 30 seconds.
      rng_seed: The seed to use for the PRNG key. By default this is set to a
        fixed constant.
      rng_impl: The implementation of the PRNG key. By default this is set to
        None which means that the default implementation (generally
        partitionable threefry) will be used.
      enable_checkpointing: Whether to enable checkpointing. Defaults to True.
    """

    if not isinstance(steps_per_loop, int) or steps_per_loop < 1:
      raise ValueError(
          f"`steps_per_loop` ({steps_per_loop}) must be a positive integer."
      )

    self._partitioner = partitioner or partitioning.NullPartitioner()
    self._model_dir = model_dir or "/tmp"
    self._train_steps = train_steps
    self._steps_per_loop = steps_per_loop
    self._steps_per_eval = steps_per_eval
    self._continuous_eval_timeout = continuous_eval_timeout
    self._checkpoint_interval = checkpoint_interval or steps_per_loop
    self._max_checkpoints_to_keep = max_checkpoints_to_keep
    self._rng_impl = rng_impl
    self._rng_seed = rng_seed
    self._enable_checkpointing = enable_checkpointing

  @functools.cached_property
  def checkpoint_manager(self) -> ocp.CheckpointManager:
    """Returns the checkpoint manager."""
    save_on_steps = []
    for step in range(self._train_steps):
      if (step + 1) % self._checkpoint_interval == 0:
        save_on_steps.append(step)

    # Always save the last checkpoint.
    if not save_on_steps or save_on_steps[-1] != self._train_steps - 1:
      save_on_steps.append(self._train_steps - 1)

    save_on_steps = set(save_on_steps)
    
    if self._enable_checkpointing:

      return ocp.CheckpointManager(
          directory=os.path.join(self._model_dir, core.CHECKPOINT_DIR),
          options=ocp.CheckpointManagerOptions(
              should_save_fn=lambda step, _: step in save_on_steps,
              max_to_keep=self._max_checkpoints_to_keep,
          ),
      )
    else:
      
      return None

  @functools.cached_property
  def train_summary_writer(self) -> metrics_tools.AsyncMultiWriter:
    """Returns the summary writer for training."""
    return metrics_tools.AsyncMultiWriter(
        log_dir=os.path.join(self._model_dir, core.LOG_DIR),
        name=core.TRAIN_LOG_DIRNAME,
    )

  @functools.cached_property
  def report_progress(self) -> periodic_actions.ReportProgress:
    """Returns the report progress action."""
    return periodic_actions.ReportProgress(
        writer=self.train_summary_writer, every_steps=self._steps_per_loop
    )

  def _create_eval_summary_writers(
      self, eval_datasets: Mapping[str, clu_data.DatasetIterator]
  ) -> Mapping[str, metrics_tools.AsyncMultiWriter]:
    """Returns the summary writers for evaluation."""
    return {
        name: metrics_tools.AsyncMultiWriter(
            log_dir=os.path.join(self._model_dir, core.LOG_DIR),
            name=_val_logdir(name),
        )
        for name in eval_datasets
    }

  def _maybe_save_checkpoint(
      self,
      step: int,
      state: State,
      metrics: Mapping[str, Any] | None = None,
  ):
    """Saves a checkpoint and returns a bool indicating whether it was saved."""
    if not self._enable_checkpointing:
      return

    items = {core.STATE_CHECKPOINT_KEY: ocp.args.StandardSave(state)}
    with self.report_progress.timed("checkpointing"):
      self.checkpoint_manager.save(
          step=step,
          args=ocp.args.Composite(**items),
          metrics=metrics,
      )

  def _maybe_restore_checkpoint(
      self, state: State, step: int | None = None
  ) -> State:
    """Restores a checkpoint from a directory if it exists."""
    directory = os.path.join(self._model_dir, core.CHECKPOINT_DIR)

    if step is None:
      logging.info("There is no checkpoint in directory %s.", directory)
      return state

    args = {core.STATE_CHECKPOINT_KEY: ocp.args.StandardRestore(state)}
    restored_items = self.checkpoint_manager.restore(
        step=step, args=ocp.args.Composite(**args)
    )

    restored_state = restored_items[core.STATE_CHECKPOINT_KEY]
    if restored_state is not None:
      logging.info("Restored model from dir: %s.", directory)

    return restored_state

  def _write_marker_file(self):
    """Writes a marker file to disk."""
    with tf.io.gfile.GFile(
        os.path.join(self._model_dir, core.TRAINING_COMPLETE_MARKER_FILE), "w"
    ) as f:
      f.write("COMPLETED")

  def _train_n_steps(
      self,
      train_iter: Iterator[PyTree],
      train_step: partitioning.StepFn,
      state: State,
      start_step: int,
      num_steps: int,
      summary_writer: metrics_tools.AsyncMultiWriter,
  ) -> tuple[State, Mapping[str, Any]]:
    """Performs a training loop and returns the updated state and metrics."""
    metrics_accum = metrics_tools.MetricAccumulator(summary_writer)
    for step in range(start_step, start_step + num_steps):
      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        train_batch = next(train_iter)
        inputs = self._partitioner.shard_inputs(train_batch)
        state, metrics_update = train_step(inputs, state)
        metrics_accum.accumulate(metrics_update, step)
        self.report_progress(step)
        if (step != start_step + num_steps - 1) and self._enable_checkpointing:
          self._maybe_save_checkpoint(step, state)

    metrics = metrics_accum.compute_and_log_scalars(start_step + num_steps - 1)
    return state, metrics

  def _evaluate_n_steps(
      self,
      eval_iter: Iterator[PyTree],
      eval_step: partitioning.StepFn,
      state: State,
      num_steps: int | None,
      step_number: int,
      summary_writer: metrics_tools.AsyncMultiWriter,
  ) -> Mapping[str, Any]:
    """Performs an evaluation loop and returns the metrics."""

    step_count = 0
    metrics_accum = metrics_tools.MetricAccumulator(summary_writer)
    while num_steps is None or step_count < num_steps:
      with jax.profiler.StepTraceAnnotation("eval", step_num=step_count):
        try:
          eval_batch = next(eval_iter)
        except (StopIteration, tf.errors.OutOfRangeError):
          logging.info("Evaluated dataset to completion.")
          break
        inputs = self._partitioner.shard_inputs(eval_batch)
        metrics_update = eval_step(inputs, state)
        metrics_accum.accumulate(metrics_update, step_number)
      step_count += 1

    metrics = metrics_accum.compute_and_log_scalars(step_number)
    return metrics

  def process_task(
      self, task: JaxTask, *, training: bool, check_for_checkpoints: bool
  ) -> tuple[
      iterator_lib.Iterator,
      Mapping[str, iterator_lib.Iterator],
      State,
      partitioning.StepFn,
      partitioning.StepFn,
      int,
  ]:
    """Initializes the objects required for training from the task."""

    init_rng, step_rng = jax.random.split(
        jax.random.key(self._rng_seed, impl=self._rng_impl)
    )

    def _create_state(inputs: PyTree) -> State:
      return task.create_state(inputs, init_rng)

    def _train_step(
        inputs: PyTree, state: State
    ) -> tuple[State, Mapping[str, clu_metrics.Metric]]:
      rng = jax.random.fold_in(step_rng, state.step)  # pytype: disable=attribute-error
      state, metrics = task.train_step(inputs, state, rng)
      return state, {**_state_metrics(state), **metrics}

    def _eval_step(
        inputs: PyTree, state: State
    ) -> Mapping[str, clu_metrics.Metric]:
      return task.eval_step(inputs, state)

    train_iter, eval_iters = core.get_iterators(task.create_datasets())
    element_spec = (
        train_iter.element_spec
        if training
        else next(iter(eval_iters.values())).element_spec
    )
    abstract_batch = jax.tree.map(
        lambda x: np.zeros(x.shape, dtype=x.dtype), element_spec
    )

    sharded_abstract_batch = self._partitioner.shard_inputs(abstract_batch)
    init_fn = self._partitioner.partition_init(
        _create_state, abstract_batch=sharded_abstract_batch
    )
    train_step = self._partitioner.partition_step(_train_step, training=True)
    eval_step = self._partitioner.partition_step(_eval_step)

    sharded_abstract_batch = self._partitioner.shard_inputs(abstract_batch)
    state = init_fn(sharded_abstract_batch)

    if (
        check_for_checkpoints
        and self._enable_checkpointing
        and self.checkpoint_manager.latest_step() is not None
    ):
      step_to_resume_from = self.checkpoint_manager.latest_step()
      state = self._maybe_restore_checkpoint(
          state=state, step=step_to_resume_from
      )
      step_to_resume_from = step_to_resume_from or 0
    else:
      state = task.restore_state(state)
      step_to_resume_from = int(jax.device_get(state.step))

    return (
        train_iter,
        eval_iters,
        state,
        train_step,
        eval_step,
        step_to_resume_from,
    )

  def train(self, task: JaxTask) -> core.Logs:
    """Trains the model."""
    train_iter, _, state, train_step, _, step = self.process_task(
        task, training=True, check_for_checkpoints=False
    )

    logging.info(
        f"train | step: {step: 6d} | training for {self._train_steps} steps..."
    )
    metrics = {}
    while step < self._train_steps:
      num_steps = min(self._train_steps - step, self._steps_per_loop)
      state, train_metrics = self._train_n_steps(
          train_iter=train_iter,
          train_step=train_step,
          state=state,
          start_step=step,
          num_steps=num_steps,
          summary_writer=self.train_summary_writer,
      )
      curr_step = step + num_steps - 1

      logging.info(
          f"train | step: {curr_step: 6d} | metrics:"
          f" {_format_output(train_metrics)}"
      )
      metrics[core.TRAIN_LOG_DIRNAME] = train_metrics
      if self._enable_checkpointing:
        self._maybe_save_checkpoint(curr_step, state, metrics=metrics)
      step = curr_step + 1

    if self._enable_checkpointing:
      self.checkpoint_manager.wait_until_finished()

    if jax.process_index() == 0:
      self._write_marker_file()
      task.export_model(state, self._model_dir)

    if self._enable_checkpointing:
      self.checkpoint_manager.close()
      del self.checkpoint_manager

    return metrics

  def evaluate(self, task: JaxTask) -> core.Logs:
    """Evaluates the model."""
    _, eval_iters, state, _, eval_step, step = self.process_task(
        task, training=False, check_for_checkpoints=False
    )
    eval_summary_writers = self._create_eval_summary_writers(eval_iters)

    if self._steps_per_eval is not None:
      steps_msg = f"running {self._steps_per_eval} steps of evaluation..."
    else:
      steps_msg = "running complete evaluation..."

    logging.info(f"eval | step: {step: 6d} | {steps_msg}")
    metrics = {}
    with self.report_progress.timed("eval"):
      for key, eval_iter in eval_iters.items():
        eval_iter.reset()
        eval_metrics = self._evaluate_n_steps(
            eval_iter=eval_iter,
            eval_step=eval_step,
            state=state,
            num_steps=self._steps_per_eval,
            step_number=step,
            summary_writer=eval_summary_writers[key],
        )
        logging.info(
            f"eval {key} | step: {step: 6d} | metrics:"
            f" {_format_output(eval_metrics)}"
        )
        metrics[_val_logdir(key)] = eval_metrics

    return metrics

  def train_and_evaluate(self, task: JaxTask) -> core.Logs:
    """Trains and evaluates the model."""
    train_iter, eval_iters, state, train_step, eval_step, step = (
        self.process_task(task, training=True, check_for_checkpoints=False)
    )
    eval_summary_writers = self._create_eval_summary_writers(eval_iters)

    metrics = {}
    while step < self._train_steps:
      num_steps = min(self._train_steps - step, self._steps_per_loop)
      state, train_metrics = self._train_n_steps(
          train_iter=train_iter,
          train_step=train_step,
          state=state,
          start_step=step,
          num_steps=num_steps,
          summary_writer=self.train_summary_writer,
      )
      curr_step = step + num_steps - 1

      logging.info(
          f"train | step: {curr_step: 6d} | metrics:"
          f" {_format_output(train_metrics)}"
      )
      metrics[core.TRAIN_LOG_DIRNAME] = train_metrics

      if self._steps_per_eval is not None:
        steps_msg = f"running {self._steps_per_eval} steps of evaluation..."
      else:
        steps_msg = "running complete evaluation..."

      logging.info(f"eval | step: {step: 6d} | {steps_msg}")
      with self.report_progress.timed("eval"):
        for key, eval_iter in eval_iters.items():
          eval_iter.reset()
          eval_metrics = self._evaluate_n_steps(
              eval_iter=eval_iter,
              eval_step=eval_step,
              state=state,
              num_steps=self._steps_per_eval,
              step_number=curr_step,
              summary_writer=eval_summary_writers[key],
          )
          logging.info(
              f"eval {key} | step: {curr_step: 6d} | metrics:"
              f" {_format_output(eval_metrics)}"
          )
          metrics[_val_logdir(key)] = eval_metrics
      if self._enable_checkpointing:
        self._maybe_save_checkpoint(curr_step, state, metrics=metrics)
      step = curr_step + 1

    if self._enable_checkpointing:
      self.checkpoint_manager.wait_until_finished()

    if jax.process_index() == 0:
      self._write_marker_file()
      task.export_model(state, self._model_dir)

    if self._enable_checkpointing:
      self.checkpoint_manager.close()
      del self.checkpoint_manager

    return metrics

  def evaluate_continuously(self, task: JaxTask) -> core.Logs:
    """Continuously evaluates the model."""
    _, eval_iters, state, _, eval_step, _ = self.process_task(
        task, training=False, check_for_checkpoints=False
    )
    eval_summary_writers = self._create_eval_summary_writers(eval_iters)

    def timeout_fn() -> bool:
      return tf.io.gfile.exists(
          os.path.join(self._model_dir, core.TRAINING_COMPLETE_MARKER_FILE)
      )

    if self._steps_per_eval is not None:
      steps_msg = f"running {self._steps_per_eval} steps of evaluation..."
    else:
      steps_msg = "running complete evaluation..."

    metrics = {}
    for step in ocp.checkpoint_utils.checkpoints_iterator(
        os.path.join(self._model_dir, core.CHECKPOINT_DIR),
        timeout=self._continuous_eval_timeout,
        timeout_fn=timeout_fn,
    ):
      try:
        if self._enable_checkpointing:
          state = self._maybe_restore_checkpoint(state, step)
        logging.info(f"eval | step: {step: 6d} | {steps_msg}")
        with self.report_progress.timed("eval"):
          for key, eval_iter in eval_iters.items():
            eval_iter.reset()
            eval_metrics = self._evaluate_n_steps(
                eval_iter=eval_iter,
                eval_step=eval_step,
                state=state,
                num_steps=self._steps_per_eval,
                step_number=step,
                summary_writer=eval_summary_writers[key],
            )
            logging.info(
                f"eval {key} | step: {step: 6d} | metrics:"
                f" {_format_output(eval_metrics)}"
            )
            metrics[_val_logdir(key)] = eval_metrics

      except FileNotFoundError:
        logging.info("Checkpoint step: %s did not finish writing...", step)

    return metrics


def _state_metrics(state: State) -> Mapping[str, base_metrics.Metric]:
  """Utility method to add state statistics to metrics."""

  def _param_count(params: PyTree) -> int:
    return sum([math.prod(x.shape) for x in jax.tree.flatten(params)[0]])

  def _name(prefix: str, key: str) -> str:
    return f"{prefix}_{key}" if prefix else key

  metrics = {}

  if isinstance(state, JaxState):
    metrics["model/num_trainable_variables"] = base_metrics.scalar(
        _param_count(state.params)
    )
    metrics["model/num_non_trainable_variables"] = base_metrics.scalar(
        _param_count(state.mutable)
    )
  elif isinstance(state, KerasState):
    metrics["model/num_trainable_variables"] = base_metrics.scalar(
        _param_count(state.tvars)
    )
    metrics["model/num_non_trainable_variables"] = base_metrics.scalar(
        _param_count(state.ntvars)
    )
  else:
    if hasattr(state, "params"):
      metrics["model/num_trainable_variables"] = base_metrics.scalar(
          _param_count(state.params)
      )

  def _add_optimizer_metrics(opt_state: optax.OptState, prefix: str):
    if isinstance(opt_state, optax.MultiTransformState):
      for name, inner_state in opt_state.inner_states.items():
        _add_optimizer_metrics(inner_state, _name(prefix, name))
    elif isinstance(
        opt_state,
        (optax.InjectStatefulHyperparamsState, optax.InjectHyperparamsState),
    ):
      for key, hparam in opt_state.hyperparams.items():
        if isinstance(hparam, (int, float)) or (
            isinstance(hparam, (np.ndarray, jax.Array))
            and np.prod(hparam.shape) == 1
        ):
          metrics[f"optimizer/{_name(prefix, key)}"] = base_metrics.scalar(
              hparam
          )
    elif isinstance(opt_state, (list, tuple)):
      for opt_state in opt_state:
        _add_optimizer_metrics(opt_state, prefix)

  _add_optimizer_metrics(state.opt_state, prefix="")
  metrics["optimizer/num_params"] = base_metrics.scalar(
      _param_count(state.opt_state)
  )
  return metrics


def _val_logdir(key: str) -> str:
  return (
      "_".join([core.EVAL_LOG_DIRNAME, key]) if key else core.EVAL_LOG_DIRNAME
  )


def _format_output(output: Any, indent: int = 4, width: int = 80) -> str:
  """Formats `output`, either on one line, or indented across multiple lines."""
  formatted = pprint.pformat(output, width=width)
  lines = formatted.splitlines()
  if len(lines) == 1:
    return formatted
  lines = [" " * indent + line for line in lines]
  return "\n" + "\n".join(lines)


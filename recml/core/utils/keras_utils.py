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
"""Utilities for training Keras models on Jax backend."""

from collections.abc import Mapping
import dataclasses
import functools
import os
from typing import Any

from absl import logging
import getpass
import jax
import keras
import orbax.checkpoint as ocp
import tensorflow as tf

STATE_CHECKPOINT_KEY = "state"
TRAINABLE_VARIABLES_KEY = "trainable_variables"
NON_TRAINABLE_VARIABLES_KEY = "non_trainable_variables"
OPTIMIZER_VARIABLES_KEY = "optimizer_variables"
CONFIG_CHECKPOINT_KEY = "config"
ORBAX_CHECKPOINT_DEFAULT_KEY = "default"
S2_MARKER_KEY = "s2_resumable_marker"


def _assert_variables_built(model: keras.Model):
  if not model.built or not model.optimizer.built:
    raise ValueError(
        "To use methods on `KerasOrbaxCheckpointManager`, your model and"
        f" optimizer must be built. Model built: {model.built}, Optimizer"
        f" built: {model.optimizer.built}"
    )


def _assert_all_layers_built(model: keras.Model):
  flattened_layers = model._flatten_layers(include_self=True)  # pylint: disable=protected-access
  if not all(layer.built for layer in flattened_layers):
    raise ValueError(
        "To save or restore a checkpoint with a Keras model, the model and"
        " all of its layers must be built. The layers that are not built"
        " properly are the following:"
        f" {[layer for layer in flattened_layers if not layer.built]}."
    )


def _to_shape_dtype_struct(x: keras.Variable) -> jax.ShapeDtypeStruct:
  if not isinstance(x, keras.Variable):
    raise ValueError(f"Expected a `keras.Variable`, got {type(x)}.")
  return jax.ShapeDtypeStruct(
      shape=x.value.shape,
      dtype=x.value.dtype,
      sharding=x.value.sharding,
  )


class KerasOrbaxCheckpointManagerV2(ocp.CheckpointManager):
  """An Orbax checkpoint manager for Keras 3."""

  def __init__(
      self,
      checkpoint_dir: str,
      max_to_keep: int = 5,
      save_interval_epochs: int = 1,
  ):
    """Initializes a KerasOrbaxCheckpointManager.

    Args:
      checkpoint_dir: The directory to save checkpoints to.
      max_to_keep: The maximum number of checkpoints to keep.
      save_interval_epochs: The interval (in epochs) to save checkpoints.
    """
    if keras.backend.backend() != "jax":
      raise ValueError(
          "`KerasOrbaxCheckpointManagerV2` is only supported on a `jax`"
          " backend."
      )
    super().__init__(
        directory=checkpoint_dir,
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=save_interval_epochs,
            max_to_keep=max_to_keep,
        ),
    )

  def save_model_variables(
      self,
      model: keras.Model,
      epoch: int,
      logs: Mapping[str, Any] | None = None,
  ):
    """Saves the model variables and optimizer variables to a checkpoint."""
    _assert_variables_built(model)
    _assert_all_layers_built(model)

    if not model._jax_state_synced:  # pylint: disable=protected-access
      model.jax_state_sync()

    variables = {
        TRAINABLE_VARIABLES_KEY: model.trainable_variables,
        NON_TRAINABLE_VARIABLES_KEY: model.non_trainable_variables,
        OPTIMIZER_VARIABLES_KEY: model.optimizer.variables,
    }
    state = jax.tree.map(lambda x: x.value, variables)
    config = keras.utils.serialize_keras_object(model)

    logging.info("Saving checkpoint for epoch %s...", epoch)

    args = {
        STATE_CHECKPOINT_KEY: ocp.args.StandardSave(state),
        CONFIG_CHECKPOINT_KEY: ocp.args.JsonSave(config),
    }

    self.save(
        step=epoch,
        args=ocp.args.Composite(**args),
        metrics=logs,
    )

  def restore_model_variables(self, model: keras.Model, epoch: int):
    """Restores the model variables and optimizer variables during training."""

    _assert_variables_built(model)
    _assert_all_layers_built(model)

    if not model._jax_state_synced:  # pylint: disable=protected-access
      model.jax_state_sync()

    variables = {
        TRAINABLE_VARIABLES_KEY: model.trainable_variables,
        NON_TRAINABLE_VARIABLES_KEY: model.non_trainable_variables,
        OPTIMIZER_VARIABLES_KEY: model.optimizer.variables,
    }

    # TODO(zixiangzhou): Update variables to use a nested dictionary and index
    # map instead of flattened list.

    # Construct abstract variables to ensure the checkpoint is restored with
    # the same sharding as the current variables. This is so we can delete the
    # variables from device memory to reduce peak memory usage.
    abstract_variables = jax.tree.map(_to_shape_dtype_struct, variables)
    for var in jax.tree.flatten(variables)[0]:
      var.value.delete()
      var._value = None  # pylint: disable=protected-access

    logging.info("Restoring checkpoint for epoch %s...", epoch)

    restored_items = self.restore(
        step=epoch,
        args=ocp.args.Composite(**{
            STATE_CHECKPOINT_KEY: ocp.args.StandardRestore(abstract_variables)
        }),
    )
    restored_variables = restored_items[STATE_CHECKPOINT_KEY]

    logging.info("Restored checkpoint for epoch %s.", epoch)

    model._initial_epoch = epoch + 1  # pylint: disable=protected-access

    keras.tree.assert_same_structure(variables, restored_variables)
    for var, restored_var in zip(
        jax.tree.flatten(variables)[0], jax.tree.flatten(restored_variables)[0]
    ):
      var._value = restored_var  # pylint: disable=protected-access


def restore_keras_checkpoint(
    checkpoint_dir: str,
    *,
    model: keras.Model | None = None,
    epoch: int | None = None,
    compile: bool = False,  # pylint: disable=redefined-builtin
    restore_optimizer_vars: bool = False,
    restore_model_epoch: bool = False,
    restore_iterations: bool = True,
) -> keras.Model:
  """Restores a Keras 3 Jax backend model from an Orbax checkpoint.

  Args:
    checkpoint_dir: The directory containing the Orbax checkpoint(s).
    model: The Keras model to restore. If not provided, the model will be
      instantiated from the config stored in the checkpoint if available.
      Otherwise and error will be thrown.
    epoch: The epoch to restore the checkpoint from. If None, the latest
      checkpoint will be used.
    compile: Whether to compile the model when it is instantiated from the
      checkpoint config. If `model` is provided, this argument is ignored.
      Defaults to False.
    restore_optimizer_vars: Whether to restore the optimizer variables from the
      checkpoint. Defaults to False.
    restore_model_epoch: Whether to restore the epoch on the model. If set, the
      epoch on the model will be restored to `epoch + 1` so the model can
      continue training from where it left off. Defaults to False.
    restore_iterations: Whether to restore the optimizer iterations from the
      checkpoint when `restore_optimizer_vars` is True. This is an optimizer
      variable used for controlling the learning rate schedule. Defaults to
      True.

  Returns:
    A Keras model with the weights restored from the checkpoint. If the model
    was provided, a reference to the same model is returned.

  Raises:
    ValueError: If the Keras backend is not "jax" or if the checkpoint does not
      contain a model config and `model` is not provided.
    FileNotFoundError: If no checkpoints are found in the checkpoint directory.
    ValueError: If the specified `epoch` is not found in the checkpoint
      directory.
    ValueError: If the model is not built when `restore_optimizer_vars` is True.
  """

  if keras.backend.backend() != "jax":
    raise ValueError(
        "This function only supports restoring a Keras 3 Jax backend model."
    )
  if restore_optimizer_vars and model is None:
    raise ValueError(
        "To use `restore_keras_checkpoint` with `restore_optimizer_vars` set to"
        " True, a model must be provided."
    )

  metadata = ocp.path.step.latest_step_metadata(
      checkpoint_dir, ocp.path.step.standard_name_format()
  )
  if metadata is None:
    raise FileNotFoundError(
        f"No checkpoints found in {checkpoint_dir}. Please ensure that the"
        " checkpoint directory contains Orbax checkpoints."
    )
  if epoch is None:
    epoch = metadata.step
  elif epoch not in ocp.path.step.checkpoint_steps(checkpoint_dir):
    raise ValueError(
        f"Step {epoch} not found in {checkpoint_dir}. Please ensure you specify"
        " a valid step. Available steps:"
        f" {ocp.path.step.checkpoint_steps(checkpoint_dir)}"
    )

  checkpoint_path = ocp.path.step.build_step_path(
      checkpoint_dir, ocp.path.step.standard_name_format(), epoch
  )

  if model is None:
    cfg = {**load_keras_model_config(checkpoint_dir, epoch=epoch)}
    if not compile and "compile_config" in cfg:
      cfg.pop("compile_config")

    model: keras.Model = keras.utils.deserialize_keras_object(cfg)
    if not model.built:
      if "build_config" not in cfg:
        raise ValueError(
            "To use `restore_keras_checkpoint` on a model checkpoint without"
            " passing a model the `build_config` must be present in the config."
            " Make sure the you have implemented `get_build_config` correctly."
            " Generally, you shouldn't need to do this and the default"
            " implementation should work for most cases."
        )
      model.build_from_config(cfg["build_config"])
  elif not model._jax_state_synced:  # pylint: disable=protected-access
    model.jax_state_sync()

  _assert_all_layers_built(model)

  variables = {
      TRAINABLE_VARIABLES_KEY: model.trainable_variables,
      NON_TRAINABLE_VARIABLES_KEY: model.non_trainable_variables,
  }
  if restore_optimizer_vars:
    if not model.optimizer.built:
      raise ValueError(
          "To use `restore_keras_checkpoint` on an existing model with"
          " `restore_optimizer_vars` set to True, the optimizer must be"
          " built."
      )
    variables[OPTIMIZER_VARIABLES_KEY] = model.optimizer.variables

  # TODO(zixiangzhou): Update variables to use a nested dictionary and index map
  # instead of flattened list.

  # Construct abstract variables to ensure the checkpoint is restored with
  # the same sharding as the current variables.
  abstract_state = jax.tree.map(_to_shape_dtype_struct, variables)

  # Delete the variables from device memory to reduce peak memory usage.
  for var in jax.tree.flatten(variables)[0]:
    var.value.delete()
    var._value = None  # pylint: disable=protected-access

  # TODO(aahil): Look into converging the logic here with the checkpointing
  # logic in KerasOrbaxCheckpointManagerV2.
  checkpointer = ocp.Checkpointer(
      ocp.CompositeCheckpointHandler(**{
          STATE_CHECKPOINT_KEY: ocp.handlers.PyTreeCheckpointHandler(
              restore_concurrent_gb=96,
          ),
      })
  )
  restored_state = checkpointer.restore(
      checkpoint_path,
      args=ocp.args.Composite(**{
          STATE_CHECKPOINT_KEY: ocp.args.PyTreeRestore(
              abstract_state,
              transforms={},
              restore_args=ocp.checkpoint_utils.construct_restore_args(
                  abstract_state
              ),
          ),
      }),
  )[STATE_CHECKPOINT_KEY]
  checkpointer.close()

  # TODO(zixiangzhou): Unflatten the variables based on index here.
  keras.tree.assert_same_structure(variables, restored_state)
  for var, restored_var in zip(
      jax.tree.flatten(variables)[0], jax.tree.flatten(restored_state)[0]
  ):
    var._value = restored_var  # pylint: disable=protected-access

  if restore_model_epoch:
    model._initial_epoch = epoch + 1  # pylint: disable=protected-access
  if restore_optimizer_vars and not restore_iterations:
    model.optimizer.iterations.assign(0)

  return model


@functools.lru_cache
def load_keras_model_config(
    checkpoint_dir: str, epoch: int | None = None
) -> Mapping[str, Any]:
  """Loads a Keras model from a checkpoint directory."""
  if keras.backend.backend() != "jax":
    raise ValueError(
        "This function only supports loading a Keras 3 Jax backend model."
    )

  metadata = ocp.path.step.latest_step_metadata(
      checkpoint_dir, ocp.path.step.standard_name_format()
  )
  if metadata is None:
    raise FileNotFoundError(
        f"No checkpoints found in {checkpoint_dir}. Please ensure that the"
        " checkpoint directory contains Orbax checkpoints."
    )
  if epoch is None:
    epoch = metadata.step
  elif epoch not in ocp.path.step.checkpoint_steps(checkpoint_dir):
    raise ValueError(
        f"Step {epoch} not found in {checkpoint_dir}. Please ensure you specify"
        " a valid step. Available steps:"
        f" {ocp.path.step.checkpoint_steps(checkpoint_dir)}"
    )

  checkpoint_path = ocp.path.step.build_step_path(
      checkpoint_dir, ocp.path.step.standard_name_format(), epoch
  )

  json_checkpointer = ocp.Checkpointer(
      ocp.CompositeCheckpointHandler(
          **{CONFIG_CHECKPOINT_KEY: ocp.handlers.JsonCheckpointHandler()}
      )
  )
  cfg = json_checkpointer.restore(
      checkpoint_path,
      args=ocp.args.Composite(
          **{CONFIG_CHECKPOINT_KEY: ocp.args.JsonRestore()}
      ),
  )[CONFIG_CHECKPOINT_KEY]
  json_checkpointer.close()
  return cfg


class KerasOrbaxCheckpointManager(ocp.CheckpointManager):
  """An Orbax checkpoint manager for Keras 3."""

  def __init__(
      self,
      checkpoint_dir: str,
      max_to_keep: int = 5,
      save_interval_epochs: int = 1,
  ):
    """Initializes a KerasOrbaxCheckpointManager.

    Args:
      checkpoint_dir: The directory to save checkpoints to.
      max_to_keep: The maximum number of checkpoints to keep.
      save_interval_epochs: The interval (in epochs) to save checkpoints.
    """
    super().__init__(
        directory=checkpoint_dir,
        checkpointers=ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler()),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=save_interval_epochs,
            max_to_keep=max_to_keep,
        ),
    )

  def save_model_variables(
      self,
      model: keras.Model,
      epoch: int,
      logs: Mapping[str, Any] | None = None,
  ):

    _assert_variables_built(model)
    state = model._get_jax_state(  # pylint: disable=protected-access
        trainable_variables=True,
        non_trainable_variables=True,
        optimizer_variables=True,
        # metrics_variables is default to False because we don't want to save
        # metrics variables in the checkpoint. The metrics varibles are reset
        # after each epoch. We need to recalculate them after restoring from
        # the checkpoint.
        metrics_variables=False,
    )
    logging.info("Writing checkpoint for epoch %s...", epoch)

    self.save(step=epoch, items=state, metrics=logs)

  def restore_model_variables(self, model: keras.Model, epoch: int):
    _assert_variables_built(model)
    state = model._get_jax_state(  # pylint: disable=protected-access
        trainable_variables=True,
        non_trainable_variables=True,
        optimizer_variables=True,
        purge_model_variables=True,
    )
    logging.info("Restoring checkpoint for epoch %s...", epoch)
    model._jax_state_synced = False  # pylint: disable=protected-access

    def _restore(value):
      if isinstance(value, jax.Array):
        return ocp.type_handlers.ArrayRestoreArgs(
            restore_type=jax.Array,
            sharding=value.sharding,
            global_shape=value.shape,
            dtype=value.dtype,
        )
      return ocp.type_handlers.RestoreArgs(
          restore_type=type(value),
          dtype=value.dtype if hasattr(value, "dtype") else None,
      )

    restore_args = jax.tree.map(_restore, state)
    # TODO(zixiangzhou): 'transforms' is a walkaround to avoid the error of
    # loading a checkpoint that has a different number of variables than the
    # current state because we don't want to load metrics_variables. But this
    # might lead to future bugs when the checkpoint does not exactly match the
    # defined model state. Currently, 'transforms' won't work if the order of
    # the variables is different from the checkpoint or new variables are added.
    # A better solution is to add keys for variables when checkpointing to use
    # the 'transforms' API (mapping by variable keys).
    restored_state = self.restore(
        step=epoch,
        args=ocp.args.PyTreeRestore(
            state,
            transforms={},
            restore_args=restore_args,
        ),
        directory=str(self.directory),
    )
    logging.info("Restored checkpoint for epoch %s.", epoch)
    model._initial_epoch = epoch + 1  # pylint: disable=protected-access
    (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
    ) = restored_state
    model._jax_state = {  # pylint: disable=protected-access
        "trainable_variables": trainable_variables,
        "non_trainable_variables": non_trainable_variables,
        "optimizer_variables": optimizer_variables,
    }
    model.jax_state_sync()

class EpochOrbaxCheckpointAndRestoreCallback(keras.callbacks.Callback):
  """A callback for checkpointing and restoring state using Orbax."""

  def __init__(
      self,
      checkpoint_manager: (
          KerasOrbaxCheckpointManager | KerasOrbaxCheckpointManagerV2
      ),
      marker_path: str | None = None,
  ):
    if keras.backend.backend() != "jax":
      raise ValueError(
          "`EpochOrbaxCheckpointAndRestoreCallback` is only supported on a"
          " `jax` backend."
      )

    self._checkpoint_manager = checkpoint_manager
    self._marker_path = marker_path
    # Marks the callback as async safe so batch end callbacks can be dispatched
    # asynchronously.
    self.async_safe = True

  def on_train_begin(self, logs: Mapping[str, Any] | None = None):
    if not self.model.built or not self.model.optimizer.built:
      raise ValueError(
          "To use `EpochOrbaxCheckpointAndRestoreCallback`, "
          "your model and optimizer must be built before you call `fit()`."
      )

    latest_epoch = self._checkpoint_manager.latest_step()
    if latest_epoch is not None:
      self._checkpoint_manager.restore_model_variables(self.model, latest_epoch)
    else:
      # save the model checkpoint at the begining of the training.
      # So that the continuous eval job finds it and logs the eval at step 0
      s2_marker = None
      self._checkpoint_manager.save_model_variables(
          self.model, 0, logs, s2_marker=s2_marker
      )

  def on_epoch_end(self, epoch: int, logs: Mapping[str, Any] | None = None):
    s2_marker = None
    self._checkpoint_manager.save_model_variables(
        self.model, epoch, logs, s2_marker=s2_marker
    )

  def on_train_end(self, logs: Mapping[str, Any] | None = None):
    self._checkpoint_manager.wait_until_finished()
    if self._marker_path is not None and jax.process_index() == 0:
      with tf.io.gfile.GFile(self._marker_path, "w") as f:
        f.write("COMPLETED")


def restore_keras_model(
    model: keras.Model,
    checkpoint_dir: str,
    step: int | None = None,
    restore_optimizer_vars: bool = True,
    restore_steps: bool = True,
    restore_iterations: bool = True,
):
  """Restores a Keras 3 Jax backend model from an Orbax checkpoint.

  This is only compatible with `KerasOrbaxCheckpointManager`. If you are using
  `KerasOrbaxCheckpointManagerV2`, use `restore_keras_checkpoint` instead.

  Args:
    model: The Keras model to restore.
    checkpoint_dir: The directory containing the Orbax checkpoints.
    step: The checkpoint step to resume training from. If set, it requires a
      checkpoint with the same step number to be present in the model directory.
      If not set, will resume training from the last checkpoint. Depending on
      the value of `max_checkpoints_to_keep`, the model directory only contains
      a certain number of the latest checkpoints.
    restore_optimizer_vars: Whether to restore the optimizer variables.
    restore_steps: Whether to restore the model's steps. If `True` then the
      model will continue training from the step the checkpoint was saved at. If
      `False` then the model will start training from the first step.
    restore_iterations: Whether to restore the model's iterations. If `True`
      then the model will continue training from the iteration the checkpoint
      was saved at. This is an optimizer variable used for controlling the
      learning rate schedule. This is not supported if restore_optimizer_vars is
      `False`.

  Raises:
    FileNotFoundError: If no checkpoints are found in the checkpoint directory.
    ValueError: If the specified step is not found in the checkpoint directory
      or if the model or the optimizer is not built.
  """
  if keras.backend.backend() != "jax":
    raise ValueError(
        "This function only supports restoring a Keras 3 Jax backend model from"
        " a TF Saved Model."
    )

  _assert_variables_built(model)

  metadata = ocp.path.step.latest_step_metadata(
      checkpoint_dir, ocp.path.step.standard_name_format()
  )
  if metadata is None:
    raise FileNotFoundError(
        f"No checkpoints found in {checkpoint_dir}. Please ensure that the"
        " checkpoint directory contains Orbax checkpoints."
    )
  if step is None:
    step = metadata.step
  elif step not in ocp.path.step.checkpoint_steps(checkpoint_dir):
    raise ValueError(
        f"Step {step} not found in {checkpoint_dir}. Please ensure you specify "
        "a valid step. Available steps: "
        f"{ocp.path.step.checkpoint_steps(checkpoint_dir)}"
    )

  checkpointer = ocp.Checkpointer(
      ocp.CompositeCheckpointHandler(**{
          ORBAX_CHECKPOINT_DEFAULT_KEY: ocp.handlers.PyTreeCheckpointHandler()
      })
  )
  state = model._get_jax_state(  # pylint: disable=protected-access
      trainable_variables=True,
      non_trainable_variables=True,
      optimizer_variables=restore_optimizer_vars,
      purge_model_variables=True,
  )
  model._jax_state_synced = False  # pylint: disable=protected-access

  # Delete the state to save memory.
  abstract_state = jax.tree.map(ocp.utils.to_shape_dtype_struct, state)
  jax.tree.map(
      lambda x: x.delete() if isinstance(x, jax.Array) else None, state
  )
  checkpoint_path = ocp.path.step.build_step_path(
      checkpoint_dir, ocp.path.step.standard_name_format(), step
  )
  # TODO(zixiangzhou): 'transforms' is a walkaround to avoid the error of
  # loading a checkpoint that has a different number of variables than the
  # current state because we don't want to load metrics_variables. But this
  # might lead to future bugs when the checkpoint does not exactly match the
  # defined model state. Currently, 'transforms' won't work if the order of
  # the variables is different from the checkpoint or new variables are added.
  # A better solution is to add keys for variables when checkpointing to use
  # the 'transforms' API (mapping by variable keys).
  restored_state = checkpointer.restore(
      checkpoint_path,
      args=ocp.args.Composite(**{
          ORBAX_CHECKPOINT_DEFAULT_KEY: ocp.args.PyTreeRestore(
              item=abstract_state,
              transforms={},
              restore_args=ocp.checkpoint_utils.construct_restore_args(
                  abstract_state
              ),
          ),
      }),
  )[ORBAX_CHECKPOINT_DEFAULT_KEY]
  (
      trainable_variables,
      non_trainable_variables,
  ) = restored_state[:2]
  model._jax_state = {  # pylint: disable=protected-access
      "trainable_variables": trainable_variables,
      "non_trainable_variables": non_trainable_variables,
  }
  if restore_optimizer_vars:
    optimizer_variables = restored_state[2]
    model._jax_state["optimizer_variables"] = optimizer_variables  # pylint: disable=protected-access
  model.jax_state_sync()
  if restore_steps:
    model._initial_epoch = step + 1  # pylint: disable=protected-access
  if restore_optimizer_vars and not restore_iterations:
    model.optimizer.iterations.assign(0)


# TODO(b/343544467): Support logging metrics more frequently.
class EpochSummaryCallback(keras.callbacks.TensorBoard):
  """A custom summary callback that only reports epoch metrics."""

  def __init__(
      self,
      log_dir: str,
      steps_per_epoch: int,
      write_steps_per_second: bool = True,
  ):
    super().__init__(
        log_dir,
        write_steps_per_second=write_steps_per_second,
        update_freq="epoch",
        write_graph=False,
    )
    self._steps_per_epoch = steps_per_epoch
    self._num_params = None
    # Marks the callback as async safe so batch end callbacks can be dispatched
    # asynchronously.
    self.async_safe = True

  def _get_num_params(self, training: bool) -> dict[str, int]:
    if self._num_params is None:
      self._num_params = {
          "num_params/trainable": keras.src.utils.summary_utils.count_params(
              self.model.trainable_variables
          ),
          "num_params/non_trainable": (
              keras.src.utils.summary_utils.count_params(
                  self.model.non_trainable_variables
              )
          ),
          "num_params/optimizer": keras.src.utils.summary_utils.count_params(
              self.model.optimizer.variables
          ),
      }
      self._num_params["num_params/total"] = sum(self._num_params.values())
    if not training:
      return {"val_" + k: v for k, v in self._num_params.items()}
    return self._num_params

  def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None):
    if not logs:
      return

    step = epoch * self._steps_per_epoch
    train_logs = {k: v for k, v in logs.items() if not k.startswith("val_")}
    val_logs = {k: v for k, v in logs.items() if k.startswith("val_")}
    train_logs = self._collect_learning_rate(train_logs)
    if self.write_steps_per_second:
      train_logs["steps_per_second"] = self._compute_steps_per_second()

    if train_logs:
      num_params = self._get_num_params(training=True)
      logs.update(num_params)
      train_logs.update(num_params)
      with self._train_writer.as_default():
        for name, value in train_logs.items():
          self.summary.scalar(name, value, step=step)

    if val_logs:
      num_params = self._get_num_params(training=False)
      logs.update(num_params)
      val_logs.update(num_params)
      with self._val_writer.as_default():
        for name, value in val_logs.items():
          self.summary.scalar(name.removeprefix("val_"), value, step=step)

  def _collect_learning_rate(self, logs: Any) -> Any:
    if not self.model:
      return logs
    optimizer = self.model.optimizer
    if isinstance(optimizer, keras.optimizers.Optimizer):
      if hasattr(optimizer, "learning_rates"):
        learning_rates = optimizer.learning_rates
        if isinstance(learning_rates, Mapping):
          for k, v in learning_rates.items():
            logs["learning_rate/" + k] = float(keras.ops.convert_to_numpy(v))
      else:
        logs["learning_rate"] = float(
            keras.ops.convert_to_numpy(optimizer.learning_rate)
        )
    return logs

  def on_test_end(self, logs=None):
    self._pop_writer()

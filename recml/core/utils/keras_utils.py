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

from collections.abc import Mapping, Sequence
import dataclasses
import datetime
import enum
import os
import re
from typing import Any

from absl import logging
from etils import epath
import jax
import keras
import orbax.checkpoint as ocp
import tensorflow as tf


STATE_CHECKPOINT_KEY = "state"
TRAINABLE_VARIABLES_KEY = "trainable_variables"
NON_TRAINABLE_VARIABLES_KEY = "non_trainable_variables"
OPTIMIZER_VARIABLES_KEY = "optimizer_variables"
CONFIG_CHECKPOINT_KEY = "config"
FORMAT_VERSION_KEY = "format_version"
NON_TRAINABLE_PATHS_KEY = "non_trainable_paths"
OPTIMIZER_PATHS_KEY = "optimizer_paths"
ORBAX_CHECKPOINT_DEFAULT_KEY = "default"


class CheckpointVersion(enum.StrEnum):
  V1 = "v1"
  V2 = "v2"
  V3 = "v3"


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


def _variables_to_path_dict(
    variables: Sequence[keras.Variable],
    collection_name: str,
) -> dict[str, keras.Variable]:
  """Converts a sequence of variables to a dict mapped by path, checking for duplicates."""
  var_dict = {}
  duplicates = []
  for v in variables:
    if v.path in var_dict:
      duplicates.append(v.path)
    else:
      var_dict[v.path] = v
  if duplicates:
    raise ValueError(
        f"Duplicate variable paths detected in {collection_name}. Ensure"
        " unique layer names (e.g. name_layers=True). Duplicates: "
        f"{duplicates}"
    )
  return var_dict


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
    self.save(
        step=epoch,
        args=ocp.args.Composite(**{
            STATE_CHECKPOINT_KEY: ocp.args.StandardSave(state),
            CONFIG_CHECKPOINT_KEY: ocp.args.JsonSave(config),
        }),
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


class KerasOrbaxCheckpointManagerV3(ocp.CheckpointManager):
  """An Orbax checkpoint manager for Keras 3 with dictionary state.

  This manager saves the full training state (trainable, non-trainable, and
  optimizer variables). For training resume and preemption recovery, the full
  state is restored via `restore_keras_checkpoint`.

  For selective weight transfer (warm-starting from a checkpoint of a model
  with a different architecture), use `restore_partial_checkpoint`. Note that
  partial restoration is restricted to trainable variables (weights).
  Non-trainable and optimizer variables are specific to the training run and
  are not supported for partial transfer.
  """

  def __init__(
      self,
      checkpoint_dir: str,
      max_to_keep: int = 5,
      save_interval_epochs: int = 1,
  ):
    """Initializes a KerasOrbaxCheckpointManagerV3.

    Args:
      checkpoint_dir: The directory to save checkpoints to.
      max_to_keep: The maximum number of checkpoints to keep.
      save_interval_epochs: The interval (in epochs) to save checkpoints.
    """
    if keras.backend.backend() != "jax":
      raise ValueError(
          "`KerasOrbaxCheckpointManagerV3` is only supported on a `jax`"
          " backend."
      )
    super().__init__(
        directory=checkpoint_dir,
        item_names=(
            STATE_CHECKPOINT_KEY,
            CONFIG_CHECKPOINT_KEY,
            FORMAT_VERSION_KEY,
            NON_TRAINABLE_PATHS_KEY,
            OPTIMIZER_PATHS_KEY,
        ),
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

    trainable_variables = _variables_to_path_dict(
        model.trainable_variables, TRAINABLE_VARIABLES_KEY
    )
    non_trainable_variables = _variables_to_path_dict(
        model.non_trainable_variables, NON_TRAINABLE_VARIABLES_KEY
    )
    optimizer_variables = _variables_to_path_dict(
        model.optimizer.variables, OPTIMIZER_VARIABLES_KEY
    )

    # Extract values from keras.Variable instances
    state = {
        TRAINABLE_VARIABLES_KEY: {
            k: v.value for k, v in trainable_variables.items()
        },
        NON_TRAINABLE_VARIABLES_KEY: {
            k: v.value for k, v in non_trainable_variables.items()
        },
        OPTIMIZER_VARIABLES_KEY: {
            k: v.value for k, v in optimizer_variables.items()
        },
    }
    config = keras.utils.serialize_keras_object(model)
    non_trainable_paths = {
        "paths": [v.path for v in model.non_trainable_variables]
    }
    optimizer_paths = {"paths": [v.path for v in model.optimizer.variables]}
    logging.info("SAVED non_trainable_paths: %s", non_trainable_paths)
    logging.info("SAVED optimizer_paths: %s", optimizer_paths)

    logging.info("Saving checkpoint for epoch %s...", epoch)
    self.save(
        step=epoch,
        args=ocp.args.Composite(**{
            STATE_CHECKPOINT_KEY: ocp.args.PyTreeSave(state),
            CONFIG_CHECKPOINT_KEY: ocp.args.JsonSave(config),
            FORMAT_VERSION_KEY: ocp.args.JsonSave({"version": 3}),
            NON_TRAINABLE_PATHS_KEY: ocp.args.JsonSave(non_trainable_paths),
            OPTIMIZER_PATHS_KEY: ocp.args.JsonSave(optimizer_paths),
        }),
        metrics=logs,
    )

  def restore_model_variables(self, model: keras.Model, epoch: int):
    """Restores the model variables and optimizer variables during training."""

    _assert_variables_built(model)
    _assert_all_layers_built(model)

    if not model._jax_state_synced:  # pylint: disable=protected-access
      model.jax_state_sync()

    trainable_variables = _variables_to_path_dict(
        model.trainable_variables, TRAINABLE_VARIABLES_KEY
    )
    non_trainable_variables = _variables_to_path_dict(
        model.non_trainable_variables, NON_TRAINABLE_VARIABLES_KEY
    )
    optimizer_variables = _variables_to_path_dict(
        model.optimizer.variables, OPTIMIZER_VARIABLES_KEY
    )

    variables = {
        TRAINABLE_VARIABLES_KEY: trainable_variables,
        NON_TRAINABLE_VARIABLES_KEY: non_trainable_variables,
        OPTIMIZER_VARIABLES_KEY: optimizer_variables,
    }

    # Construct abstract variables to ensure the checkpoint is restored with
    # the same sharding as the current variables.
    abstract_variables = jax.tree.map(_to_shape_dtype_struct, variables)
    for var in jax.tree.flatten(variables)[0]:
      var.value.delete()
      var._value = None  # pylint: disable=protected-access

    logging.info("Restoring checkpoint for epoch %s...", epoch)

    step_path = os.path.join(self.directory, str(epoch))
    abstract_variables, state_transforms = _prepare_v3_restore(
        step_path,
        abstract_variables,
        model,
        restore_optimizer_vars=True,
    )

    restored_items = self.restore(
        step=epoch,
        args=ocp.args.Composite(**{
            STATE_CHECKPOINT_KEY: ocp.args.PyTreeRestore(
                abstract_variables,
                transforms=state_transforms,
                restore_args=ocp.checkpoint_utils.construct_restore_args(
                    abstract_variables
                ),
            )
        }),
    )
    restored_variables = restored_items[STATE_CHECKPOINT_KEY]

    logging.info("Restored checkpoint for epoch %s.", epoch)

    model._initial_epoch = epoch + 1  # pylint: disable=protected-access

    keras.tree.assert_same_structure(variables, restored_variables)

    for key in [
        TRAINABLE_VARIABLES_KEY,
        NON_TRAINABLE_VARIABLES_KEY,
        OPTIMIZER_VARIABLES_KEY,
    ]:
      var_dict = variables[key]
      restored_var_dict = restored_variables[key]
      for path, var in var_dict.items():
        var._value = restored_var_dict[path]  # pylint: disable=protected-access


def resolve_orbax_checkpoint_path(
    checkpoint_dir: str, epoch: int | None = None
) -> tuple[str, int | None]:
  """Resolves the checkpoint path and epoch for an Orbax checkpoint.

  This function handles two cases:
  1. Flat Orbax Checkpoint: If `checkpoint_dir` is itself a valid Orbax
     checkpoint (as determined by `ocp.path.format_utils.is_orbax_checkpoint`),
     it is returned as-is along with the provided epoch.
  2. Nested Step Directories: If `checkpoint_dir` contains step subdirectories
     (e.g., `0`, `1000`), it resolves to the latest step if `epoch` is
     None, or the specified `epoch`.

  Args:
    checkpoint_dir: The directory of or containing the Orbax checkpoints.
    epoch: Optional epoch (step) number to resolve. Defaults to None, which
      resolves to the latest step for nested directories. Ignored if the
      checkpoint_dir is detected as a flat Orbax checkpoint directly.

  Returns:
    A tuple (resolved_checkpoint_path, resolved_epoch), where
    resolved_checkpoint_path is the directory of the resolved checkpoint, and
    resolved_epoch is the resolved epoch number.

  Raises:
    FileNotFoundError: If no checkpoints are found in `checkpoint_dir`.
    ValueError: If the specified `epoch` is not found in `checkpoint_dir`.
  """
  if ocp.path.format_utils.is_orbax_checkpoint(checkpoint_dir):
    return checkpoint_dir, epoch

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
        f"Step {epoch} not found in {checkpoint_dir}. Please ensure you"
        " specify a valid step. Available steps:"
        f" {ocp.path.step.checkpoint_steps(checkpoint_dir)}"
    )

  checkpoint_path = ocp.path.step.build_step_path(
      checkpoint_dir, ocp.path.step.standard_name_format(), epoch
  )
  return os.fspath(checkpoint_path), epoch


def _is_v1_checkpoint_path(checkpoint_path: str) -> bool:
  """Checks if a resolved checkpoint path is in V1 format."""
  return gfile.Exists(
      os.path.join(checkpoint_path, ORBAX_CHECKPOINT_DEFAULT_KEY)
  )


def _is_v3_checkpoint_path(checkpoint_path: str) -> bool:
  """Checks if a resolved checkpoint path is in V3 format."""
  if not gfile.Exists(os.path.join(checkpoint_path, FORMAT_VERSION_KEY)):
    return False

  version_checkpointer = ocp.Checkpointer(
      ocp.CompositeCheckpointHandler(
          **{FORMAT_VERSION_KEY: ocp.handlers.JsonCheckpointHandler()}  # pyrefly: ignore[bad-argument-type]
      )
  )
  is_v3 = False
  try:
    version_info = version_checkpointer.restore(
        checkpoint_path,
        args=ocp.args.Composite(**{FORMAT_VERSION_KEY: ocp.args.JsonRestore()}),
    )[FORMAT_VERSION_KEY]
    if version_info.get("version") == 3:
      is_v3 = True
  except (ValueError, KeyError, OSError) as e:
    logging.warning(
        "Failed to read format version from %s: %s", checkpoint_path, e
    )
  finally:
    version_checkpointer.close()
  return is_v3


def is_v3_checkpoint(checkpoint_dir: str, epoch: int | None = None) -> bool:
  """Checks if a checkpoint is in V3 format."""
  try:
    checkpoint_path, _ = resolve_orbax_checkpoint_path(checkpoint_dir, epoch)
    return _is_v3_checkpoint_path(checkpoint_path)
  except (FileNotFoundError, ValueError):
    return False


def is_v1_checkpoint(checkpoint_dir: str, epoch: int | None = None) -> bool:
  """Checks if a checkpoint is in V1 format."""
  try:
    checkpoint_path, _ = resolve_orbax_checkpoint_path(checkpoint_dir, epoch)
    return _is_v1_checkpoint_path(checkpoint_path)
  except (FileNotFoundError, ValueError):
    return False


def _prepare_v3_restore(
    checkpoint_path: str,
    abstract_state: Mapping[str, Any],
    model: keras.Model | None = None,
    restore_optimizer_vars: bool = False,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
  """Prepares abstract state and state transforms for V3 checkpoint restore."""
  metadata_handler = ocp.handlers.PyTreeCheckpointHandler()
  state_checkpoint_path = epath.Path(checkpoint_path) / STATE_CHECKPOINT_KEY
  saved_state_metadata = metadata_handler.metadata(state_checkpoint_path)

  has_paths = gfile.Exists(
      os.path.join(checkpoint_path, NON_TRAINABLE_PATHS_KEY)
  )
  non_trainable_paths = None
  optimizer_paths = None
  if has_paths and model is not None:
    paths_checkpointer = ocp.Checkpointer(
        ocp.CompositeCheckpointHandler(**{  # pyrefly: ignore[bad-argument-type]
            NON_TRAINABLE_PATHS_KEY: ocp.handlers.JsonCheckpointHandler(),
            OPTIMIZER_PATHS_KEY: ocp.handlers.JsonCheckpointHandler(),
        })
    )
    restored_paths = paths_checkpointer.restore(
        checkpoint_path,
        args=ocp.args.Composite(**{
            NON_TRAINABLE_PATHS_KEY: ocp.args.JsonRestore(),
            OPTIMIZER_PATHS_KEY: ocp.args.JsonRestore(),
        }),
    )
    non_trainable_paths = restored_paths[NON_TRAINABLE_PATHS_KEY]["paths"]
    optimizer_paths = restored_paths[OPTIMIZER_PATHS_KEY]["paths"]
    paths_checkpointer.close()

  filtered_abstract_state = {}
  state_transforms = {}
  keys = [TRAINABLE_VARIABLES_KEY, NON_TRAINABLE_VARIABLES_KEY]
  if restore_optimizer_vars:
    keys.append(OPTIMIZER_VARIABLES_KEY)

  for key in keys:
    if key in abstract_state and key in saved_state_metadata:
      filtered_abstract_state[key] = {}
      state_transforms[key] = {}

      if key in [NON_TRAINABLE_VARIABLES_KEY, OPTIMIZER_VARIABLES_KEY]:
        if model is None:
          raise ValueError(f"Model must be provided to restore key {key}.")
        target_paths = (
            [v.path for v in model.non_trainable_variables]
            if key == NON_TRAINABLE_VARIABLES_KEY
            else [v.path for v in model.optimizer.variables]
        )
        source_paths = (
            non_trainable_paths
            if key == NON_TRAINABLE_VARIABLES_KEY
            else optimizer_paths
        )

        if has_paths:
          assert source_paths is not None
          # Map by index using the saved paths ordering
          for i, target_path in enumerate(target_paths):
            struct = abstract_state[key][target_path]
            if i < len(source_paths):
              source_path = source_paths[i]
              filtered_abstract_state[key][target_path] = struct
              if target_path != source_path:
                state_transforms[key][target_path] = (
                    ocp.transform_utils.Transform(
                        original_key=f"{key}/{source_path}"
                    )
                )
                logging.info(
                    "Mapping target path %s to source path %s by index %d",
                    target_path,
                    source_path,
                    i,
                )
            else:
              logging.warning(
                  "No source path at index %d for target path %s",
                  i,
                  target_path,
              )
        else:
          raise ValueError(
              "Index information (non_trainable_paths / optimizer_paths) "
              f"not found in checkpoint for key {key}. "
              "Unable to perform index-based mapping."
          )
      else:
        # Strict matching for trainable variables
        missing_paths = []
        for target_path, struct in abstract_state[key].items():
          if target_path in saved_state_metadata[key]:
            filtered_abstract_state[key][target_path] = struct
          else:
            missing_paths.append(target_path)

        if missing_paths:
          raise ValueError(
              f"Failed to restore variables for key {key}. "
              f"Missing paths in checkpoint: {missing_paths}"
          )
    elif key in abstract_state:
      logging.warning("Key %s not found in checkpoint metadata", key)

  return filtered_abstract_state, state_transforms


def _assign_v3_restored_values(
    variables: Mapping[str, Any],
    restored_state: Mapping[str, Any],
    restore_optimizer_vars: bool,
):
  """Assigns restored V3 values back to Keras variables."""
  for key in [
      TRAINABLE_VARIABLES_KEY,
      NON_TRAINABLE_VARIABLES_KEY,
  ]:
    if key in variables:
      var_dict = variables[key]
      restored_var_dict = restored_state.get(key)
      if restored_var_dict is not None:
        for path, var in var_dict.items():
          if path in restored_var_dict:
            logging.info("Restoring variable %s for key %s", path, key)
            var._value = restored_var_dict[path]  # pylint: disable=protected-access
          else:
            logging.warning(
                "Path %s not found in restored state for key %s", path, key
            )
      else:
        logging.warning("Key %s not found in restored state", key)
  if restore_optimizer_vars:
    key = OPTIMIZER_VARIABLES_KEY
    var_dict = variables[key]
    restored_var_dict = restored_state.get(key)
    if restored_var_dict is None:
      raise ValueError(
          f"Optimizer variables key {key} not found in restored state."
      )

    # Try exact match first
    missing_paths = []
    for path, var in var_dict.items():
      if path in restored_var_dict:
        var._value = restored_var_dict[path]  # pylint: disable=protected-access
      else:
        missing_paths.append((path, var))

    if missing_paths:
      logging.warning(
          "Some optimizer paths did not match exactly. Trying heuristic"
          " matching."
      )
      # Heuristic match: compare suffixes or ignore optimizer name prefix
      # e.g. 'adam_3/var_name' vs 'adam_1/var_name'
      # Let's try matching by the part after the first '/'
      restored_by_suffix = {}
      for k, v in restored_var_dict.items():
        parts = k.split("/", 1)
        if len(parts) > 1:
          restored_by_suffix[parts[1]] = v
        else:
          restored_by_suffix[k] = v

      still_missing = []
      for path, var in missing_paths:
        parts = path.split("/", 1)
        suffix = parts[1] if len(parts) > 1 else path
        if suffix in restored_by_suffix:
          var._value = restored_by_suffix[suffix]  # pylint: disable=protected-access
          logging.info(
              "Matched optimizer variable %s by suffix %s", path, suffix
          )
        else:
          still_missing.append(path)

      if still_missing:
        raise ValueError(
            f"Failed to restore optimizer variables for paths: {still_missing}"
        )


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

  checkpoint_path, epoch = resolve_orbax_checkpoint_path(checkpoint_dir, epoch)

  if _is_v1_checkpoint_path(checkpoint_path):
    raise ValueError(
        f"The checkpoint in {checkpoint_dir} is in V1 format (list-based)"
        f" at step {epoch}."
        " `restore_keras_checkpoint` is only compatible with V2/V3 checkpoints."
        " Please use `restore_keras_model` instead."
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

  is_v3 = _is_v3_checkpoint_path(checkpoint_path)

  if is_v3:
    variables = {
        TRAINABLE_VARIABLES_KEY: _variables_to_path_dict(
            model.trainable_variables, TRAINABLE_VARIABLES_KEY
        ),
        NON_TRAINABLE_VARIABLES_KEY: _variables_to_path_dict(
            model.non_trainable_variables, NON_TRAINABLE_VARIABLES_KEY
        ),
    }
    if restore_optimizer_vars:
      if not model.optimizer.built:
        raise ValueError(
            "To use `restore_keras_checkpoint` on an existing model with"
            " `restore_optimizer_vars` set to True, the optimizer must be"
            " built."
        )
      variables[OPTIMIZER_VARIABLES_KEY] = _variables_to_path_dict(
          model.optimizer.variables, OPTIMIZER_VARIABLES_KEY
      )
  else:
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

  state_transforms = {}
  if is_v3:
    abstract_state, state_transforms = _prepare_v3_restore(
        checkpoint_path, abstract_state, model, restore_optimizer_vars
    )

  # Delete the variables from device memory to reduce peak memory usage.
  # Only delete variables that we are actually trying to restore.
  if is_v3:
    for key, path_dict in abstract_state.items():
      for path in path_dict.keys():
        var = variables[key][path]
        var.value.delete()
        var._value = None  # pylint: disable=protected-access
  else:
    for var in jax.tree.flatten(variables)[0]:
      var.value.delete()
      var._value = None  # pylint: disable=protected-access

  # Always use PyTreeCheckpointHandler for restoring, as it is more flexible
  # (supports transforms)
  state_handler = ocp.handlers.PyTreeCheckpointHandler(
      restore_concurrent_gb=96,
  )
  checkpointer = ocp.Checkpointer(
      ocp.CompositeCheckpointHandler(**{  # pyrefly: ignore[bad-argument-type]
          STATE_CHECKPOINT_KEY: state_handler,
      })
  )

  restore_args = ocp.args.Composite(**{
      STATE_CHECKPOINT_KEY: ocp.args.PyTreeRestore(
          abstract_state,
          transforms=state_transforms if is_v3 else {},
          restore_args=ocp.checkpoint_utils.construct_restore_args(
              abstract_state
          ),
      ),
  })

  restored_state = checkpointer.restore(
      checkpoint_path,
      args=restore_args,
  )[STATE_CHECKPOINT_KEY]
  checkpointer.close()

  if is_v3:
    _assign_v3_restored_values(
        variables, restored_state, restore_optimizer_vars
    )
  else:
    keras.tree.assert_same_structure(variables, restored_state)
    for var, restored_var in zip(
        jax.tree.flatten(variables)[0], jax.tree.flatten(restored_state)[0]
    ):
      var._value = restored_var  # pylint: disable=protected-access

  if restore_model_epoch:
    model._initial_epoch = epoch + 1  # pylint: disable=protected-access  # pyrefly: ignore[unsupported-operation]
  if restore_optimizer_vars and not restore_iterations:
    model.optimizer.iterations.assign(0)

  return model


def restore_partial_checkpoint(
    checkpoint_dir: str,
    partial_variables: Mapping[str, Any],
    epoch: int | None = None,
) -> Mapping[str, Any]:
  """Restores partial variables from an Orbax checkpoint.

  Args:
      checkpoint_dir: The directory containing the Orbax checkpoint(s).
      partial_variables: A dictionary mapping keys (e.g.
        TRAINABLE_VARIABLES_KEY) to dictionaries mapping variable paths to
        keras.Variable instances.
      epoch: The epoch to restore. If None, latest is used.

  Returns:
      The restored state dictionary (containing Jax Arrays).
  """
  checkpoint_path, _ = resolve_orbax_checkpoint_path(checkpoint_dir, epoch)

  is_v3 = _is_v3_checkpoint_path(checkpoint_path)
  if not is_v3:
    raise ValueError(
        "restore_partial_checkpoint only supports V3 (dictionary-based)"
        " checkpoints."
    )

  # Partial restoration is restricted to trainable variables because they are
  # the primary targets for selective weight transfer (e.g. sequence encoder).
  # Non-trainable and optimizer variables are training-run specific and their
  # mapping by index is fragile, so they are not supported for partial restore.
  if (
      NON_TRAINABLE_VARIABLES_KEY in partial_variables
      and partial_variables[NON_TRAINABLE_VARIABLES_KEY]
  ) or (
      OPTIMIZER_VARIABLES_KEY in partial_variables
      and partial_variables[OPTIMIZER_VARIABLES_KEY]
  ):
    raise ValueError(
        "Partial restoration is only supported for trainable variables."
    )

  abstract_state = jax.tree.map(_to_shape_dtype_struct, partial_variables)

  # Delete variables from device memory to reduce peak memory usage.
  for var in jax.tree.flatten(partial_variables)[0]:
    if var._value is not None:  # pylint: disable=protected-access
      var.value.delete()
      var._value = None  # pylint: disable=protected-access

  abstract_state, state_transforms = _prepare_v3_restore(
      checkpoint_path, abstract_state, model=None, restore_optimizer_vars=False
  )

  checkpointer = ocp.Checkpointer(
      ocp.CompositeCheckpointHandler(**{  # pyrefly: ignore[bad-argument-type]
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
              transforms=state_transforms,
              restore_args=ocp.checkpoint_utils.construct_restore_args(
                  abstract_state
              ),
          )
      }),
  )[STATE_CHECKPOINT_KEY]
  # Assign restored values back to partial_variables in-place
  key = TRAINABLE_VARIABLES_KEY
  if key in partial_variables and key in restored_state:
    var_dict = partial_variables[key]
    restored_var_dict = restored_state[key]
    for path, var in var_dict.items():
      if path in restored_var_dict:
        var._value = restored_var_dict[path]  # pylint: disable=protected-access
      else:
        logging.warning("Path %s was NOT restored for key %s", path, key)

  checkpointer.close()
  return restored_state


def load_keras_model_config(
    checkpoint_dir: str, epoch: int | None = None
) -> Mapping[str, Any]:
  """Loads a Keras model from a checkpoint directory."""
  if keras.backend.backend() != "jax":
    raise ValueError(
        "This function only supports loading a Keras 3 Jax backend model."
    )

  checkpoint_path, _ = resolve_orbax_checkpoint_path(checkpoint_dir, epoch)

  json_checkpointer = ocp.Checkpointer(
      ocp.CompositeCheckpointHandler(
          **{CONFIG_CHECKPOINT_KEY: ocp.handlers.JsonCheckpointHandler()}  # pyrefly: ignore[bad-argument-type]
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


def check_all_layers_built(model: keras.layers.Layer):
  """Checks if any layers in a Keras model are not built."""
  unbuilt_layers = []
  for layer in model._flatten_layers(include_self=True):  # pylint: disable=protected-access
    if not layer.built:
      unbuilt_layers.append(layer)

  if unbuilt_layers:
    raise ValueError(
        "The following layers are not built:"
        f" {[layer.name for layer in unbuilt_layers]}."
    )


def check_no_layers_built(model: keras.layers.Layer):
  """Checks if any layers in a Keras model already built."""
  built_layers = []
  for layer in model._flatten_layers(include_self=True):  # pylint: disable=protected-access
    if layer.built:
      built_layers.append(layer)

  if built_layers:
    raise ValueError(
        "The following layers are already built:"
        f" {[layer.name for layer in built_layers]}."
    )


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
          KerasOrbaxCheckpointManager
          | KerasOrbaxCheckpointManagerV2
          | KerasOrbaxCheckpointManagerV3
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
      # So that the continuous eval job finds it and logs the eval at step 0.
      self._checkpoint_manager.save_model_variables(self.model, 0, logs)

  def on_epoch_end(self, epoch: int, logs: Mapping[str, Any] | None = None):
    self._checkpoint_manager.save_model_variables(self.model, epoch, logs)

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
  `KerasOrbaxCheckpointManagerV2` or `KerasOrbaxCheckpointManagerV3`, use
  `restore_keras_checkpoint` instead.

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

  checkpoint_path = ocp.path.step.build_step_path(
      checkpoint_dir, ocp.path.step.standard_name_format(), step
  )

  if gfile.Exists(os.path.join(checkpoint_path, STATE_CHECKPOINT_KEY)):
    raise ValueError(
        f"The checkpoint in {checkpoint_dir} is in V2/V3 format"
        f" (dictionary-based) at step {step}."
        " `restore_keras_model` is only compatible with legacy V1 checkpoints."
        " Please use `restore_keras_checkpoint` instead."
    )

  checkpointer = ocp.Checkpointer(
      ocp.CompositeCheckpointHandler(**{  # pyrefly: ignore[bad-argument-type]
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
      eval_subdir: str = "validation",
  ):
    super().__init__(
        log_dir,
        write_steps_per_second=write_steps_per_second,
        update_freq="epoch",
        write_graph=False,
    )
    self._steps_per_epoch = steps_per_epoch
    self._num_params = None
    self._eval_subdir = eval_subdir
    # Marks the callback as async safe so batch end callbacks can be dispatched
    # asynchronously.
    self.async_safe = True

  def set_model(self, model: keras.Model):
    """Sets Keras model and writes graph if specified."""
    super().set_model(model)
    if self._eval_subdir != "validation":
      # We need to manually set `_val_dir` to point to the correct subdirectory.
      # `super().set_model(model)` sets `_val_dir` to `log_dir/validation`.
      self._val_dir = os.path.join(self.log_dir, self._eval_subdir)
      # `super().set_model(model)` lazily creates the writers so we need to
      # reset them here to make sure they point to the correct subdirectories.
      self._writers = {}

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

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

from collections.abc import Container, Mapping, Sequence
import dataclasses
import datetime
import enum
import os
import re
import time
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
SHARED_VARIABLE_MAP_KEY = "shared_variable_map"
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
        f"Duplicate variable paths detected in {collection_name}. Ensure "
        "unique layer names (e.g. set name_layers=True if using a GUM "
        f"model). Duplicates: {duplicates}"
    )
  return var_dict


def extract_shared_variable_map(model: keras.Model) -> dict[str, str]:
  """Extracts mapping of shared variables across the model layer tree.

  When multiple layers in a model reference the same `keras.Variable` instance,
  Keras's `model.trainable_variables` assigns the variable a single canonical
  `.path` corresponding to the first layer that tracked it.

  This function traverses the full layer hierarchy to discover alternative
  logical paths referencing those shared variables and maps them to their
  canonical path.

  An alias path is built as `{sharing_layer_path}/{var.name}`. Layer names in
  the path include Keras uniquifying suffixes (e.g. `dense_1`); variable names
  are never uniquified by Keras, so `var.name` is the name the variable was
  created with.

  Supported scope:
    Only layer-level sharing is supported: a layer instance is reused in
    several places. The variable is created inside the shared layer, so its
    name is the same under every parent, and the alias matches the path an
    unshared model would use, e.g.
      `{"model/surface_b/text_emb/embeddings":
        "model/surface_a/text_emb/embeddings"}`.

  Caveat:
    Variable-level sharing is NOT supported. This is when a layer stores
    another layer's variable as an attribute under its own name, e.g.
    `self.token_table = other_layer.embeddings`. The variable is still
    recorded here, but its alias uses the variable's own name
    (`model/tower/embeddings`), not the attribute name (`token_table`). A
    target model that owns the weight as `model/tower/token_table` will not
    match the alias, and restore fails with a missing-path error. `prefix_map`
    does not help, since it keeps variable names unchanged; such models need an
    explicit per-variable transform.

  Args:
    model: The Keras model instance.

  Returns:
    A dictionary mapping shared_path -> canonical_path (e.g.
    `{"model/search_surface/text_emb/embeddings":
    "model/chrome_surface/text_emb/embeddings"}`).

  Raises:
    ValueError: If two different shared variables produce the same alias path,
      e.g. a layer that stores the `kernel` of two other layers.
  """
  shared_var_map = {}

  def _traverse(layer: keras.layers.Layer, prefix: str):
    # Walk tracked child layers
    for child in getattr(layer, "_layers", []):
      # child.name includes any uniquifying suffix (e.g. 'dense_1') assigned
      # during layer construction.
      child_prefix = f"{prefix}/{child.name}" if prefix else child.name
      _traverse(child, child_prefix)

    # Check variables directly attached to this layer.
    own_vars = getattr(layer, "_trainable_variables", []) + getattr(
        layer, "_non_trainable_variables", []
    )
    for var in own_vars:
      # var.name is identical across both paths since it is the same Variable.
      logical_path = f"{prefix}/{var.name}" if prefix else var.name
      if logical_path == var.path:
        continue
      existing = shared_var_map.get(logical_path)
      if existing is not None and existing != var.path:
        raise ValueError(
            f"Shared variables {existing} and {var.path} both map to alias "
            f"path {logical_path}. This happens when a layer stores several "
            "shared variables with the same name as attributes; share whole "
            "layers instead."
        )
      shared_var_map[logical_path] = var.path

  _traverse(model, model.name or "")
  return shared_var_map


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
      choose_store_cell: bool = True,
  ):
    """Initializes a KerasOrbaxCheckpointManager.

    Args:
      checkpoint_dir: The directory to save checkpoints to.
      max_to_keep: The maximum number of checkpoints to keep.
      save_interval_epochs: The interval (in epochs) to save checkpoints.
      choose_store_cell: Whether to dynamically select the CNS2 store cell.
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
            file_options=ocp.options.FileOptions(
                cns2_storage_options=ocp.options.Cns2StorageOptions(
                    choose_store_cell=choose_store_cell,
                ),
            ),
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
      choose_store_cell: bool = True,
  ):
    """Initializes a KerasOrbaxCheckpointManagerV3.

    Args:
      checkpoint_dir: The directory to save checkpoints to.
      max_to_keep: The maximum number of checkpoints to keep.
      save_interval_epochs: The interval (in epochs) to save checkpoints.
      choose_store_cell: Whether to dynamically select the CNS2 store cell.
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
            SHARED_VARIABLE_MAP_KEY,
        ),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=save_interval_epochs,
            max_to_keep=max_to_keep,
            file_options=ocp.options.FileOptions(
                cns2_storage_options=ocp.options.Cns2StorageOptions(
                    choose_store_cell=choose_store_cell,
                ),
            ),
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
    shared_var_map = extract_shared_variable_map(model)
    shared_variable_map = {SHARED_VARIABLE_MAP_KEY: shared_var_map}
    logging.info("SAVED non_trainable_paths: %s", non_trainable_paths)
    logging.info("SAVED optimizer_paths: %s", optimizer_paths)
    logging.info("SAVED shared_variable_map: %s", shared_variable_map)

    logging.info("Saving checkpoint for epoch %s...", epoch)
    self.save(
        step=epoch,
        args=ocp.args.Composite(**{
            STATE_CHECKPOINT_KEY: ocp.args.PyTreeSave(state),
            CONFIG_CHECKPOINT_KEY: ocp.args.JsonSave(config),
            FORMAT_VERSION_KEY: ocp.args.JsonSave({"version": 3}),
            NON_TRAINABLE_PATHS_KEY: ocp.args.JsonSave(non_trainable_paths),
            OPTIMIZER_PATHS_KEY: ocp.args.JsonSave(optimizer_paths),
            SHARED_VARIABLE_MAP_KEY: ocp.args.JsonSave(shared_variable_map),
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
  try:
    version_info = version_checkpointer.restore(
        checkpoint_path,
        args=ocp.args.Composite(**{FORMAT_VERSION_KEY: ocp.args.JsonRestore()}),
    )[FORMAT_VERSION_KEY]
    return version_info.get("version") == 3
  finally:
    version_checkpointer.close()


def _detect_checkpoint_version(checkpoint_path: str) -> CheckpointVersion:
  """Detects the version of the checkpoint at the given path.

  Detection order and discriminator criteria:
  - V1 checkpoints use the legacy item directory layout ('default/').
  - V3 checkpoints contain an explicit 'format_version' item with
    {"version": 3}.
  - V2 checkpoints contain 'state/' without the V3 'format_version' marker.

  Args:
    checkpoint_path: Path to the checkpoint directory.

  Returns:
    The detected CheckpointVersion enum value.
  """
  # V1 uses the legacy 'default' directory layout.
  if _is_v1_checkpoint_path(checkpoint_path):
    return CheckpointVersion.V1
  # V3 checkpoints are explicitly tagged with format_version.
  if _is_v3_checkpoint_path(checkpoint_path):
    return CheckpointVersion.V3
  # V2 checkpoints have a 'state' directory without the V3 format_version
  # marker.
  if gfile.Exists(os.path.join(checkpoint_path, STATE_CHECKPOINT_KEY)):
    return CheckpointVersion.V2
  raise ValueError(f"Unknown checkpoint format at {checkpoint_path}")


def is_v3_checkpoint(checkpoint_dir: str, epoch: int | None = None) -> bool:
  """Checks if a checkpoint is in V3 format."""
  checkpoint_path, _ = resolve_orbax_checkpoint_path(checkpoint_dir, epoch)
  return _detect_checkpoint_version(checkpoint_path) == CheckpointVersion.V3


def _validate_v3_checkpoint(
    checkpoint_path: str,
    saved_state_metadata: Any | None = None,
) -> Any:
  """Validates that the V3 checkpoint is healthy and returns its state metadata.

  A healthy V3 checkpoint must contain the state directory and its companion
  metadata items on disk (non_trainable_paths, optimizer_paths, and
  shared_variable_map), and its state metadata must contain all three keys
  (trainable_variables, non_trainable_variables, and optimizer_variables).

  `shared_variable_map` is required even for models without shared variables,
  where it is an empty map. Without it, a restore could not resolve shared
  variables referenced by an alias path, so a checkpoint that lacks it is
  treated as incomplete rather than silently restored without alias
  resolution.

  Args:
    checkpoint_path: Path to the checkpoint directory.
    saved_state_metadata: Optional pre-loaded state metadata. If None, it will
      be read from the state checkpoint directory.

  Returns:
    The saved state metadata (TreeMetadata) from the state checkpoint item.

  Raises:
    ValueError: If the checkpoint is missing required files or state keys.
  """
  root = epath.Path(checkpoint_path)
  required_items = (
      STATE_CHECKPOINT_KEY,
      NON_TRAINABLE_PATHS_KEY,
      OPTIMIZER_PATHS_KEY,
      SHARED_VARIABLE_MAP_KEY,
  )
  missing_items = [f for f in required_items if not (root / f).exists()]
  if missing_items:
    raise ValueError(
        f"Corrupted or incomplete V3 checkpoint at {checkpoint_path}: "
        f"missing required checkpoint item(s): {missing_items}."
    )

  if saved_state_metadata is None:
    metadata_handler = ocp.handlers.PyTreeCheckpointHandler()
    state_checkpoint_path = root / STATE_CHECKPOINT_KEY
    saved_state_metadata = metadata_handler.metadata(state_checkpoint_path)

  required_state_keys = (
      TRAINABLE_VARIABLES_KEY,
      NON_TRAINABLE_VARIABLES_KEY,
      OPTIMIZER_VARIABLES_KEY,
  )
  missing_keys = [
      k for k in required_state_keys if k not in saved_state_metadata
  ]
  if missing_keys:
    raise ValueError(
        f"Corrupted or incomplete V3 checkpoint at {checkpoint_path}: "
        f"missing required state keys in checkpoint metadata: {missing_keys}."
    )
  return saved_state_metadata


def _find_stored_path(
    path: str,
    stored_paths: Container[str],
    shared_var_map: Mapping[str, str],
) -> str | None:
  """Returns the path under which `path`'s value is stored, or None.

  A variable is stored under its own path, unless it is a shared variable
  reached through an alias, in which case it is stored under the canonical path
  recorded in `shared_var_map`. A direct match takes precedence, so a real
  variable is never shadowed by an alias that happens to share its path.

  Args:
    path: A variable path, possibly an alias of a shared variable.
    stored_paths: The paths stored in the checkpoint.
    shared_var_map: Map from alias path to canonical path.

  Returns:
    The stored path holding the value, or None if there is none.
  """
  if path in stored_paths:
    return path
  canonical = shared_var_map.get(path)
  if canonical is not None and canonical in stored_paths:
    return canonical
  return None


def _resolve_transform_source(
    transform: Any,
    key: str,
    stored_paths: Container[str],
    shared_var_map: Mapping[str, str],
) -> Any:
  """Rewrites a transform's source key to its stored path if it names an alias.

  In V3 checkpoints, shared weights are deduplicated so only the canonical path
  is physically stored on disk (e.g. `surface_b`), while aliases are
  recorded in `shared_var_map` (e.g. `surface_a -> surface_b`).

  If a caller supplies a `Transform(original_key="surface_a")` to restore a
  renamed variable (e.g. `surface_a_1`), Orbax cannot find `surface_a` on disk.
  This function resolves `original_key` to its physical storage path
  (`surface_b`) so Orbax can load the tensor.

  Example:
    If `surface_a` was deduplicated to `surface_b` in the checkpoint:
      `shared_var_map`: `{"surface_a/kernel": "surface_b/kernel"}`
      `stored_paths`: `{"surface_b/kernel"}`

      # User maps renamed variable surface_a_1 -> surface_a:
      resolved = _resolve_transform_source(
          transform=Transform(
              original_key="trainable_variables/surface_a/kernel"
          ),
          key="trainable_variables",
          stored_paths=stored_paths,
          shared_var_map=shared_var_map,
      )
      # rewritten to: original_key="trainable_variables/surface_b/kernel"

  Args:
    transform: A user-supplied transform for one target variable.
    key: The state key, e.g. `trainable_variables`. `original_key` may or may
      not carry it as a `{key}/` prefix.
    stored_paths: The paths stored in the checkpoint under `key`.
    shared_var_map: Map from alias path to canonical path.

  Returns:
    The transform, with `original_key` rewritten if it named an alias.
  """
  if not isinstance(transform, ocp.transform_utils.Transform) or not isinstance(
      transform.original_key, str
  ):
    return transform
  prefix = f"{key}/"
  has_prefix = transform.original_key.startswith(prefix)
  source = transform.original_key.removeprefix(prefix)
  stored = _find_stored_path(source, stored_paths, shared_var_map)
  if stored is None or stored == source:
    # Not an alias. Leave it for Orbax to resolve, or to reject if missing.
    return transform
  logging.info(
      "Resolved transform source %s to stored path %s via shared variable map",
      source,
      stored,
  )
  return dataclasses.replace(
      transform, original_key=f"{prefix}{stored}" if has_prefix else stored
  )


def _map_prefix(
    path: str, prefix_map: Mapping[str, str]
) -> tuple[str, str] | None:
  """Rewrites `path` using the longest matching prefix in `prefix_map`.

  Prefixes match whole `/`-separated segments only, so `model/tower` matches
  `model/tower/dense/kernel` but not `model/tower_2/dense/kernel`.

  Args:
    path: A target variable path.
    prefix_map: Map from target path prefix to source path prefix.

  Returns:
    A tuple of (matched target prefix, source path), or None if no prefix
    matches.
  """
  best = None
  for target_prefix in prefix_map:
    if path == target_prefix or path.startswith(f"{target_prefix}/"):
      if best is None or len(target_prefix) > len(best):
        best = target_prefix
  if best is None:
    return None
  return best, prefix_map[best] + path[len(best) :]


def _prepare_v3_restore(
    checkpoint_path: str,
    abstract_state: Mapping[str, Any],
    model: keras.Model | None = None,
    restore_optimizer_vars: bool = False,
    transforms: Mapping[str, Any] | None = None,
    prefix_map: Mapping[str, str] | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
  """Prepares the abstract state and constructs transforms for V3 restore.

  For trainable variables, this performs strict name-based matching against
  the checkpoint metadata unless explicit transforms are provided.
  For non-trainable and optimizer variables, if path metadata is available
  in the checkpoint, it maps variables by index (similar to V2 list-based
  restoration) to handle potential name changes (e.g., due to different
  layer naming across runs). Note that for this index-based fallback to work
  correctly, the source and target variables must have the exact same order
  and length.

  Args:
    checkpoint_path: The resolved checkpoint directory path.
    abstract_state: The model's full abstract state structure.
    model: The Keras model instance. Required for non-trainable/optimizer
      variables mapping.
    restore_optimizer_vars: Whether to prepare optimizer variables.
    transforms: An optional mapping of custom transforms for variable mapping.
    prefix_map: An optional map from target path prefix to source path prefix
      for trainable variables. See `restore_partial_checkpoint`.

  Returns:
    A tuple of (filtered_abstract_state, state_transforms) to be passed to
    the Orbax restore call.
  """
  # Validate that underlying checkpoint is healthy and retrieve state metadata.
  saved_state_metadata = _validate_v3_checkpoint(checkpoint_path)

  # The shared variable map is a required V3 item, checked above.
  map_path = epath.Path(checkpoint_path) / SHARED_VARIABLE_MAP_KEY
  checkpointer = ocp.Checkpointer(ocp.handlers.JsonCheckpointHandler())
  try:
    shared_var_map = checkpointer.restore(
        os.fspath(map_path), args=ocp.args.JsonRestore()
    )[SHARED_VARIABLE_MAP_KEY]
  finally:
    checkpointer.close()

  # Load index path metadata for non-trainable and optimizer variables if
  # needed.
  non_trainable_paths = None
  if model is not None and abstract_state.get(NON_TRAINABLE_VARIABLES_KEY):
    nt_path = epath.Path(checkpoint_path) / NON_TRAINABLE_PATHS_KEY
    checkpointer = ocp.Checkpointer(ocp.handlers.JsonCheckpointHandler())
    try:
      non_trainable_paths = checkpointer.restore(
          os.fspath(nt_path), args=ocp.args.JsonRestore()
      )["paths"]
    finally:
      checkpointer.close()

  optimizer_paths = None
  if (
      model is not None
      and restore_optimizer_vars
      and abstract_state.get(OPTIMIZER_VARIABLES_KEY)
  ):
    opt_path = epath.Path(checkpoint_path) / OPTIMIZER_PATHS_KEY
    checkpointer = ocp.Checkpointer(ocp.handlers.JsonCheckpointHandler())
    try:
      optimizer_paths = checkpointer.restore(
          os.fspath(opt_path), args=ocp.args.JsonRestore()
      )["paths"]
    finally:
      checkpointer.close()

  keys = [TRAINABLE_VARIABLES_KEY, NON_TRAINABLE_VARIABLES_KEY]
  if restore_optimizer_vars:
    keys.append(OPTIMIZER_VARIABLES_KEY)

  filtered_abstract_state = {}
  state_transforms = {}

  for key in keys:
    if not abstract_state.get(key):
      continue

    filtered_abstract_state[key] = {}
    state_transforms[key] = {}

    if key in (NON_TRAINABLE_VARIABLES_KEY, OPTIMIZER_VARIABLES_KEY):
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
      if source_paths is None:
        readable_key = key.replace("_", " ")
        raise ValueError(
            f"Failed to restore {readable_key}: checkpoint metadata paths"
            " missing from checkpoint."
        )
      if len(source_paths) != len(target_paths):
        readable_key = key.replace("_", " ")
        raise ValueError(
            f"Failed to restore {readable_key}: variable count mismatch. "
            f"Source checkpoint has {len(source_paths)} paths, but target "
            f"model has {len(target_paths)} paths."
        )

      # Map by index using the saved paths ordering
      for i, target_path in enumerate(target_paths):
        struct = abstract_state[key][target_path]
        source_path = source_paths[i]
        filtered_abstract_state[key][target_path] = struct
        if target_path != source_path:
          state_transforms[key][target_path] = ocp.transform_utils.Transform(
              original_key=f"{key}/{source_path}"
          )
          logging.info(
              "Mapping target path %s to source path %s by index %d",
              target_path,
              source_path,
              i,
          )
    else:
      # Trainable variables are matched by path. Each target is restored from
      # the source a user transform names, else from its prefix-mapped path,
      # else from its own path; any of these may be an alias of a shared
      # variable, resolved by _find_stored_path.
      missing_paths = []
      key_transforms = {}
      if transforms:
        key_transforms = (
            transforms[key]
            if key in transforms and isinstance(transforms[key], Mapping)
            else transforms
        )
      stored_paths = saved_state_metadata[key]
      normalized_prefix_map = {
          t.rstrip("/"): s.rstrip("/") for t, s in (prefix_map or {}).items()
      }
      if "" in normalized_prefix_map:
        raise ValueError("prefix_map keys must be non-empty path prefixes.")
      unused_prefixes = set(normalized_prefix_map)
      for target_path, struct in abstract_state[key].items():
        mapped_source = None
        prefix_match = _map_prefix(target_path, normalized_prefix_map)
        if prefix_match is not None:
          matched_prefix, mapped_source = prefix_match
          unused_prefixes.discard(matched_prefix)
        # Branch 1: Explicit user transform (remapping/surgery).
        # Target and source paths must match exact model variable paths
        # (including suffixes like `dense_1`). If `original_key` is an alias,
        # it is resolved to its physical storage path.
        if target_path in key_transforms:
          transform = _resolve_transform_source(
              key_transforms[target_path], key, stored_paths, shared_var_map
          )
        # Branch 2: Prefix mapping. The target is restored from the source path
        # obtained by swapping its longest matching prefix; the source may be
        # an alias of a shared variable.
        elif mapped_source is not None:
          stored = _find_stored_path(
              mapped_source, stored_paths, shared_var_map
          )
          if stored is None:
            missing_paths.append(f"{target_path} (from {mapped_source})")
            continue
          transform = (
              None
              if stored == target_path
              else ocp.transform_utils.Transform(original_key=f"{key}/{stored}")
          )
        # Branch 3: Default 1-to-1 match.
        # Resolves target_path if it is an alias, synthesizing a Transform to
        # its canonical storage path.
        else:
          stored = _find_stored_path(target_path, stored_paths, shared_var_map)
          if stored is None:
            missing_paths.append(target_path)
            continue
          transform = (
              None
              if stored == target_path
              else ocp.transform_utils.Transform(original_key=f"{key}/{stored}")
          )
        filtered_abstract_state[key][target_path] = struct
        if transform is not None:
          state_transforms[key][target_path] = transform

      if unused_prefixes:
        logging.warning(
            "prefix_map entries matched no target variables for key %s: %s",
            key,
            sorted(unused_prefixes),
        )
      if missing_paths:
        raise ValueError(
            f"Failed to restore variables for key {key}. "
            f"Missing paths in checkpoint: {missing_paths}"
        )

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
      if restored_var_dict is None:
        if var_dict:
          raise ValueError(f"Key {key} not found in restored state.")
        continue

      missing_paths = []
      for path, var in var_dict.items():
        if path in restored_var_dict:
          logging.info("Restoring variable %s for key %s", path, key)
          var._value = restored_var_dict[path]  # pylint: disable=protected-access
        else:
          missing_paths.append(path)

      if missing_paths:
        raise ValueError(
            f"Failed to restore {key} variables for paths: {missing_paths}"
        )

  if restore_optimizer_vars:
    key = OPTIMIZER_VARIABLES_KEY
    var_dict = variables[key]
    restored_var_dict = restored_state.get(key)
    if restored_var_dict is None:
      if var_dict:
        raise ValueError(
            f"Optimizer variables key {key} not found in restored state."
        )
      return

    missing_paths = []
    for path, var in var_dict.items():
      if path in restored_var_dict:
        logging.info("Restoring variable %s for key %s", path, key)
        var._value = restored_var_dict[path]  # pylint: disable=protected-access
      else:
        missing_paths.append(path)

    if missing_paths:
      raise ValueError(
          f"Failed to restore optimizer variables for paths: {missing_paths}"
      )


def _restore_state_pytree(
    checkpoint_path: str,
    abstract_state: Mapping[str, Any],
    state_transforms: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
  """Restores the state PyTree from the checkpoint path."""
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
          transforms=state_transforms or {},
          restore_args=ocp.checkpoint_utils.construct_restore_args(
              abstract_state
          ),
      ),
  })
  try:
    return checkpointer.restore(
        checkpoint_path,
        args=restore_args,
    )[STATE_CHECKPOINT_KEY]
  finally:
    checkpointer.close()


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

  version = _detect_checkpoint_version(checkpoint_path)
  if version == CheckpointVersion.V1:
    raise ValueError(
        f"The checkpoint in {checkpoint_dir} is in V1 format (list-based)"
        f" at step {epoch}."
        " `restore_keras_checkpoint` is only compatible with V2/V3 checkpoints."
        " Please use `restore_keras_model` instead."
    )
  is_v3 = version == CheckpointVersion.V3

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

  restored_state = _restore_state_pytree(
      checkpoint_path,
      abstract_state,
      state_transforms=state_transforms,
  )

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
    transforms: Mapping[str, Any] | None = None,
    prefix_map: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
  """Restores partial variables from an Orbax checkpoint.

  Each target variable is restored from, in order of precedence:
    1. The source named by its entry in `transforms`, if any.
    2. The source obtained by rewriting its path with `prefix_map`, if a prefix
       matches.
    3. Its own path (default 1-to-1 match).
  User-specified remappings (1 and 2) take precedence even if the target path
  already exists in the checkpoint (e.g. initializing one surface from another
  within the same model). Sources in all three cases may be aliases of shared
  variables; they are resolved to their stored paths through the checkpoint's
  shared variable map.

  Args:
      checkpoint_dir: The directory containing the Orbax checkpoint(s).
      partial_variables: A dictionary mapping keys (e.g.
        TRAINABLE_VARIABLES_KEY) to dictionaries mapping variable paths to
        keras.Variable instances.
      epoch: The epoch to restore. If None, latest is used.
      transforms: An optional mapping of custom transforms (e.g. mapping target
        variable paths to `ocp.transform_utils.Transform` objects) for explicit
        variable re-mapping. Keys and `original_key` values must be exact
        variable paths, including Keras layer name suffixes (e.g. `dense_1`).
        Prefer `prefix_map` or the default resolution unless you need
        per-variable control.
      prefix_map: An optional map from target path prefix to source path prefix
        for subtree remapping, e.g. `{"model/target_tower": "model/src_tower"}`.
        Paths omit the state key (e.g. `trainable_variables/`).
        Rules:
        - Depth matching: A prefix matches all variables at any depth beneath
          it on whole path segments (`layer_a` matches `layer_a/var_a` and
          `layer_a/layer_b/var_b`, but not `layer_a_2`).
        - Overlapping prefixes: The longest (most specific) prefix wins,
          allowing subtree overrides (`model/new/tower` overrides `model/new`).
        - Subtree structure: The path remainder below the prefix is preserved,
          so source and target subtrees must share identical structure.
        Logs a warning if a prefix matches no target variables; raises if a
        mapped source is missing; shape mismatches fail at restore.

  Returns:
      The restored state dictionary (containing Jax Arrays).
  """
  checkpoint_path, _ = resolve_orbax_checkpoint_path(checkpoint_dir, epoch)

  version = _detect_checkpoint_version(checkpoint_path)
  if version != CheckpointVersion.V3:
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
      checkpoint_path,
      abstract_state,
      model=None,
      restore_optimizer_vars=False,
      transforms=transforms,
      prefix_map=prefix_map,
  )

  restored_state = _restore_state_pytree(
      checkpoint_path,
      abstract_state,
      state_transforms=state_transforms,
  )
  _assign_v3_restored_values(
      partial_variables, restored_state, restore_optimizer_vars=False
  )

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
      choose_store_cell: bool = True,
  ):
    """Initializes a KerasOrbaxCheckpointManager.

    Args:
      checkpoint_dir: The directory to save checkpoints to.
      max_to_keep: The maximum number of checkpoints to keep.
      save_interval_epochs: The interval (in epochs) to save checkpoints.
      choose_store_cell: Whether to dynamically select the CNS2 store cell.
    """
    super().__init__(
        directory=checkpoint_dir,
        checkpointers=ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler()),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=save_interval_epochs,
            max_to_keep=max_to_keep,
            file_options=ocp.options.FileOptions(
                cns2_storage_options=ocp.options.Cns2StorageOptions(
                    choose_store_cell=choose_store_cell,
                ),
            ),
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

  # This callback only writes summaries in `on_epoch_end`. The inherited
  # `TensorBoard` batch hooks still call `summary.scalar` for every key in
  # `logs` on every step, even though `update_freq="epoch"` means there is no
  # default writer and nothing is recorded. Use the no-op `Callback` hooks
  # instead. Keras treats them as unset, so they add no per-step host work.
  on_train_batch_begin = keras.callbacks.Callback.on_train_batch_begin
  on_train_batch_end = keras.callbacks.Callback.on_train_batch_end
  on_test_batch_begin = keras.callbacks.Callback.on_test_batch_begin


class MetricsCallback(keras.callbacks.Callback):
  """Base class for callbacks that add scalar metrics to the epoch summary.

  In training, a subclass can add metrics to `logs` in `on_epoch_end`. It must
  run before `EpochSummaryCallback`, which writes `logs` to TensorBoard.

  In evaluation, Keras passes `on_test_end` a copy of `logs`, so changes made
  there do not reach the dict that `model.evaluate` returns. A subclass must
  also save its metrics in `latest_metrics`. `KerasTrainer` reads that property
  after `model.evaluate` and adds the values, with a `val_` prefix, to the
  validation logs that it writes.
  """

  def __init__(self):
    super().__init__()
    self._latest_metrics: dict[str, float] = {}

  @property
  def latest_metrics(self) -> dict[str, float]:
    """Returns a copy of the metrics from the most recent window."""
    return dict(self._latest_metrics)

  def _record(self, logs: dict[str, Any] | None, name: str, value: float):
    """Saves a metric and adds it to `logs` if `logs` is not None."""
    self._latest_metrics[name] = value
    if logs is not None:
      logs[name] = value


def _require_positive(name: str, value: int) -> int:
  """Returns `value`, or raises a `ValueError` if it is not positive.

  A silent fallback to 1 would inflate the per-device metrics by up to the pod
  size and nobody would notice on the dashboard.

  Args:
    name: Name of the value, for the error message.
    value: The value to check.
  """
  if value < 1:
    raise ValueError(f"{name} must be positive, got {value}.")
  return value


class _TokenThroughputCallback(MetricsCallback):
  """Adds token throughput metrics for one timed window.

  Metrics, all under `throughput/`:

    * `k_valid_tokens_per_sec_per_device`: non-padding tokens per second per
      accelerator, in thousands. It goes up when packing improves and does not
      depend on the number of chips.
    * `k_tokens_per_sec_per_device`: the same, but padding tokens are counted
      too. This is the rate that the hardware sees.
    * `packing_efficiency_pct`: `valid_seq_len / padded_seq_len * 100`.

  `k_valid / k_tokens == packing_efficiency_pct / 100` by construction. So the
  three metrics have only two degrees of freedom, and the fact that they agree
  does not prove that they are correct.

  The metric names are the same for training and evaluation. `KerasTrainer`
  adds a `val_` prefix to evaluation metrics and `EpochSummaryCallback` removes
  it again, so both series show in the same TensorBoard cards.

  The callback never reads `logs` in its batch hooks and sets `async_safe`.
  Thus Keras can keep dispatching batch callbacks asynchronously. Without
  `async_safe`, one callback with a batch hook makes Keras convert all the
  step logs to Python floats on the main thread on every step, which blocks
  the accelerator.

  Subclasses select the Keras hooks that open and close the window.
  """

  def __init__(
      self,
      *,
      global_batch_size: int,
      padded_seq_len: int,
      valid_seq_len_key: str,
      device_count: int | None = None,
  ):
    """Initializes the callback.

    Args:
      global_batch_size: Number of examples per step, across all devices.
      padded_seq_len: Padded sequence length of one example.
      valid_seq_len_key: Key of the model metric that holds the mean number of
        non-padding tokens per example. A per-batch sum gives wrong values. If
        the result is out of range, the token metrics are skipped and an error
        is logged.
      device_count: Global accelerator count. If None, it is read from
        `jax.device_count()` when the first metrics are computed.
    """
    super().__init__()
    self.async_safe = True
    self._global_batch_size = _require_positive(
        "global_batch_size", global_batch_size
    )
    self._padded_seq_len = _require_positive("padded_seq_len", padded_seq_len)
    self._valid_seq_len_key = valid_seq_len_key
    self._device_count = (
        None
        if device_count is None
        else _require_positive("device_count", device_count)
    )
    self._window_start: float | None = None
    self._steps = 0

  def _open_window(self) -> None:
    """Starts the timer and resets the step count."""
    self._window_start = time.time()
    # Fallback for when no batch hook fires in the window.
    self._steps = (self.params or {}).get("steps") or 0

  def _count_step(self, batch: int) -> None:
    """Records progress from a batch-end hook.

    Args:
      batch: Index of the last step that completed. Keras gives the end step,
        not a count, so this stays correct when `steps_per_execution > 1`.
    """
    self._steps = batch + 1

  def _get_device_count(self) -> int:
    if self._device_count is None:
      self._device_count = _require_positive(
          "jax.device_count()", jax.device_count()
      )
    return self._device_count

  def _close_window(self, logs: dict[str, Any] | None) -> None:
    """Computes the metrics for the window and adds them to `logs`."""
    elapsed = (
        time.time() - self._window_start
        if self._window_start is not None
        else 0.0
    )
    self._latest_metrics = {}
    if not logs or self._valid_seq_len_key not in logs:
      logging.warning(
          "%s: %s is not in logs; throughput metrics are skipped.",
          type(self).__name__,
          self._valid_seq_len_key,
      )
      return
    valid_seq_len = float(logs[self._valid_seq_len_key])
    packing_efficiency_pct = valid_seq_len / self._padded_seq_len * 100.0
    if not 0.0 <= packing_efficiency_pct <= 100.0:
      # Do not stop a long training job because of a logging metric, and do
      # not write values that are wrong by a factor of the batch size.
      logging.error(
          "%s: packing_efficiency_pct=%.1f is not in [0, 100]. %s is probably"
          " a per-batch sum, not a per-example mean. Throughput metrics are"
          " skipped.",
          type(self).__name__,
          packing_efficiency_pct,
          self._valid_seq_len_key,
      )
      return
    # This metric has no time term, so record it even without a step rate.
    self._record(
        logs, "throughput/packing_efficiency_pct", packing_efficiency_pct
    )

    if not self._steps or elapsed <= 0:
      logging.warning(
          "%s: no step rate is available; tokens/sec metrics are skipped.",
          type(self).__name__,
      )
      return
    examples_per_sec_per_device = (
        self._global_batch_size * self._steps / elapsed
    ) / self._get_device_count()
    self._record(
        logs,
        "throughput/k_valid_tokens_per_sec_per_device",
        examples_per_sec_per_device * valid_seq_len / 1000.0,
    )
    self._record(
        logs,
        "throughput/k_tokens_per_sec_per_device",
        examples_per_sec_per_device * self._padded_seq_len / 1000.0,
    )


class TrainThroughputCallback(_TokenThroughputCallback):
  """Adds token throughput metrics once per training epoch.

  The window starts in `on_epoch_begin` and ends in `on_epoch_end`, i.e. it
  covers `steps_per_loop` steps. With `KerasTrainer.train_and_evaluate`, Keras
  runs validation inside the epoch, so the window also includes validation time.
  """

  def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None):
    self._open_window()

  def on_train_batch_end(self, batch: int, logs: dict[str, Any] | None = None):
    self._count_step(batch)

  def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None):
    self._close_window(logs)


class EvalThroughputCallback(_TokenThroughputCallback):
  """Adds token throughput metrics once per evaluation pass.

  `model.evaluate` does not call `on_epoch_end`, so the window starts in
  `on_test_begin` and ends in `on_test_end`. `KerasTrainer` writes the result
  from `latest_metrics` (see `MetricsCallback`).
  """

  def on_test_begin(self, logs: dict[str, Any] | None = None):
    self._open_window()

  def on_test_batch_end(self, batch: int, logs: dict[str, Any] | None = None):
    self._count_step(batch)

  def on_test_end(self, logs: dict[str, Any] | None = None):
    self._close_window(logs)

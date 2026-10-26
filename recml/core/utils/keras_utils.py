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
import copy
import dataclasses
import datetime
import enum
import math
import os
import re
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
# How the SparseCore stacked tables were sharded (see `_sparsecore_topology`).
SPARSECORE_TOPOLOGY_KEY = "sparsecore_topology"
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
            **_sparsecore_topology_save_args(model),
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

    sparsecore_restores = (
        _prepare_sparsecore_restore(model, str(self.directory), epoch)
        if _has_sparsecore_tables(model)
        else []
    )

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

    if sparsecore_restores:
      _finish_sparsecore_restore(sparsecore_restores)


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
            SPARSECORE_TOPOLOGY_KEY,
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
            **_sparsecore_topology_save_args(model),
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

    sparsecore_restores = (
        _prepare_sparsecore_restore(model, str(self.directory), epoch)
        if _has_sparsecore_tables(model)
        else []
    )

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

    if sparsecore_restores:
      _finish_sparsecore_restore(sparsecore_restores)


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

  sparsecore_restores = (
      _prepare_sparsecore_restore(model, checkpoint_dir, epoch)
      if _has_sparsecore_tables(model)
      else []
  )

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

  if sparsecore_restores:
    _finish_sparsecore_restore(sparsecore_restores)

  if restore_model_epoch:
    model._initial_epoch = epoch + 1  # pylint: disable=protected-access  # pyrefly: ignore[unsupported-operation]
  if restore_optimizer_vars and not restore_iterations:
    model.optimizer.iterations.assign(0)

  return model


class _RestoreValuePlaceholder:
  """Stands in for `keras.Variable._value` to restore a differently shaped value.

  Used by `_prepare_sparsecore_restore` for SparseCore variables saved for
  another topology. The restore paths of this module build the Orbax restore
  target from `var.value.shape`, `.dtype` and `.sharding` (see
  `_to_shape_dtype_struct`), so these attributes make Orbax restore the saved
  shape. They then call `var.value.delete()` to free each variable before
  restoring; the placeholder holds no buffer, so `delete` is a no-op.
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


def _settings_in_stack(
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


def _reshard_stacked_array(
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
  # TODO(b/577354074): Replace with keras_rs's cross-topology resharding once
  # supported.
  dst_settings = _settings_in_stack(
      table_stack, dst_num_shards, dst_num_sc_per_device
  )
  for spec in table_stack:
    if spec.setting_in_stack != dst_settings[spec.name]:
      raise ValueError(
          f"Unexpected SparseCore stacking layout for table '{spec.name}':"
          f" {spec.setting_in_stack} != {dst_settings[spec.name]}. Cannot"
          " reshard the checkpoint across TPU topologies."
      )
  src_settings = _settings_in_stack(
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


def _sparsecore_topology(model: keras.Model) -> dict[str, dict[str, int]]:
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
  for config, _, table_vars in _sparsecore_layer_states(model):
    num_sc_per_device = int(config.num_sc_per_device)
    num_shards = int(config.mesh.devices.size) * num_sc_per_device
    for emb_vars in table_vars.values():
      topology[emb_vars.table.path] = {
          "num_shards": num_shards,
          "num_sc_per_device": num_sc_per_device,
      }
  return topology


def _sparsecore_topology_save_args(model: keras.Model) -> dict[str, Any]:
  """Returns the Orbax save args of `_sparsecore_topology`, if any.

  Only models with SparseCore tables get the item, so the checkpoints of other
  models are unchanged.

  Args:
    model: The model being saved.
  """
  topology = _sparsecore_topology(model)
  if not topology:
    return {}
  return {
      SPARSECORE_TOPOLOGY_KEY: ocp.args.JsonSave(
          {SPARSECORE_TOPOLOGY_KEY: topology}
      )
  }


def _saved_sparsecore_topology(
    checkpoint_path: str,
) -> dict[str, dict[str, int]] | None:
  """Returns the saved `_sparsecore_topology`, or None if it wasn't saved.

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


def _sparsecore_layer_states(
    model: keras.Model,
) -> list[tuple[Any, Any, Mapping[str, Any]]]:
  """Returns the SparseCore state of the SparseCore layers of `model`.

  Matches the JAX `keras_rs.layers.DistributedEmbedding` layers with tables on
  SparseCore, including the one wrapped by `SparseCoreEmbedding`.

  Args:
    model: The model being restored.

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


def _has_sparsecore_tables(model: keras.Model) -> bool:
  """Returns whether `model` has embedding tables on SparseCore."""
  return keras.backend.backend() == "jax" and bool(
      _sparsecore_layer_states(model)
  )


@dataclasses.dataclass(frozen=True)
class _StackRestore:
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


def _saved_trainable_metadata(
    model: keras.Model, checkpoint_dir: str, epoch: int | None
) -> dict[str, Any]:
  """Returns the checkpoint metadata of the model's trainable variables by path.

  Args:
    model: The model being restored.
    checkpoint_dir: Directory containing the Orbax checkpoint(s).
    epoch: Optional step/epoch being restored.

  Returns:
    Variable path to its Orbax `ArrayMetadata`, or an empty dict if the
    checkpoint has no Orbax state item.
  """
  try:
    checkpoint_path, _ = resolve_orbax_checkpoint_path(checkpoint_dir, epoch)
    state_dir = epath.Path(checkpoint_path) / STATE_CHECKPOINT_KEY
    if not state_dir.exists():
      return {}
    saved_state_metadata = ocp.handlers.PyTreeCheckpointHandler().metadata(
        state_dir
    )
  except Exception:
    # Without the metadata, the SparseCore tables can't be resharded, and
    # restoring them as is would fail with a shape mismatch or silently load
    # rows sharded for another topology.
    logging.exception(
        "Failed to read the metadata of the checkpoint in %s (epoch %s) to"
        " restore the SparseCore tables across TPU topologies.",
        checkpoint_dir,
        epoch,
    )
    raise

  saved_trainable_meta = saved_state_metadata.get(TRAINABLE_VARIABLES_KEY)
  if isinstance(saved_trainable_meta, Mapping):  # V3: keyed by variable path.
    return {
        var.path: saved_trainable_meta[var.path]
        for var in model.trainable_variables
        if var.path in saved_trainable_meta
    }
  if isinstance(saved_trainable_meta, Sequence) and len(
      saved_trainable_meta
  ) == len(
      model.trainable_variables
  ):  # V2: in variable order.
    return {
        var.path: meta
        for var, meta in zip(model.trainable_variables, saved_trainable_meta)
    }
  return {}


def _plan_sparsecore_restore(
    model: keras.Model, checkpoint_dir: str, epoch: int | None
) -> list[_StackRestore]:
  """Returns the SparseCore stacks of `model` that must be resharded.

  Stacks saved for the current topology are left out: Orbax restores them as
  is. Stacks of checkpoints saved without their topology are restored as is if
  their shapes match.

  Args:
    model: The model being restored.
    checkpoint_dir: Directory containing the Orbax checkpoint(s).
    epoch: Optional step/epoch being restored.

  Raises:
    ValueError: If a stack saved with another shape has no saved topology.
  """
  sc_layer_states = _sparsecore_layer_states(model)
  if not sc_layer_states:
    return []
  if not model._jax_state_synced:  # pylint: disable=protected-access
    model.jax_state_sync()
  meta_by_path = _saved_trainable_metadata(model, checkpoint_dir, epoch)
  if not meta_by_path:
    return []
  checkpoint_path, _ = resolve_orbax_checkpoint_path(checkpoint_dir, epoch)
  saved_topology = _saved_sparsecore_topology(checkpoint_path) or {}

  plans = []
  for config, sc_layout, table_vars in sc_layer_states:
    dst_num_sc_per_device = config.num_sc_per_device
    dst_num_shards = config.mesh.devices.size * dst_num_sc_per_device
    table_stacks = jte_table_stacking.get_table_stacks(
        jte_embedding.get_table_specs(config.feature_specs)
    )
    for stack_name, table_stack in table_stacks.items():
      emb_vars = table_vars[stack_name]
      stored_meta = meta_by_path.get(emb_vars.table.path)
      if stored_meta is None:
        continue
      variables = [emb_vars.table, *keras.tree.flatten(emb_vars.slot)]
      # Slots have the shape of their table.
      saved_shapes = [
          tuple(meta_by_path.get(var.path, stored_meta).shape)
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
      plan = _StackRestore(
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


def _row_sharding(
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


# TODO(b/577354074): Remove this cross-topology restore (and the
# jax_tpu_embedding dependency of this module) once keras_rs
# `DistributedEmbedding` supports resharding SparseCore tables and optimizer
# slots across TPU topologies. Until then, consider moving it behind a
# layer-provided checkpoint restore adapter next to `SparseCoreEmbedding`.
def _prepare_sparsecore_restore(
    model: keras.Model, checkpoint_dir: str, epoch: int | None
) -> list[_StackRestore]:
  """Prepares the SparseCore tables of `model` for a cross-topology restore.

  `jax_tpu_embedding` stacks, pads and mod-shards the SparseCore tables (and
  their optimizer slots) for the total number of SparseCores of the topology
  (`global_device_count * num_sc_per_device`), so both the shape and the row
  layout of a stacked variable change across topologies (e.g. `glp=4x4` train
  vs. `glp=2x2` eval). The restore paths of this module restore each variable
  into `jax.ShapeDtypeStruct(var.value.shape, ...)`, so restoring such a
  checkpoint as is would fail with a shape mismatch, or load rows sharded for
  the wrong topology when the shapes happen to match.

  For models with SparseCore tables (see `_has_sparsecore_tables`), call this
  before building the Orbax restore target and `_finish_sparsecore_restore`
  after the restore. This reads the source topology saved with the checkpoint
  (see `_sparsecore_topology`) and, for each stacked variable saved with
  another shape, replaces its value with a `_RestoreValuePlaceholder` of the
  saved shape, so that Orbax restores the raw saved tensor. Stacks saved for
  the current topology are not touched.

  If the restore fails in between, the model must be rebuilt: like the restore
  paths themselves, this frees the current buffers of the restored variables.

  Args:
    model: The model being restored.
    checkpoint_dir: Directory containing the Orbax checkpoint(s).
    epoch: Optional step/epoch being restored.

  Returns:
    The stacks to reshard, to pass to `_finish_sparsecore_restore`.

  Raises:
    ValueError: If a stack saved with another shape has no saved topology.
    Exception: If the checkpoint metadata can't be read.
  """
  plans = _plan_sparsecore_restore(model, checkpoint_dir, epoch)
  for plan in plans:
    sc_sharding = plan.layout.backend_layout.sharding
    for var, saved_shape in zip(plan.variables, plan.saved_shapes):
      # Restore the raw saved value instead of laying it out for the current
      # topology.
      var._layout = None  # pylint: disable=protected-access
      if saved_shape != tuple(var.shape):
        dtype = var.value.dtype
        sharding = _row_sharding(sc_sharding, saved_shape[0])
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


def _finish_sparsecore_restore(plans: Sequence[_StackRestore]) -> None:
  """Reshards the restored SparseCore stacks for the current topology.

  Args:
    plans: The result of `_prepare_sparsecore_restore`, after the restore.
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
        restored = _reshard_stacked_array(
            restored,
            plan.table_stack,
            src_num_shards=plan.src_num_shards,
            src_num_sc_per_device=plan.src_num_sc_per_device,
            dst_num_shards=plan.dst_num_shards,
            dst_num_sc_per_device=plan.dst_num_sc_per_device,
            out_sharding=plan.layout.backend_layout.sharding,
        )
      var.assign(restored)


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

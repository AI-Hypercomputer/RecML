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
import json
import re
from typing import Any

from absl import logging
import jax
import keras
import orbax.checkpoint as ocp
import tensorflow as tf

ORBAX_CHECKPOINT_DEFAULT_KEY = "default"


def _assert_variables_built(model: keras.Model):
  if not model.built or not model.optimizer.built:
    raise ValueError(
        "To use methods on `KerasOrbaxCheckpointManager`, your model and"
        f" optimizer must be built. Model built: {model.built}, Optimizer"
        f" built: {model.optimizer.built}"
    )


class PathTrie:
  """A class to create a Prefix Tree (Trie) from file paths."""

  def __init__(self):
    """Initializes the Trie with an empty dictionary as its root.

    Also initializes data structures for the context-aware re-indexing logic.
    """
    self.root = {}
    # Stores mapping from an original full path part (e.g., 'aaa/bcd/ee_1')
    # to its new processed name (e.g., 'ee_1'). This acts as a cache.
    self.reindex_map = {}
    # Stores the next available index for a given full base name key.
    # e.g., {'aaa/bcd/ee': 2} means the next part with this prefix
    # and base name will become 'ee_2'.
    self.base_name_counts = {}

  def reset(self):
    """Resets the Trie to its initial state."""
    self.root = {}
    self.reindex_map = {}
    self.base_name_counts = {}

  def insert(self, path: str):
    """Inserts a complete path string into the trie.

    It processes each path component, re-indexing it based on its context,
    which is defined by the full path of processed parts leading up to it.

    Args:
        path: A string representing the path, with components separated by '/'.

    Returns:
        The reindexed path.
    """
    node = self.root
    original_parts = path.strip("/").split("/")
    processed_prefix_parts = []  # The path built from *new* names

    for i, part in enumerate(original_parts):
      # The key for memoization must be unique to the original part in its
      # context. The full original path up to and including this part serves as
      # this unique key.
      original_path_to_part = "/".join(original_parts[: i + 1])

      # Check if we've processed this exact part in its exact context before.
      if original_path_to_part in self.reindex_map:
        processed_part = self.reindex_map[original_path_to_part]
      else:
        # This is a new, unique part we haven't seen before.

        # Determine the local base name by stripping any '_<number>' suffix.
        match = re.match(r"(.+)_(\d+)$", part)
        local_base_name = match.group(1) if match else part

        # The key for counting is the path of processed parts leading to
        # the current part's base name. This provides the context.
        full_base_name_key = "/".join(
            processed_prefix_parts + [local_base_name]
        )

        # Get the current count for this base name to determine its new index.
        count = self.base_name_counts.get(full_base_name_key, 0)

        # The first occurrence (count=0) of a base name is the local base name.
        if count == 0:
          processed_part = local_base_name
        # Subsequent parts with the same base name context get a numeric suffix.
        else:
          processed_part = f"{local_base_name}_{count}"

        # Store the result in our memoization map for the original path part.
        self.reindex_map[original_path_to_part] = processed_part

        # Increment the count for this base name context.
        self.base_name_counts[full_base_name_key] = count + 1

      # Move down the tree using the newly processed part name.
      node = node.setdefault(processed_part, {})
      # Add the processed part to our prefix for the next iteration's context.
      processed_prefix_parts.append(processed_part)
    return ("/").join(processed_prefix_parts)

  def get_all_paths(self) -> list[str]:
    """Traverses the trie to reconstruct and return all full paths.

    Returns:
        A list of strings, where each string is a reconstructed path.
    """
    all_paths = []
    self._traverse_paths(self.root, [], all_paths)
    return all_paths

  def _traverse_paths(
      self,
      node: dict[str, Any],
      current_path_parts: list[Any],
      all_paths: list[Any],
  ):
    """A recursive helper function to perform a depth-first traversal of the trie.

    Args:
        node: The current node (dictionary) in the trie.
        current_path_parts: The list of parts forming the path to the current
          node.
        all_paths: The master list to which complete paths are added.
    """
    # Iterate through all children of the current node.
    for part, child_node in node.items():
      # Add the current part to our path tracker
      current_path_parts.append(part)

      # If the child node is empty, it signifies the end of a complete path.
      if not child_node:
        all_paths.append("/".join(current_path_parts))
      else:
        # If there are more parts, recurse deeper into the tree.
        self._traverse_paths(child_node, current_path_parts, all_paths)

      # Backtrack: remove the current part to explore other branches (siblings).
      current_path_parts.pop()

  def __str__(self):
    """Returns a string representation of the trie in a readable JSON format."""
    return json.dumps(self.root, indent=4)


def _get_jax_state_with_keys(
    model: keras.Model,
    trainable_variables: bool = False,
    non_trainable_variables: bool = False,
    optimizer_variables: bool = False,
    metrics_variables: bool = False,
    purge_model_variables: bool = False,
) -> tuple[Sequence[Mapping[str, jax.Array]], Sequence[Mapping[str, int]]]:
  """Returns a dictionary of variables with keys.

  Modified from _get_jax_state.

  Args:
    model: The Keras model to get the variables from.
    trainable_variables: Whether to get the trainable variables.
    non_trainable_variables: Whether to get the non-trainable variables.
    optimizer_variables: Whether to get the optimizer variables.
    metrics_variables: Whether to get the metrics variables.
    purge_model_variables: Whether to purge the model variables.

  Returns:
    A list of dictionaries of variables with keys.
    A list of indexes of the variables with keys.
  """
  variable_list = []
  variable_index_list = []
  variables_path_trie = PathTrie()
  for include_variables, variables in [
      (trainable_variables, model.trainable_variables),
      (non_trainable_variables, model.non_trainable_variables),
      (optimizer_variables, model.optimizer.variables),
      (metrics_variables, model.metrics_variables),
  ]:
    if include_variables:
      index = 0
      variable_dict = {}
      variable_index = {}
      for v in variables:
        variable_key = variables_path_trie.insert(v.path)
        variable_dict[variable_key] = v.value
        variable_index[variable_key] = index
        index += 1
      variable_list.append(variable_dict)
      variable_index_list.append(variable_index)
  if purge_model_variables:
    model._purge_model_variables(  # pylint: disable=protected-access
        trainable_variables=trainable_variables,
        non_trainable_variables=non_trainable_variables,
        optimizer_variables=optimizer_variables,
        metrics_variables=metrics_variables,
    )
  return variable_list, variable_index_list


class KerasOrbaxCheckpointManager(ocp.CheckpointManager):
  """An Orbax checkpoint manager for Keras 3."""

  def __init__(
      self,
      checkpoint_dir: str,
      max_to_keep: int = 5,
      save_interval_epochs: int = 1,
      legacy_format: bool = True,
  ):
    """Initializes a KerasOrbaxCheckpointManager.

    Args:
      checkpoint_dir: The directory to save checkpoints to.
      max_to_keep: The maximum number of checkpoints to keep.
      save_interval_epochs: The interval (in epochs) to save checkpoints.
      legacy_format: Whether to use the legacy checkpoint format.
    """
    super().__init__(
        directory=checkpoint_dir,
        checkpointers=ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler()),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=save_interval_epochs,
            max_to_keep=max_to_keep,
        ),
    )
    self._legacy_format = legacy_format

  def save_model_variables(
      self,
      model: keras.Model,
      epoch: int,
      logs: Mapping[str, Any] | None = None,
  ):
    _assert_variables_built(model)
    logging.info("Writing checkpoint for epoch %s...", epoch)
    if self._legacy_format:
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
    else:
      state, _ = _get_jax_state_with_keys(
          model,
          trainable_variables=True,
          non_trainable_variables=True,
          optimizer_variables=True,
      )

      logging.info(
          "Save Checkpointing state with keys '%s'", [v.keys() for v in state]
      )

    self.save(step=epoch, items=state, metrics=logs)

  def restore_model_variables(self, model: keras.Model, epoch: int):
    _assert_variables_built(model)
    if self._legacy_format:
      state = model._get_jax_state(  # pylint: disable=protected-access
          trainable_variables=True,
          non_trainable_variables=True,
          optimizer_variables=True,
          purge_model_variables=True,
      )
      variable_index_list = None
    else:
      state, variable_index_list = _get_jax_state_with_keys(
          model,
          trainable_variables=True,
          non_trainable_variables=True,
          optimizer_variables=False,
          purge_model_variables=True,
      )
      logging.info(
          "Restore Checkpointing state with keys '%s'",
          [v.keys() for v in state],
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
    if not self._legacy_format and variable_index_list is not None:
      for i in range(len(variable_index_list)):
        restored_state[i] = [
            restored_state[i][k] for k, _ in variable_index_list[i].items()
        ]
    (
        trainable_variables,
        non_trainable_variables,
    ) = restored_state[:2]
    model._jax_state = {  # pylint: disable=protected-access
        "trainable_variables": trainable_variables,
        "non_trainable_variables": non_trainable_variables,
    }
    if self._legacy_format:
      model._jax_state["optimizer_variables"] = restored_state[2]  # pylint: disable=protected-access
    model.jax_state_sync()


class EpochOrbaxCheckpointAndRestoreCallback(keras.callbacks.Callback):
  """A callback for checkpointing and restoring state using Orbax."""

  def __init__(
      self,
      checkpoint_manager: KerasOrbaxCheckpointManager,
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
    legacy_format: bool = True,
    transforms: Mapping[str, Any] | None = None,
):
  """Restores a Keras 3 Jax backend model from an Orbax checkpoint.

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
    legacy_format: Whether to use the legacy format for restoring the model.
    transforms: A mapping from variable keys to the corresponding restore
      arguments. If None, the model will be restored with the same variable
      structure as the checkpoint. If provided, the model will be restored with
      the provided transforms.

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
  if not legacy_format:
    # TODO(zixiangzhou): Remove this once the optimizer format is supported.
    restore_optimizer_vars = False

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
  if legacy_format:
    state = model._get_jax_state(  # pylint: disable=protected-access
        trainable_variables=True,
        non_trainable_variables=True,
        optimizer_variables=restore_optimizer_vars,
        purge_model_variables=True,
    )
    variable_index_list = None
  else:
    state, variable_index_list = _get_jax_state_with_keys(
        model,
        trainable_variables=True,
        non_trainable_variables=True,
        optimizer_variables=restore_optimizer_vars,
        purge_model_variables=True,
    )
    logging.info(
        "Restore Checkpointing state with keys '%s'", [v.keys() for v in state]
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
              transforms={} if transforms is None else transforms,
              restore_args=ocp.checkpoint_utils.construct_restore_args(
                  abstract_state
              ),
          ),
      }),
  )[ORBAX_CHECKPOINT_DEFAULT_KEY]
  if not legacy_format and variable_index_list is not None:
    for i in range(len(variable_index_list)):
      restored_state[i] = [
          restored_state[i][k] for k, _ in variable_index_list[i].items()
      ]
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
    model._jax_state["optimizer_variables"] = (  # pylint: disable=protected-access
        optimizer_variables
    )
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

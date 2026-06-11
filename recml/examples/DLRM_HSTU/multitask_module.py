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
"""Contains modules and functions for handling multitask predictions and losses."""

import abc
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Tuple

from flax import linen as nn
import jax.numpy as jnp
import numpy as np
import optax


# These data classes are pure Python and can be used directly.
class MultitaskTaskType(IntEnum):
  BINARY_CLASSIFICATION = 0
  REGRESSION = 1


@dataclass
class TaskConfig:
  task_name: str
  task_weight: int
  task_type: MultitaskTaskType


class MultitaskModule(nn.Module):
  """Abstract base class for multitask modules in Flax."""

  def __call__(
      self,
      encoded_user_embeddings: jnp.ndarray,
      item_embeddings: jnp.ndarray,
      supervision_labels: Dict[str, jnp.ndarray],
      supervision_weights: Dict[str, jnp.ndarray],
      deterministic: bool,
  ) -> Tuple[
      jnp.ndarray,
      Optional[jnp.ndarray],
      Optional[jnp.ndarray],
      Optional[jnp.ndarray],
  ]:
    """Computes multi-task predictions.

    Args:
        encoded_user_embeddings: (B, N, D) float array.
        item_embeddings: (B, N, D) float array.
        supervision_labels: Dictionary of (B, N) float or int arrays.
        supervision_weights: Dictionary of (B, N) float or int arrays.
        deterministic: If True, losses are not computed (inference mode).

    Returns:
        A tuple of (predictions, labels, weights, losses).
        Predictions are of shape (num_tasks, B, N).
    """
    raise NotImplementedError


def _compute_pred_and_logits(
    prediction_module: nn.Module,
    encoded_user_embeddings: jnp.ndarray,
    item_embeddings: jnp.ndarray,
    task_offsets: List[int],
    has_multiple_task_types: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Computes predictions and raw logits from user and item embeddings."""
  # Logits are computed by applying the prediction module to the
  # element-wise product.
  # Input shape: (B, N, D), Output shape: (B, N, num_tasks)
  mt_logits_unnposed = prediction_module(
      encoded_user_embeddings * item_embeddings
  )
  # Transpose to (num_tasks, B, N) to match PyTorch logic.
  mt_logits = jnp.transpose(mt_logits_unnposed, (2, 0, 1))

  mt_preds_list: List[jnp.ndarray] = []
  for task_type in MultitaskTaskType:
    start_offset, end_offset = (
        task_offsets[task_type],
        task_offsets[task_type + 1],
    )
    if end_offset > start_offset:
      task_logits = mt_logits[start_offset:end_offset, ...]
      if task_type == MultitaskTaskType.REGRESSION:
        # For regression, predictions are the raw logits.
        mt_preds_list.append(task_logits)
      else:
        # For classification, predictions are the sigmoid of the logits.
        mt_preds_list.append(nn.sigmoid(task_logits))

  mt_preds = (
      jnp.concatenate(mt_preds_list, axis=0)
      if has_multiple_task_types
      else mt_preds_list[0]
  )

  return mt_preds, mt_logits


def _compute_labels_and_weights(
    supervision_labels: Dict[str, jnp.ndarray],
    supervision_weights: Dict[str, jnp.ndarray],
    task_configs: List[TaskConfig],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Aggregates label and weight tensors from input dictionaries."""
  # Get a sample tensor to determine shape and dtype for the default weight.
  first_label = next(iter(supervision_labels.values()))
  default_supervision_weight = jnp.ones_like(first_label)

  mt_labels_list: List[jnp.ndarray] = []
  mt_weights_list: List[jnp.ndarray] = []
  for task in task_configs:
    mt_labels_list.append(supervision_labels[task.task_name])
    mt_weights_list.append(
        supervision_weights.get(task.task_name, default_supervision_weight)
    )

  # Stack along a new 'task' dimension.
  mt_labels = jnp.stack(mt_labels_list, axis=0)
  mt_weights = jnp.stack(mt_weights_list, axis=0)

  return mt_labels, mt_weights


def _compute_loss(
    task_offsets: List[int],
    causal_multitask_weights: float,
    mt_logits: jnp.ndarray,
    mt_labels: jnp.ndarray,
    mt_weights: jnp.ndarray,
    has_multiple_task_types: bool,
) -> jnp.ndarray:
  """Computes the final loss across all tasks."""
  mt_losses_list: List[jnp.ndarray] = []
  for task_type in MultitaskTaskType:
    start_offset, end_offset = (
        task_offsets[task_type],
        task_offsets[task_type + 1],
    )
    if end_offset > start_offset:
      task_logits = mt_logits[start_offset:end_offset, ...]
      task_labels = mt_labels[start_offset:end_offset, ...]
      task_weights = mt_weights[start_offset:end_offset, ...]

      if task_type == MultitaskTaskType.REGRESSION:
        # Equivalent to mse_loss with reduction='none'.
        task_losses = (task_logits - task_labels) ** 2
      else:
        # Equivalent to binary_cross_entropy_with_logits with reduction='none'.
        task_losses = optax.sigmoid_binary_cross_entropy(
            task_logits, task_labels
        )

      # Apply task-specific weights.
      mt_losses_list.append(task_losses * task_weights)

  mt_losses = (
      jnp.concatenate(mt_losses_list, axis=0)
      if has_multiple_task_types
      else mt_losses_list[0]
  )

  # Normalize loss per task by the sum of weights for that task.
  # Sum over the item dimension (axis=-1).
  sum_losses = mt_losses.sum(axis=-1)
  sum_weights = mt_weights.sum(axis=-1)

  # Clamp sum_weights to avoid division by zero for empty examples.
  normalized_losses = sum_losses / jnp.maximum(sum_weights, 1.0)

  # Apply a global weight for this entire multitask head.
  return normalized_losses * causal_multitask_weights


class DefaultMultitaskModule(MultitaskModule):
  """
  JAX/Flax implementation of the default multitask module.

  Attributes:
      task_configs: A list of TaskConfig objects, which must be pre-sorted
          by task_type.
      embedding_dim: The dimensionality of the input embeddings.
      prediction_fn: A function that returns a Flax module for predictions,
          e.g., a simple MLP. It takes embedding_dim and num_tasks as input.
      causal_multitask_weights: A global weight for the final computed loss.
  """
  task_configs: List[TaskConfig]
  embedding_dim: int
  prediction_fn: Callable[[int, int], nn.Module]
  causal_multitask_weights: float

  def setup(self):
    if not self.task_configs:
      raise ValueError("task_configs must be non-empty.")

    # Check if tasks are sorted by type, as required by the original logic.
    is_sorted = all(
        self.task_configs[i].task_type <= self.task_configs[i + 1].task_type
        for i in range(len(self.task_configs) - 1)
    )
    if not is_sorted:
      raise ValueError("task_configs must be sorted by task_type.")

    # Calculate offsets for slicing tensors based on task type.
    task_offsets_list = [0] * (len(MultitaskTaskType) + 1)
    for task in self.task_configs:
      task_offsets_list[task.task_type + 1] += 1

    self._has_multiple_task_types: bool = (
        task_offsets_list.count(0) < len(MultitaskTaskType)
    )
    self._task_offsets: List[int] = np.cumsum(task_offsets_list).tolist()

    # Instantiate the prediction module.
    self._prediction_module = self.prediction_fn(
        self.embedding_dim, len(self.task_configs)
    )

  def __call__(
      self,
      encoded_user_embeddings: jnp.ndarray,
      item_embeddings: jnp.ndarray,
      supervision_labels: Dict[str, jnp.ndarray],
      supervision_weights: Dict[str, jnp.ndarray],
      deterministic: bool,
  ) -> Tuple[
      jnp.ndarray,
      Optional[jnp.ndarray],
      Optional[jnp.ndarray],
      Optional[jnp.ndarray],
  ]:

    mt_preds, mt_logits = _compute_pred_and_logits(
        prediction_module=self._prediction_module,
        encoded_user_embeddings=encoded_user_embeddings,
        item_embeddings=item_embeddings,
        task_offsets=self._task_offsets,
        has_multiple_task_types=self._has_multiple_task_types,
    )

    mt_labels: Optional[jnp.ndarray] = None
    mt_weights: Optional[jnp.ndarray] = None
    mt_losses: Optional[jnp.ndarray] = None

    if not deterministic:
      mt_labels, mt_weights = _compute_labels_and_weights(
          supervision_labels=supervision_labels,
          supervision_weights=supervision_weights,
          task_configs=self.task_configs,
      )
      mt_losses = _compute_loss(
          task_offsets=self._task_offsets,
          causal_multitask_weights=self.causal_multitask_weights,
          mt_logits=mt_logits,
          mt_labels=mt_labels,
          mt_weights=mt_weights,
          has_multiple_task_types=self._has_multiple_task_types,
      )

    return mt_preds, mt_labels, mt_weights, mt_losses

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
"""JAX implementation of the ActionEncoder module."""

from typing import Dict, List, Optional, Tuple

import flax.linen as nn
from flax.linen import initializers
import jax
import jax.numpy as jnp


class ActionEncoder(nn.Module):
  """Encodes categorical actions and continuous watch times into a fixed-size embedding.

  assumes dense tensors of shape (batch_size, sequence_length) for all inputs.
  """

  action_embedding_dim: int
  action_feature_name: str
  action_weights: List[int]
  watchtime_feature_name: str = ""
  watchtime_to_action_thresholds_and_weights: Optional[
      List[Tuple[int, int]]
  ] = None

  def setup(self):
    """Initializes parameters and constants for the module."""
    wt_thresholds_and_weights = (
        self.watchtime_to_action_thresholds_and_weights or []
    )

    self.combined_action_weights = jnp.array(
        list(self.action_weights) + [w for _, w in wt_thresholds_and_weights]
    )

    self.num_action_types: int = (
        len(self.action_weights) + len(wt_thresholds_and_weights)
    )

    self.action_embedding_table = self.param(
        "action_embedding_table",
        initializers.normal(stddev=0.1),
        (self.num_action_types, self.action_embedding_dim),
    )

    self.target_action_embedding_table = self.param(
        "target_action_embedding_table",
        initializers.normal(stddev=0.1),
        (1, self.output_embedding_dim),
    )

  @property
  def output_embedding_dim(self) -> int:
    """The dimension of the final output embedding."""
    num_watchtime_actions = (
        len(self.watchtime_to_action_thresholds_and_weights)
        if self.watchtime_to_action_thresholds_and_weights
        else 0
    )
    num_action_types = len(self.action_weights) + num_watchtime_actions
    return self.action_embedding_dim * num_action_types

  def __call__(
      self,
      seq_payloads: Dict[str, jax.Array],
      is_target_mask: jax.Array,
  ) -> jax.Array:
    """Processes a batch of sequences to generate action embeddings.

    Args:
        seq_payloads: A dictionary of feature names to dense tensors of shape
          `(batch_size, sequence_length)`.
        is_target_mask: A boolean tensor of shape `(batch_size,
          sequence_length)` where `True` indicates a target item.

    Returns:
        A dense tensor of action embeddings of shape
        `(batch_size, sequence_length, output_embedding_dim)`.
    """

    seq_actions = seq_payloads[self.action_feature_name]

    wt_thresholds_and_weights = (
        self.watchtime_to_action_thresholds_and_weights or []
    )
    if wt_thresholds_and_weights:
      watchtimes = seq_payloads[self.watchtime_feature_name]
      for threshold, weight in wt_thresholds_and_weights:
        watch_action = (watchtimes >= threshold).astype(jnp.int64) * weight
        seq_actions = jnp.bitwise_or(seq_actions, watch_action)

    exploded_actions = (
        jnp.bitwise_and(seq_actions[..., None], self.combined_action_weights)
        > 0
    )

    history_embeddings = (
        exploded_actions[..., None] * self.action_embedding_table
    ).reshape(*seq_actions.shape, -1)

    target_embeddings = jnp.broadcast_to(
        self.target_action_embedding_table, history_embeddings.shape
    )

    final_embeddings = jnp.where(
        is_target_mask[..., None],
        target_embeddings,
        history_embeddings,
    )

    return final_embeddings

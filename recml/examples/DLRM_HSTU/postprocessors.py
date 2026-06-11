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
"""Postprocessors for user embeddings after HSTU layers."""

import math
from typing import Any, Dict, List, Tuple

import flax.linen as nn
from flax.linen.initializers import xavier_uniform
from flax.linen.initializers import zeros
import jax.numpy as jnp


Array = jnp.ndarray
Dtype = Any


class OutputPostprocessor(nn.Module):
  """An abstract class for post-processing user embeddings after HSTU layers."""

  def __call__(
      self,
      seq_embeddings: Array,
      seq_timestamps: Array,
      seq_payloads: Dict[str, Array],
  ) -> Array:
    """Processes the final sequence embeddings.

    Args:
      seq_embeddings: (B, N, D) or (L, D) final embeddings from the model.
      seq_timestamps: (B, N) or (L,) corresponding timestamps.
      seq_payloads: A dictionary of other features.

    Returns:
      The post-processed sequence embeddings.
    """
    raise NotImplementedError


class L2NormPostprocessor(OutputPostprocessor):
    """Postprocesses user embeddings with L2 normalization."""
    epsilon: float = 1e-6

    @nn.compact
    def __call__(
        self,
        seq_embeddings: Array,
        seq_timestamps: Array,
        seq_payloads: Dict[str, Array],
    ) -> Array:
        norm = jnp.linalg.norm(seq_embeddings, ord=2, axis=-1, keepdims=True)
        # Prevent division by zero
        safe_norm = jnp.maximum(norm, self.epsilon)
        return seq_embeddings / safe_norm


class LayerNormPostprocessor(OutputPostprocessor):
    """Postprocesses user embeddings with LayerNorm."""
    embedding_dim: int
    eps: float = 1e-5
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        seq_embeddings: Array,
        seq_timestamps: Array,
        seq_payloads: Dict[str, Array],
    ) -> Array:
        ln = nn.LayerNorm(epsilon=self.eps, dtype=self.dtype)
        return ln(seq_embeddings)


class TimestampLayerNormPostprocessor(OutputPostprocessor):
    """Postprocesses user embeddings with a timestamp-based MLP and LayerNorm."""
    embedding_dim: int
    time_duration_features: List[Tuple[int, int]]
    eps: float = 1e-5
    dtype: Dtype = jnp.float32

    def setup(self):
      self._layer_norm = nn.LayerNorm(epsilon=self.eps, dtype=self.dtype)

      num_time_features = len(self.time_duration_features)
      combiner_input_dim = self.embedding_dim + 2 * num_time_features

      self._time_feature_combiner = nn.Dense(
          features=self.embedding_dim,
          dtype=self.dtype,
          kernel_init=xavier_uniform(),
          bias_init=zeros,
      )

      # Store time feature constants directly. No need for buffers in Flax.
      self._period_units = jnp.array(
          [f[0] for f in self.time_duration_features], dtype=self.dtype
      )
      self._units_per_period = jnp.array(
          [f[1] for f in self.time_duration_features], dtype=self.dtype
      )

    def __call__(
        self,
        seq_embeddings: Array,
        seq_timestamps: Array,
        seq_payloads: Dict[str, Array],
    ) -> Array:
      """Processes sequence embeddings with timestamp features and LayerNorm.

      Creates circular time features, concatenates them to the embeddings,
      processes through an MLP, and applies LayerNorm.

      Args:
        seq_embeddings: (B, N, D) or (L, D) final embeddings from the model.
        seq_timestamps: (B, N) or (L,) corresponding timestamps.
        seq_payloads: A dictionary of other features.

      Returns:
        The post-processed sequence embeddings.
      """

      # 1. Create circular time features from timestamps.
      # Ensure timestamps have a feature dimension for broadcasting.
      if seq_timestamps.ndim != seq_embeddings.ndim:
        timestamps = jnp.expand_dims(seq_timestamps, axis=-1)
      else:
        timestamps = seq_timestamps

      # Ensure correct broadcast shape for time constants.
      # Original shape: (num_features,) -> (1, ..., 1, num_features)
      broadcast_shape = (1,) * (timestamps.ndim - 1) + (-1,)
      period_units = self._period_units.reshape(broadcast_shape)
      units_per_period = self._units_per_period.reshape(broadcast_shape)

      # Calculate the phase angle for the circular representation.
      units_since_epoch = jnp.floor(timestamps / period_units)
      remainder = jnp.remainder(units_since_epoch, units_per_period)
      angle = (remainder / units_per_period) * 2 * math.pi

      # Create sin/cos features. Cast to float32 for precision if needed.
      original_dtype = angle.dtype
      if original_dtype != jnp.float32:
          angle = angle.astype(jnp.float32)

      cos_features = jnp.cos(angle)
      sin_features = jnp.sin(angle)

      time_features = jnp.stack([cos_features, sin_features], axis=-1)

      # New shape will have a final dimension of num_time_features * 2
      final_shape = seq_embeddings.shape[:-1] + (-1,)
      time_features = time_features.reshape(final_shape).astype(original_dtype)
      # 2. Concatenate with sequence embeddings.
      combined_embeddings = jnp.concatenate(
          [seq_embeddings, time_features], axis=-1
      )
      # 3. Process through the MLP and LayerNorm.
      user_embeddings = self._time_feature_combiner(combined_embeddings)
      final_embeddings = self._layer_norm(user_embeddings)
      return final_embeddings

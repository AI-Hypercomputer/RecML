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
"""JAX/Flax implementation of ContextualInterleavePreprocessor."""

from typing import Callable, Dict, Tuple

from flax import linen as nn
import jax.numpy as jnp
from recml.examples.DLRM_HSTU.action_encoder import ActionEncoder
from recml.examples.DLRM_HSTU.content_encoder import ContentEncoder
from recml.examples.DLRM_HSTU.contextualize_mlps import ContextualizedMLP
from recml.examples.DLRM_HSTU.preprocessors import get_contextual_input_embeddings
from recml.examples.DLRM_HSTU.preprocessors import InputPreprocessor


class ContextualInterleavePreprocessor(InputPreprocessor):
  """A JAX/Flax implementation of the ContextualInterleavePreprocessor.

  This preprocessor orchestrates content encoding, action encoding, and
  contextualization using parameterized MLPs, working on dense, padded tensors.
  """

  input_embedding_dim: int
  output_embedding_dim: int
  contextual_feature_to_max_length: Dict[str, int]
  contextual_feature_to_min_uih_length: Dict[str, int]
  content_encoder: ContentEncoder
  content_contextualize_mlp_fn: Callable[[], ContextualizedMLP]
  action_encoder: ActionEncoder
  action_contextualize_mlp_fn: Callable[[], ContextualizedMLP]
  pmlp_contextual_dropout_ratio: float = 0.0
  enable_interleaving: bool = False

  def setup(self):
    self._max_contextual_seq_len = sum(
        self.contextual_feature_to_max_length.values()
    )

    self._content_embedding_mlp = self.content_contextualize_mlp_fn()
    self._action_embedding_mlp = self.action_contextualize_mlp_fn()

    if self._max_contextual_seq_len > 0:
      self._batched_contextual_linear_weights = self.param(
          "batched_contextual_linear_weights",
          nn.initializers.xavier_uniform(),
          (
              self._max_contextual_seq_len,
              self.input_embedding_dim,
              self.output_embedding_dim,
          ),
      )
      self._batched_contextual_linear_bias = self.param(
          "batched_contextual_linear_bias",
          nn.initializers.zeros,
          (self._max_contextual_seq_len, self.output_embedding_dim),
      )
      self._pmlp_dropout = nn.Dropout(rate=self.pmlp_contextual_dropout_ratio)

  def __call__(
      self,
      max_uih_len: int,
      seq_embeddings: jnp.ndarray,
      seq_mask: jnp.ndarray,
      seq_timestamps: jnp.ndarray,
      num_targets: jnp.ndarray,
      seq_payloads: Dict[str, jnp.ndarray],
      *,
      deterministic: bool,
  ) -> Tuple[
      jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]
  ]:
    batch_size, max_seq_len, _ = seq_embeddings.shape

    pmlp_contextual_embeddings = None
    contextual_embeddings = None
    if self._max_contextual_seq_len > 0:
      contextual_input_embeddings = get_contextual_input_embeddings(
          seq_mask=seq_mask,
          seq_payloads=seq_payloads,
          contextual_feature_to_max_length=self.contextual_feature_to_max_length,
          contextual_feature_to_min_uih_length=self.contextual_feature_to_min_uih_length,
          dtype=seq_embeddings.dtype,
      )

      pmlp_contextual_embeddings = self._pmlp_dropout(
          contextual_input_embeddings, deterministic=deterministic
      )

      contextual_embeddings = jnp.einsum(
          "bci,cio->bco",
          contextual_input_embeddings.reshape(
              batch_size, self._max_contextual_seq_len, self.input_embedding_dim
          ),
          self._batched_contextual_linear_weights,
      ) + jnp.expand_dims(self._batched_contextual_linear_bias, axis=0)

    # Content Embeddings
    content_embeddings = self.content_encoder(
        max_uih_len=max_uih_len,
        seq_embeddings=seq_embeddings,
        seq_payloads=seq_payloads,
    )
    content_embeddings = self._content_embedding_mlp(
        seq_embeddings=content_embeddings,
        contextual_embeddings=pmlp_contextual_embeddings,
    )

    # Action Embeddings
    seq_lengths = jnp.sum(seq_mask, axis=1, dtype=jnp.int32)
    indices = jnp.arange(max_seq_len)
    start_target_idx = jnp.expand_dims(seq_lengths - num_targets, axis=1)
    is_target_mask = (indices >= start_target_idx) & seq_mask

    action_embeddings = self.action_encoder(
        seq_payloads=seq_payloads,
        is_target_mask=is_target_mask,
    )
    action_embeddings = self._action_embedding_mlp(
        seq_embeddings=action_embeddings,
        contextual_embeddings=pmlp_contextual_embeddings,
    )

    # Combine
    output_seq_embeddings = content_embeddings + action_embeddings
    output_seq_embeddings *= jnp.expand_dims(seq_mask, axis=-1)
    output_mask = seq_mask
    output_timestamps = seq_timestamps

    # Prepend contextual embeddings
    if self._max_contextual_seq_len > 0:
      output_seq_embeddings = jnp.concatenate(
          [contextual_embeddings, output_seq_embeddings], axis=1
      )
      contextual_mask = jnp.ones(
          (batch_size, self._max_contextual_seq_len), dtype=jnp.bool_
      )
      output_mask = jnp.concatenate([contextual_mask, seq_mask], axis=1)

      contextual_timestamps = jnp.zeros(
          (batch_size, self._max_contextual_seq_len),
          dtype=seq_timestamps.dtype,
      )
      output_timestamps = jnp.concatenate(
          [contextual_timestamps, seq_timestamps], axis=1
      )

    return (
        output_seq_embeddings,
        output_mask,
        output_timestamps,
        num_targets,
        seq_payloads,
    )

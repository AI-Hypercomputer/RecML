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
"""Self-Targeting Unit (STU) module.

This module implements the STU layer and a stack of STU
layers. The STU layer is designed to capture long-range dependencies in
sequential data by incorporating self-attention mechanisms with a gating
mechanism.
"""

import dataclasses
from typing import Optional, Sequence

from flax import linen as nn
import jax
import jax.numpy as jnp


dataclass = dataclasses.dataclass


@dataclass
class STULayerConfig:
  """Configuration for the STU layer.

  Attributes:
      embedding_dim: Input embedding dimension.
      num_heads: Number of attention heads.
      hidden_dim: Hidden dimension of the STU layer.
      attention_dim: Dimension of the attention projections.
      output_dropout_ratio: Dropout ratio for the output.
      causal: Whether to use causal attention.
      target_aware: Whether to use target-aware attention.
      max_attn_len: Maximum attention length.
      attn_alpha: Scaling factor for attention scores.
      use_group_norm: Whether to use group normalization.
      recompute_normed_x: Whether to recompute normalized input.
      recompute_uvqk: Whether to recompute u, v, q, k projections.
      recompute_y: Whether to recompute the output.
      sort_by_length: Whether to sort sequences by length.
      contextual_seq_len: Contextual sequence length.
      min_full_attn_seq_len: Minimum sequence length to apply full attention.
      norm_epsilon: Epsilon value for normalization.
      deterministic: Whether to apply dropout in deterministic mode.
      max_decode_length: The maximum length for decoding.
  """

  embedding_dim: int
  num_heads: int
  hidden_dim: int
  attention_dim: int
  output_dropout_ratio: float = 0.0
  causal: bool = True
  target_aware: bool = True
  max_attn_len: Optional[int] = None
  attn_alpha: Optional[float] = None
  use_group_norm: bool = False
  recompute_normed_x: bool = True
  recompute_uvqk: bool = True
  recompute_y: bool = True
  sort_by_length: bool = True
  contextual_seq_len: int = 0
  norm_epsilon: float = 1e-6
  min_full_attn_seq_len: int = 0
  deterministic: bool = True
  max_decode_length: int = 2048


class STULayer(nn.Module):
  """Self-Targeting Unit layer.

  Attributes:
      config: STULayerConfig, configuration of the STU layer.
  """

  config: STULayerConfig

  def setup(self):
    self.num_heads: int = self.config.num_heads
    self.embedding_dim: int = self.config.embedding_dim
    self.hidden_dim: int = self.config.hidden_dim
    self.attention_dim: int = self.config.attention_dim
    self.output_dropout_ratio: float = self.config.output_dropout_ratio
    self.target_aware: bool = self.config.target_aware
    self.causal: bool = self.config.causal
    self.max_attn_len: int = self.config.max_attn_len or 0
    self.attn_alpha: float = self.config.attn_alpha or 1.0 / (
        self.attention_dim**0.5
    )
    self.use_group_norm: bool = self.config.use_group_norm
    self.norm_epsilon: float = self.config.norm_epsilon
    self.contextual_seq_len: int = self.config.contextual_seq_len
    self.min_full_attn_seq_len: int = self.config.min_full_attn_seq_len

    self.uvqk_weight = self.param(
        '_uvqk_weight',
        nn.initializers.xavier_normal(),
        (
            self.embedding_dim,
            (self.hidden_dim * 2 + self.attention_dim * 2) * self.num_heads,
        ),
    )
    self.uvqk_beta = self.param(
        '_uvqk_beta',
        nn.initializers.zeros,
        (self.hidden_dim * 2 + self.attention_dim * 2) * self.num_heads,
    )

    self.output_weight = self.param(
        '_output_weight',
        nn.initializers.xavier_uniform(),
        (self.hidden_dim * self.num_heads * 3, self.embedding_dim),
    )

    self.dropout_layer = nn.Dropout(rate=self.output_dropout_ratio)
    self.group_norm_layer = nn.GroupNorm(
        num_groups=self.num_heads,
        use_scale=True,
        use_bias=True,
        epsilon=self.norm_epsilon,
    )
    self.input_norm_layer = nn.LayerNorm(
        use_scale=True, use_bias=True, epsilon=self.norm_epsilon
    )
    self.output_norm_layer = nn.LayerNorm(
        use_scale=True, use_bias=True, epsilon=self.norm_epsilon
    )
    self.cached_key = self.variable('cache', 'cached_key', lambda: None)
    self.cached_value = self.variable('cache', 'cached_value', lambda: None)
    self.cache_index = self.variable(
        'cache', 'cache_index', lambda: jnp.zeros((), jnp.int32)
    )

  def _get_valid_attn_mask(self, x, num_targets: Optional[jnp.ndarray]):
    batch_size, seq_len, _ = x.shape
    seq_lengths = jnp.full((batch_size,), seq_len, dtype=jnp.int32)
    ids = jnp.arange(seq_len)[None, :]
    max_ids = seq_lengths[:, None, None]

    if self.contextual_seq_len > 0:
      ids = ids - self.contextual_seq_len + 1
      ids = jnp.maximum(ids, 0)
      max_ids = max_ids - self.contextual_seq_len + 1

    if num_targets is not None:
      max_ids = (max_ids - num_targets[:, None, None]).squeeze(axis=-1)
      ids = jnp.minimum(ids, max_ids)
      row_ids = ids[:, :, None]
      col_ids = ids[:, None, :]
    else:
      row_ids_base = jnp.arange(seq_len)[None, :, None]
      col_ids_base = jnp.arange(seq_len)[None, None, :]
      row_ids = jnp.broadcast_to(row_ids_base, (1, seq_len, seq_len))
      col_ids = jnp.broadcast_to(col_ids_base, (1, seq_len, seq_len))

    row_col_dist = row_ids - col_ids
    valid_attn_mask = jnp.eye(seq_len, dtype=jnp.bool_)[None, :, :]
    if not self.causal:
      row_col_dist = jnp.abs(row_col_dist)
    valid_attn_mask = jnp.logical_or(valid_attn_mask, row_col_dist > 0)
    if self.max_attn_len > 0:
      if self.min_full_attn_seq_len > 0:
        valid_attn_mask = jnp.logical_and(
            valid_attn_mask,
            jnp.logical_or(
                row_col_dist <= self.max_attn_len,
                row_ids >= max_ids - self.min_full_attn_seq_len,
            ),
        )
      else:
        valid_attn_mask = jnp.logical_and(
            valid_attn_mask, row_col_dist <= self.max_attn_len
        )
    if self.contextual_seq_len > 0:
      valid_attn_mask = jnp.logical_or(
          valid_attn_mask, jnp.logical_and(row_ids == 0, col_ids < max_ids)
      )

    return valid_attn_mask

  def hstu_compute_output(self, attn, u, x, deterministic: bool):
    """Computes the output of the STU layer with corrected logic."""
    if self.use_group_norm:
      norm_input = attn.reshape(
          attn.shape[0], attn.shape[1], self.num_heads, self.hidden_dim
      )
      normed_attn = self.group_norm_layer(norm_input).reshape(
          attn.shape[0], attn.shape[1], -1
      )
    else:
      normed_attn = self.output_norm_layer(attn)

    gated_attn = u * normed_attn
    proj_input = jnp.concatenate([u, attn, gated_attn], axis=-1)
    projected_output = proj_input @ self.output_weight
    dropped_out = self.dropout_layer(
        projected_output, deterministic=deterministic
    )
    return x + dropped_out

  def hstu_preprocess_and_attention(
      self,
      x: jnp.ndarray,
      num_targets: Optional[jnp.ndarray],
      deterministic: bool,
      decode: bool = False,
  ):
    """Replicated STU preprocess and attention."""
    normed_x = self.input_norm_layer(x)
    uvqk = normed_x @ self.uvqk_weight + self.uvqk_beta
    u_proj, v_proj, q_proj, k_proj = jnp.split(
        uvqk,
        [
            self.hidden_dim * self.num_heads,
            self.hidden_dim * self.num_heads * 2,
            self.hidden_dim * self.num_heads * 2
            + self.attention_dim * self.num_heads,
        ],
        axis=-1,
    )

    u = nn.silu(u_proj)
    batch_size, seq_len, _ = x.shape
    q = q_proj.reshape(
        batch_size, seq_len, self.num_heads, self.attention_dim
    ).transpose(0, 2, 1, 3)
    k = k_proj.reshape(
        batch_size, seq_len, self.num_heads, self.attention_dim
    ).transpose(0, 2, 1, 3)
    v = v_proj.reshape(
        batch_size, seq_len, self.num_heads, self.hidden_dim
    ).transpose(0, 2, 1, 3)

    cache_index = 0
    if decode:
      is_initialized = (
          self.has_variable('cache', 'cached_key')
          and self.cached_key.value is not None
      )
      cache_index = self.cache_index.value
      if not is_initialized and self.is_mutable_collection('cache'):
        k_cache_shape = (
            batch_size,
            self.num_heads,
            self.config.max_decode_length,
            self.attention_dim,
        )
        v_cache_shape = (
            batch_size,
            self.num_heads,
            self.config.max_decode_length,
            self.hidden_dim,
        )
        self.cached_key.value = jnp.zeros(k_cache_shape, k.dtype)
        self.cached_value.value = jnp.zeros(v_cache_shape, v.dtype)

      if self.is_mutable_collection('cache'):
        k_cache = jax.lax.dynamic_update_slice(
            self.cached_key.value,
            k.astype(self.cached_key.value.dtype),
            (0, 0, cache_index, 0),
        )
        v_cache = jax.lax.dynamic_update_slice(
            self.cached_value.value,
            v.astype(self.cached_value.value.dtype),
            (0, 0, cache_index, 0),
        )
        self.cached_key.value = k_cache
        self.cached_value.value = v_cache
        self.cache_index.value = cache_index + seq_len
        k = k_cache
        v = v_cache
      elif is_initialized:
        k = self.cached_key.value
        v = self.cached_value.value
      else:
        raise ValueError('Cache not initialized and not mutable.')

    attn_scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) * self.attn_alpha
    if decode:
      attn_weights = nn.silu(attn_scores) / self.config.max_decode_length
      mask = (jnp.arange(self.config.max_decode_length) <= cache_index)[
          None, None, None, :
      ]
      attn_weights = attn_weights * mask
    else:
      attn_weights = nn.silu(attn_scores) / seq_len
      mask = self._get_valid_attn_mask(x, num_targets)
      attn_weights = attn_weights * mask[:, None, :, :]

    attn_weights = self.dropout_layer(attn_weights, deterministic=deterministic)
    attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
        batch_size, seq_len, -1
    )

    return u, attn_output, k_proj, v_proj

  def __call__(
      self,
      x: jnp.ndarray,
      num_targets: Optional[jnp.ndarray] = None,
      deterministic: bool = True,
      decode: bool = False,
  ):
    """Computes the STU layer."""
    actual_num_targets = num_targets if self.target_aware else None
    u, attn_output, _, _ = self.hstu_preprocess_and_attention(
        x,
        actual_num_targets,
        deterministic=deterministic,
        decode=decode,
    )
    final_output = self.hstu_compute_output(
        attn=attn_output, u=u, x=x, deterministic=deterministic
    )
    return final_output


class STUStack(nn.Module):
  """STU stack.

  This module creates a stack of STU layers.

  Attributes:
    configs: A sequence of STU layer configs.
  """

  configs: Sequence[STULayerConfig]

  def setup(self):
    self.stu_layers = [
        STULayer(config=c, name=f'stu_layer_{i}')
        for i, c in enumerate(self.configs)
    ]

  def __call__(
      self,
      x: jnp.ndarray,
      num_targets: Optional[jnp.ndarray] = None,
      deterministic: bool = True,
      decode: bool = False,
  ):
    for i in range(len(self.stu_layers)):
      x = self.stu_layers[i](x, num_targets, deterministic, decode)
    return x

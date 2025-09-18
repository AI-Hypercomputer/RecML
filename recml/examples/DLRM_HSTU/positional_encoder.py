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
"""JAX implementation of positional and timestamp encoding for sequences."""

from math import sqrt
from typing import Optional

from flax import linen as nn
import jax
import jax.numpy as jnp


def _get_col_indices(
    max_seq_len: int,
    max_contextual_seq_len: int,
    max_pos_ind: int,
    seq_lengths: jnp.ndarray,
    num_targets: Optional[jnp.ndarray],
    interleave_targets: bool,
) -> jnp.ndarray:
  """Calculates the positional indices for each element in the sequence.

  JAX translation of `_get_col_indices` from pt_position.py.

  Args:
      max_seq_len: The maximum sequence length.
      max_contextual_seq_len: The maximum length of the contextual prefix.
      max_pos_ind: The maximum positional index.
      seq_lengths: A 1D tensor of shape (batch_size,) with the true length of
        each sequence.
      num_targets: An optional 1D tensor of shape (batch_size,) indicating the
        number of target items at the end of each sequence.
      interleave_targets: A boolean indicating whether to interleave targets.

  Returns:
      A 2D tensor of shape (batch_size, max_seq_len) containing the positional
      indices for each element in the sequence.
  """
  batch_size = seq_lengths.shape[0]
  col_indices = jnp.tile(
      jnp.arange(max_seq_len, dtype=jnp.int32), (batch_size, 1)
  )

  if num_targets is not None:
    if interleave_targets:
      high_inds = seq_lengths - num_targets * 2
    else:
      high_inds = seq_lengths - num_targets

    col_indices = jnp.minimum(col_indices, high_inds[:, jnp.newaxis])
    col_indices = high_inds[:, jnp.newaxis] - col_indices
  else:
    col_indices = seq_lengths[:, jnp.newaxis] - col_indices

  col_indices = col_indices + max_contextual_seq_len
  col_indices = jnp.clip(col_indices, a_min=0, a_max=max_pos_ind - 1)

  if max_contextual_seq_len > 0:
    contextual_indices = jnp.arange(max_contextual_seq_len, dtype=jnp.int32)[
        jnp.newaxis, :
    ]
    col_indices = col_indices.at[:, :max_contextual_seq_len].set(
        contextual_indices
    )

  return col_indices


def add_timestamp_positional_embeddings(
    seq_embeddings: jnp.ndarray,
    pos_embeddings: jnp.ndarray,
    ts_embeddings: jnp.ndarray,
    timestamps: jnp.ndarray,
    max_seq_len: int,
    max_contextual_seq_len: int,
    seq_lengths: jnp.ndarray,
    num_targets: Optional[jnp.ndarray],
    interleave_targets: bool,
    time_bucket_fn: str,
) -> jnp.ndarray:
  """Adds timestamp and positional embeddings to sequence embeddings.

  JAX translation of `pytorch_add_timestamp_positional_embeddings`. Assumes
  inputs are padded dense tensors.

  Args:
      seq_embeddings: A 3D padded tensor of shape (batch_size, max_seq_len,
        embedding_dim) containing the input item embeddings.
      pos_embeddings: The learned positional embedding weights.
      ts_embeddings: The learned timestamp embedding weights.
      timestamps: A 2D padded tensor of shape (batch_size, max_seq_len)
        containing timestamps for each item.
      max_seq_len: The maximum sequence length for padding.
      max_contextual_seq_len: The maximum length of the contextual prefix.
      seq_lengths: A 1D tensor of shape (batch_size,) with the true length of
        each sequence.
      num_targets: An optional 1D tensor of shape (batch_size,) indicating the
        number of target items at the end of each sequence.
      interleave_targets: A boolean indicating whether to interleave targets.
      time_bucket_fn: The function to use for time bucketing ("log" or "sqrt").

  Returns:
      A 3D tensor of the same shape as `seq_embeddings` with positional
      and time embeddings added.
  """
  # Position encoding
  max_pos_ind = pos_embeddings.shape[0]
  pos_inds = _get_col_indices(
      max_seq_len=max_seq_len,
      max_contextual_seq_len=max_contextual_seq_len,
      max_pos_ind=max_pos_ind,
      seq_lengths=seq_lengths,
      num_targets=num_targets,
      interleave_targets=interleave_targets,
  )
  position_embeddings = pos_embeddings[pos_inds]

  # Timestamp encoding
  batch_size = seq_lengths.shape[0]
  num_time_buckets = ts_embeddings.shape[0] - 1
  time_bucket_increments = 60.0
  time_bucket_divisor = 1.0
  time_delta = 0

  # Get the last valid timestamp from each padded sequence for query_time
  query_indices = jnp.maximum(0, seq_lengths - 1)
  query_time = timestamps[jnp.arange(batch_size), query_indices][:, jnp.newaxis]

  ts = query_time - timestamps
  ts = ts + time_delta
  ts = jnp.maximum(ts, 1e-6) / time_bucket_increments

  if time_bucket_fn == "log":
    ts = jnp.log(ts)
  elif time_bucket_fn == "sqrt":
    ts = jnp.sqrt(ts)
  else:
    raise ValueError(f"Unsupported time_bucket_fn: {time_bucket_fn}")

  ts = (ts / time_bucket_divisor).clip(min=0).astype(jnp.int32)
  ts = jnp.clip(ts, a_min=0, a_max=num_time_buckets)

  time_embeddings = ts_embeddings[ts]

  # Combine embeddings
  added_embeddings = (position_embeddings + time_embeddings).astype(
      seq_embeddings.dtype
  )

  # The original op implies addition to only the valid (non-padded) parts.
  # In a dense representation, this is equivalent to masking the added
  # embeddings.
  mask = (
      jnp.arange(max_seq_len, dtype=jnp.int32)[jnp.newaxis, :]
      < seq_lengths[:, jnp.newaxis]
  )
  masked_added_embeddings = added_embeddings * mask[..., jnp.newaxis]

  return seq_embeddings + masked_added_embeddings


class HSTUPositionalEncoder(nn.Module):
  """JAX implementation of HSTUPositionalEncoder.

  This module computes and adds positional and timestamp-based embeddings
  to a sequence of input embeddings.

  Attributes:
      num_position_buckets: The total number of position buckets.
      num_time_buckets: The total number of time buckets.
      embedding_dim: The dimensionality of the embeddings.
      contextual_seq_len: The length of the contextual prefix in sequences.
  """

  num_position_buckets: int
  num_time_buckets: int
  embedding_dim: int
  contextual_seq_len: int

  @nn.compact
  def __call__(
      self,
      max_seq_len: int,
      seq_lengths: jnp.ndarray,
      seq_timestamps: jnp.ndarray,
      seq_embeddings: jnp.ndarray,
      num_targets: Optional[jnp.ndarray],
  ) -> jnp.ndarray:
    """Adds positional and timestamp embeddings to the input sequence embeddings.

    Args:
        max_seq_len: The maximum sequence length for padding.
        seq_lengths: A 1D tensor of shape (batch_size,) with the true length of
          each sequence.
        seq_timestamps: A 2D padded tensor of shape (batch_size, max_seq_len)
          containing timestamps for each item.
        seq_embeddings: A 3D padded tensor of shape (batch_size, max_seq_len,
          embedding_dim) containing the input item embeddings.
        num_targets: An optional 1D tensor of shape (batch_size,) indicating the
          number of target items at the end of each sequence.

    Returns:
        A 3D tensor of the same shape as `seq_embeddings` with positional
        and time embeddings added.
    """
    position_embeddings_weight = self.param(
        "_position_embeddings_weight",
        nn.initializers.uniform(scale=sqrt(1.0 / self.num_position_buckets)),
        (self.num_position_buckets, self.embedding_dim),
    )
    timestamp_embeddings_weight = self.param(
        "_timestamp_embeddings_weight",
        nn.initializers.uniform(scale=sqrt(1.0 / self.num_time_buckets)),
        (self.num_time_buckets + 1, self.embedding_dim),
    )

    scaled_seq_embeddings = seq_embeddings * sqrt(self.embedding_dim)

    final_embeddings = add_timestamp_positional_embeddings(
        seq_embeddings=scaled_seq_embeddings,
        pos_embeddings=position_embeddings_weight,
        ts_embeddings=timestamp_embeddings_weight,
        timestamps=seq_timestamps,
        max_seq_len=max_seq_len,
        max_contextual_seq_len=self.contextual_seq_len,
        seq_lengths=seq_lengths,
        num_targets=num_targets,
        interleave_targets=False,
        time_bucket_fn="sqrt",
    )
    return final_embeddings

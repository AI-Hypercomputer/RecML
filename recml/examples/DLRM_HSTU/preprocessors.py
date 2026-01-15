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
#!/usr/bin/env python3

"""Input preprocessors for HSTU models."""

from typing import Any, Dict, Tuple

import flax.linen as nn
from flax.linen.initializers import xavier_uniform
from flax.linen.initializers import zeros
import jax.numpy as jnp
from recml.examples.DLRM_HSTU.action_encoder import ActionEncoder


Array = jnp.ndarray
Dtype = Any


class SwishLayerNorm(nn.Module):
  """JAX/Flax implementation of SwishLayerNorm.

  Corresponds to generative_recommenders/ops/layer_norm.py -> SwishLayerNorm
  The PyTorch implementation is: x * sigmoid(layer_norm(x))
  """

  epsilon: float = 1e-5
  dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, x: Array) -> Array:
    """Applies swish layer normalization to the input."""
    ln = nn.LayerNorm(
        epsilon=self.epsilon,
        use_bias=True,
        use_scale=True,
        dtype=self.dtype,
    )
    normed_x = ln(x)
    return x * nn.sigmoid(normed_x)


class InputPreprocessor(nn.Module):
  """An abstract class for pre-processing sequence embeddings before HSTU layers."""

  def __call__(
      self,
      max_uih_len: int,
      seq_embeddings: Array,
      seq_mask: Array,
      seq_timestamps: Array,
      num_targets: Array,
      seq_payloads: Dict[str, Array],
      *,
      deterministic: bool,
  ) -> Tuple[Array, Array, Array, Array, Dict[str, Array]]:
    """Processes input sequences and their features.

    Args:
        max_uih_len: Maximum length of the user item history.
        seq_embeddings: (B, N, D) Padded sequence embeddings.
        seq_mask: (B, N) Boolean mask for seq_embeddings.
        seq_timestamps: (B, N) Padded timestamps.
        num_targets: (B,) Number of targets for each sequence.
        seq_payloads: Dict of other features, also as padded tensors with
          masks.
        deterministic: Controls dropout behavior.

    Returns:
        A tuple containing the processed (
            output_embeddings,
            output_mask,
            output_timestamps,
            output_num_targets,
            output_payloads
        ).
    """
    raise NotImplementedError

  def interleave_targets(self) -> bool:
    return False


def get_contextual_input_embeddings(
    seq_mask: Array,
    seq_payloads: Dict[str, Array],
    contextual_feature_to_max_length: Dict[str, int],
    contextual_feature_to_min_uih_length: Dict[str, int],
    dtype: Dtype,
) -> Array:
  """Constructs the input for contextual embeddings from dense tensors.

  Args:
      seq_mask: Boolean mask for the sequence.
      seq_payloads: Dictionary of all feature tensors.
      contextual_feature_to_max_length: Maps feature names to their max length.
      contextual_feature_to_min_uih_length: Maps features to a min uih length
        for them to be active.
      dtype: Data type for the output.

  Returns:
      A dense tensor of shape (batch_size, sum_of_dims).
  """
  padded_values = []
  seq_lengths = jnp.sum(seq_mask, axis=1, dtype=jnp.int32)

  for key, max_len in contextual_feature_to_max_length.items():
    # Assuming the payload is already a dense tensor of shape (B, L, D)
    v = seq_payloads[key].astype(dtype)

    min_uih_length = contextual_feature_to_min_uih_length.get(key, 0)
    if min_uih_length > 0:
      # Create a mask to zero out embeddings for sequences that are too short
      mask = (seq_lengths >= min_uih_length).reshape(-1, 1, 1)
      v *= mask

    # Flatten the feature dimension
    padded_values.append(v.reshape(v.shape[0], -1))

  return jnp.concatenate(padded_values, axis=1)

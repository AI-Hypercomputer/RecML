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
"""JAX/Flax implementation of ContentEncoder for dense tensors."""

from typing import Dict, List, Optional

import flax.linen as nn
from jax import numpy as jnp


class ContentEncoder(nn.Module):
  """JAX/Flax implementation of ContentEncoder for dense tensors.

  This module concatenates input embeddings with additional features. It
  handles two types of features:
  1. `additional_content_features`: Features available for the entire
     sequence.
  2. `target_enrich_features`: Features available only for target items, with
     a learned dummy embedding used as a placeholder for history items.
  """
  input_embedding_dim: int
  additional_content_features: Optional[Dict[str, int]] = None
  target_enrich_features: Optional[Dict[str, int]] = None

  def setup(self) -> None:
    self._additional_content_features_internal: Dict[str, int] = (
        self.additional_content_features
        if self.additional_content_features is not None
        else {}
    )
    self._target_enrich_features_internal: Dict[str, int] = (
        self.target_enrich_features
        if self.target_enrich_features is not None
        else {}
    )

    self._target_enrich_dummy_embeddings = {
        name: self.param(
            f"target_enrich_dummy_param_{name}",
            nn.initializers.normal(stddev=0.1),
            (1, dim),  # Shape is (1, feature_dim) for broadcasting
        )
        for name, dim in self._target_enrich_features_internal.items()
    }

  @property
  def output_embedding_dim(self) -> int:
    """The total dimension of the output embeddings after concatenation."""
    additional_dim = sum(
        self.additional_content_features.values()
        if self.additional_content_features
        else []
    )
    enrich_dim = sum(
        self.target_enrich_features.values()
        if self.target_enrich_features
        else []
    )
    return self.input_embedding_dim + additional_dim + enrich_dim

  @nn.compact
  def __call__(
      self,
      max_uih_len: int,
      seq_embeddings: jnp.ndarray,
      seq_payloads: Dict[str, jnp.ndarray],
  ) -> jnp.ndarray:
    """Forward pass for the ContentEncoder.

    Args:
      max_uih_len: The length of the user interaction history (non-target part)
        in the padded sequence.
      seq_embeddings: The base embeddings for the sequence with shape
        (batch_size, seq_len, input_embedding_dim).
      seq_payloads: A dictionary mapping feature names to their tensors. - For
        `additional_content_features`, shape is (batch_size, seq_len,
        feature_dim). - For `target_enrich_features`, shape is (batch_size,
        max_targets, feature_dim).

    Returns:
      The concatenated content embeddings.
      Shape: (batch_size, seq_len, output_embedding_dim).
    """
    content_embeddings_list: List[jnp.ndarray] = [seq_embeddings]

    if self._additional_content_features_internal:
      for x in self._additional_content_features_internal.keys():
        content_embeddings_list.append(
            seq_payloads[x].astype(seq_embeddings.dtype)
        )

    if self._target_enrich_dummy_embeddings:
      batch_size = seq_embeddings.shape[0]

      for name, param in self._target_enrich_dummy_embeddings.items():
        # If a feature is used for both additional content and target
        # enrichment, the payload will contain the full sequence. We need to
        # slice the target part.
        if name in self._additional_content_features_internal:
            full_sequence_feature = seq_payloads[name]
            enrich_embeddings_target = full_sequence_feature[
                :, max_uih_len:, :
            ].astype(seq_embeddings.dtype)
        else:
            # Otherwise, the payload contains only the target features.
            enrich_embeddings_target = seq_payloads[name].astype(
                seq_embeddings.dtype
            )
        enrich_embeddings_uih = jnp.broadcast_to(
            param, (batch_size, max_uih_len, param.shape[-1])
        ).astype(seq_embeddings.dtype)

        # Pad targets if necessary to match sequence length
        num_targets = enrich_embeddings_target.shape[1]
        num_history = max_uih_len
        if num_history + num_targets < seq_embeddings.shape[1]:
            padding_needed = seq_embeddings.shape[1] - (
                num_history + num_targets
            )
            padding = jnp.zeros(
                (
                    batch_size,
                    padding_needed,
                    enrich_embeddings_target.shape[-1],
                ),
                dtype=enrich_embeddings_target.dtype,
            )
            enrich_embeddings_target = jnp.concatenate(
                [enrich_embeddings_target, padding], axis=1
            )

        enrich_embeddings = jnp.concatenate(
            [enrich_embeddings_uih, enrich_embeddings_target], axis=1
        )
        if enrich_embeddings.shape[1] < seq_embeddings.shape[1]:
          padding = jnp.zeros(
              (
                  batch_size,
                  seq_embeddings.shape[1] - enrich_embeddings.shape[1],
                  enrich_embeddings.shape[2],
              ),
              dtype=enrich_embeddings.dtype,
          )
          enrich_embeddings = jnp.concatenate(
              [enrich_embeddings, padding], axis=1
          )
        content_embeddings_list.append(enrich_embeddings)

    if (
        not self._additional_content_features_internal
        and not self._target_enrich_features_internal
    ):
      return seq_embeddings
    else:
      content_embeddings = jnp.concatenate(
          content_embeddings_list,
          axis=-1,
      )
      return content_embeddings

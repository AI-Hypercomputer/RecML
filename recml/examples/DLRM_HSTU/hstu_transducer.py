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
"""JAX/Flax implementation of HSTUTransducer for dense tensors."""

import logging
from typing import Dict, Optional, Tuple, Type

import flax.linen as nn
import jax.numpy as jnp
from recml.examples.DLRM_HSTU.positional_encoder import HSTUPositionalEncoder
from recml.examples.DLRM_HSTU.postprocessors import L2NormPostprocessor
from recml.examples.DLRM_HSTU.postprocessors import OutputPostprocessor
from recml.examples.DLRM_HSTU.preprocessors import InputPreprocessor
from recml.examples.DLRM_HSTU.stu import STUStack


logger = logging.getLogger(__name__)


class HSTUTransducer(nn.Module):
  """JAX/Flax implementation of the HSTU Transducer module, using dense tensors.

  This implementation mirrors structure but replaces jagged tensor operations
  with dense tensor operations using masking.
  """

  stu_module: STUStack
  input_preprocessor: InputPreprocessor
  output_postprocessor_cls: Type[OutputPostprocessor] = L2NormPostprocessor
  input_dropout_ratio: float = 0.0
  positional_encoder: Optional[HSTUPositionalEncoder] = None
  return_full_embeddings: bool = False
  listwise: bool = False

  def setup(self):
    self._output_postprocessor: OutputPostprocessor = (
        self.output_postprocessor_cls()
    )
    self._input_dropout = nn.Dropout(rate=self.input_dropout_ratio)

  def _preprocess(
      self,
      max_uih_len: int,
      seq_embeddings: jnp.ndarray,
      seq_mask: jnp.ndarray,
      seq_timestamps: jnp.ndarray,
      num_targets: jnp.ndarray,
      seq_payloads: Dict[str, jnp.ndarray],
      is_training: bool,
  ) -> Tuple[
      jnp.ndarray,
      jnp.ndarray,
      jnp.ndarray,
      jnp.ndarray,
      Dict[str, jnp.ndarray],
  ]:
    """Preprocesses the input sequence embeddings."""
    (
        output_seq_embeddings,
        output_seq_mask,
        output_seq_timestamps,
        output_num_targets,
        output_seq_payloads,
    ) = self.input_preprocessor(
        max_uih_len=max_uih_len,
        seq_embeddings=seq_embeddings,
        seq_mask=seq_mask,
        seq_timestamps=seq_timestamps,
        num_targets=num_targets,
        seq_payloads=seq_payloads,
        deterministic=not is_training,
    )

    output_seq_lengths = jnp.sum(output_seq_mask, axis=1, dtype=jnp.int32)

    if self.positional_encoder is not None:
      output_seq_embeddings = self.positional_encoder(
          max_seq_len=output_seq_embeddings.shape[1],
          seq_lengths=output_seq_lengths,
          seq_timestamps=output_seq_timestamps,
          seq_embeddings=output_seq_embeddings,
          num_targets=(
              None if self.listwise and is_training else output_num_targets
          ),
      )

    output_seq_embeddings = self._input_dropout(
        output_seq_embeddings, deterministic=not is_training
    )

    return (
        output_seq_embeddings,
        output_seq_mask,
        output_seq_timestamps,
        output_num_targets,
        output_seq_payloads,
    )

  def _hstu_compute(
      self,
      seq_embeddings: jnp.ndarray,
      num_targets: jnp.ndarray,
      is_training: bool,
      decode: bool = False,
  ) -> jnp.ndarray:
    """Computes the HSTU embeddings."""
    seq_embeddings = self.stu_module(
        x=seq_embeddings,
        num_targets=None if self.listwise and is_training else num_targets,
        deterministic=not is_training,
        decode=decode,
    )
    return seq_embeddings

  def _postprocess(
      self,
      seq_embeddings: jnp.ndarray,
      seq_mask: jnp.ndarray,
      seq_timestamps: jnp.ndarray,
      num_targets: jnp.ndarray,
      seq_payloads: Dict[str, jnp.ndarray],
  ) -> Tuple[Optional[jnp.ndarray], jnp.ndarray]:
    """Postprocesses the output sequence embeddings."""
    if self.return_full_embeddings:
      seq_embeddings = self._output_postprocessor(
          seq_embeddings=seq_embeddings,
          seq_timestamps=seq_timestamps,
          seq_payloads=seq_payloads,
      )

    batch_size, max_seq_len, embedding_dim = seq_embeddings.shape
    seq_lengths = jnp.sum(seq_mask, axis=1, dtype=jnp.int32)
    indices = jnp.arange(max_seq_len)
    start_target_idx = seq_lengths - num_targets
    candidate_mask = (indices >= start_target_idx[:, jnp.newaxis]) & (
        indices < seq_lengths[:, jnp.newaxis]
    )

    candidate_embeddings_masked = (
        seq_embeddings * candidate_mask[..., jnp.newaxis]
    )
    candidate_timestamps_masked = seq_timestamps * candidate_mask

    if self.input_preprocessor.interleave_targets():
      raise NotImplementedError(
          "Interleaved targets not supported in dense post-processing yet."
      )

    if not self.return_full_embeddings:
      candidate_embeddings = self._output_postprocessor(
          seq_embeddings=candidate_embeddings_masked,
          seq_timestamps=candidate_timestamps_masked,
          seq_payloads=seq_payloads,
      )
      candidate_embeddings = (
          candidate_embeddings * candidate_mask[..., jnp.newaxis]
      )
    else:
      candidate_embeddings = candidate_embeddings_masked

    return (
        seq_embeddings if self.return_full_embeddings else None,
        candidate_embeddings,
    )

  def __call__(
      self,
      max_uih_len: int,
      max_targets: int,
      total_uih_len: int,
      total_targets: int,
      seq_lengths: jnp.ndarray,
      seq_embeddings: jnp.ndarray,
      seq_timestamps: jnp.ndarray,
      num_targets: jnp.ndarray,
      seq_payloads: Dict[str, jnp.ndarray],
      *,
      deterministic: Optional[bool] = None,
      decode: bool = False,
  ) -> Tuple[
      jnp.ndarray,
      Optional[jnp.ndarray],
  ]:
    """Forward pass for HSTUTransducer."""
    if decode and not deterministic:
      raise ValueError("If decode=True, deterministic must be True.")
    is_training = (
        not deterministic if deterministic is None else not deterministic
    )

    batch_size, max_len, _ = seq_embeddings.shape
    seq_mask = (
        jnp.arange(max_len, dtype=jnp.int32)[None, :] < seq_lengths[:, None]
    )

    (
        processed_seq_embeddings,
        processed_seq_mask,
        processed_seq_timestamps,
        processed_num_targets,
        processed_seq_payloads,
    ) = self._preprocess(
        max_uih_len=max_uih_len,
        seq_embeddings=seq_embeddings,
        seq_mask=seq_mask,
        seq_timestamps=seq_timestamps,
        num_targets=num_targets,
        seq_payloads=seq_payloads,
        is_training=is_training,
    )

    encoded_embeddings = self._hstu_compute(
        seq_embeddings=processed_seq_embeddings,
        num_targets=processed_num_targets,
        is_training=is_training,
        decode=decode,
    )

    encoded_embeddings = (
        encoded_embeddings * processed_seq_mask[..., jnp.newaxis]
    )

    full_embeddings, candidate_embeddings = self._postprocess(
        seq_embeddings=encoded_embeddings,
        seq_mask=processed_seq_mask,
        seq_timestamps=processed_seq_timestamps,
        num_targets=processed_num_targets,
        seq_payloads=processed_seq_payloads,
    )

    return candidate_embeddings, full_embeddings

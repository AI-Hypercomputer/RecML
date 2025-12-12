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
"""JAX/Flax implementation of DLRM-HSTU."""

from dataclasses import dataclass
from dataclasses import field
from functools import partial
import logging
from typing import Any, Dict, List, Optional, Tuple

import flax.linen as nn
from flax.linen.initializers import xavier_uniform
from flax.linen.initializers import zeros
import jax.numpy as jnp
from recml.examples.DLRM_HSTU.action_encoder import ActionEncoder
from recml.examples.DLRM_HSTU.content_encoder import ContentEncoder
from recml.examples.DLRM_HSTU.contextual_interleave_preprocessor import ContextualInterleavePreprocessor
from recml.examples.DLRM_HSTU.contextualize_mlps import ContextualizedMLP
from recml.examples.DLRM_HSTU.contextualize_mlps import ParameterizedContextualizedMLP
from recml.examples.DLRM_HSTU.contextualize_mlps import SimpleContextualizedMLP
from recml.examples.DLRM_HSTU.hstu_transducer import HSTUTransducer
from recml.examples.DLRM_HSTU.multitask_module import DefaultMultitaskModule
from recml.examples.DLRM_HSTU.multitask_module import MultitaskTaskType
from recml.examples.DLRM_HSTU.multitask_module import TaskConfig
from recml.examples.DLRM_HSTU.positional_encoder import HSTUPositionalEncoder
from recml.examples.DLRM_HSTU.postprocessors import L2NormPostprocessor
from recml.examples.DLRM_HSTU.postprocessors import LayerNormPostprocessor
from recml.examples.DLRM_HSTU.postprocessors import TimestampLayerNormPostprocessor
from recml.examples.DLRM_HSTU.preprocessors import SwishLayerNorm
from recml.examples.DLRM_HSTU.stu import STULayer
from recml.examples.DLRM_HSTU.stu import STULayerConfig
from recml.examples.DLRM_HSTU.stu import STUStack


logger = logging.getLogger(__name__)

Dtype = Any
Array = jnp.ndarray


@dataclass
class EmbeddingConfig:
  """Simplified embedding config for JAX."""

  name: str
  num_embeddings: int
  embedding_dim: int


@dataclass
class DlrmHSTUConfig:
  """Configuration for DLRM-HSTU model."""

  max_seq_len: int = 2056
  max_num_candidates: int = 10
  max_num_candidates_inference: int = 5
  hstu_num_heads: int = 1
  hstu_attn_linear_dim: int = 256
  hstu_attn_qk_dim: int = 128
  hstu_attn_num_layers: int = 12
  hstu_embedding_table_dim: int = 192
  hstu_preprocessor_hidden_dim: int = 256
  hstu_transducer_embedding_dim: int = 256  # changed from 0
  hstu_group_norm: bool = False
  hstu_input_dropout_ratio: float = 0.2
  hstu_linear_dropout_rate: float = 0.2
  hstu_max_attn_len: int = 0
  contextual_feature_to_max_length: Dict[str, int] = field(default_factory=dict)
  contextual_feature_to_min_uih_length: Dict[str, int] = field(
      default_factory=dict
  )
  additional_content_features: Optional[Dict[str, int]] = None
  target_enrich_features: Optional[Dict[str, int]] = None
  pmlp_contextual_dropout_ratio: float = 0.0
  candidates_weight_feature_name: str = ""
  candidates_watchtime_feature_name: str = ""
  candidates_querytime_feature_name: str = ""
  watchtime_feature_name: str = ""
  causal_multitask_weights: float = 0.2
  multitask_configs: List[TaskConfig] = field(default_factory=list)
  user_embedding_feature_names: List[str] = field(default_factory=list)
  item_embedding_feature_names: List[str] = field(default_factory=list)
  uih_post_id_feature_name: str = ""
  uih_action_time_feature_name: str = ""
  uih_weight_feature_name: str = ""
  hstu_uih_feature_names: List[str] = field(default_factory=list)
  hstu_candidate_feature_names: List[str] = field(default_factory=list)
  merge_uih_candidate_feature_mapping: List[Tuple[str, str]] = field(
      default_factory=list
  )
  action_weights: Optional[List[int]] = None
  watchtime_to_action_thresholds_and_weights: Optional[
      List[Tuple[int, int]]
  ] = None
  enable_postprocessor: bool = True
  use_layer_norm_postprocessor: bool = False


def _get_supervision_labels_and_weights(
    supervision_bitmasks: jnp.ndarray,
    watchtime_sequence: jnp.ndarray,
    task_configs: List[TaskConfig],
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
  """Computes supervision labels and weights for multitask learning."""
  supervision_labels: Dict[str, jnp.ndarray] = {}
  supervision_weights: Dict[str, jnp.ndarray] = {}
  for task in task_configs:
    if task.task_type == MultitaskTaskType.REGRESSION:
      supervision_labels[task.task_name] = watchtime_sequence.astype(
          jnp.float32
      )
    elif task.task_type == MultitaskTaskType.BINARY_CLASSIFICATION:
      supervision_labels[task.task_name] = (
          jnp.bitwise_and(supervision_bitmasks, task.task_weight) > 0
      ).astype(jnp.float32)
    else:
      raise RuntimeError("Unsupported MultitaskTaskType")
  return supervision_labels, supervision_weights


class EmbeddingCollection(nn.Module):
  """A module to hold and query multiple embedding tables."""

  embedding_configs: Dict[str, EmbeddingConfig]

  def setup(self):
    self.embeddings = {
        name: nn.Embed(
            num_embeddings=cfg.num_embeddings,
            features=cfg.embedding_dim,
            name=name,
        )
        for name, cfg in self.embedding_configs.items()
    }

  def __call__(
      self, features: Dict[str, jnp.ndarray]
  ) -> Dict[str, jnp.ndarray]:
    """Looks up embeddings for features given as dense ID tensors."""
    return {
        name: self.embeddings[name](ids)
        for name, ids in features.items()
        if name in self.embeddings
    }


class PredictionMLP(nn.Module):
  """MLP for multitask prediction head."""

  hidden_dim: int
  num_tasks: int
  dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, x: Array) -> Array:
    x = nn.Dense(
        features=self.hidden_dim,
        dtype=self.dtype,
        kernel_init=xavier_uniform(),
        bias_init=zeros,
    )(x)
    x = SwishLayerNorm(dtype=self.dtype)(x)
    x = nn.Dense(
        features=self.num_tasks,
        dtype=self.dtype,
        kernel_init=xavier_uniform(),
        bias_init=zeros,
    )(x)
    return x


class ItemMLP(nn.Module):
  """MLP for processing item embeddings."""

  hidden_dim: int
  output_dim: int
  dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, x: Array) -> Array:
    x = nn.Dense(
        features=self.hidden_dim,
        dtype=self.dtype,
        kernel_init=xavier_uniform(),
        bias_init=zeros,
    )(x)
    x = SwishLayerNorm(dtype=self.dtype)(x)
    x = nn.Dense(
        features=self.output_dim,
        dtype=self.dtype,
        kernel_init=xavier_uniform(),
        bias_init=zeros,
    )(x)
    x = nn.LayerNorm(dtype=self.dtype)(x)
    return x


class DlrmHSTU(nn.Module):
  """JAX/Flax implementation of DLRM with HSTU user encoder.

  Operates on dense tensors.
  """

  hstu_configs: DlrmHSTUConfig
  embedding_tables: Dict[str, EmbeddingConfig]
  dtype: Dtype = jnp.float32

  def setup(self):
    self._embedding_collection = EmbeddingCollection(self.embedding_tables)
    self._multitask_configs: List[TaskConfig] = (
        self.hstu_configs.multitask_configs
    )

    self._multitask_module = DefaultMultitaskModule(
        task_configs=self._multitask_configs,
        embedding_dim=self.hstu_configs.hstu_transducer_embedding_dim,
        prediction_fn=lambda in_dim, num_tasks: PredictionMLP(
            hidden_dim=512, num_tasks=num_tasks, dtype=self.dtype
        ),
        causal_multitask_weights=self.hstu_configs.causal_multitask_weights,
    )

    hstu_config = self.hstu_configs

    content_encoder = ContentEncoder(
        input_embedding_dim=hstu_config.hstu_embedding_table_dim,
        additional_content_features=hstu_config.additional_content_features,
        target_enrich_features=hstu_config.target_enrich_features,
    )

    action_encoder = ActionEncoder(
        action_embedding_dim=hstu_config.hstu_transducer_embedding_dim,
        action_feature_name=hstu_config.uih_weight_feature_name,
        action_weights=hstu_config.action_weights,
        watchtime_feature_name=hstu_config.watchtime_feature_name,
        watchtime_to_action_thresholds_and_weights=hstu_config.watchtime_to_action_thresholds_and_weights,
    )

    contextual_embedding_dim = sum(
        hstu_config.contextual_feature_to_max_length.values()
    ) * hstu_config.hstu_embedding_table_dim

    def mlp_fn(
        sequential_input_dim: int,
    ) -> ContextualizedMLP:
      if contextual_embedding_dim > 0:
        return ParameterizedContextualizedMLP(
            contextual_embedding_dim=contextual_embedding_dim,
            sequential_input_dim=sequential_input_dim,
            sequential_output_dim=hstu_config.hstu_transducer_embedding_dim,
            hidden_dim=hstu_config.hstu_preprocessor_hidden_dim,
        )
      else:
        return SimpleContextualizedMLP(
            sequential_input_dim=sequential_input_dim,
            sequential_output_dim=hstu_config.hstu_transducer_embedding_dim,
            hidden_dim=hstu_config.hstu_preprocessor_hidden_dim,
        )

    preprocessor = ContextualInterleavePreprocessor(
        input_embedding_dim=hstu_config.hstu_embedding_table_dim,
        output_embedding_dim=hstu_config.hstu_transducer_embedding_dim,
        contextual_feature_to_max_length=hstu_config.contextual_feature_to_max_length,
        contextual_feature_to_min_uih_length=hstu_config.contextual_feature_to_min_uih_length,
        content_encoder=content_encoder,
        content_contextualize_mlp_fn=partial(
            mlp_fn, sequential_input_dim=content_encoder.output_embedding_dim
        ),
        action_encoder=action_encoder,
        action_contextualize_mlp_fn=partial(
            mlp_fn, sequential_input_dim=action_encoder.output_embedding_dim
        ),
        pmlp_contextual_dropout_ratio=hstu_config.pmlp_contextual_dropout_ratio,
    )

    contextual_seq_len = sum(
        hstu_config.contextual_feature_to_max_length.values()
    )
    positional_encoder = HSTUPositionalEncoder(
        num_position_buckets=8192,
        num_time_buckets=2048,
        embedding_dim=hstu_config.hstu_transducer_embedding_dim,
        contextual_seq_len=contextual_seq_len,
    )

    if hstu_config.enable_postprocessor:
      if hstu_config.use_layer_norm_postprocessor:
        postproc_cls = partial(
            LayerNormPostprocessor,
            embedding_dim=hstu_config.hstu_transducer_embedding_dim,
            eps=1e-5,
            dtype=self.dtype,
        )
      else:
        postproc_cls = partial(
            TimestampLayerNormPostprocessor,
            embedding_dim=hstu_config.hstu_transducer_embedding_dim,
            time_duration_features=[(60 * 60, 24), (24 * 60 * 60, 7)],
            eps=1e-5,
            dtype=self.dtype,
        )
    else:
      postproc_cls = L2NormPostprocessor

    stu_layers = []
    for _ in range(hstu_config.hstu_attn_num_layers):
      stu_layer_config = STULayerConfig(
          embedding_dim=hstu_config.hstu_transducer_embedding_dim,
          num_heads=hstu_config.hstu_num_heads,
          hidden_dim=hstu_config.hstu_attn_linear_dim,
          attention_dim=hstu_config.hstu_attn_qk_dim,
          output_dropout_ratio=hstu_config.hstu_linear_dropout_rate,
          causal=True,
          target_aware=True,
          use_group_norm=hstu_config.hstu_group_norm,
          contextual_seq_len=contextual_seq_len,
          max_attn_len=hstu_config.hstu_max_attn_len,
      )
      stu_layers.append(STULayer(config=stu_layer_config))
    stu_module = STUStack(
        stu_layers=stu_layers
    )

    self._hstu_transducer = HSTUTransducer(
        stu_module=stu_module,
        input_preprocessor=preprocessor,
        output_postprocessor_cls=postproc_cls,
        input_dropout_ratio=hstu_config.hstu_input_dropout_ratio,
        positional_encoder=positional_encoder,
        return_full_embeddings=False,
        listwise=False,
    )

    self._item_embedding_mlp = ItemMLP(
        hidden_dim=512,
        output_dim=hstu_config.hstu_transducer_embedding_dim,
        dtype=self.dtype,
    )

  def _concat_features(
      self, uih_tensor: Array, cand_tensor: Array
  ) -> Array:
    """Concatenates dense UIH and candidate tensors along sequence dim."""
    return jnp.concatenate([uih_tensor, cand_tensor], axis=1)

  def _construct_payload(
      self,
      uih_features: Dict[str, Array],
      cand_features: Dict[str, Array],
      uih_embeddings: Dict[str, Array],
      cand_embeddings: Dict[str, Array],
  ) -> Dict[str, Array]:
    """Constructs payload dictionary for HSTUTransducer."""
    payload = {}
    for name in self.hstu_configs.contextual_feature_to_max_length:
      if name in uih_embeddings:
        payload[name] = uih_embeddings[name]
      elif name in cand_embeddings:
        payload[name] = cand_embeddings[name]
      elif name in uih_features and name in self.embedding_tables:
        payload[name] = self._embedding_collection({name: uih_features[name]})[
            name
        ]
      elif name in cand_features and name in self.embedding_tables:
        payload[name] = self._embedding_collection({name: cand_features[name]})[
            name
        ]
      elif name in uih_features:  # non-embedding contextual feature
        payload[name] = uih_features[name]
      elif name in cand_features:
        payload[name] = cand_features[name]

    for (
        uih_name,
        cand_name,
    ) in self.hstu_configs.merge_uih_candidate_feature_mapping:
      # Handle non-embedding features that need to be merged.
      if uih_name in uih_features and cand_name in cand_features:
        if uih_name not in self.embedding_tables and uih_name not in payload:
          payload[uih_name] = self._concat_features(
              uih_features[uih_name], cand_features[cand_name]
          )
      # Handle embedding features that need to be in the payload.
      if uih_name in uih_embeddings and cand_name in cand_embeddings:
        if uih_name not in payload:
          payload[uih_name] = self._concat_features(
              uih_embeddings[uih_name], cand_embeddings[cand_name]
          )

    # Handle features that only exist for candidates (for target enrichment)
    if self.hstu_configs.target_enrich_features:
      for feat_name in self.hstu_configs.target_enrich_features:
        if feat_name in cand_embeddings and feat_name not in payload:
          payload[feat_name] = cand_embeddings[feat_name]
    return payload

  def __call__(
      self,
      uih_features: Dict[str, Array],
      candidate_features: Dict[str, Array],
      uih_lengths: Array,
      num_candidates: Array,
      *,
      deterministic: bool,
  ) -> Tuple[
      Array,
      Array,
      Dict[str, Array],
      Optional[Array],
      Optional[Array],
      Optional[Array],
  ]:
    """Forward pass for DLRM-HSTU.

    Args:
        uih_features: Dict of dense UIH feature tensors (B, max_uih_len, ...).
        candidate_features: Dict of dense candidate feature tensors
          (B, max_cand, ...).
        uih_lengths: Length of UIH sequences (B,).
        num_candidates: Number of candidates per example (B,).
        deterministic: If true, disable dropout.

    Returns:
        Tuple of (user_embeddings, item_embeddings, aux_losses,
                 preds, labels, weights).
    """
    max_uih_len = uih_features[
        self.hstu_configs.uih_post_id_feature_name
    ].shape[1]
    cand_key = self.hstu_configs.item_embedding_feature_names[0]
    max_candidates = candidate_features[cand_key].shape[1]

    uih_embeddings = self._embedding_collection(uih_features)
    cand_embeddings = self._embedding_collection(candidate_features)

    merged_embeddings: Dict[str, Array] = {}
    for (
        uih_name,
        cand_name,
    ) in self.hstu_configs.merge_uih_candidate_feature_mapping:
      if uih_name in uih_embeddings and cand_name in cand_embeddings:
        merged_embeddings[uih_name] = self._concat_features(
            uih_embeddings[uih_name], cand_embeddings[cand_name]
        )
    if self.hstu_configs.uih_post_id_feature_name not in merged_embeddings:
      raise ValueError(
          "Post ID feature "
          f"{self.hstu_configs.uih_post_id_feature_name} not found in "
          "merged embeddings."
      )
    cand_item_embeddings_for_mlp = jnp.concatenate(
        [
            cand_embeddings[k]
            for k in self.hstu_configs.item_embedding_feature_names
        ],
        axis=-1,
    )
    item_embeddings_candidates = self._item_embedding_mlp(
        cand_item_embeddings_for_mlp
    )

    payload = self._construct_payload(
        uih_features, candidate_features, uih_embeddings, cand_embeddings
    )
    hstu_seq_lengths = uih_lengths + num_candidates
    hstu_seq_embeddings = merged_embeddings[
        self.hstu_configs.uih_post_id_feature_name
    ]
    candidate_querytime_feature_name = (
        self.hstu_configs.candidates_querytime_feature_name
    )
    hstu_seq_timestamps = self._concat_features(
        uih_features[self.hstu_configs.uih_action_time_feature_name],
        candidate_features[candidate_querytime_feature_name],
    )

    user_embeddings_candidates, _ = self._hstu_transducer(
        max_uih_len=max_uih_len,
        max_targets=max_candidates,
        total_uih_len=0,  # Not used in dense tensor implementation
        total_targets=0,  # Not used in dense tensor implementation
        seq_lengths=hstu_seq_lengths,
        seq_embeddings=hstu_seq_embeddings,
        seq_timestamps=hstu_seq_timestamps,
        num_targets=num_candidates,
        seq_payloads=payload,
        deterministic=deterministic,
    )

    supervision_bitmasks = candidate_features[
        self.hstu_configs.candidates_weight_feature_name
    ]
    watchtime_sequence = candidate_features[
        self.hstu_configs.candidates_watchtime_feature_name
    ]
    supervision_labels, supervision_weights = (
        _get_supervision_labels_and_weights(
            supervision_bitmasks,
            watchtime_sequence,
            self._multitask_configs,
        )
    )

    # The HSTU transducer returns embeddings for the full sequence, with
    # non-candidate parts masked. We need to slice out the candidate parts
    # to match the shape of the item embeddings.
    user_embeddings_candidates = user_embeddings_candidates[
        :, -max_candidates:, :
    ]

    mt_target_preds, mt_target_labels, mt_target_weights, mt_losses = (
        self._multitask_module(
            encoded_user_embeddings=user_embeddings_candidates,
            item_embeddings=item_embeddings_candidates,
            supervision_labels=supervision_labels,
            supervision_weights=supervision_weights,
            deterministic=deterministic,
        )
    )

    aux_losses: Dict[str, Array] = {}
    if not deterministic and mt_losses is not None:
      for i, task in enumerate(self._multitask_configs):
        aux_losses[task.task_name] = mt_losses[i]

    return (
        user_embeddings_candidates,
        item_embeddings_candidates,
        aux_losses,
        mt_target_preds,
        mt_target_labels,
        mt_target_weights,
    )

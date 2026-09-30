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
"""Memory-efficient JAX operations for binary focal cross-entropy loss.

Computes exact binary focal cross-entropy loss over large vocabulary
without materializing full [batch, seq_len, vocab_size] logit tensors in HBM.
"""

import dataclasses
import functools
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from recml.core.ops import binary_cross_entropy_ops as bce_ops

EPS = bce_ops.EPS
_auto_block_v = bce_ops._auto_block_v  # pylint: disable=protected-access
_balanced_chunk_n = (
    bce_ops._balanced_chunk_n  # pylint: disable=protected-access
)
_bce_bwd_scan_groups = (
    bce_ops._bce_bwd_scan_groups  # pylint: disable=protected-access
)
_local_tokens = bce_ops._local_tokens  # pylint: disable=protected-access
_check_vocab_replicated_in_d = (
    bce_ops._check_vocab_replicated_in_d  # pylint: disable=protected-access
)
_effective_block_v = (
    bce_ops._effective_block_v  # pylint: disable=protected-access
)
_get_sharding = bce_ops._get_sharding  # pylint: disable=protected-access
_max_safe_block_v = bce_ops._max_safe_block_v  # pylint: disable=protected-access
_max_safe_chunk_n = bce_ops._max_safe_chunk_n  # pylint: disable=protected-access
_pallas_interpret = bce_ops._pallas_interpret  # pylint: disable=protected-access
_pallas_lane = bce_ops._pallas_lane  # pylint: disable=protected-access
_pallas_sublane = bce_ops._pallas_sublane  # pylint: disable=protected-access
_pallas_vmem_budget = (
    bce_ops._pallas_vmem_budget  # pylint: disable=protected-access
)
_replicate_hidden_dim = (
    bce_ops._replicate_hidden_dim  # pylint: disable=protected-access
)
_BLOCK_N = bce_ops._BLOCK_N  # pylint: disable=protected-access


@dataclasses.dataclass(frozen=True)
class FocalBCEConfig(bce_ops.BCEConfig):
  """Configuration for the binary focal cross-entropy loss.

  Inherits common fields from `bce_ops.BCEConfig`.

  Attributes:
    gamma: Focusing parameter of the focal loss. The per-target BCE loss is
      scaled by ``(1 - p_t) ** gamma``, which down-weights well-classified
      targets; ``gamma = 0`` recovers the plain BCE loss.
    alpha: Class balancing weight in ``[0, 1]`` applied to positive targets,
      with ``1 - alpha`` applied to negative targets. Only used when
      `apply_class_balancing` is True.
    apply_class_balancing: Whether to weight the loss by `alpha` as described
      above.
  """

  gamma: float = 2.0
  alpha: float = 0.25
  apply_class_balancing: bool = False


def _focal_bce_fwd_chunk(
    config: FocalBCEConfig,
    activations_2d: jt.Float[jt.Array, "N D"],
    embeddings: jt.Float[jt.Array, "V D"],
    targets_2d: jt.Int[jt.Array, "N L"],
    j: jt.Int[jt.Array, ""],
    block_v: int,
    vocab: int,
) -> tuple[
    jt.Float[jt.Array, "N"],
    jt.Float[jt.Array, "N block_v"],
    jt.Bool[jt.Array, "N block_v"],
    jt.Bool[jt.Array, "block_v"],
]:
  """Computes logits and base focal loss for a single vocabulary chunk."""
  if block_v > vocab:
    raise ValueError(
        f"block_v ({block_v}) must not exceed the vocabulary size ({vocab});"
        " callers must clamp block_v before chunking."
    )
  actual_start = jnp.maximum(0, jnp.minimum(j * block_v, vocab - block_v))
  emb_chunk = jax.lax.dynamic_slice_in_dim(embeddings, actual_start, block_v)
  logits = jax.lax.dot(
      activations_2d,
      emb_chunk,
      dimension_numbers=(((1,), (1,)), ((), ())),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.DEFAULT,
  )

  chunk_indices = actual_start + jnp.arange(block_v)
  valid_mask = chunk_indices >= j * block_v

  n = activations_2d.shape[0]
  targets_chunk = jnp.zeros((n, block_v), dtype=jnp.bool_)
  rel_targets = targets_2d - actual_start
  chunk_cols = jnp.arange(block_v)[None, :]
  for l_idx in range(targets_2d.shape[-1]):
    targets_chunk = targets_chunk | (
        rel_targets[:, l_idx : l_idx + 1] == chunk_cols
    )
  targets_float = targets_chunk.astype(logits.dtype)

  probs = jax.nn.sigmoid(logits)
  p_t = targets_float * probs + (1.0 - targets_float) * (1.0 - probs)
  focal_factor = jnp.power(1.0 - p_t, config.gamma)

  # Fused BCE Loss: BCE(x, y) = BCE(x, 0) - y * x
  loss_zero = jnp.maximum(logits, 0.0) + jnp.log1p(jnp.exp(-jnp.abs(logits)))
  bce_loss_chunk = loss_zero - targets_float * logits
  loss_chunk = focal_factor * bce_loss_chunk
  if config.apply_class_balancing:
    weight = targets_float * config.alpha + (1.0 - targets_float) * (
        1.0 - config.alpha
    )
    loss_chunk = weight * loss_chunk

  loss_chunk = loss_chunk * valid_mask[None, :]
  loss_sum = jnp.sum(loss_chunk, axis=-1)
  return loss_sum, logits, targets_chunk, valid_mask


def _focal_bce_fwd_local(
    config: FocalBCEConfig,
    activations: jt.Float[jt.Array, "B N D"],
    embeddings: jt.Float[jt.Array, "V D"],
    targets: jt.Int[jt.Array, "B N L"],
) -> (
    tuple[
        jt.Float[jt.Array, "B N"],
        jt.Float[jt.Array, "B N"],
        jt.Float[jt.Array, "B N"],
        jt.Float[jt.Array, "B N"],
        jt.Float[jt.Array, "B N"],
    ]
    | jt.Float[jt.Array, "B N"]
):
  """Computes the sum of FocalLoss(x_v, target_v) over all V block-wise, and metrics."""
  block_v = config.block_v
  batch, seq_len, hidden = activations.shape
  vocab = embeddings.shape[0]

  n = batch * seq_len
  v_blocks = int(np.ceil(vocab / block_v))

  activations_2d = jnp.reshape(activations, (n, hidden))
  targets_2d = jnp.reshape(targets, (n, -1))

  if config.compute_metrics:

    def v_body(
        carry: tuple[jt.Float[jt.Array, "N"], ...],
        j: jt.Int[jt.Array, ""],
    ) -> tuple[tuple[jt.Float[jt.Array, "N"], ...], None]:
      loss_acc, tp_acc, fp_acc, fn_acc, tn_acc = carry
      loss_sum, logits, targets_chunk, valid_mask = _focal_bce_fwd_chunk(
          config, activations_2d, embeddings, targets_2d, j, block_v, vocab
      )

      predictions_chunk = (logits > 0.0) & valid_mask[None, :]
      targets_chunk = targets_chunk & valid_mask[None, :]

      tp_chunk = targets_chunk & predictions_chunk
      fp_chunk = predictions_chunk ^ tp_chunk
      fn_chunk = targets_chunk ^ tp_chunk
      tn_chunk = valid_mask[None, :] & (~(targets_chunk | predictions_chunk))

      tp_sum = jnp.sum(tp_chunk, axis=-1).astype(jnp.float32)
      fp_sum = jnp.sum(fp_chunk, axis=-1).astype(jnp.float32)
      fn_sum = jnp.sum(fn_chunk, axis=-1).astype(jnp.float32)
      tn_sum = jnp.sum(tn_chunk, axis=-1).astype(jnp.float32)

      return (
          loss_acc + loss_sum,
          tp_acc + tp_sum,
          fp_acc + fp_sum,
          fn_acc + fn_sum,
          tn_acc + tn_sum,
      ), None

    init = (
        jnp.zeros((n,), dtype=jnp.float32),
        jnp.zeros((n,), dtype=jnp.float32),
        jnp.zeros((n,), dtype=jnp.float32),
        jnp.zeros((n,), dtype=jnp.float32),
        jnp.zeros((n,), dtype=jnp.float32),
    )
    (loss_final, tp_final, fp_final, fn_final, tn_final), _ = jax.lax.scan(
        v_body, init, jnp.arange(v_blocks)
    )
    return (
        jnp.reshape(loss_final, (batch, seq_len)),
        jnp.reshape(tp_final, (batch, seq_len)),
        jnp.reshape(fp_final, (batch, seq_len)),
        jnp.reshape(fn_final, (batch, seq_len)),
        jnp.reshape(tn_final, (batch, seq_len)),
    )
  else:

    def v_body_no_metrics(
        loss_acc: jt.Float[jt.Array, "N"],
        j: jt.Int[jt.Array, ""],
    ) -> tuple[jt.Float[jt.Array, "N"], None]:
      loss_sum, _, _, _ = _focal_bce_fwd_chunk(
          config, activations_2d, embeddings, targets_2d, j, block_v, vocab
      )
      return loss_acc + loss_sum, None

    init = jnp.zeros((n,), dtype=jnp.float32)
    loss_final, _ = jax.lax.scan(v_body_no_metrics, init, jnp.arange(v_blocks))
    return jnp.reshape(loss_final, (batch, seq_len))


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _cut_binary_focal_cross_entropy(
    config: FocalBCEConfig,
    activations: jt.Float[jt.Array, "... B N D"],
    embeddings: jt.Float[jt.Array, "V D"],
    targets: jt.Int[jt.Array, "... B N L"],
) -> (
    tuple[
        jt.Float[jt.Array, "... B N"],
        jt.Float[jt.Array, "... B N"],
        jt.Float[jt.Array, "... B N"],
        jt.Float[jt.Array, "... B N"],
        jt.Float[jt.Array, "... B N"],
    ]
    | jt.Float[jt.Array, "... B N"]
):
  """Computes the non-differentiable path of cut Focal BCE loss and metrics."""
  outputs, _ = _cut_binary_focal_cross_entropy_fwd(
      config, activations, embeddings, targets
  )
  return outputs


def _cut_binary_focal_cross_entropy_fwd(
    config: FocalBCEConfig,
    activations: jt.Float[jt.Array, "B N D"],
    embeddings: jt.Float[jt.Array, "V D"],
    targets: jt.Int[jt.Array, "B N L"],
) -> tuple[
    tuple[
        jt.Float[jt.Array, "B N"],
        jt.Float[jt.Array, "B N"],
        jt.Float[jt.Array, "B N"],
        jt.Float[jt.Array, "B N"],
        jt.Float[jt.Array, "B N"],
    ]
    | jt.Float[jt.Array, "B N"],
    tuple[
        jt.Float[jt.Array, "B N D"],
        jt.Float[jt.Array, "V D"],
        jt.Int[jt.Array, "B N L"],
    ],
]:
  """Computes forward mode of cut Focal BCE loss."""
  replicated_activations = _replicate_hidden_dim(activations)
  # `_focal_bce_fwd_local` is written against a 3D `[B, N, D]` array: it unpacks
  # exactly three dims and flattens `[B, N]` into the `N` rows of the blocked
  # matmul. 4D `[X, B, N, D]` inputs are therefore mapped rather than reshaped
  # to `[X * B, N, D]`, which keeps the shared-targets case (3D `[B, N, L]`
  # alongside 4D activations) a plain `in_axes=None` instead of an explicit
  # broadcast. The two spellings cost the same memory: vmap vectorizes, so one
  # chunk of logits is a single `[X, B * N, block_v]` buffer rather than `X`
  # separate `[B * N, block_v]` tiles, exactly as a reshape to `[X * B, N, D]`
  # would give `[X * B * N, block_v]`. `block_v` is budgeted against those
  # `X * B * N` rows (see `_local_tokens`); vmap does not sequence the `X`
  # slices.
  if activations.ndim == 4:
    if targets.ndim == 3:
      targets_in_axis = None
    else:
      targets_in_axis = 0
    fwd_vmap = jax.vmap(
        functools.partial(_focal_bce_fwd_local, config),
        in_axes=(0, None, targets_in_axis),
    )
    res = fwd_vmap(replicated_activations, embeddings, targets)
  else:
    res = _focal_bce_fwd_local(
        config, replicated_activations, embeddings, targets
    )
  vocab_size = embeddings.shape[0]
  if isinstance(res, tuple):
    loss_y0, tp, fp, fn, tn = res
    losses = loss_y0 * (1.0 / vocab_size)
    return (losses, tp, fp, fn, tn), (
        activations,
        embeddings,
        targets,
    )
  else:
    losses = res * (1.0 / vocab_size)
    return losses, (
        activations,
        embeddings,
        targets,
    )


def _cut_binary_focal_cross_entropy_bwd(
    config: FocalBCEConfig,
    res: tuple[
        jt.Float[jt.Array, "... B N D"],
        jt.Float[jt.Array, "V D"],
        jt.Int[jt.Array, "... B N L"],
    ],
    d_outputs: (
        tuple[
            jt.Float[jt.Array, "... B N"],
            jt.Float[jt.Array, "... B N"],
            jt.Float[jt.Array, "... B N"],
            jt.Float[jt.Array, "... B N"],
            jt.Float[jt.Array, "... B N"],
        ]
        | jt.Float[jt.Array, "... B N"]
    ),
) -> tuple[
    jt.Float[jt.Array, "... B N D"],
    jt.Float[jt.Array, "V D"],
    None,
]:
  """Computes the backward mode of cut Focal BCE loss."""
  if isinstance(d_outputs, tuple):
    d_losses = d_outputs[0]
  else:
    d_losses = d_outputs
  activations, embeddings, targets = res
  d_activations, d_embeddings = _focal_bce_bwd_sharded(
      config, d_losses, activations, embeddings, targets
  )
  return d_activations, d_embeddings, None


_cut_binary_focal_cross_entropy.defvjp(
    _cut_binary_focal_cross_entropy_fwd, _cut_binary_focal_cross_entropy_bwd
)


def cut_binary_focal_cross_entropy(
    activations: jt.Float[jt.Array, "... B N D"],
    embeddings: jt.Float[jt.Array, "V D"],
    targets: jt.Int[jt.Array, "... B N L"],
    weights: jt.Float[jt.Array, "... B N"] | None = None,
    *,
    gamma: float = 2.0,
    alpha: float = 0.25,
    apply_class_balancing: bool = False,
    return_per_target_losses: bool = False,
    return_metrics: bool = False,
    block_v: int | None = None,
    block_n: int | None = None,
    mesh: jax.sharding.Mesh | None = None,
    act_spec: jax.sharding.PartitionSpec | None = None,
    emb_spec: jax.sharding.PartitionSpec | None = None,
    use_pallas: bool = True,
) -> (
    jt.Float[jt.Array, ""]
    | tuple[jt.Float[jt.Array, ""], jt.Float[jt.Array, "... B N"]]
    | tuple[
        jt.Float[jt.Array, ""],
        jt.Float[jt.Array, ""],
        jt.Float[jt.Array, ""],
        jt.Float[jt.Array, ""],
        jt.Float[jt.Array, ""],
    ]
    | tuple[
        jt.Float[jt.Array, ""],
        jt.Float[jt.Array, "... B N"],
        jt.Float[jt.Array, ""],
        jt.Float[jt.Array, ""],
        jt.Float[jt.Array, ""],
        jt.Float[jt.Array, ""],
    ]
):
  """Computes binary focal cross entropy loss over unmaterialized logits.

  Only 3D `[B, N, D]` and 4D `[X, B, N, D]` activations are supported; rank 5
  and above raise a `ValueError`. Reshape or vmap over the extra leading axes
  before calling this op.

  Args:
    activations: Hidden-state outputs of shape ``[B, N, D]``, or ``[X, B, N,
      D]`` with a single extra leading axis.
    embeddings: Output embedding / unembedding weights of shape ``[V, D]``.
    targets: Target token ids of shape ``[B, N, L]``, i.e. the ``L`` positive
      labels of each sequence position. For 4D activations, either a matching
      4D ``[X, B, N, L]`` array or a 3D ``[B, N, L]`` array shared across the
      leading axis.
    weights: Per-sequence-position loss weights, typically a padding/validity
      mask. Must have exactly the shape of the per-position losses, i.e.
      ``activations.shape[:-1]`` (``[B, N]``, or ``[X, B, N]`` for 4D
      activations); any other shape raises. This is one weight per position,
      *not* per target id: the ``L`` target ids of a position all share that
      position's weight.
    gamma: Gamma parameter for focal loss.
    alpha: Alpha parameter for focal loss.
    apply_class_balancing: If True, apply class balancing.
    return_per_target_losses: If True, also return the per-target loss tensor.
    return_metrics: If True, also return TP, FP, FN, TN metric counts.
    block_v: Vocab-axis block size. Auto-picked if omitted.
    block_n: Sequence-axis block size for backward kernel.
    mesh: Optional mesh to use for sharding.
    act_spec: Optional partition spec for activations.
    emb_spec: Optional partition spec for embeddings.
    use_pallas: If True, use Pallas kernels; if False, use pure JAX
      implementation.

  Returns:
    Scalar loss, optionally paired with per-target losses and/or metrics.

  Raises:
    ValueError: If ``activations`` has rank 5 or above, or if ``weights`` does
      not exactly match the per-position loss shape.
  """
  # `_focal_bce_fwd_local` operates on a single `[B, N, D]` array and the
  # forward pass only vmaps over one extra leading axis, so ranks >= 5 are
  # unsupported.
  if activations.ndim >= 5:
    raise ValueError(
        "`activations` must be a 3D `[B, N, D]` or 4D `[X, B, N, D]` array;"
        f" rank {activations.ndim} is not supported. Got shape"
        f" {activations.shape}. Reshape or vmap over the extra leading axes"
        " before calling this op."
    )

  vocab_size = embeddings.shape[0]
  # The per-position losses have one entry per `[..., B, N]` position.
  losses_shape = tuple(activations.shape[:-1])

  if weights is not None and tuple(weights.shape) != losses_shape:
    # `losses * weights` would broadcast, but `jnp.sum(weights)` would not, so a
    # mismatch silently normalizes the loss by the wrong number of elements.
    raise ValueError(
        f"`weights` must have shape {losses_shape}, i.e. one weight per"
        " sequence position (not one per target id). Got shape"
        f" {tuple(weights.shape)}."
    )

  sharding = _get_sharding(embeddings)
  if (
      isinstance(sharding, jax.sharding.NamedSharding)
      and not sharding.mesh.empty
  ):
    replicated_sharding = jax.sharding.NamedSharding(
        sharding.mesh, jax.sharding.PartitionSpec()
    )
    embeddings = jax.lax.with_sharding_constraint(
        embeddings, replicated_sharding
    )

  if block_v is None:
    block_v = _auto_block_v(
        _local_tokens(activations, mesh, act_spec), vocab_size
    )
  else:
    block_v = min(block_v, vocab_size)

  effective_block_n = _pallas_lane() if block_n is None else block_n
  config = FocalBCEConfig(
      block_v=block_v,
      block_n=effective_block_n,
      compute_metrics=return_metrics,
      gamma=gamma,
      alpha=alpha,
      apply_class_balancing=apply_class_balancing,
      mesh=mesh,
      act_spec=act_spec,
      emb_spec=emb_spec,
      use_pallas=use_pallas,
      global_vocab=vocab_size,
  )

  res = _cut_binary_focal_cross_entropy(
      config,
      activations,
      embeddings,
      targets,
  )

  if return_metrics:
    losses, tp, fp, fn, tn = res

    if weights is not None:
      losses = losses * weights
      weight_sum = jnp.sum(weights)
      tp_sum = jnp.sum(tp * weights)
      fp_sum = jnp.sum(fp * weights)
      fn_sum = jnp.sum(fn * weights)
      tn_sum = jnp.sum(tn * weights)
    else:
      weight_sum = np.prod(losses_shape)
      tp_sum = jnp.sum(tp)
      fp_sum = jnp.sum(fp)
      fn_sum = jnp.sum(fn)
      tn_sum = jnp.sum(tn)

    loss = jnp.sum(losses) / (weight_sum + EPS)

    if return_per_target_losses:
      return loss, losses, tp_sum, fp_sum, fn_sum, tn_sum
    return loss, tp_sum, fp_sum, fn_sum, tn_sum
  else:
    losses = res

    if weights is not None:
      losses = losses * weights
      weight_sum = jnp.sum(weights)
    else:
      weight_sum = np.prod(losses_shape)

    loss = jnp.sum(losses) / (weight_sum + EPS)

    if return_per_target_losses:
      return loss, losses
    return loss


cut_binary_cross_entropy = cut_binary_focal_cross_entropy


def _focal_bce_bwd_kernel(
    emb_ref: jax.Ref,  # [block_v, padded_d] VMEM
    act_ref: jax.Ref,  # [padded_n, padded_d] VMEM, resident across the grid
    tgt_ref: jax.Ref,  # [l_padded, padded_n] VMEM, resident across the grid
    dloss_ref: jax.Ref,  # [1, padded_n] VMEM, resident across the grid
    d_emb_ref: jax.Ref,  # [block_v, padded_d] VMEM output block
    d_act_partials_ref: jax.Ref,  # [1, padded_n, padded_d] VMEM output block
    d_emb_scratch: jax.Ref,  # [block_v, padded_d] VMEM scratch
    *,
    block_v: int,
    block_n: int,
    n_blocks: int,
    vocab: int,
    labels: int,
    n_real: int,
    gamma: float,
    alpha: float,
    apply_class_balancing: bool,
    global_vocab: int,
):
  """Pallas TPU per-shard chunked backward for Focal BCE."""
  v_idx = pl.program_id(0)

  # Initialize accumulators in VMEM to 0
  d_emb_scratch[...] = jnp.zeros(d_emb_scratch.shape, jnp.float32)

  # Valid vocab mask and local vocab ids: (1, block_v)
  chunk_indices_local = v_idx * block_v + jnp.arange(block_v)[None, :]
  valid_vocab_mask = chunk_indices_local < vocab
  valid_vocab_mask_t = valid_vocab_mask.T

  # Load inputs from VMEM
  emb = emb_ref[...]
  emb = jnp.where(valid_vocab_mask_t, emb, 0.0)

  # Precomputed inverse vocab, batch iota, and alpha balancing constants
  # `vocab` is shard-local; the forward normalizes by the global vocab size.
  inv_vocab = 1.0 / global_vocab
  local_n_iota = jnp.arange(block_n)[:, None]
  alpha_diff = 2.0 * alpha - 1.0
  alpha_base = 1.0 - alpha

  def loop_body(n_idx, _):
    act = act_ref[pl.ds(n_idx * block_n, block_n), :]
    tgt_val = tgt_ref[:labels, pl.ds(n_idx * block_n, block_n)]
    # (labels, block_n)
    dloss_val = dloss_ref[0, pl.ds(n_idx * block_n, block_n)]  # (block_n,)

    # Compute logits: (block_n, block_v)
    logits = jax.lax.dot(
        act,
        emb,
        dimension_numbers=(((1,), (1,)), ((), ())),
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32,
    )
    probs = jax.nn.sigmoid(logits)

    # Target matching across labels: (block_n, block_v)
    y_true_chunk = jnp.any(tgt_val[:, :, None] == chunk_indices_local, axis=0)

    # Valid batch mask for this loop step: (block_n, 1)
    batch_indices_local = n_idx * block_n + local_n_iota
    valid_batch_mask = batch_indices_local < n_real

    # Gradients w.r.t logits, masked
    y_true_float = y_true_chunk.astype(probs.dtype)
    g_bce = probs - y_true_float
    abs_g = jnp.abs(g_bce)
    p_t = 1.0 - abs_g
    focal_factor = jnp.power(abs_g, gamma)
    loss_zero = jnp.maximum(logits, 0.0) + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    bce_loss_chunk = loss_zero - y_true_float * logits
    # d/dz [(1 - p_t)^gamma * BCE] = g_bce * [(1 - p_t)^gamma
    #     + gamma * (1 - p_t)^(gamma - 1) * p_t * BCE].
    # `gamma` is a static Python float, so this branch is resolved at trace
    # time and only one form is compiled.
    if gamma >= 1.0:
      # Non-negative exponent: the direct form is exact and is the cheapest
      # for the common gamma = 2 case.
      focal_factor_m1 = jnp.power(abs_g, gamma - 1.0)
      g_focal = g_bce * (
          focal_factor + gamma * focal_factor_m1 * p_t * bce_loss_chunk
      )
    elif gamma == 0.0:
      # Plain BCE: (1 - p_t)^0 = 1 and the second term vanishes.
      g_focal = g_bce
    else:
      # 0 < gamma < 1: (1 - p_t)^(gamma - 1) is a negative power that blows up
      # as p_t -> 1. Since |g_bce| = 1 - p_t, g_bce * (1 - p_t)^(gamma - 1)
      # equals sign(g_bce) * (1 - p_t)^gamma, and with sign(g_bce) = 1 - 2y,
      # sign(g_bce) * p_t = 1 - y - p (where g_bce == 0 both forms yield 0).
      # This is exact and finite, and needs neither p_t nor a sign op.
      g_focal = focal_factor * (
          g_bce + gamma * ((1.0 - y_true_float) - probs) * bce_loss_chunk
      )
    if apply_class_balancing:
      weight = y_true_float * alpha_diff + alpha_base
      g_focal = weight * g_focal

    scale = jnp.where(valid_batch_mask, (dloss_val * inv_vocab)[:, None], 0.0)
    deriv = scale * g_focal

    # Accumulate d_emb
    d_emb_contrib = jax.lax.dot(
        deriv.astype(act.dtype),
        act,
        dimension_numbers=(((0,), (0,)), ((), ())),
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32,
    )
    d_emb_scratch[...] = d_emb_scratch[...] + d_emb_contrib

    # Compute d_act and write directly to HBM
    d_act_contrib = jax.lax.dot(
        deriv.astype(emb.dtype),
        emb,
        dimension_numbers=(((1,), (0,)), ((), ())),
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32,
    )
    d_act_partials_ref[0, pl.ds(n_idx * block_n, block_n), :] = d_act_contrib

    return None

  # Run reduction loop over n_blocks
  jax.lax.fori_loop(0, n_blocks, loop_body, None)

  # Store accumulated results to HBM
  d_emb_ref[...] = d_emb_scratch[...]


def _focal_bce_bwd_pallas_chunked_n(
    config: FocalBCEConfig,
    d_loss: jt.Float[jt.Array, "B N"],
    activations: jt.Float[jt.Array, "B N D"],
    embeddings: jt.Float[jt.Array, "V D"],
    targets: jt.Int[jt.Array, "B N L"],
    vocab_offset: int = 0,
) -> tuple[
    jt.Float[jt.Array, "B N D"],
    jt.Float[jt.Array, "V D"],
]:
  """Wraps _focal_bce_bwd_pallas by chunking the N (sequence) dimension in JAX."""
  batch, seq_len, hidden = activations.shape
  n = batch * seq_len

  vocab = embeddings.shape[0]
  padded_d, block_v = _effective_block_v(config, hidden, vocab)
  vmem_budget = _pallas_vmem_budget()
  max_chunk_n = _max_safe_chunk_n(
      vmem_budget, padded_d, block_v, config.block_n
  )
  if n <= max_chunk_n:
    return _focal_bce_bwd_pallas(
        config, d_loss, activations, embeddings, targets, vocab_offset
    )

  chunk_n = _balanced_chunk_n(n, max_chunk_n)

  activations_2d = jnp.reshape(activations, (n, hidden))
  dloss_2d = jnp.reshape(d_loss, (n, 1))
  targets_2d = jnp.reshape(targets, (n, -1))

  n_chunks = (n + chunk_n - 1) // chunk_n
  padded_n = n_chunks * chunk_n

  if padded_n > n:
    activations_2d = jnp.pad(activations_2d, ((0, padded_n - n), (0, 0)))
    dloss_2d = jnp.pad(dloss_2d, ((0, padded_n - n), (0, 0)))
    targets_2d = jnp.pad(
        targets_2d, ((0, padded_n - n), (0, 0)), constant_values=-1
    )

  act_chunks_3d = jnp.reshape(activations_2d, (n_chunks, 1, chunk_n, hidden))
  dloss_chunks_3d = jnp.reshape(dloss_2d, (n_chunks, 1, chunk_n))
  tgt_chunks_3d = jnp.reshape(targets_2d, (n_chunks, 1, chunk_n, -1))

  def loop_body(carry, x):
    (d_emb_acc,) = carry
    act_chunk, dloss_chunk, tgt_chunk = x

    d_act_chunk, d_emb_contrib = _focal_bce_bwd_pallas(
        config,
        dloss_chunk,
        act_chunk,
        embeddings,
        tgt_chunk,
        vocab_offset=vocab_offset,
    )

    return (d_emb_acc + d_emb_contrib,), d_act_chunk

  init = (jnp.zeros_like(embeddings),)
  (d_embeddings,), d_act_chunks = jax.lax.scan(
      loop_body, init, (act_chunks_3d, dloss_chunks_3d, tgt_chunks_3d)
  )

  d_act_flat = jnp.reshape(d_act_chunks, (padded_n, hidden))[:n]
  d_act = jnp.reshape(d_act_flat, (batch, seq_len, hidden))
  return d_act, d_embeddings


def _focal_bce_bwd_pallas(
    config: FocalBCEConfig,
    d_loss: jt.Float[jt.Array, "B N"],
    activations: jt.Float[jt.Array, "B N D"],
    embeddings: jt.Float[jt.Array, "V D"],
    targets: jt.Int[jt.Array, "B N L"],
    vocab_offset: int = 0,
) -> tuple[
    jt.Float[jt.Array, "B N D"],
    jt.Float[jt.Array, "V D"],
]:
  """Pallas TPU per-shard chunked backward for Focal BCE."""
  block_n = config.block_n
  batch, seq_len, hidden = activations.shape
  vocab = embeddings.shape[0]
  padded_d, block_v = _effective_block_v(config, hidden, vocab)
  global_vocab = vocab if config.global_vocab is None else config.global_vocab
  n = batch * seq_len
  n_blocks = (n + block_n - 1) // block_n
  v_blocks = (vocab + block_v - 1) // block_v
  vocab_padded = v_blocks * block_v
  labels = targets.shape[-1]

  # Rebase global target ids onto this shard's local rows here rather than
  # inside the kernel: under vocab sharding `vocab_offset` is a tracer, and a
  # tracer cannot be bound into a Pallas kernel as a static argument. Ids
  # belonging to other shards land outside [0, vocab) and simply never match.
  targets = targets - vocab_offset

  # Pad activations, targets, d_loss to multiples of block_n if necessary
  padded_n = n_blocks * block_n
  if padded_n > n:
    pad_len = padded_n - n
    activations_2d = jnp.pad(
        jnp.reshape(activations, (n, hidden)), ((0, pad_len), (0, 0))
    )
    dloss_2d = jnp.pad(jnp.reshape(d_loss, (n, 1)), ((0, pad_len), (0, 0)))
    targets_2d = jnp.pad(
        jnp.reshape(targets, (n, labels)),
        ((0, pad_len), (0, 0)),
        constant_values=-1,
    )
  else:
    activations_2d = jnp.reshape(activations, (n, hidden))
    dloss_2d = jnp.reshape(d_loss, (n, 1))
    targets_2d = jnp.reshape(targets, (n, labels))
  # Pad activations and embeddings to padded_d columns
  if padded_d > hidden:
    activations_padded = jnp.pad(
        activations_2d, ((0, 0), (0, padded_d - hidden))
    )
  else:
    activations_padded = activations_2d

  if vocab_padded > vocab or padded_d > hidden:
    embeddings_padded = jnp.pad(
        embeddings, ((0, vocab_padded - vocab), (0, padded_d - hidden))
    )
  else:
    embeddings_padded = embeddings

  # Transpose and pad targets: (n, labels) -> (labels, n) ->
  # (l_padded, padded_n)
  num_sublanes = _pallas_sublane()
  l_padded = max(num_sublanes, pl.align_to(labels, num_sublanes))
  targets_t = jnp.transpose(targets_2d, (1, 0))
  if l_padded > labels or padded_n > n:
    targets_padded = jnp.pad(
        targets_t,
        ((0, l_padded - labels), (0, padded_n - n)),
        constant_values=-1,
    ).astype(jnp.int32)
  else:
    targets_padded = targets_t.astype(jnp.int32)

  # Transpose and pad dloss: (n, 1) -> (1, n) -> (1, padded_n)
  dloss_t = jnp.transpose(dloss_2d, (1, 0))
  if padded_n > n:
    dloss_padded = jnp.pad(dloss_t, ((0, 0), (0, padded_n - n))).astype(
        embeddings.dtype
    )
  else:
    dloss_padded = dloss_t.astype(embeddings.dtype)

  d_emb_padded, d_act_partials = pl.pallas_call(
      functools.partial(
          _focal_bce_bwd_kernel,
          block_v=block_v,
          block_n=block_n,
          n_blocks=n_blocks,
          vocab=vocab,
          labels=labels,
          n_real=n,
          gamma=config.gamma,
          alpha=config.alpha,
          apply_class_balancing=config.apply_class_balancing,
          global_vocab=global_vocab,
      ),
      out_shape=[
          jax.ShapeDtypeStruct((vocab_padded, padded_d), embeddings.dtype),
          jax.ShapeDtypeStruct(
              (v_blocks, padded_n, padded_d), activations.dtype
          ),
      ],
      grid=(v_blocks,),
      in_specs=[
          pl.BlockSpec((block_v, padded_d), lambda v: (v, 0)),  # emb
          pl.BlockSpec((padded_n, padded_d), lambda v: (0, 0)),  # act
          pl.BlockSpec((l_padded, padded_n), lambda v: (0, 0)),  # tgt
          pl.BlockSpec((1, padded_n), lambda v: (0, 0)),  # dloss
      ],
      out_specs=[
          pl.BlockSpec((block_v, padded_d), lambda v: (v, 0)),  # d_emb
          pl.BlockSpec((1, padded_n, padded_d), lambda v: (v, 0, 0)),  # d_act
      ],
      scratch_shapes=[
          pltpu.VMEM((block_v, padded_d), jnp.float32),  # d_emb_scratch
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary",),
          vmem_limit_bytes=_pallas_vmem_budget(),
      ),
      interpret=_pallas_interpret(),
  )(
      embeddings_padded,
      activations_padded,
      targets_padded,
      dloss_padded,
  )

  if vocab < vocab_padded or hidden < padded_d:
    d_emb = d_emb_padded[:vocab, :hidden]
  else:
    d_emb = d_emb_padded

  if hidden < padded_d:
    d_act_2d = jnp.sum(d_act_partials[:, :, :hidden], axis=0)
  else:
    d_act_2d = jnp.sum(d_act_partials, axis=0)

  if padded_n > n:
    d_act_2d = d_act_2d[:n, :]
  d_activations = jnp.reshape(d_act_2d, (batch, seq_len, hidden))
  return d_activations, d_emb


def _focal_bce_bwd_pure_jax(
    config: FocalBCEConfig,
    d_loss: jt.Float[jt.Array, "... B N"],
    activations: jt.Float[jt.Array, "... B N D"],
    embeddings: jt.Float[jt.Array, "V D"],
    targets: jt.Int[jt.Array, "... B N L"],
    vocab_offset: int = 0,
) -> tuple[
    jt.Float[jt.Array, "... B N D"],
    jt.Float[jt.Array, "V D"],
]:
  """Pure JAX backward pass for focal binary cross entropy."""
  if activations.ndim == 4:
    return _bce_bwd_scan_groups(
        lambda d_loss_g, act_g, tgt_g: _focal_bce_bwd_pure_jax(
            config, d_loss_g, act_g, embeddings, tgt_g, vocab_offset
        ),
        d_loss,
        activations,
        embeddings,
        targets,
    )

  batch, seq_len, hidden = activations.shape
  vocab, _ = embeddings.shape
  n = batch * seq_len
  # `vocab` is shard-local; the forward normalizes by the global vocab size.
  global_vocab = vocab if config.global_vocab is None else config.global_vocab
  inv_vocab = 1.0 / global_vocab
  block_v = min(config.block_v, vocab)

  activations_2d = jnp.reshape(activations, (n, hidden))
  targets_2d = jnp.reshape(targets, (n, -1))
  dloss_2d = jnp.reshape(d_loss, (n, 1))

  # Chunked scan backward pass without tensor padding
  v_blocks = int(np.ceil(vocab / block_v))

  def scan_body(carry, j):
    d_act_acc, d_emb_acc = carry
    actual_start = jnp.maximum(0, jnp.minimum(j * block_v, vocab - block_v))
    emb_chunk = jax.lax.dynamic_slice_in_dim(embeddings, actual_start, block_v)

    logits = jax.lax.dot(
        activations_2d,
        emb_chunk,
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.DEFAULT,
    )
    probs = jax.nn.sigmoid(logits)

    chunk_indices = actual_start + jnp.arange(block_v)
    valid_mask = chunk_indices >= j * block_v

    targets_chunk = jnp.zeros((n, block_v), dtype=jnp.bool_)
    rel_targets = targets_2d - (actual_start + vocab_offset)
    chunk_cols = jnp.arange(block_v)[None, :]
    for l_idx in range(targets_2d.shape[-1]):
      targets_chunk = targets_chunk | (
          rel_targets[:, l_idx : l_idx + 1] == chunk_cols
      )
    g_bce = jnp.where(targets_chunk, probs - 1.0, probs)
    loss_zero = jnp.maximum(logits, 0.0) + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    bce_loss_chunk = jnp.where(targets_chunk, loss_zero - logits, loss_zero)
    # See `_focal_bce_bwd_kernel`; `config.gamma` is static, so only one branch
    # is traced.
    if config.gamma >= 1.0:
      p_t = jnp.where(targets_chunk, probs, 1.0 - probs)
      focal_factor = jnp.power(1.0 - p_t, config.gamma)
      focal_factor_m1 = jnp.power(1.0 - p_t, config.gamma - 1.0)
      g_focal = g_bce * (
          focal_factor + config.gamma * focal_factor_m1 * p_t * bce_loss_chunk
      )
    elif config.gamma == 0.0:
      # Plain BCE: (1 - p_t)^0 = 1 and the second term vanishes.
      g_focal = g_bce
    else:
      # |g_bce| = 1 - p_t and sign(g_bce) = 1 - 2y, so
      # g_bce * (1 - p_t)^(gamma - 1) = (1 - 2y) * (1 - p_t)^gamma, which is
      # exact and finite for 0 < gamma < 1.
      abs_g = jnp.abs(g_bce)
      p_t = 1.0 - abs_g
      focal_factor = jnp.power(abs_g, config.gamma)
      signed_gamma = jnp.where(targets_chunk, -config.gamma, config.gamma)
      g_focal = focal_factor * (g_bce + signed_gamma * p_t * bce_loss_chunk)
    if config.apply_class_balancing:
      weight = jnp.where(targets_chunk, config.alpha, 1.0 - config.alpha)
      g_focal = weight * g_focal

    d_logits_chunk = g_focal * valid_mask[None, :] * (dloss_2d * inv_vocab)

    d_act_contrib = jax.lax.dot(
        d_logits_chunk,
        emb_chunk,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=activations.dtype,
        precision=jax.lax.Precision.DEFAULT,
    )
    d_emb_chunk = jax.lax.dot(
        d_logits_chunk,
        activations_2d,
        dimension_numbers=(((0,), (0,)), ((), ())),
        preferred_element_type=embeddings.dtype,
        precision=jax.lax.Precision.DEFAULT,
    )
    curr_slice = jax.lax.dynamic_slice(
        d_emb_acc, (actual_start, 0), (block_v, hidden)
    )
    d_emb_acc = jax.lax.dynamic_update_slice(
        d_emb_acc, curr_slice + d_emb_chunk, (actual_start, 0)
    )

    return (d_act_acc + d_act_contrib, d_emb_acc), None

  init = (jnp.zeros_like(activations_2d), jnp.zeros_like(embeddings))
  (d_act_final, d_embeddings), _ = jax.lax.scan(
      scan_body, init, jnp.arange(v_blocks)
  )

  d_activations = jnp.reshape(d_act_final, (batch, seq_len, hidden))
  return d_activations, d_embeddings


def _focal_bce_bwd_sharded(
    config: FocalBCEConfig,
    d_loss: jt.Float[jt.Array, "... B N"],
    activations: jt.Float[jt.Array, "... B N D"],
    embeddings: jt.Float[jt.Array, "V D"],
    targets: jt.Int[jt.Array, "... B N L"],
) -> tuple[
    jt.Float[jt.Array, "... B N D"],
    jt.Float[jt.Array, "V D"],
]:
  """Sharding-aware dispatcher for Focal BCE backward."""
  mesh = None
  act_spec = None
  emb_spec = None

  if config.mesh is not None and config.act_spec is not None:
    mesh = config.mesh
    act_spec = config.act_spec
    emb_spec = config.emb_spec or jax.sharding.PartitionSpec()
    act_sharding = jax.sharding.NamedSharding(mesh, act_spec)
    is_sharded = True
  else:
    act_sharding = _get_sharding(activations)
    emb_sharding = _get_sharding(embeddings)
    is_sharded = (
        isinstance(act_sharding, jax.sharding.NamedSharding)
        and not act_sharding.mesh.empty
        and isinstance(emb_sharding, jax.sharding.NamedSharding)
    )
    if is_sharded:
      act_spec = act_sharding.spec  # pyrefly: ignore[missing-attribute]
      emb_spec = emb_sharding.spec
      mesh = act_sharding.mesh  # pyrefly: ignore[missing-attribute]

  if not is_sharded:
    if not config.use_pallas:
      return _focal_bce_bwd_pure_jax(
          config, d_loss, activations, embeddings, targets
      )
    if activations.ndim == 4:
      return _focal_bce_bwd_loop_fallback(
          config, d_loss, activations, embeddings, targets
      )
    return _focal_bce_bwd_pallas_chunked_n(
        config, d_loss, activations, embeddings, targets
    )

  if emb_spec is None:
    emb_spec = jax.sharding.PartitionSpec()
  _check_vocab_replicated_in_d(emb_spec)

  full_act_spec = tuple(act_spec) + (None,) * (  # pyrefly: ignore[bad-argument-type]
      activations.ndim - len(act_spec)  # pyrefly: ignore[bad-argument-type]
  )
  act_spec = jax.sharding.PartitionSpec(*full_act_spec)

  hidden_axis_name = act_spec[-1]
  if hidden_axis_name is not None:
    replicated_activations = _replicate_hidden_dim(activations)
    new_spec_list = list(act_spec)
    new_spec_list[-1] = None
    local_act_spec = jax.sharding.PartitionSpec(*new_spec_list)
  else:
    replicated_activations = activations
    local_act_spec = act_spec

  vocab_axis_name = emb_spec[0] if emb_spec else None

  dp_axes = []
  for axis in local_act_spec:  # pyrefly: ignore[not-iterable]
    if axis is not None and axis != vocab_axis_name:
      dp_axes.append(axis)

  def _bwd_local_with_reduction(d_loss_, act_, emb_, tgt_):
    if vocab_axis_name is not None:
      vocab_offset = jax.lax.axis_index(vocab_axis_name) * emb_.shape[0]
    else:
      vocab_offset = 0
    if not config.use_pallas:
      d_act, d_emb = _focal_bce_bwd_pure_jax(
          config, d_loss_, act_, emb_, tgt_, vocab_offset=vocab_offset
      )
    else:
      if act_.ndim == 4:
        d_act, d_emb = _focal_bce_bwd_loop_fallback(
            config, d_loss_, act_, emb_, tgt_, vocab_offset=vocab_offset
        )
      else:
        d_act, d_emb = _focal_bce_bwd_pallas_chunked_n(
            config, d_loss_, act_, emb_, tgt_, vocab_offset=vocab_offset
        )
    if dp_axes:
      d_emb = jax.lax.psum(d_emb, axis_name=dp_axes)
    if vocab_axis_name is not None:
      d_act = jax.lax.psum(d_act, axis_name=vocab_axis_name)
    return d_act, d_emb

  # A 3D `[B, N, L]` `targets` is shared across the leading group axis of 4D
  # activations, so it needs its own spec: reusing `local_act_spec` would hand
  # `shard_map` a rank-4 spec for a rank-3 operand. Dropping the leading entry
  # also expresses the right semantics, replicating `targets` over whatever
  # mesh axis the group axis is sharded on.
  if targets.ndim == replicated_activations.ndim:
    local_tgt_spec = local_act_spec
  else:
    local_tgt_spec = jax.sharding.PartitionSpec(*local_act_spec[1:])  # pyrefly: ignore[unsupported-operation]

  d_act_replicated, d_emb = jax.shard_map(
      _bwd_local_with_reduction,
      mesh=mesh,
      in_specs=(
          jax.sharding.PartitionSpec(*local_act_spec[:-1]),  # pyrefly: ignore[unsupported-operation]
          local_act_spec,
          emb_spec,
          local_tgt_spec,
      ),
      out_specs=(
          local_act_spec,
          emb_spec,
      ),
      check_vma=False,
  )(
      d_loss,
      replicated_activations,
      embeddings,
      targets,
  )

  if hidden_axis_name is not None:
    d_activations = jax.lax.with_sharding_constraint(
        d_act_replicated, act_sharding
    )
  else:
    d_activations = d_act_replicated

  return d_activations, d_emb


def _focal_bce_bwd_loop_fallback(
    config: FocalBCEConfig,
    d_loss: jt.Float[jt.Array, "X B N"],
    activations: jt.Float[jt.Array, "X B N D"],
    embeddings: jt.Float[jt.Array, "V D"],
    targets: jt.Int[jt.Array, "X B N L"] | jt.Int[jt.Array, "B N L"],
    vocab_offset: int = 0,
) -> tuple[
    jt.Float[jt.Array, "X B N D"],
    jt.Float[jt.Array, "V D"],
]:
  """Sequential loop fallback over the group dimension.

  Applicable to activations with an extra dimension before the batch axis;
  `_focal_bce_bwd_pallas_chunked_n` only accepts a 3D `[B, N, D]` array. The
  groups are serialized by `_bce_bwd_scan_groups` so that only one `[V, D]`
  embedding gradient is live at a time.

  Args:
    config: Focal BCE config.
    d_loss: Gradient of the loss with respect to the per-position losses, of
      shape ``[X, B, N]``.
    activations: Hidden-state outputs of shape ``[X, B, N, D]``.
    embeddings: Output embedding / unembedding weights of shape ``[V, D]``.
    targets: Target token ids, either ``[X, B, N, L]`` or a 3D ``[B, N, L]``
      array shared across the group axis.
    vocab_offset: Vocab offset for the current chunk.

  Returns:
    Gradient of the loss with respect to the activations and embeddings.
  """
  return _bce_bwd_scan_groups(
      lambda d_loss_g, act_g, tgt_g: _focal_bce_bwd_pallas_chunked_n(
          config, d_loss_g, act_g, embeddings, tgt_g, vocab_offset
      ),
      d_loss,
      activations,
      embeddings,
      targets,
  )

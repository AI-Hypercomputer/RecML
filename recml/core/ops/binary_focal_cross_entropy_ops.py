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
_check_vocab_replicated_in_d = (
    bce_ops._check_vocab_replicated_in_d  # pylint: disable=protected-access
)
_get_sharding = bce_ops._get_sharding  # pylint: disable=protected-access
_max_safe_block_v = bce_ops._max_safe_block_v  # pylint: disable=protected-access
_max_safe_chunk_n = bce_ops._max_safe_chunk_n  # pylint: disable=protected-access
_pallas_interpret = bce_ops._pallas_interpret  # pylint: disable=protected-access
_pallas_lane = bce_ops._pallas_lane  # pylint: disable=protected-access
_pallas_vmem_budget = (
    bce_ops._pallas_vmem_budget  # pylint: disable=protected-access
)
_replicate_hidden_dim = (
    bce_ops._replicate_hidden_dim  # pylint: disable=protected-access
)
_BLOCK_N = bce_ops._BLOCK_N  # pylint: disable=protected-access


@dataclasses.dataclass
class FocalBCEConfig:
  """Configuration for the binary focal cross-entropy loss."""

  block_v: int
  block_n: int = _BLOCK_N
  compute_metrics: bool = False
  pure_jax: bool = True

  gamma: float = 2.0
  alpha: float = 0.25
  apply_class_balancing: bool = False
  # Sharding specs for VJP backward pass optimization
  mesh: jax.sharding.Mesh | None = None
  act_spec: jax.sharding.PartitionSpec | None = None
  emb_spec: jax.sharding.PartitionSpec | None = None


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
      actual_start = jnp.maximum(0, jnp.minimum(j * block_v, vocab - block_v))
      emb_chunk = jax.lax.dynamic_slice_in_dim(
          embeddings, actual_start, block_v
      )
      logits = jax.lax.dot_general(
          activations_2d,
          emb_chunk,
          (((1,), (1,)), ((), ())),
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.DEFAULT,
      )

      chunk_indices = actual_start + jnp.arange(block_v)
      valid_mask = (chunk_indices >= j * block_v) & (chunk_indices < vocab)

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
      loss_zero = jnp.maximum(logits, 0.0) + jnp.log1p(
          jnp.exp(-jnp.abs(logits))
      )
      bce_loss_chunk = loss_zero - targets_float * logits
      loss_chunk = focal_factor * bce_loss_chunk
      if config.apply_class_balancing:
        weight = targets_float * config.alpha + (1.0 - targets_float) * (
            1.0 - config.alpha
        )
        loss_chunk = weight * loss_chunk

      loss_chunk = loss_chunk * valid_mask[None, :]
      loss_sum = jnp.sum(loss_chunk, axis=-1)

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
      actual_start = jnp.maximum(0, jnp.minimum(j * block_v, vocab - block_v))
      emb_chunk = jax.lax.dynamic_slice_in_dim(
          embeddings, actual_start, block_v
      )
      logits = jax.lax.dot_general(
          activations_2d,
          emb_chunk,
          (((1,), (1,)), ((), ())),
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.DEFAULT,
      )

      chunk_indices = actual_start + jnp.arange(block_v)
      valid_mask = (chunk_indices >= j * block_v) & (chunk_indices < vocab)

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
      loss_zero = jnp.maximum(logits, 0.0) + jnp.log1p(
          jnp.exp(-jnp.abs(logits))
      )
      bce_loss_chunk = loss_zero - targets_float * logits
      loss_chunk = focal_factor * bce_loss_chunk
      if config.apply_class_balancing:
        weight = targets_float * config.alpha + (1.0 - targets_float) * (
            1.0 - config.alpha
        )
        loss_chunk = weight * loss_chunk

      loss_chunk = loss_chunk * valid_mask[None, :]
      loss_sum = jnp.sum(loss_chunk, axis=-1)
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
    mesh: jax.sharding.Mesh | None = None,
    act_spec: jax.sharding.PartitionSpec | None = None,
    emb_spec: jax.sharding.PartitionSpec | None = None,
    pure_jax: bool = True,
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
  """Computes binary focal cross entropy loss over unmaterialized logits."""
  vocab_size = embeddings.shape[0]

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
    n = activations.shape[-3] * activations.shape[-2]
    block_v = _auto_block_v(n, vocab_size)
  else:
    block_v = min(block_v, vocab_size)

  if return_metrics:
    losses, tp, fp, fn, tn = _cut_binary_focal_cross_entropy(
        FocalBCEConfig(
            block_v=block_v,
            compute_metrics=True,
            gamma=gamma,
            alpha=alpha,
            apply_class_balancing=apply_class_balancing,
            mesh=mesh,
            act_spec=act_spec,
            emb_spec=emb_spec,
            pure_jax=pure_jax,
        ),
        activations,
        embeddings,
        targets,
    )

    if weights is not None:
      losses = losses * weights
      weight_sum = jnp.sum(weights)
      tp_sum = jnp.sum(tp * weights)
      fp_sum = jnp.sum(fp * weights)
      fn_sum = jnp.sum(fn * weights)
      tn_sum = jnp.sum(tn * weights)
    else:
      weight_sum = np.prod(targets.shape[:-1])
      tp_sum = jnp.sum(tp)
      fp_sum = jnp.sum(fp)
      fn_sum = jnp.sum(fn)
      tn_sum = jnp.sum(tn)

    loss = jnp.sum(losses) / (weight_sum + EPS)

    if return_per_target_losses:
      return loss, losses, tp_sum, fp_sum, fn_sum, tn_sum
    return loss, tp_sum, fp_sum, fn_sum, tn_sum
  else:
    losses = _cut_binary_focal_cross_entropy(
        FocalBCEConfig(
            block_v=block_v,
            compute_metrics=False,
            gamma=gamma,
            alpha=alpha,
            apply_class_balancing=apply_class_balancing,
            mesh=mesh,
            act_spec=act_spec,
            emb_spec=emb_spec,
            pure_jax=pure_jax,
        ),
        activations,
        embeddings,
        targets,
    )

    if weights is not None:
      losses = losses * weights
      weight_sum = jnp.sum(weights)
    else:
      weight_sum = np.prod(targets.shape[:-1])

    loss = jnp.sum(losses) / (weight_sum + EPS)

    if return_per_target_losses:
      return loss, losses
    return loss


cut_binary_cross_entropy = cut_binary_focal_cross_entropy


def _focal_bce_bwd_kernel(
    emb_ref,  # [block_v, 128] VMEM
    act_ref,  # [padded_n, 256] HBM
    tgt_ref,  # [l_padded, padded_n] HBM
    dloss_ref,  # [l_padded, padded_n] HBM
    d_emb_ref,  # [block_v, 128] HBM output
    d_act_partials_ref,  # [1, padded_n, 256] HBM output
    d_emb_scratch,  # [block_v, 128] VMEM scratch
    *,
    block_v: int,
    block_n: int,
    n_blocks: int,
    vocab: int,
    vocab_offset: int,
    labels: int,
    n_real: int,
    gamma: float,
    alpha: float,
    apply_class_balancing: bool,
):
  """Pallas TPU per-shard chunked backward for Focal BCE."""
  v_idx = pl.program_id(0)

  # Initialize accumulators in VMEM to 0
  d_emb_scratch[...] = jnp.zeros(d_emb_scratch.shape, jnp.float32)

  # Valid vocab mask and global vocab ids: (1, block_v)
  chunk_indices_local = v_idx * block_v + jnp.arange(block_v)[None, :]
  valid_vocab_mask = chunk_indices_local < vocab
  valid_vocab_mask_t = valid_vocab_mask.T
  vocab_ids_global = chunk_indices_local + vocab_offset

  # Load inputs from VMEM
  emb = emb_ref[...]
  emb = jnp.where(valid_vocab_mask_t, emb, 0.0)

  # Precomputed inverse vocab, batch iota, and alpha balancing constants
  inv_vocab = 1.0 / vocab
  local_n_iota = jnp.arange(block_n)[:, None]
  alpha_diff = 2.0 * alpha - 1.0
  alpha_base = 1.0 - alpha

  def loop_body(n_idx, _):
    # Load act, tgt, dloss slices for this n_idx from HBM to VMEM
    act = act_ref[pl.ds(n_idx * block_n, block_n), :]
    tgt_val = tgt_ref[:labels, pl.ds(n_idx * block_n, block_n)]
    # (labels, block_n)
    dloss_val = dloss_ref[0, pl.ds(n_idx * block_n, block_n)]  # (block_n,)

    # Compute logits: (block_n, block_v)
    logits = jax.lax.dot_general(
        act,
        emb,
        (((1,), (1,)), ((), ())),
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32,
    )
    probs = jax.nn.sigmoid(logits)

    # Target matching across labels: (block_n, block_v)
    y_true_chunk = jnp.any(tgt_val[:, :, None] == vocab_ids_global, axis=0)

    # Valid batch mask for this loop step: (block_n, 1)
    batch_indices_local = n_idx * block_n + local_n_iota
    valid_batch_mask = batch_indices_local < n_real

    # Gradients w.r.t logits, masked
    y_true_float = y_true_chunk.astype(probs.dtype)
    g_bce = probs - y_true_float
    abs_g = jnp.abs(g_bce)
    p_t = 1.0 - abs_g
    focal_factor = jnp.power(abs_g, gamma)
    focal_factor_m1 = jnp.where(
        gamma == 0.0,
        0.0,
        jnp.power(abs_g, jnp.maximum(0.0, gamma - 1.0)),
    )
    loss_zero = jnp.maximum(logits, 0.0) + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    bce_loss_chunk = loss_zero - y_true_float * logits
    g_focal = g_bce * (
        focal_factor + gamma * focal_factor_m1 * p_t * bce_loss_chunk
    )
    if apply_class_balancing:
      weight = y_true_float * alpha_diff + alpha_base
      g_focal = weight * g_focal

    scale = jnp.where(valid_batch_mask, (dloss_val * inv_vocab)[:, None], 0.0)
    deriv = scale * g_focal

    # Accumulate d_emb
    d_emb_contrib = jax.lax.dot_general(
        deriv.astype(act.dtype),
        act,
        (((0,), (0,)), ((), ())),
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32,
    )
    d_emb_scratch[...] = d_emb_scratch[...] + d_emb_contrib

    # Compute d_act and write directly to HBM
    d_act_contrib = jax.lax.dot_general(
        deriv.astype(emb.dtype),
        emb,
        (((1,), (0,)), ((), ())),
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

  padded_d = ((hidden + 127) // 128) * 128
  vmem_budget = _pallas_vmem_budget()
  chunk_n = _max_safe_chunk_n(vmem_budget, padded_d)
  if n <= chunk_n:
    return _focal_bce_bwd_pallas(
        config, d_loss, activations, embeddings, targets, vocab_offset
    )
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

  kernel_config = FocalBCEConfig(
      block_v=config.block_v,
      block_n=bce_ops._BLOCK_N,  # pylint:disable=protected-access
      compute_metrics=config.compute_metrics,
      gamma=config.gamma,
      alpha=config.alpha,
      apply_class_balancing=config.apply_class_balancing,
  )

  def loop_body(carry, x):
    (d_emb_acc,) = carry
    act_chunk, dloss_chunk, tgt_chunk = x

    d_act_chunk, d_emb_contrib = _focal_bce_bwd_pallas(
        kernel_config,
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
  padded_d = ((hidden + 127) // 128) * 128
  vocab = embeddings.shape[0]
  vmem_budget = _pallas_vmem_budget()
  max_safe = _max_safe_block_v(vmem_budget, padded_d)
  vocab_padded_128 = ((vocab + 127) // 128) * 128
  block_v = min(config.block_v, vocab, max_safe)
  block_v = ((block_v + 127) // 128) * 128
  block_v = min(block_v, vocab_padded_128, max_safe)
  n = batch * seq_len
  n_blocks = (n + block_n - 1) // block_n
  v_blocks = (vocab + block_v - 1) // block_v
  vocab_padded = v_blocks * block_v
  labels = targets.shape[-1]

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
  l_padded = max(8, ((labels + 7) // 8) * 8)
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
          vocab_offset=vocab_offset,
          labels=labels,
          n_real=n,
          gamma=config.gamma,
          alpha=config.alpha,
          apply_class_balancing=config.apply_class_balancing,
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
    groups = activations.shape[0]
    d_acts = []
    d_emb = jnp.zeros_like(embeddings)
    for g in range(groups):
      tgt_g = targets if targets.ndim == 3 else targets[g]
      d_act_g, d_emb_g = _focal_bce_bwd_pure_jax(
          config, d_loss[g], activations[g], embeddings, tgt_g, vocab_offset
      )
      d_acts.append(d_act_g)
      d_emb = d_emb + d_emb_g
    return jnp.stack(d_acts, axis=0), d_emb

  batch, seq_len, hidden = activations.shape
  vocab, _ = embeddings.shape
  n = batch * seq_len
  inv_vocab = 1.0 / vocab
  block_v = config.block_v

  activations_2d = jnp.reshape(activations, (n, hidden))
  targets_2d = jnp.reshape(targets, (n, -1))
  dloss_2d = jnp.reshape(d_loss, (n, 1))

  # Chunked scan backward pass without tensor padding
  v_blocks = int(np.ceil(vocab / block_v))

  def scan_body(carry, j):
    d_act_acc, d_emb_acc = carry
    actual_start = jnp.maximum(0, jnp.minimum(j * block_v, vocab - block_v))
    emb_chunk = jax.lax.dynamic_slice_in_dim(embeddings, actual_start, block_v)

    logits = jax.lax.dot_general(
        activations_2d,
        emb_chunk,
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.DEFAULT,
    )
    probs = jax.nn.sigmoid(logits)

    chunk_indices = actual_start + jnp.arange(block_v)
    valid_mask = (chunk_indices >= j * block_v) & (chunk_indices < vocab)

    targets_chunk = jnp.zeros((n, block_v), dtype=jnp.bool_)
    rel_targets = targets_2d - (actual_start + vocab_offset)
    chunk_cols = jnp.arange(block_v)[None, :]
    for l_idx in range(targets_2d.shape[-1]):
      targets_chunk = targets_chunk | (
          rel_targets[:, l_idx : l_idx + 1] == chunk_cols
      )
    g_bce = jnp.where(targets_chunk, probs - 1.0, probs)
    p_t = jnp.where(targets_chunk, probs, 1.0 - probs)
    focal_factor = jnp.power(1.0 - p_t, config.gamma)
    focal_factor_m1 = jnp.where(
        config.gamma == 0.0,
        0.0,
        jnp.power(1.0 - p_t, jnp.maximum(0.0, config.gamma - 1.0)),
    )
    loss_zero = jnp.maximum(logits, 0.0) + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    bce_loss_chunk = jnp.where(targets_chunk, loss_zero - logits, loss_zero)
    g_focal = g_bce * (
        focal_factor + config.gamma * focal_factor_m1 * p_t * bce_loss_chunk
    )
    if config.apply_class_balancing:
      weight = jnp.where(targets_chunk, config.alpha, 1.0 - config.alpha)
      g_focal = weight * g_focal

    d_logits_chunk = g_focal * valid_mask[None, :] * (dloss_2d * inv_vocab)

    d_act_contrib = jax.lax.dot_general(
        d_logits_chunk,
        emb_chunk,
        (((1,), (0,)), ((), ())),
        preferred_element_type=activations.dtype,
        precision=jax.lax.Precision.DEFAULT,
    )
    d_emb_chunk = jax.lax.dot_general(
        d_logits_chunk,
        activations_2d,
        (((0,), (0,)), ((), ())),
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
    emb_spec = config.emb_spec
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
    if config.pure_jax:
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

  _check_vocab_replicated_in_d(emb_spec)  # pyrefly: ignore[bad-argument-type]

  hidden_axis_name = act_spec[-1]  # pyrefly: ignore[unsupported-operation]
  if hidden_axis_name is not None:
    replicated_activations = _replicate_hidden_dim(activations)
    new_spec_list = list(act_spec)  # pyrefly: ignore[bad-argument-type]
    new_spec_list[-1] = None
    local_act_spec = jax.sharding.PartitionSpec(*new_spec_list)
  else:
    replicated_activations = activations
    local_act_spec = act_spec

  vocab_axis_name = emb_spec[0]  # pyrefly: ignore[unsupported-operation]

  dp_axes = []
  for axis in local_act_spec:  # pyrefly: ignore[not-iterable]
    if axis is not None and axis != vocab_axis_name:
      dp_axes.append(axis)

  def _bwd_local_with_reduction(d_loss_, act_, emb_, tgt_):
    if vocab_axis_name is not None:
      vocab_offset = jax.lax.axis_index(vocab_axis_name) * emb_.shape[0]
    else:
      vocab_offset = 0
    if config.pure_jax:
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

  d_act_replicated, d_emb = jax.shard_map(
      _bwd_local_with_reduction,
      mesh=mesh,
      in_specs=(
          jax.sharding.PartitionSpec(*local_act_spec[:-1]),  # pyrefly: ignore[unsupported-operation]
          local_act_spec,
          emb_spec,
          local_act_spec,
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
    d_loss: jt.Float[jt.Array, "... B N"],
    activations: jt.Float[jt.Array, "... B N D"],
    embeddings: jt.Float[jt.Array, "V D"],
    targets: jt.Int[jt.Array, "... B N L"],
    vocab_offset: int = 0,
) -> tuple[
    jt.Float[jt.Array, "... B N D"],
    jt.Float[jt.Array, "V D"],
]:
  """Sequential loop fallback over group dimension.

  Applicable to activations with an extra dimension reshaped before batch axis.

  Args:
    config: Focal BCE config.
    d_loss: Gradient of the loss with respect to the logits.
    activations: Hidden-state outputs of shape ``[B, N, D]``.
    embeddings: Output embedding / unembedding weights of shape ``[V, D]``.
    targets: Target token ids of shape ``[B, N, L]``.
    vocab_offset: Vocab offset for the current chunk.

  Returns:
    Gradient of the loss with respect to the activations and embeddings.
  """
  groups = activations.shape[0]
  d_acts = []
  d_emb = jnp.zeros_like(embeddings)
  for g in range(groups):
    tgt_g = targets if targets.ndim == 3 else targets[g]
    d_act_g, d_emb_g = _focal_bce_bwd_pallas_chunked_n(
        config,
        d_loss[g],
        activations[g],
        embeddings,
        tgt_g,
        vocab_offset,
    )
    d_acts.append(d_act_g)
    d_emb += d_emb_g
  return jnp.stack(d_acts, axis=0), d_emb

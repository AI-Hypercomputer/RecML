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
"""Binary cross-entropy loss implementation with optimized memory footprint.

This implementation computes BCE loss without materializing the [B, N, V] logits
matrix in memory, by chunking the vocabulary dimension.
"""

from collections.abc import Callable
import dataclasses
import functools
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np

EPS = 1e-8
_BYTES_IN_MB = 1024 * 1024

# Hyperparameters for Pallas VMEM budgeting and inner-loop tile sizing.
#
# _RESERVE_MB:
#   The safety headroom (in megabytes) subtracted from the live TPU's total
#   hardware VMEM capacity (pltpu.get_tpu_info().vmem_capacity_bytes) to derive
#   the usable Pallas allocation budget (_pallas_vmem_budget). TPU VMEM is
#   shared between user-declared scratch buffers and compiler-generated
#   artifacts. Subtracting 16 MB leaves 112 MB on 128 MB chips (TPU v5e and v6e)
#   and 48 MB on 64 MB chips (TPU v5p and TPU 7/GF) for large vocabulary blocks
#   while preventing compiler layout-assignment VMEM OOMs. On legacy 16 MB VMEM
#   chips where subtracting 16 MB would leave zero, _pallas_vmem_budget floors
#   at 16 MB.
#
# _TARGET_RATIO:
#   The fraction of available VMEM budget allocated to the sequence/token
#   dimension (N = batch * seq_len) in _max_safe_chunk_n when chip total VMEM
#   exceeds 32 MB. The conservative memory formula assumes triple-buffering
#   (1 load + 2 stores) across activations and gradient accumulators. Allocating
#   80% (0.8) maximizes chunk_n, allowing standard recommendation batches
#   to execute in a single Pallas kernel call without falling back to an outer
#   JAX scan loop. On <=32 MB chips, this falls back to _SMALL_VMEM_SHARE.
#
# _BLOCK_N:
#   The inner-loop token/sequence tile size in _bce_bwd_kernel. Default to 128
#   for hardware alignment: 128 matches the TPU vector lane width
#   (num_lanes = 128) across all TPU generations, ensuring slices along N map
#   directly to full vector registers without partial-lane edge masking. This
#   constant is only the fallback used when no TPU is attached; the live lane
#   width is read from tpu_info via _pallas_lane().
#
# VMEM tile inventory (_VOCAB_VMEM_TILES / _LOGIT_VMEM_TILES /
# _TOKEN_VMEM_TILES):
#   Counts of the live buffers _bce_bwd_kernel keeps in VMEM at once, so that
#   the sizing helpers below can derive block sizes from the chip's reported
#   VMEM capacity instead of byte products measured on one generation.
#   - _VOCAB_VMEM_TILES: per-block_v rows of width padded_d, namely the
#     embedding block (emb_ref), the gradient output block (d_emb_ref), the
#     float32 accumulator (d_emb_scratch) and the two staging buffers Mosaic
#     adds for the grid's double-buffered DMAs.
#   - _LOGIT_VMEM_TILES: per-block_v columns of height block_n, namely the
#     [block_n, block_v] logits intermediate plus the sigmoid, target-match and
#     gradient temporaries derived from it inside loop_body.
#   - _TOKEN_VMEM_TILES: per-token rows of width padded_d that stay resident
#     for the whole vocabulary grid: the activation block plus the two
#     d_act_partials staging buffers (1 load + 2 stores).
#
# _VOCAB_VMEM_SHARE / _SMALL_VMEM_SHARE:
#   Fraction of the VMEM budget the vocabulary-side working set may claim. The
#   remainder is left for the token-side buffers and compiler temporaries. On
#   <=32 MB chips this drops to _SMALL_VMEM_SHARE because fixed compiler
#   overhead consumes a much larger fraction of a small VMEM.
#
# _MIN_CHUNK_TILES:
#   Floor on chunk_n, expressed in whole vector-lane tiles rather than a raw
#   element count: a smaller chunk cannot amortize kernel launch and DMA
#   dispatch overhead.

_RESERVE_MB: float = 16.0
_TARGET_RATIO: float = 0.8
_MEDIUM_TARGET_RATIO: float = 0.65
_BLOCK_N: int = 128
_FWD_MAX_CHUNK_N: int = 2048
_MAX_SMEM_POS_ENTRIES: int = 163_840
_MAX_PACKED_DIM: int = 32_768
_VOCAB_VMEM_TILES: int = 5
_LOGIT_VMEM_TILES: int = 4
_TOKEN_VMEM_TILES: int = 3
_VOCAB_VMEM_SHARE: float = 0.5
_MEDIUM_VOCAB_VMEM_SHARE: float = 0.42
_SMALL_VMEM_SHARE: float = 0.3
_MIN_CHUNK_TILES: int = 4


def _get_mxu_size() -> int:
  """Returns the MXU tile alignment size based on the active TPU generation."""
  if not any(d.platform == "tpu" for d in jax.devices()):
    return 128

  tpu_info = pltpu.get_tpu_info()
  chip = str(tpu_info.chip_version)
  match chip:
    case "v2" | "v3" | "v4" | "v4i" | "v5e" | "v5p":
      return 128
    case "v6e" | "7" | "7x" | "8i" | "8t":
      return 256
    case _:
      raise NotImplementedError(
          f"Unsupported TPU chip version: {chip}. Please explicitly verify MXU "
          "systolic dimensions and extend _get_mxu_size."
      )


def _auto_block_v(
    local_n: int, vocab_size: int, dtype: jax.typing.DTypeLike = jnp.float32
) -> int:
  """Automatically picks block_v for intermediate logits."""
  local_n = max(local_n, 1)

  # Scale target memory based on chip VMEM capacity:
  # - On <=32 MB chips (TPU v3/v4): target 4 MB to leave ample headroom.
  # - On 64 MB chips (TPU v5p, TPU 7): target 16 MB.
  # - On >=128 MB chips (TPU v5e, TPU v6e, TPU v8i, TPU v8t): target 32 MB.
  if any(d.platform == "tpu" for d in jax.devices()):
    vmem_bytes = pltpu.get_tpu_info().vmem_capacity_bytes
    if vmem_bytes <= 32 * _BYTES_IN_MB:
      target_bytes = 4 * _BYTES_IN_MB
    elif vmem_bytes <= 64 * _BYTES_IN_MB:
      target_bytes = 16 * _BYTES_IN_MB
    else:
      target_bytes = 32 * _BYTES_IN_MB
  else:
    target_bytes = 32 * _BYTES_IN_MB

  bytes_per_element = jnp.dtype(dtype).itemsize
  target_elements = target_bytes // bytes_per_element
  block_v = target_elements // local_n

  # Align to MXU systolic dimension (128 for legacy TPUs, 256 for TPU v6e+)
  mxu_size = _get_mxu_size()
  block_v = max(mxu_size, (block_v // mxu_size) * mxu_size)
  # Don't exceed vocab_size
  block_v = min(vocab_size, block_v)
  return block_v


def _get_sharding(x: jt.ArrayLike) -> jax.sharding.Sharding | None:
  if hasattr(x, "sharding"):
    return x.sharding
  if hasattr(x, "aval") and hasattr(x.aval, "sharding"):
    return x.aval.sharding
  return None


def _token_shard_count(
    x: jt.Float[jt.Array, "... D"],
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None,
    spec: jax.sharding.PartitionSpec | None = None,
) -> int | None:
  """Returns how many ways the token axes of `x` are sharded, or None."""
  if mesh is None or spec is None:
    sharding = _get_sharding(x)
    if not isinstance(sharding, jax.sharding.NamedSharding):
      return None
    mesh, spec = sharding.mesh, sharding.spec
  if mesh.empty:
    return None

  # A PartitionSpec may be shorter than `x.ndim`; trailing axes are unsharded.
  factor = 1
  for axis in tuple(spec)[: x.ndim - 1]:
    if axis is None:
      continue
    for name in axis if isinstance(axis, tuple) else (axis,):
      if name not in mesh.shape:  # e.g. PartitionSpec.UNCONSTRAINED
        return None
      factor *= mesh.shape[name]

  if factor == 1 and isinstance(x, jax.core.Tracer):
    return None
  return factor


def _local_tokens(
    x: jt.Float[jt.Array, "... D"],
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None,
    spec: jax.sharding.PartitionSpec | None = None,
) -> int:
  """Estimates the number of logits rows a single device holds."""
  n = int(np.prod(x.shape[:-1]))
  shards = _token_shard_count(x, mesh, spec)
  if shards is None:
    shards = jax.device_count()
  return max(n // shards, 1)


def _replicate_hidden_dim(
    x: jt.Float[jt.Array, "... D"],
) -> jt.Float[jt.Array, "... D"]:
  """Replicates the hidden dimension of the input tensor if sharded."""
  sharding = _get_sharding(x)
  if isinstance(sharding, jax.sharding.NamedSharding):
    mesh = sharding.mesh
    if not mesh.empty:
      spec = sharding.spec
      new_spec_list = list(spec) + [None] * (x.ndim - len(spec))
      if new_spec_list:
        new_spec_list[-1] = None
      new_spec = jax.sharding.PartitionSpec(*new_spec_list)
      return jax.lax.with_sharding_constraint(
          x, jax.sharding.NamedSharding(mesh, new_spec)
      )
  return x


@dataclasses.dataclass(frozen=True)
class BCEConfig:
  """Configuration for the binary cross-entropy loss.

  This config is passed as a static (non-differentiable) argument to the
  `jax.custom_vjp` wrapped loss, so it is immutable and hashable.

  Attributes:
    block_v: Vocabulary-axis block size used to chunk the logits computation so
      the full ``[N, V]`` logits matrix is never materialized.
    block_n: Sequence/token-axis (``N = B * seq_len``) tile size of the Pallas
      backward kernel inner loop.
    compute_metrics: Whether the forward pass also accumulates the TP, FP, FN
      and TN counts alongside the loss.
    use_pallas: Whether to use the Pallas TPU kernels for the backward pass. If
      False, a pure JAX implementation is used instead.
    mesh: Optional device mesh describing how the inputs are sharded. Used
      together with `act_spec` and `emb_spec` to shard the backward pass; if
      None, the sharding is inferred from the input arrays.
    act_spec: Optional partition spec of the activations, of the same rank as
      the activations.
    emb_spec: Optional partition spec of the embeddings. Sharding along the
      hidden dimension ``D`` is not supported.
    global_vocab: Size of the full vocabulary. Under `_bce_bwd_sharded` the
      backward helpers run inside a `shard_map` and only observe this shard's
      slice of the embedding table, but the forward pass normalizes the loss by
      the full vocabulary, so the backward must use this value rather than
      ``embeddings.shape[0]``.
    optimize_large_l: Whether to use the target-free vocabulary pass + sparse
      positive-target correction. Defaults to False.
    force_bwd_block_v: Whether to force the backward pass to use `block_v`
      directly (aligned to lane width and capped by vocabulary size) instead of
      clamping to `_max_safe_block_v`. Defaults to False.
  """

  block_v: int
  block_n: int = _BLOCK_N
  compute_metrics: bool = False
  use_pallas: bool = True
  global_vocab: int | None = None
  optimize_large_l: bool = False
  force_bwd_block_v: bool = False

  # Sharding specs for VJP backward pass optimization
  mesh: jax.sharding.Mesh | None = None
  act_spec: jax.sharding.PartitionSpec | None = None
  emb_spec: jax.sharding.PartitionSpec | None = None


def _unique_valid_targets(
    targets_2d: jt.Int[jt.Array, "N L"],
    vocab: int,
) -> tuple[jt.Int[jt.Array, "N L"], jt.Bool[jt.Array, "N L"]]:
  """Returns sorted in-bounds target indices in [0, vocab) and a unique mask."""
  sorted_t = jnp.sort(targets_2d, axis=-1)
  valid_pos = (sorted_t >= 0) & (sorted_t < vocab)
  if targets_2d.shape[-1] > 1:
    valid_pos = valid_pos & jnp.concatenate(
        [jnp.ones_like(valid_pos[:, :1]), sorted_t[:, 1:] != sorted_t[:, :-1]],
        axis=-1,
    )
  return jnp.where(valid_pos, sorted_t, 0), valid_pos


def _fwd_pos_correction(
    activations_2d: jt.Float[jt.Array, "N D"],
    embeddings: jt.Float[jt.Array, "V D"],
    targets_2d: jt.Int[jt.Array, "N L"],
    loss_diff_fn: (
        Callable[[jt.Float[jt.Array, "N L"]], jt.Float[jt.Array, "N L"]] | None
    ) = None,
) -> tuple[
    jt.Float[jt.Array, "N"],
    jt.Float[jt.Array, "N"],
    jt.Float[jt.Array, "N"],
]:
  """Returns (delta_loss_sum, tp_pos, num_pos) for positive targets."""
  safe_t, valid_pos = _unique_valid_targets(targets_2d, embeddings.shape[0])
  pos_logits = jnp.einsum(
      "nd,nld->nl",
      activations_2d,
      embeddings[safe_t],
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.DEFAULT,
  )
  delta = -pos_logits if loss_diff_fn is None else loss_diff_fn(pos_logits)
  delta_loss_sum = jnp.sum(jnp.where(valid_pos, delta, 0.0), axis=-1)
  tp_pos = jnp.sum((pos_logits > 0.0) & valid_pos, axis=-1).astype(jnp.float32)
  num_pos = jnp.sum(valid_pos, axis=-1).astype(jnp.float32)
  return delta_loss_sum, tp_pos, num_pos


def _bce_fwd_chunk(
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
  """Computes logits and base fused BCE loss for a single vocabulary chunk."""
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
  # Fused BCE Loss: BCE(x, y) = BCE(x, 0) - y * x
  loss_zero = jnp.maximum(logits, 0.0) + jnp.log1p(jnp.exp(-jnp.abs(logits)))

  n = activations_2d.shape[0]
  targets_chunk = jnp.zeros((n, block_v), dtype=jnp.bool_)
  rel_targets = targets_2d - actual_start
  chunk_cols = jnp.arange(block_v)[None, :]
  for l_idx in range(targets_2d.shape[-1]):
    targets_chunk = targets_chunk | (
        rel_targets[:, l_idx : l_idx + 1] == chunk_cols
    )

  loss_chunk = loss_zero - targets_chunk * logits
  loss_chunk = loss_chunk * valid_mask[None, :]
  loss_sum = jnp.sum(loss_chunk, axis=-1)
  return loss_sum, logits, targets_chunk, valid_mask


def _select_fwd_block_n(chunk_n: int, lane: int) -> int:
  """Picks a pipelined token sub-block dividing `chunk_n` for forward MXU util."""
  for candidate in (512, 256):
    if candidate >= lane and chunk_n >= candidate and chunk_n % candidate == 0:
      return candidate
  return min(chunk_n, lane)


def _bce_fwd_kernel(
    act_ref: jax.Ref,  # [chunk_n, padded_d] VMEM
    emb_ref: jax.Ref,  # [block_v, padded_d] VMEM
    tgt_ref: jax.Ref | None,  # [l_padded, chunk_n] VMEM or None
    loss_out_ref: jax.Ref,  # [num_sublanes, chunk_n] VMEM output
    tp_out_ref: jax.Ref | None,
    fp_out_ref: jax.Ref | None,
    fn_out_ref: jax.Ref | None,
    tn_out_ref: jax.Ref | None,
    loss_scratch: jax.Ref,  # [num_sublanes, chunk_n] VMEM scratch
    tp_scratch: jax.Ref | None,
    fp_scratch: jax.Ref | None,
    fn_scratch: jax.Ref | None,
    tn_scratch: jax.Ref | None,
    *,
    chunk_n: int,
    fwd_block_n: int,
    block_v: int,
    v_blocks: int,
    vocab: int,
    labels: int,
    compute_metrics: bool,
) -> None:
  """2D-tiled Pallas TPU forward kernel with full-block MXU/VPU pipelining."""
  j = pl.program_id(1)
  num_sublanes = loss_scratch.shape[0]

  @pl.when(j == 0)
  def _init_accumulators():
    z = jnp.zeros(loss_scratch.shape, jnp.float32)
    loss_scratch[...] = z
    if compute_metrics:
      assert fp_scratch is not None
      fp_scratch[...] = z
      if labels > 0:
        assert (
            tp_scratch is not None
            and fn_scratch is not None
            and tn_scratch is not None
        )
        tp_scratch[...] = z
        fn_scratch[...] = z
        tn_scratch[...] = z

  n_sub_blocks = chunk_n // fwd_block_n
  col_ids = j * block_v + jax.lax.broadcasted_iota(jnp.int32, (1, block_v), 1)
  valid_v = col_ids < vocab if vocab % block_v != 0 else None
  emb_block = (
      jnp.where(valid_v.T, emb_ref[...], 0.0)
      if valid_v is not None
      else emb_ref[...]
  )

  def n_loop_body(nc: jax.Array, _: None) -> None:
    n_slice = pl.ds(nc * fwd_block_n, fwd_block_n)
    act_sub = act_ref[n_slice, :]
    logits = jax.lax.dot(
        act_sub,
        emb_block,
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.DEFAULT,
    )
    loss_chunk = jnp.maximum(logits, 0.0) + jnp.log1p(jnp.exp(-jnp.abs(logits)))

    y_true = None
    if labels > 0:
      assert tgt_ref is not None
      tgt_sub = tgt_ref[:, n_slice]
      y_true = jnp.zeros((fwd_block_n, block_v), dtype=jnp.bool_)
      for l_idx in range(labels):
        y_true = y_true | (tgt_sub[l_idx : l_idx + 1, :].T == col_ids)
      loss_chunk = loss_chunk - jnp.where(y_true, logits, 0.0)

    if valid_v is not None:
      loss_chunk = jnp.where(valid_v, loss_chunk, 0.0)

    loss_1d = jnp.sum(loss_chunk, axis=-1)
    loss_scratch[:, n_slice] = loss_scratch[:, n_slice] + jnp.broadcast_to(
        loss_1d[None, :], (num_sublanes, fwd_block_n)
    )
    if not compute_metrics:
      return

    preds = logits > 0.0
    if valid_v is not None:
      preds = preds & valid_v
    assert fp_scratch is not None
    if labels == 0:
      fp_1d = jnp.sum(preds, axis=-1).astype(jnp.float32)
      fp_scratch[:, n_slice] = fp_scratch[:, n_slice] + jnp.broadcast_to(
          fp_1d[None, :], (num_sublanes, fwd_block_n)
      )
      return

    assert (
        y_true is not None
        and tp_scratch is not None
        and fn_scratch is not None
        and tn_scratch is not None
    )
    tgt_masked = y_true & valid_v if valid_v is not None else y_true
    tp_chunk = tgt_masked & preds
    fp_chunk = preds ^ tp_chunk
    fn_chunk = tgt_masked ^ tp_chunk
    tn_unmasked = ~(tgt_masked | preds)
    tn_chunk = tn_unmasked & valid_v if valid_v is not None else tn_unmasked

    tp_1d = jnp.sum(tp_chunk, axis=-1).astype(jnp.float32)
    fp_1d = jnp.sum(fp_chunk, axis=-1).astype(jnp.float32)
    fn_1d = jnp.sum(fn_chunk, axis=-1).astype(jnp.float32)
    tn_1d = jnp.sum(tn_chunk, axis=-1).astype(jnp.float32)
    tp_scratch[:, n_slice] = tp_scratch[:, n_slice] + jnp.broadcast_to(
        tp_1d[None, :], (num_sublanes, fwd_block_n)
    )
    fp_scratch[:, n_slice] = fp_scratch[:, n_slice] + jnp.broadcast_to(
        fp_1d[None, :], (num_sublanes, fwd_block_n)
    )
    fn_scratch[:, n_slice] = fn_scratch[:, n_slice] + jnp.broadcast_to(
        fn_1d[None, :], (num_sublanes, fwd_block_n)
    )
    tn_scratch[:, n_slice] = tn_scratch[:, n_slice] + jnp.broadcast_to(
        tn_1d[None, :], (num_sublanes, fwd_block_n)
    )

  if n_sub_blocks == 1:
    n_loop_body(jnp.int32(0), None)
  else:
    jax.lax.fori_loop(0, n_sub_blocks, n_loop_body, None)

  @pl.when(j == v_blocks - 1)
  def _store_outputs():
    loss_out_ref[...] = loss_scratch[...]
    if compute_metrics:
      assert fp_out_ref is not None and fp_scratch is not None
      fp_out_ref[...] = fp_scratch[...]
      if labels > 0:
        assert (
            tp_out_ref is not None
            and fn_out_ref is not None
            and tn_out_ref is not None
            and tp_scratch is not None
            and fn_scratch is not None
            and tn_scratch is not None
        )
        tp_out_ref[...] = tp_scratch[...]
        fn_out_ref[...] = fn_scratch[...]
        tn_out_ref[...] = tn_scratch[...]


def _bce_fwd_pallas(
    config: BCEConfig,
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
  """Computes forward BCE loss and optional metrics using a 2D Pallas kernel."""
  batch, seq_len, hidden = activations.shape
  vocab = embeddings.shape[0]
  n = batch * seq_len
  lane = _pallas_lane()
  num_sublanes = _pallas_sublane()

  if n <= _FWD_MAX_CHUNK_N:
    chunk_n = pl.align_to(max(n, lane), lane)
  else:
    chunk_n = _balanced_chunk_n(n, _FWD_MAX_CHUNK_N)
  n_outer = (n + chunk_n - 1) // chunk_n
  padded_n = n_outer * chunk_n
  fwd_block_n = _select_fwd_block_n(chunk_n, lane)

  fwd_cfg = dataclasses.replace(config, block_n=fwd_block_n)
  padded_d, block_v = _effective_block_v(fwd_cfg, hidden, vocab)
  v_blocks = (vocab + block_v - 1) // block_v
  vocab_hbm = (
      v_blocks * block_v
      if _pallas_interpret()
      else max(pl.align_to(vocab, num_sublanes), block_v)
  )

  labels = 0 if config.optimize_large_l else targets.shape[-1]
  activations_2d = jnp.reshape(activations, (n, hidden))
  targets_2d = jnp.reshape(targets, (n, -1))

  if padded_n > n or padded_d > hidden:
    activations_padded = jnp.pad(
        activations_2d, ((0, padded_n - n), (0, padded_d - hidden))
    )
  else:
    activations_padded = activations_2d

  if vocab_hbm > vocab or padded_d > hidden:
    embeddings_padded = jnp.pad(
        embeddings, ((0, vocab_hbm - vocab), (0, padded_d - hidden))
    )
  else:
    embeddings_padded = embeddings

  if labels > 0:
    l_padded = max(num_sublanes, pl.align_to(labels, num_sublanes))
    targets_t = jnp.transpose(targets_2d[:, :labels], (1, 0))
    if l_padded > labels or padded_n > n:
      targets_padded = jnp.pad(
          targets_t,
          ((0, l_padded - labels), (0, padded_n - n)),
          constant_values=-1,
      ).astype(jnp.int32)
    else:
      targets_padded = targets_t.astype(jnp.int32)
    tgt_spec = pl.BlockSpec((l_padded, chunk_n), lambda i, j: (0, i))
  else:
    targets_padded = None
    tgt_spec = None

  vec_shape = jax.ShapeDtypeStruct((num_sublanes, padded_n), jnp.float32)
  vec_spec = pl.BlockSpec((num_sublanes, chunk_n), lambda i, j: (0, i))
  vec_vmem = pltpu.VMEM((num_sublanes, chunk_n), jnp.float32)

  if not config.compute_metrics:
    out_shape = [vec_shape, None, None, None, None]
    out_specs = [vec_spec, None, None, None, None]
    scratch_shapes = [vec_vmem, None, None, None, None]
  elif labels == 0:
    out_shape = [vec_shape, None, vec_shape, None, None]
    out_specs = [vec_spec, None, vec_spec, None, None]
    scratch_shapes = [vec_vmem, None, vec_vmem, None, None]
  else:
    out_shape = [vec_shape, vec_shape, vec_shape, vec_shape, vec_shape]
    out_specs = [vec_spec, vec_spec, vec_spec, vec_spec, vec_spec]
    scratch_shapes = [vec_vmem, vec_vmem, vec_vmem, vec_vmem, vec_vmem]

  loss_out, tp_out, fp_out, fn_out, tn_out = pl.pallas_call(
      functools.partial(
          _bce_fwd_kernel,
          chunk_n=chunk_n,
          fwd_block_n=fwd_block_n,
          block_v=block_v,
          v_blocks=v_blocks,
          vocab=vocab,
          labels=labels,
          compute_metrics=config.compute_metrics,
      ),
      out_shape=out_shape,
      grid=(n_outer, v_blocks),
      in_specs=[
          pl.BlockSpec((chunk_n, padded_d), lambda i, j: (i, 0)),
          pl.BlockSpec((block_v, padded_d), lambda i, j: (j, 0)),
          tgt_spec,
      ],
      out_specs=out_specs,
      scratch_shapes=scratch_shapes,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          vmem_limit_bytes=_pallas_vmem_budget(),
      ),
      interpret=_pallas_interpret(),
  )(activations_padded, embeddings_padded, targets_padded)

  loss_final = loss_out[0, :n]
  if not config.compute_metrics:
    if config.optimize_large_l:
      loss_final = (
          loss_final
          + _fwd_pos_correction(activations_2d, embeddings, targets_2d)[0]
      )
    return jnp.reshape(loss_final, (batch, seq_len))

  assert fp_out is not None
  fp_final = fp_out[0, :n]
  if labels == 0:
    delta_loss, tp_final, num_pos = _fwd_pos_correction(
        activations_2d, embeddings, targets_2d
    )
    loss_final = loss_final + delta_loss
    tn_zero = float(vocab) - fp_final
    fp_final = fp_final - tp_final
    fn_final = num_pos - tp_final
    tn_final = tn_zero - fn_final
  else:
    assert tp_out is not None and fn_out is not None and tn_out is not None
    tp_final = tp_out[0, :n]
    fn_final = fn_out[0, :n]
    tn_final = tn_out[0, :n]

  return (
      jnp.reshape(loss_final, (batch, seq_len)),
      jnp.reshape(tp_final, (batch, seq_len)),
      jnp.reshape(fp_final, (batch, seq_len)),
      jnp.reshape(fn_final, (batch, seq_len)),
      jnp.reshape(tn_final, (batch, seq_len)),
  )


def _bce_fwd_local(
    config: BCEConfig,
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
  """Computes the sum of Loss(x_v, target_v) over all V block-wise, and metrics."""
  if config.use_pallas and not _pallas_interpret():
    return _bce_fwd_pallas(config, activations, embeddings, targets)

  block_v = config.block_v
  batch, seq_len, hidden = activations.shape
  vocab = embeddings.shape[0]

  n = batch * seq_len
  # NOMUTANTS -- v_blocks is calculated from block_v and vocab.
  v_blocks = int(np.ceil(vocab / block_v))

  activations_2d = jnp.reshape(activations, (n, hidden))
  targets_2d = jnp.reshape(targets, (n, -1))
  scan_targets = targets_2d[:, :0] if config.optimize_large_l else targets_2d

  if config.compute_metrics:

    def v_body(
        carry: tuple[jt.Float[jt.Array, "N"], ...],
        j: jt.Int[jt.Array, ""],
    ) -> tuple[tuple[jt.Float[jt.Array, "N"], ...], None]:
      loss_acc, tp_acc, fp_acc, fn_acc, tn_acc = carry
      loss_sum, logits, targets_chunk, valid_mask = _bce_fwd_chunk(
          activations_2d, embeddings, scan_targets, j, block_v, vocab
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
    if config.optimize_large_l:
      delta_loss, tp_final, num_pos = _fwd_pos_correction(
          activations_2d, embeddings, targets_2d
      )
      loss_final = loss_final + delta_loss
      fp_final = fp_final - tp_final
      fn_final = num_pos - tp_final
      tn_final = tn_final - fn_final
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
      loss_sum, _, _, _ = _bce_fwd_chunk(
          activations_2d, embeddings, scan_targets, j, block_v, vocab
      )
      return loss_acc + loss_sum, None

    init = jnp.zeros((n,), dtype=jnp.float32)
    loss_final, _ = jax.lax.scan(v_body_no_metrics, init, jnp.arange(v_blocks))
    if config.optimize_large_l:
      loss_final = (
          loss_final
          + _fwd_pos_correction(activations_2d, embeddings, targets_2d)[0]
      )
    return jnp.reshape(loss_final, (batch, seq_len))


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _cut_binary_cross_entropy(
    config: BCEConfig,
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
  """Computes the non-differentiable path of cut BCE loss and metrics."""
  outputs, _ = _cut_binary_cross_entropy_fwd(
      config, activations, embeddings, targets
  )
  return outputs


def _run_fwd_unsharded(
    config: BCEConfig,
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
  """Runs `_bce_fwd_local` on per-shard 3D or 4D inputs."""
  if activations.ndim == 4:
    targets_in_axis = None if targets.ndim == 3 else 0
    fwd_vmap = jax.vmap(
        functools.partial(_bce_fwd_local, config),
        in_axes=(0, None, targets_in_axis),
    )
    return fwd_vmap(activations, embeddings, targets)
  return _bce_fwd_local(config, activations, embeddings, targets)


def _bce_fwd_sharded(
    config: BCEConfig,
    replicated_activations: jt.Float[jt.Array, "... B N D"],
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
  """Sharding-aware dispatcher wrapping Pallas forward in `jax.shard_map`."""
  if not config.use_pallas or _pallas_interpret():
    return _run_fwd_unsharded(
        config, replicated_activations, embeddings, targets
    )

  mesh = None
  act_spec = None
  if config.mesh is not None and config.act_spec is not None:
    mesh = config.mesh
    act_spec = config.act_spec
    is_sharded = not mesh.empty
  else:
    act_sharding = _get_sharding(replicated_activations)
    emb_sharding = _get_sharding(embeddings)
    if (
        isinstance(act_sharding, jax.sharding.NamedSharding)
        and not act_sharding.mesh.empty
    ):
      mesh = act_sharding.mesh
      act_spec = act_sharding.spec
      is_sharded = True
    elif (
        isinstance(emb_sharding, jax.sharding.NamedSharding)
        and not emb_sharding.mesh.empty
    ):
      mesh = emb_sharding.mesh
      act_spec = jax.sharding.PartitionSpec()
      is_sharded = True
    else:
      is_sharded = False

  if not is_sharded:
    return _run_fwd_unsharded(
        config, replicated_activations, embeddings, targets
    )

  assert mesh is not None and act_spec is not None
  full_act_spec = tuple(act_spec) + (None,) * (
      replicated_activations.ndim - len(act_spec)
  )
  local_act_spec = jax.sharding.PartitionSpec(*(full_act_spec[:-1] + (None,)))
  if targets.ndim == replicated_activations.ndim:
    local_tgt_spec = local_act_spec
  else:
    local_tgt_spec = jax.sharding.PartitionSpec(*local_act_spec[1:])

  replicated_emb_spec = jax.sharding.PartitionSpec()
  embeddings_rep = jax.lax.with_sharding_constraint(
      embeddings, jax.sharding.NamedSharding(mesh, replicated_emb_spec)
  )
  pos_out_spec = jax.sharding.PartitionSpec(*local_act_spec[:-1])
  out_specs = (
      (pos_out_spec, pos_out_spec, pos_out_spec, pos_out_spec, pos_out_spec)
      if config.compute_metrics
      else pos_out_spec
  )
  return jax.shard_map(
      functools.partial(_run_fwd_unsharded, config),
      mesh=mesh,
      in_specs=(local_act_spec, replicated_emb_spec, local_tgt_spec),
      out_specs=out_specs,
      check_vma=False,
  )(replicated_activations, embeddings_rep, targets)


def _cut_binary_cross_entropy_fwd(
    config: BCEConfig,
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
  """Computes forward mode of cut BCE loss."""
  replicated_activations = _replicate_hidden_dim(activations)
  res = _bce_fwd_sharded(config, replicated_activations, embeddings, targets)
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


def _cut_binary_cross_entropy_bwd(
    config: BCEConfig,
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
  """Computes the backward mode of cut BCE loss."""
  if isinstance(d_outputs, tuple):
    d_losses = d_outputs[0]
  else:
    d_losses = d_outputs
  activations, embeddings, targets = res
  d_activations, d_embeddings = _bce_bwd_sharded(
      config, d_losses, activations, embeddings, targets
  )
  return d_activations, d_embeddings, None


_cut_binary_cross_entropy.defvjp(
    _cut_binary_cross_entropy_fwd, _cut_binary_cross_entropy_bwd
)


def cut_binary_cross_entropy(
    activations: jt.Float[jt.Array, "... B N D"],
    embeddings: jt.Float[jt.Array, "V D"],
    targets: jt.Int[jt.Array, "... B N L"],
    weights: jt.Float[jt.Array, "... B N"] | None = None,
    *,
    return_per_target_losses: bool = False,
    return_metrics: bool = False,
    return_per_target_metrics: bool = False,
    block_v: int | None = None,
    block_n: int | None = None,
    mesh: jax.sharding.Mesh | None = None,
    act_spec: jax.sharding.PartitionSpec | None = None,
    emb_spec: jax.sharding.PartitionSpec | None = None,
    use_pallas: bool = True,
    optimize_large_l: bool = False,
    force_bwd_block_v: bool = False,
) -> (
    jt.Float[jt.Array, ""]
    | tuple[jt.Float[jt.Array, ""], jt.Float[jt.Array, "... B N"]]
    | tuple[
        jt.Float[jt.Array, ""],
        jt.Float[jt.Array, "... B N"] | jt.Float[jt.Array, ""],
        jt.Float[jt.Array, "... B N"] | jt.Float[jt.Array, ""],
        jt.Float[jt.Array, "... B N"] | jt.Float[jt.Array, ""],
        jt.Float[jt.Array, "... B N"] | jt.Float[jt.Array, ""],
    ]
    | tuple[
        jt.Float[jt.Array, ""],
        jt.Float[jt.Array, "... B N"],
        jt.Float[jt.Array, "... B N"] | jt.Float[jt.Array, ""],
        jt.Float[jt.Array, "... B N"] | jt.Float[jt.Array, ""],
        jt.Float[jt.Array, "... B N"] | jt.Float[jt.Array, ""],
        jt.Float[jt.Array, "... B N"] | jt.Float[jt.Array, ""],
    ]
):
  """Computes binary cross entropy loss over unmaterialized logits.

  Only 3D `[B, N, D]` and 4D `[X, B, N, D]` activations are supported; rank 5
  and above raise a `ValueError`. Reshape or vmap over the extra leading axes
  before calling this op.

  Args:
    activations: Hidden-state outputs of shape ``[B, N, D]``, or ``[X, B, N,
      D]`` with a single extra leading axis.
    embeddings: Output embedding / unembedding weights of shape ``[V, D]``.
    targets: Target token ids of shape ``[B, N, L]``, i.e. the ``L`` positive
      labels of each sequence position. For 4D activations, either a matching 4D
      ``[X, B, N, L]`` array or a 3D ``[B, N, L]`` array shared across the
      leading axis.
    weights: Per-sequence-position loss weights, typically a padding/validity
      mask. Must have exactly the shape of the per-position losses, i.e.
      ``activations.shape[:-1]`` (``[B, N]``, or ``[X, B, N]`` for 4D
      activations); any other shape raises. This is one weight per position,
      *not* per target id: the ``L`` target ids of a position all share that
      position's weight.
    return_per_target_losses: If True, also return the per-target loss tensor.
    return_metrics: If True, also return TP, FP, FN, TN metric counts.
    return_per_target_metrics: If True (with ``return_metrics=True``), return
      per-position TP, FP, FN, TN tensors instead of summing them to scalars.
    block_v: Vocab-axis block size. Auto-picked if omitted.
    block_n: Sequence-axis block size for backward kernel.
    mesh: Optional mesh to use for sharding.
    act_spec: Optional partition spec for activations.
    emb_spec: Optional partition spec for embeddings.
    use_pallas: If True, use Pallas kernels; if False, use pure JAX
      implementation.
    optimize_large_l: If True, use the target-free vocabulary pass + sparse
      positive-target correction. Defaults to False.
    force_bwd_block_v: If True, force the backward pass to use `block_v`
      directly instead of clamping to `_max_safe_block_v`. Defaults to False.

  Returns:
    - ``loss`` if ``return_per_target_losses=False`` and
      ``return_metrics=False``.
    - ``(loss, losses)`` if ``return_per_target_losses=True`` and
      ``return_metrics=False``.
    - ``(loss, tp, fp, fn, tn)`` if ``return_per_target_losses=False`` and
      ``return_metrics=True``.
    - ``(loss, losses, tp, fp, fn, tn)`` if ``return_per_target_losses=True``
      and ``return_metrics=True``.
    Here ``loss`` is a scalar ``[]``, ``losses`` has shape ``[..., B, N]``, and
    ``(tp, fp, fn, tn)`` each have shape ``[..., B, N]`` when
    ``return_per_target_metrics=True`` or scalar shape ``[]`` when ``False``.

  Raises:
    ValueError: If ``activations`` has rank 5 or above, or if ``weights`` does
      not exactly match the per-position loss shape.
  """
  # `_bce_fwd_local` operates on a single `[B, N, D]` array and the forward
  # pass only vmaps over one extra leading axis, so ranks >= 5 are unsupported.
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

  # Prevent collective communications inside the loop by forcing replication of
  # weights.
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
  config = BCEConfig(
      block_v=block_v,
      block_n=effective_block_n,
      compute_metrics=return_metrics,
      mesh=mesh,
      act_spec=act_spec,
      emb_spec=emb_spec,
      use_pallas=use_pallas,
      global_vocab=vocab_size,
      optimize_large_l=optimize_large_l,
      force_bwd_block_v=force_bwd_block_v,
  )

  res = _cut_binary_cross_entropy(
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
      tp = tp * weights
      fp = fp * weights
      fn = fn * weights
      tn = tn * weights
    else:
      weight_sum = np.prod(losses_shape)

    loss = jnp.sum(losses) / (weight_sum + EPS)

    if not return_per_target_metrics:
      tp = jnp.sum(tp)
      fp = jnp.sum(fp)
      fn = jnp.sum(fn)
      tn = jnp.sum(tn)

    if return_per_target_losses:
      return loss, losses, tp, fp, fn, tn
    return loss, tp, fp, fn, tn
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


def _pallas_lane() -> int:
  """TPU MXU lane width - VMEM minor-axis tile constraint."""
  if any(d.platform == "tpu" for d in jax.devices()):
    return pltpu.get_tpu_info().num_lanes
  return 128


def _pallas_vmem_budget(reserve_mb: float | None = None) -> int:
  """Per-scoped Pallas allocation VMEM budget for the live TPU."""
  if reserve_mb is None:
    reserve_mb = _RESERVE_MB
  if any(d.platform == "tpu" for d in jax.devices()):
    cap = pltpu.get_tpu_info().vmem_capacity_bytes - int(
        reserve_mb * _BYTES_IN_MB
    )
    return max(16 * _BYTES_IN_MB, cap)
  return 16 * _BYTES_IN_MB


def _pallas_sublane() -> int:
  """TPU VMEM sublane count - second-minor-axis tile constraint."""
  if any(d.platform == "tpu" for d in jax.devices()):
    return pltpu.get_tpu_info().num_sublanes
  return 8


def _balanced_block_v(
    vocab: int, max_block_v: int, alignment: int | None = None
) -> int:
  """Spreads `vocab` evenly over the fewest aligned blocks <= `max_block_v`."""
  if vocab <= max_block_v:
    return vocab
  if alignment is None:
    alignment = _pallas_lane()
  v_blocks = (vocab + max_block_v - 1) // max_block_v
  if vocab % alignment == 0:
    tiles = vocab // alignment
    for k in range(v_blocks, v_blocks + 4):
      if tiles % k == 0 and (tiles // k) * alignment <= max_block_v:
        return (tiles // k) * alignment
  balanced = pl.align_to((vocab + v_blocks - 1) // v_blocks, alignment)
  return min(max_block_v, balanced)


def _vocab_bytes_per_block_v(
    padded_d: int,
    block_n: int,
    dtype: jax.typing.DTypeLike = jnp.float32,
    logit_vmem_tiles: int = _LOGIT_VMEM_TILES,
) -> int:
  """VMEM bytes the backward kernel needs per unit of `block_v`.

  Every extra vocabulary column adds `_VOCAB_VMEM_TILES` rows of width
  `padded_d` (embeddings, gradients, accumulator and staging buffers) plus
  `logit_vmem_tiles` columns of height `block_n` (the logits intermediate and
  the temporaries derived from it).

  Args:
    padded_d: Hidden size padded up to the vector lane width.
    block_n: Token-axis tile size of the kernel's inner loop.
    dtype: Element type of the activations and embeddings.
    logit_vmem_tiles: Number of live `[block_n, block_v]` tiles in VMEM.

  Returns:
    The number of VMEM bytes consumed by one vocabulary column.
  """
  bytes_per_element = jnp.dtype(dtype).itemsize
  return (
      _VOCAB_VMEM_TILES * padded_d + logit_vmem_tiles * block_n
  ) * bytes_per_element


def _max_safe_block_v(
    vmem_budget: int,
    padded_d: int,
    block_n: int = _BLOCK_N,
    dtype: jax.typing.DTypeLike = jnp.float32,
    logit_vmem_tiles: int = _LOGIT_VMEM_TILES,
) -> int:
  """Returns maximum safe block_v to prevent VMEM OOM.

  Dividing the share of the VMEM budget reserved for the vocabulary-side
  working set by its per-column cost yields the largest `block_v` the live chip
  can hold. This scales with the chip's reported VMEM capacity instead of
  encoding byte products measured on one particular generation.

  Args:
    vmem_budget: Usable Pallas VMEM allocation budget, in bytes.
    padded_d: Hidden size padded up to the vector lane width.
    block_n: Token-axis tile size of the kernel's inner loop.
    dtype: Element type of the activations and embeddings.
    logit_vmem_tiles: Number of live `[block_n, block_v]` tiles in VMEM.

  Returns:
    The largest MXU-aligned `block_v` that fits the budget.
  """
  if vmem_budget <= 32 * _BYTES_IN_MB:
    share = _SMALL_VMEM_SHARE
  elif vmem_budget <= 48 * _BYTES_IN_MB:
    share = _MEDIUM_VOCAB_VMEM_SHARE
  else:
    share = _VOCAB_VMEM_SHARE

  # NOMUTANTS -- max_safe is calculated based on VMEM budget.
  max_safe = int(vmem_budget * share) // _vocab_bytes_per_block_v(
      padded_d, block_n, dtype, logit_vmem_tiles=logit_vmem_tiles
  )

  # Round down to a multiple of the MXU systolic dimension. `pl.align_to`
  # rounds up, which would overshoot a ceiling, so take the multiple directly.
  mxu_size = _get_mxu_size()
  return max(mxu_size, (max_safe // mxu_size) * mxu_size)


def _effective_block_v(
    config: BCEConfig,
    hidden: int,
    vocab: int,
    logit_vmem_tiles: int = _LOGIT_VMEM_TILES,
) -> tuple[int, int]:
  """Resolves the padded hidden size and the block_v the kernel will run with.

  Shared by `_bce_fwd_pallas`, `_bce_bwd_pallas`, and
  `_bce_bwd_pallas_chunked_n` so that the token chunking is sized against the
  same vocabulary tile the kernel actually uses.

  Args:
    config: Configuration supplying the requested `block_v`.
    hidden: Unpadded hidden size ``D`` of the activations.
    vocab: Shard-local vocabulary size ``V``.
    logit_vmem_tiles: Number of live `[block_n, block_v]` tiles in VMEM.

  Returns:
    A ``(padded_d, block_v)`` pair, both aligned to the hardware tiling.
  """
  lane = _pallas_lane()
  padded_d = pl.align_to(hidden, lane)
  if config.force_bwd_block_v:
    block_v = pl.align_to(min(config.block_v, vocab), lane)
    return padded_d, min(block_v, pl.align_to(vocab, lane))
  max_safe = _max_safe_block_v(
      _pallas_vmem_budget(),
      padded_d,
      config.block_n,
      logit_vmem_tiles=logit_vmem_tiles,
  )
  capped_v = min(config.block_v, vocab, max_safe)
  if config.block_v > max_safe or (
      vocab > capped_v and vocab % lane == 0 and vocab % capped_v != 0
  ):
    limit_v = capped_v if config.block_v <= max_safe else max_safe
    balanced_v = _balanced_block_v(vocab, limit_v, lane)
  else:
    balanced_v = capped_v
  block_v = pl.align_to(balanced_v, lane)
  return padded_d, min(block_v, pl.align_to(vocab, lane), max_safe)


def _pallas_interpret() -> bool:
  """Run Pallas in interpret mode whenever no TPU is present (dev / CI)."""
  return not any(d.platform == "tpu" for d in jax.devices())


def _check_vocab_replicated_in_d(
    emb_spec: jax.sharding.PartitionSpec | None,
) -> None:
  if emb_spec is not None and len(emb_spec) > 1 and emb_spec[1] is not None:
    raise NotImplementedError(
        "Embeddings sharded along the hidden dimension D are not supported; "
        f"got embedding sharding {emb_spec}."
    )


def _build_bwd_pos_buckets(
    targets_2d: jt.Int[jt.Array, "N L"],
    vocab: int,
    block_v: int,
    v_blocks: int,
) -> tuple[jt.Int[jt.Array, "B"], jt.Int[jt.Array, "M"]]:
  """Sorts and buckets unique in-bounds positive targets per v_block tile."""
  n, num_l = targets_2d.shape
  safe_t, valid_pos = _unique_valid_targets(targets_2d, vocab)
  flat_t = jnp.reshape(jnp.where(valid_pos, safe_t, vocab), (-1,)).astype(
      jnp.int32
  )
  flat_tok = jnp.reshape(
      jax.lax.broadcasted_iota(jnp.int32, (n, num_l), 0), (-1,)
  )
  sorted_t, sorted_tok = jax.lax.sort([flat_t, flat_tok], num_keys=1)
  boundaries = jnp.minimum(
      jnp.arange(v_blocks + 1, dtype=jnp.int32) * block_v, vocab
  )
  offsets = jnp.searchsorted(sorted_t, boundaries, side="left").astype(
      jnp.int32
  )
  local_v = jnp.where(
      sorted_t < vocab, jnp.remainder(sorted_t, block_v), 0
  ).astype(jnp.int32)
  packed = jnp.bitwise_or(jnp.left_shift(local_v, 16), sorted_tok).astype(
      jnp.int32
  )
  padded_packed = jnp.pad(packed, (0, 16), constant_values=0)
  return offsets, padded_packed


def _bce_bwd_kernel(
    emb_ref: jax.Ref,  # [block_v, padded_d] VMEM
    act_ref: jax.Ref,  # [padded_n, padded_d] VMEM, resident across the grid
    tgt_ref: jax.Ref | None,  # [l_padded, padded_n] VMEM or None
    dloss_ref: jax.Ref,  # [1, padded_n] VMEM, resident across the grid
    pos_offsets_ref: jax.Ref | None,  # [v_blocks + 1] SMEM or None
    pos_packed_ref: jax.Ref | None,  # [m_pos + 4] SMEM or None
    pos_scale_ref: jax.Ref | None,  # [padded_n] SMEM or None
    d_emb_ref: jax.Ref,  # [block_v, padded_d] VMEM output block
    d_act_ref: jax.Ref,  # [padded_n, padded_d] VMEM output accumulated across v
    d_emb_scratch: jax.Ref,  # [block_v, padded_d] VMEM scratch
    d_act_scratch: jax.Ref,  # [padded_n, padded_d] VMEM scratch
    d_emb_b1: jax.Ref | None,
    d_emb_b2: jax.Ref | None,
    d_emb_b3: jax.Ref | None,
    d_act_b1: jax.Ref | None,
    d_act_b2: jax.Ref | None,
    d_act_b3: jax.Ref | None,
    *,
    block_n: int,
    block_v: int,
    padded_d: int,
    num_sublanes: int,
    n_blocks: int,
    v_blocks: int,
    vocab: int,
    labels: int,
    n_real: int,
    padded_n: int,
    global_vocab: int,
    fuse_pos: bool,
    pos_banks: int,
):
  """Pipelined Pallas TPU backward kernel with VMEM d_act & sparse pos fusion."""
  v_idx = pl.program_id(0)

  @pl.when(v_idx == 0)
  def _init_d_act():
    z_act = jnp.zeros(d_act_scratch.shape, jnp.float32)
    d_act_scratch[...] = z_act
    if fuse_pos and pos_banks == 4:
      assert (
          d_act_b1 is not None and d_act_b2 is not None and d_act_b3 is not None
      )
      d_act_b1[...] = z_act
      d_act_b2[...] = z_act
      d_act_b3[...] = z_act

  z_emb = jnp.zeros(d_emb_scratch.shape, jnp.float32)
  d_emb_scratch[...] = z_emb
  if fuse_pos and pos_banks == 4:
    assert (
        d_emb_b1 is not None and d_emb_b2 is not None and d_emb_b3 is not None
    )
    d_emb_b1[...] = z_emb
    d_emb_b2[...] = z_emb
    d_emb_b3[...] = z_emb

  inv_vocab = 1.0 / global_vocab
  col_ids = v_idx * block_v + jax.lax.broadcasted_iota(
      jnp.int32, (1, block_v), 1
  )
  valid_v = col_ids < vocab if vocab % block_v != 0 else None
  emb_block = (
      jnp.where(valid_v.T, emb_ref[...], 0.0)
      if valid_v is not None
      else emb_ref[...]
  )
  local_n_iota = jax.lax.broadcasted_iota(jnp.int32, (block_n, 1), 0)

  def n_loop_body(nc: jax.Array, _: None) -> None:
    n_slice = pl.ds(nc * block_n, block_n)
    act_sub = act_ref[n_slice, :]
    dloss_val = dloss_ref[0, n_slice]
    raw_scale = (dloss_val * inv_vocab)[:, None]
    if padded_n > n_real:
      batch_ids = nc * block_n + local_n_iota
      scale = jnp.where(batch_ids < n_real, raw_scale, 0.0)
    else:
      scale = raw_scale

    logits = jax.lax.dot(
        act_sub,
        emb_block,
        dimension_numbers=(((1,), (1,)), ((), ())),
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32,
    )
    probs = jax.nn.sigmoid(logits)

    if labels > 0:
      assert tgt_ref is not None
      tgt_sub = tgt_ref[:, n_slice]
      y_true = jnp.zeros((block_n, block_v), dtype=jnp.bool_)
      for l_idx in range(labels):
        y_true = y_true | (tgt_sub[l_idx : l_idx + 1, :].T == col_ids)
      g = probs - y_true.astype(probs.dtype)
    else:
      g = probs

    if valid_v is not None:
      g = jnp.where(valid_v, g, 0.0)

    deriv = scale * g

    d_emb_contrib = jax.lax.dot(
        deriv.astype(act_sub.dtype),
        act_sub,
        dimension_numbers=(((0,), (0,)), ((), ())),
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32,
    )
    d_emb_scratch[...] = d_emb_scratch[...] + d_emb_contrib

    d_act_contrib = jax.lax.dot(
        deriv.astype(emb_block.dtype),
        emb_block,
        dimension_numbers=(((1,), (0,)), ((), ())),
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32,
    )
    d_act_scratch[n_slice, :] = d_act_scratch[n_slice, :] + d_act_contrib

  if n_blocks == 1:
    n_loop_body(jnp.int32(0), None)
  else:
    jax.lax.fori_loop(0, n_blocks, n_loop_body, None)

  if fuse_pos:
    assert (
        pos_offsets_ref is not None
        and pos_packed_ref is not None
        and pos_scale_ref is not None
    )
    start = pos_offsets_ref[v_idx]
    end = pos_offsets_ref[v_idx + 1]
    sub_iota = jax.lax.broadcasted_iota(jnp.int32, (num_sublanes, 1), 0)
    if pos_banks == 4:
      de_banks = (d_emb_scratch, d_emb_b1, d_emb_b2, d_emb_b3)
      da_banks = (d_act_scratch, d_act_b1, d_act_b2, d_act_b3)
      num_groups = (end - start + 15) // 16

      def pos_group_body(g_idx: jax.Array, _: None) -> None:
        base = start + g_idx * 16
        for u in range(16):
          idx = base + u
          valid = idx < end
          p = pos_packed_ref[idx]
          local_v = jnp.right_shift(p, 16)
          tok = jnp.bitwise_and(p, 0xFFFF)
          s = jnp.where(valid, pos_scale_ref[tok], 0.0)
          v_tile = pl.multiple_of(
              (local_v // num_sublanes) * num_sublanes, num_sublanes
          )
          v_sub = local_v % num_sublanes
          t_tile = pl.multiple_of(
              (tok // num_sublanes) * num_sublanes, num_sublanes
          )
          t_sub = tok % num_sublanes
          v_sl = pl.ds(v_tile, num_sublanes)
          t_sl = pl.ds(t_tile, num_sublanes)
          act_sub = act_ref[t_sl, :].astype(jnp.float32)
          emb_sub = emb_ref[v_sl, :].astype(jnp.float32)
          act_row = jnp.broadcast_to(
              jnp.sum(
                  jnp.where(valid & (sub_iota == t_sub), act_sub, 0.0),
                  axis=0,
                  keepdims=True,
              ),
              (num_sublanes, padded_d),
          )
          emb_row = jnp.broadcast_to(
              jnp.sum(
                  jnp.where(valid & (sub_iota == v_sub), emb_sub, 0.0),
                  axis=0,
                  keepdims=True,
              ),
              (num_sublanes, padded_d),
          )
          bank_idx = u & 3
          de_b = de_banks[bank_idx]
          da_b = da_banks[bank_idx]
          assert de_b is not None and da_b is not None
          de_b[v_sl, :] = de_b[v_sl, :] + jnp.where(
              sub_iota == v_sub, s * act_row, 0.0
          )
          da_b[t_sl, :] = da_b[t_sl, :] + jnp.where(
              sub_iota == t_sub, s * emb_row, 0.0
          )

      jax.lax.fori_loop(0, num_groups, pos_group_body, None)
    else:

      def pos_single_body(k: jax.Array, _: None) -> None:
        p = pos_packed_ref[k]
        local_v = jnp.right_shift(p, 16)
        tok = jnp.bitwise_and(p, 0xFFFF)
        s = pos_scale_ref[tok]
        v_tile = pl.multiple_of(
            (local_v // num_sublanes) * num_sublanes, num_sublanes
        )
        v_sub = local_v % num_sublanes
        t_tile = pl.multiple_of(
            (tok // num_sublanes) * num_sublanes, num_sublanes
        )
        t_sub = tok % num_sublanes
        v_sl = pl.ds(v_tile, num_sublanes)
        t_sl = pl.ds(t_tile, num_sublanes)
        act_sub = act_ref[t_sl, :].astype(jnp.float32)
        emb_sub = emb_ref[v_sl, :].astype(jnp.float32)
        act_row = jnp.broadcast_to(
            jnp.sum(
                jnp.where(sub_iota == t_sub, act_sub, 0.0),
                axis=0,
                keepdims=True,
            ),
            (num_sublanes, padded_d),
        )
        emb_row = jnp.broadcast_to(
            jnp.sum(
                jnp.where(sub_iota == v_sub, emb_sub, 0.0),
                axis=0,
                keepdims=True,
            ),
            (num_sublanes, padded_d),
        )
        d_emb_scratch[v_sl, :] = d_emb_scratch[v_sl, :] + jnp.where(
            sub_iota == v_sub, s * act_row, 0.0
        )
        d_act_scratch[t_sl, :] = d_act_scratch[t_sl, :] + jnp.where(
            sub_iota == t_sub, s * emb_row, 0.0
        )

      jax.lax.fori_loop(start, end, pos_single_body, None)

  if fuse_pos and pos_banks == 4:
    assert (
        d_emb_b1 is not None and d_emb_b2 is not None and d_emb_b3 is not None
    )
    total_d_emb = (d_emb_scratch[...] + d_emb_b1[...]) + (
        d_emb_b2[...] + d_emb_b3[...]
    )
    d_emb_ref[...] = total_d_emb.astype(d_emb_ref.dtype)
  else:
    d_emb_ref[...] = d_emb_scratch[...].astype(d_emb_ref.dtype)

  @pl.when(v_idx == v_blocks - 1)
  def _store_d_act():
    if fuse_pos and pos_banks == 4:
      assert (
          d_act_b1 is not None and d_act_b2 is not None and d_act_b3 is not None
      )
      total_d_act = (d_act_scratch[...] + d_act_b1[...]) + (
          d_act_b2[...] + d_act_b3[...]
      )
      d_act_ref[...] = total_d_act.astype(d_act_ref.dtype)
    else:
      d_act_ref[...] = d_act_scratch[...].astype(d_act_ref.dtype)


def _apply_bwd_pos_correction(
    d_act_2d: jt.Float[jt.Array, "N D"],
    d_emb: jt.Float[jt.Array, "V D"],
    activations_2d: jt.Float[jt.Array, "N D"],
    embeddings: jt.Float[jt.Array, "V D"],
    targets_2d: jt.Int[jt.Array, "N L"],
    dloss_2d: jt.Float[jt.Array, "N 1"],
    inv_vocab: float,
    grad_diff_fn: (
        Callable[[jt.Float[jt.Array, "N L"]], jt.Float[jt.Array, "N L"]] | None
    ) = None,
) -> tuple[jt.Float[jt.Array, "N D"], jt.Float[jt.Array, "V D"]]:
  """Applies the sparse positive-target gradient correction."""
  safe_t, valid_pos = _unique_valid_targets(targets_2d, embeddings.shape[0])
  pos_emb = embeddings[safe_t]
  raw_diff = (
      -1.0
      if grad_diff_fn is None
      else grad_diff_fn(
          jnp.einsum(
              "nd,nld->nl",
              activations_2d,
              pos_emb,
              preferred_element_type=jnp.float32,
              precision=jax.lax.Precision.DEFAULT,
          )
      )
  )
  scale_2d = (dloss_2d * inv_vocab).astype(jnp.float32)
  delta_deriv = jnp.where(valid_pos, raw_diff * scale_2d, 0.0).astype(
      embeddings.dtype
  )
  d_act_pos = jnp.einsum(
      "nl,nld->nd",
      delta_deriv,
      pos_emb,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.DEFAULT,
  ).astype(d_act_2d.dtype)
  unique_t = jnp.where(valid_pos, safe_t, -1)
  d_emb_updates = delta_deriv[:, :, None] * activations_2d[:, None, :].astype(
      embeddings.dtype
  )
  return d_act_2d + d_act_pos, d_emb.at[unique_t].add(
      d_emb_updates, mode="drop"
  )


def _bce_bwd_pallas(
    config: BCEConfig,
    d_loss: jt.Float[jt.Array, "B N"],
    activations: jt.Float[jt.Array, "B N D"],
    embeddings: jt.Float[jt.Array, "V D"],
    targets: jt.Int[jt.Array, "B N L"],
    vocab_offset: int = 0,
) -> tuple[
    jt.Float[jt.Array, "B N D"],
    jt.Float[jt.Array, "V D"],
]:
  """Pallas TPU per-shard chunked backward for BCE."""
  lane = _pallas_lane()
  num_sublanes = _pallas_sublane()
  block_n = pl.align_to(config.block_n, lane)
  batch, seq_len, hidden = activations.shape
  vocab = embeddings.shape[0]
  padded_d, block_v = _effective_block_v(config, hidden, vocab)
  global_vocab = vocab if config.global_vocab is None else config.global_vocab
  n = batch * seq_len
  n_blocks = (n + block_n - 1) // block_n
  v_blocks = (vocab + block_v - 1) // block_v
  vocab_hbm = (
      v_blocks * block_v
      if _pallas_interpret()
      else max(pl.align_to(vocab, num_sublanes), block_v)
  )
  raw_labels = targets.shape[-1]
  labels = 0 if config.optimize_large_l else raw_labels

  # Rebase global target ids onto this shard's local rows here rather than
  # inside the kernel: under vocab sharding `vocab_offset` is a tracer, and a
  # tracer cannot be bound into a Pallas kernel as a static argument. Ids
  # belonging to other shards land outside [0, vocab) and simply never match.
  targets = targets - vocab_offset

  padded_n = n_blocks * block_n
  vmem_budget = _pallas_vmem_budget()
  fuse_pos = (
      config.optimize_large_l
      and not _pallas_interpret()
      and raw_labels > 0
      and block_v <= _MAX_PACKED_DIM
      and padded_n <= _MAX_PACKED_DIM
      and (n * raw_labels) <= _MAX_SMEM_POS_ENTRIES
  )
  base_vmem_bytes = (
      _vocab_bytes_per_block_v(padded_d, block_n, embeddings.dtype) * block_v
      + _TOKEN_VMEM_TILES * padded_n * padded_d * 4
  )
  extra_4bank_bytes = 3 * (block_v + padded_n) * padded_d * 4
  fits_4banks = fuse_pos and (base_vmem_bytes + extra_4bank_bytes) <= int(
      vmem_budget * 0.88
  )
  pos_banks = 4 if fits_4banks else 1

  if padded_n > n:
    pad_len = padded_n - n
    activations_2d = jnp.pad(
        jnp.reshape(activations, (n, hidden)), ((0, pad_len), (0, 0))
    )
    dloss_2d = jnp.pad(jnp.reshape(d_loss, (n, 1)), ((0, pad_len), (0, 0)))
  else:
    activations_2d = jnp.reshape(activations, (n, hidden))
    dloss_2d = jnp.reshape(d_loss, (n, 1))

  if padded_d > hidden:
    activations_padded = jnp.pad(
        activations_2d, ((0, 0), (0, padded_d - hidden))
    )
  else:
    activations_padded = activations_2d

  if vocab_hbm > vocab or padded_d > hidden:
    embeddings_padded = jnp.pad(
        embeddings, ((0, vocab_hbm - vocab), (0, padded_d - hidden))
    )
  else:
    embeddings_padded = embeddings

  if labels > 0:
    l_padded = max(num_sublanes, pl.align_to(labels, num_sublanes))
    targets_2d = jnp.reshape(targets[..., :labels], (n, labels))
    targets_t = jnp.transpose(targets_2d, (1, 0))
    if l_padded > labels or padded_n > n:
      targets_padded = jnp.pad(
          targets_t,
          ((0, l_padded - labels), (0, padded_n - n)),
          constant_values=-1,
      ).astype(jnp.int32)
    else:
      targets_padded = targets_t.astype(jnp.int32)
    tgt_spec = pl.BlockSpec((l_padded, padded_n), lambda v: (0, 0))
  else:
    targets_padded = None
    tgt_spec = None

  if fuse_pos:
    pos_offsets, pos_packed = _build_bwd_pos_buckets(
        jnp.reshape(targets, (n, -1)), vocab, block_v, v_blocks
    )
    pos_scale = (-dloss_2d[:, 0] * (1.0 / global_vocab)).astype(jnp.float32)
    pos_offsets_spec = pl.BlockSpec(
        pos_offsets.shape, lambda v: (0,), memory_space=pltpu.SMEM
    )
    pos_packed_spec = pl.BlockSpec(
        pos_packed.shape, lambda v: (0,), memory_space=pltpu.SMEM
    )
    pos_scale_spec = pl.BlockSpec(
        pos_scale.shape, lambda v: (0,), memory_space=pltpu.SMEM
    )
  else:
    pos_offsets = None
    pos_packed = None
    pos_scale = None
    pos_offsets_spec = None
    pos_packed_spec = None
    pos_scale_spec = None

  dloss_padded = jnp.transpose(dloss_2d, (1, 0)).astype(embeddings.dtype)
  emb_vmem = pltpu.VMEM((block_v, padded_d), jnp.float32)
  act_vmem = pltpu.VMEM((padded_n, padded_d), jnp.float32)
  extra_scratches = (
      [emb_vmem, emb_vmem, emb_vmem, act_vmem, act_vmem, act_vmem]
      if (fuse_pos and pos_banks == 4)
      else [None, None, None, None, None, None]
  )

  d_emb_padded, d_act_padded = pl.pallas_call(
      functools.partial(
          _bce_bwd_kernel,
          block_n=block_n,
          block_v=block_v,
          padded_d=padded_d,
          num_sublanes=num_sublanes,
          n_blocks=n_blocks,
          v_blocks=v_blocks,
          vocab=vocab,
          labels=labels,
          n_real=n,
          padded_n=padded_n,
          global_vocab=global_vocab,
          fuse_pos=fuse_pos,
          pos_banks=pos_banks,
      ),
      out_shape=[
          jax.ShapeDtypeStruct((vocab_hbm, padded_d), embeddings.dtype),
          jax.ShapeDtypeStruct((padded_n, padded_d), activations.dtype),
      ],
      grid=(v_blocks,),
      in_specs=[
          pl.BlockSpec((block_v, padded_d), lambda v: (v, 0)),  # emb
          pl.BlockSpec((padded_n, padded_d), lambda v: (0, 0)),  # act
          tgt_spec,  # tgt (None when optimize_large_l=True)
          pl.BlockSpec((1, padded_n), lambda v: (0, 0)),  # dloss
          pos_offsets_spec,
          pos_packed_spec,
          pos_scale_spec,
      ],
      out_specs=[
          pl.BlockSpec((block_v, padded_d), lambda v: (v, 0)),  # d_emb
          pl.BlockSpec((padded_n, padded_d), lambda v: (0, 0)),  # d_act
      ],
      scratch_shapes=[
          emb_vmem,  # d_emb_scratch
          act_vmem,  # d_act_scratch
      ]
      + extra_scratches,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary",),
          vmem_limit_bytes=vmem_budget,
      ),
      interpret=_pallas_interpret(),
  )(
      embeddings_padded,
      activations_padded,
      targets_padded,
      dloss_padded,
      pos_offsets,
      pos_packed,
      pos_scale,
  )

  if vocab < vocab_hbm or hidden < padded_d:
    d_emb = d_emb_padded[:vocab, :hidden]
  else:
    d_emb = d_emb_padded

  if padded_n > n or hidden < padded_d:
    d_act_2d = d_act_padded[:n, :hidden]
  else:
    d_act_2d = d_act_padded

  if config.optimize_large_l and not fuse_pos:
    d_act_2d, d_emb = _apply_bwd_pos_correction(
        d_act_2d,
        d_emb,
        activations_2d[:n],
        embeddings,
        jnp.reshape(targets, (n, -1)),
        dloss_2d[:n],
        1.0 / global_vocab,
    )

  d_activations = jnp.reshape(d_act_2d, (batch, seq_len, hidden))
  return d_activations, d_emb


def _max_safe_chunk_n(
    vmem_budget: int,
    padded_d: int,
    block_v: int,
    block_n: int = _BLOCK_N,
    target_ratio: float | None = None,
    dtype: jax.typing.DTypeLike = jnp.float32,
    logit_vmem_tiles: int = _LOGIT_VMEM_TILES,
) -> int:
  """Returns maximum safe chunk_n to prevent VMEM OOM.

  The vocabulary-side buffers stay resident in VMEM for the whole vocabulary
  grid, so the token axis may only use what they leave behind. Subtracting that
  working set from the budget gives the token axis a ceiling derived from the
  chip's actual VMEM capacity, rather than a fixed element count that happens
  to fit one particular generation.

  Args:
    vmem_budget: Usable Pallas VMEM allocation budget, in bytes.
    padded_d: Hidden size padded up to the vector lane width.
    block_v: Vocabulary tile size the kernel will run with.
    block_n: Token-axis tile size of the kernel's inner loop.
    target_ratio: Fraction of `vmem_budget` the backward pass may claim.
    dtype: Element type of the activations and embeddings.
    logit_vmem_tiles: Number of live `[block_n, block_v]` tiles in VMEM.

  Returns:
    The largest lane-aligned token chunk that fits alongside the vocabulary
    working set.
  """
  if target_ratio is None:
    if vmem_budget <= 32 * _BYTES_IN_MB:
      share = _SMALL_VMEM_SHARE
    elif vmem_budget <= 48 * _BYTES_IN_MB:
      share = _MEDIUM_TARGET_RATIO
    else:
      share = _TARGET_RATIO
  else:
    share = (
        _SMALL_VMEM_SHARE if vmem_budget <= 32 * _BYTES_IN_MB else target_ratio
    )
  bytes_per_element = jnp.dtype(dtype).itemsize

  # NOMUTANTS -- max_safe is calculated based on VMEM budget.
  vocab_bytes = (
      _vocab_bytes_per_block_v(
          padded_d, block_n, dtype, logit_vmem_tiles=logit_vmem_tiles
      )
      * block_v
  )
  available = max(0, int(vmem_budget * share) - vocab_bytes)

  # Conservative VMEM usage approx: chunk_n * padded_d * bytes_per_element * 3.
  # This allocates space for double buffering (1 load + 2 stores) to safely
  # accommodate future optimizations, even though it is currently
  # single-buffered.
  bytes_per_token = _TOKEN_VMEM_TILES * padded_d * bytes_per_element
  max_safe = available // bytes_per_token

  # Round down to the hardware vector register lane size. `pl.align_to` rounds
  # up, which would overshoot a ceiling, so the multiple is taken explicitly.
  lane = _pallas_lane()
  max_safe = (max_safe // lane) * lane

  # Floor at `_MIN_CHUNK_TILES` full vector-lane tiles so that a chunk is always
  # large enough to amortize kernel launch and DMA dispatch overhead.
  return max(_MIN_CHUNK_TILES * lane, max_safe)


def _balanced_chunk_n(n: int, max_chunk_n: int) -> int:
  """Spreads `n` tokens evenly over the fewest chunks that fit in VMEM.

  `max_chunk_n` is a VMEM ceiling, not a target. Running at the ceiling pads
  `n` up to a multiple of it and can waste close to a full chunk of compute:
  N=65536 against a 48768 ceiling processes 97536 rows, ~49% overhead, which
  measured ~46% slower end-to-end than an evenly balanced split. Dividing the
  tokens across the same number of chunks stays inside the budget while
  keeping total padding below one vector lane per chunk.

  Args:
    n: Total number of tokens (``batch * seq_len``).
    max_chunk_n: Largest chunk the VMEM budget allows.

  Returns:
    A lane-aligned chunk size no larger than `max_chunk_n`.
  """
  n_chunks = (n + max_chunk_n - 1) // max_chunk_n
  balanced = pl.align_to((n + n_chunks - 1) // n_chunks, _pallas_lane())
  return min(max_chunk_n, balanced)


def _bce_bwd_pallas_chunked_n(
    config: BCEConfig,
    d_loss: jt.Float[jt.Array, "B N"],
    activations: jt.Float[jt.Array, "B N D"],
    embeddings: jt.Float[jt.Array, "V D"],
    targets: jt.Int[jt.Array, "B N L"],
    vocab_offset: int = 0,
) -> tuple[
    jt.Float[jt.Array, "B N D"],
    jt.Float[jt.Array, "V D"],
]:
  """Wraps _bce_bwd_pallas by chunking the N (sequence) dimension in JAX."""
  batch, seq_len, hidden = activations.shape
  n = batch * seq_len

  vocab = embeddings.shape[0]
  padded_d, block_v = _effective_block_v(config, hidden, vocab)
  vmem_budget = _pallas_vmem_budget()
  max_chunk_n = _max_safe_chunk_n(
      vmem_budget, padded_d, block_v, config.block_n
  )
  if n <= max_chunk_n:
    return _bce_bwd_pallas(
        config, d_loss, activations, embeddings, targets, vocab_offset
    )

  chunk_n = _balanced_chunk_n(n, max_chunk_n)

  # Reshape inputs to 2D
  activations_2d = jnp.reshape(activations, (n, hidden))
  dloss_2d = jnp.reshape(d_loss, (n, 1))
  targets_2d = jnp.reshape(targets, (n, -1))

  n_chunks = (n + chunk_n - 1) // chunk_n
  padded_n = n_chunks * chunk_n

  if padded_n > n:
    pad_len = padded_n - n
    activations_padded = jnp.pad(activations_2d, ((0, pad_len), (0, 0)))
    dloss_padded = jnp.pad(dloss_2d, ((0, pad_len), (0, 0)))
    targets_padded = jnp.pad(
        targets_2d, ((0, pad_len), (0, 0)), constant_values=-1
    )
  else:
    activations_padded = activations_2d
    dloss_padded = dloss_2d
    targets_padded = targets_2d

  act_chunks = jnp.reshape(activations_padded, (n_chunks, chunk_n, hidden))
  dloss_chunks = jnp.reshape(dloss_padded, (n_chunks, chunk_n))
  tgt_chunks = jnp.reshape(targets_padded, (n_chunks, chunk_n, -1))

  act_chunks_3d = jnp.reshape(act_chunks, (n_chunks, 1, chunk_n, hidden))
  dloss_chunks_3d = jnp.reshape(dloss_chunks, (n_chunks, 1, chunk_n))
  tgt_chunks_3d = jnp.reshape(tgt_chunks, (n_chunks, 1, chunk_n, -1))

  def loop_body(carry, x):
    (d_emb_acc,) = carry
    act_chunk, dloss_chunk, tgt_chunk = x

    d_act_chunk, d_emb_contrib = _bce_bwd_pallas(
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
      loop_body,
      init,
      (act_chunks_3d, dloss_chunks_3d, tgt_chunks_3d),
  )

  d_act_padded = jnp.reshape(d_act_chunks, (padded_n, hidden))
  if padded_n > n:
    d_act_2d = d_act_padded[:n, :]
  else:
    d_act_2d = d_act_padded

  d_activations = jnp.reshape(d_act_2d, (batch, seq_len, hidden))
  return d_activations, d_embeddings


def _bce_bwd_scan_groups(
    step_fn: Callable[
        [
            jt.Float[jt.Array, "B N"],
            jt.Float[jt.Array, "B N D"],
            jt.Int[jt.Array, "B N L"],
        ],
        tuple[jt.Float[jt.Array, "B N D"], jt.Float[jt.Array, "V D"]],
    ],
    d_loss: jt.Float[jt.Array, "X B N"],
    activations: jt.Float[jt.Array, "X B N D"],
    embeddings: jt.Float[jt.Array, "V D"],
    targets: jt.Int[jt.Array, "X B N L"] | jt.Int[jt.Array, "B N L"],
) -> tuple[
    jt.Float[jt.Array, "X B N D"],
    jt.Float[jt.Array, "V D"],
]:
  """Runs a 3D backward `step_fn` over the leading group axis of a 4D input.

  A Python ``for`` over ``activations.shape[0]`` emits one independent program
  per group, so XLA is free to run the groups concurrently and hold one
  ``[V, D]`` embedding-gradient buffer live per group; measured peak temp then
  grows as ``groups * V * D * 4`` bytes and does not shrink with ``block_v``.
  ``jax.lax.scan`` emits a single loop body with one carried accumulator
  instead, which bounds the embedding gradient to one ``[V, D]`` buffer however
  many groups there are (and compiles the body once rather than ``groups``
  times).

  Groups are visited in index order and summed into the carry in that same
  order, exactly as the unrolled loop did, so the result is unchanged
  element-for-element.

  Args:
    step_fn: The 3D backward pass to apply to one group, called as ``step_fn(
      d_loss_g, activations_g, targets_g)`` and returning ``(d_activations_g,
      d_embeddings_g)``.
    d_loss: Gradient of the loss with respect to the per-position losses, of
      shape ``[X, B, N]``.
    activations: Hidden-state outputs of shape ``[X, B, N, D]``.
    embeddings: Output embedding / unembedding weights of shape ``[V, D]``.
    targets: Target token ids, either ``[X, B, N, L]`` or a 3D ``[B, N, L]``
      array shared across the group axis.

  Returns:
    Gradient of the loss with respect to the activations and embeddings.
  """
  # A 3D `targets` is shared across groups, so it stays a closed-over constant
  # rather than being sliced per iteration; only the group-major arrays are
  # scanned over.
  shared_targets = targets.ndim == 3

  def scan_body(
      d_emb_acc: jt.Float[jt.Array, "V D"],
      xs: tuple[jt.Array, ...],
  ) -> tuple[jt.Float[jt.Array, "V D"], jt.Float[jt.Array, "B N D"]]:
    if shared_targets:
      d_loss_g, act_g = xs
      tgt_g = targets
    else:
      d_loss_g, act_g, tgt_g = xs
    d_act_g, d_emb_g = step_fn(d_loss_g, act_g, tgt_g)
    return d_emb_acc + d_emb_g, d_act_g

  xs = (
      (d_loss, activations)
      if shared_targets
      else (d_loss, activations, targets)
  )
  d_embeddings, d_activations = jax.lax.scan(
      scan_body, jnp.zeros_like(embeddings), xs
  )
  return d_activations, d_embeddings


def _bce_bwd_pure_jax(
    config: BCEConfig,
    d_loss: jt.Float[jt.Array, "... B N"],
    activations: jt.Float[jt.Array, "... B N D"],
    embeddings: jt.Float[jt.Array, "V D"],
    targets: jt.Int[jt.Array, "... B N L"],
    vocab_offset: int = 0,
) -> tuple[
    jt.Float[jt.Array, "... B N D"],
    jt.Float[jt.Array, "V D"],
]:
  """Pure JAX backward pass for binary cross entropy."""
  if activations.ndim == 4:
    return _bce_bwd_scan_groups(
        lambda d_loss_g, act_g, tgt_g: _bce_bwd_pure_jax(
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
  scan_targets = targets_2d[:, :0] if config.optimize_large_l else targets_2d

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
    rel_targets = scan_targets - (actual_start + vocab_offset)
    chunk_cols = jnp.arange(block_v)[None, :]
    for l_idx in range(scan_targets.shape[-1]):
      targets_chunk = targets_chunk | (
          rel_targets[:, l_idx : l_idx + 1] == chunk_cols
      )

    d_logits_chunk = (
        jnp.where(targets_chunk, probs - 1.0, probs)
        * valid_mask[None, :]
        * (dloss_2d * inv_vocab)
    )

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

  if config.optimize_large_l:
    d_act_final, d_embeddings = _apply_bwd_pos_correction(
        d_act_final,
        d_embeddings,
        activations_2d,
        embeddings,
        targets_2d - vocab_offset,
        dloss_2d,
        inv_vocab,
    )

  d_activations = jnp.reshape(d_act_final, (batch, seq_len, hidden))
  return d_activations, d_embeddings


def _bce_bwd_sharded(
    config: BCEConfig,
    d_loss: jt.Float[jt.Array, "... B N"],
    activations: jt.Float[jt.Array, "... B N D"],
    embeddings: jt.Float[jt.Array, "V D"],
    targets: jt.Int[jt.Array, "... B N L"],
) -> tuple[
    jt.Float[jt.Array, "... B N D"],
    jt.Float[jt.Array, "V D"],
]:
  """Sharding-aware dispatcher for BCE backward."""
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
      return _bce_bwd_pure_jax(config, d_loss, activations, embeddings, targets)
    if activations.ndim == 4:
      return _bce_bwd_loop_fallback(
          config, d_loss, activations, embeddings, targets
      )
    return _bce_bwd_pallas_chunked_n(
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
  for axis in local_act_spec:
    if axis is not None and axis != vocab_axis_name:
      dp_axes.append(axis)

  def _bwd_local_with_reduction(d_loss_, act_, emb_, tgt_):
    if vocab_axis_name is not None:
      vocab_offset = jax.lax.axis_index(vocab_axis_name) * emb_.shape[0]
    else:
      vocab_offset = 0
    if not config.use_pallas:
      d_act, d_emb = _bce_bwd_pure_jax(
          config, d_loss_, act_, emb_, tgt_, vocab_offset=vocab_offset
      )
    else:
      if act_.ndim == 4:
        d_act, d_emb = _bce_bwd_loop_fallback(
            config, d_loss_, act_, emb_, tgt_, vocab_offset=vocab_offset
        )
      else:
        d_act, d_emb = _bce_bwd_pallas_chunked_n(
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
    local_tgt_spec = jax.sharding.PartitionSpec(*local_act_spec[1:])

  d_act_replicated, d_emb = jax.shard_map(
      _bwd_local_with_reduction,
      mesh=mesh,
      in_specs=(
          jax.sharding.PartitionSpec(*local_act_spec[:-1]),
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


def _bce_bwd_loop_fallback(
    config: BCEConfig,
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
  `_bce_bwd_pallas_chunked_n` only accepts a 3D `[B, N, D]` array. The groups
  are serialized by `_bce_bwd_scan_groups` so that only one `[V, D]` embedding
  gradient is live at a time.

  Args:
    config: BCE config.
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
      lambda d_loss_g, act_g, tgt_g: _bce_bwd_pallas_chunked_n(
          config, d_loss_g, act_g, embeddings, tgt_g, vocab_offset
      ),
      d_loss,
      activations,
      embeddings,
      targets,
  )

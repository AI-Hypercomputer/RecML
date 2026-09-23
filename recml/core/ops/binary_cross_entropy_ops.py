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

import dataclasses
import functools
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np

EPS = 1e-8
_BYTES_IN_MB = 1024 * 1024


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
      new_spec_list = list(spec)
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
    compute_metrics: Whether the forward pass also accumulates the TP, FP, FN
      and TN counts alongside the loss.
  """

  block_v: int
  compute_metrics: bool = False


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
  block_v = config.block_v
  batch, seq_len, hidden = activations.shape
  vocab = embeddings.shape[0]

  n = batch * seq_len
  # NOMUTANTS -- v_blocks is calculated from block_v and vocab.
  v_blocks = int(np.ceil(vocab / block_v))

  activations_2d = jnp.reshape(activations, (n, hidden))
  targets_2d = jnp.reshape(targets, (n, -1))

  if config.compute_metrics:

    def v_body(
        carry: tuple[jt.Float[jt.Array, "N"], ...],
        j: jt.Int[jt.Array, ""],
    ) -> tuple[tuple[jt.Float[jt.Array, "N"], ...], None]:
      loss_acc, tp_acc, fp_acc, fn_acc, tn_acc = carry
      loss_sum, logits, targets_chunk, valid_mask = _bce_fwd_chunk(
          activations_2d, embeddings, targets_2d, j, block_v, vocab
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
      loss_sum, _, _, _ = _bce_fwd_chunk(
          activations_2d, embeddings, targets_2d, j, block_v, vocab
      )
      return loss_acc + loss_sum, None

    init = jnp.zeros((n,), dtype=jnp.float32)
    loss_final, _ = jax.lax.scan(v_body_no_metrics, init, jnp.arange(v_blocks))
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
  # `_bce_fwd_local` is written against a 3D `[B, N, D]` array: it unpacks
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
        functools.partial(_bce_fwd_local, config),
        in_axes=(0, None, targets_in_axis),
    )
    res = fwd_vmap(replicated_activations, embeddings, targets)
  else:
    res = _bce_fwd_local(config, replicated_activations, embeddings, targets)
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
  del config, res, d_outputs
  raise NotImplementedError(
      "The memory-efficient backward pass of cut BCE is not implemented yet; "
      "only the forward pass is currently supported."
  )


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
    block_v: int | None = None,
    mesh: jax.sharding.Mesh | None = None,
    act_spec: jax.sharding.PartitionSpec | None = None,
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
  """Computes binary cross entropy loss over unmaterialized logits.

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
    return_per_target_losses: If True, also return the per-target loss tensor.
    return_metrics: If True, also return TP, FP, FN, TN metric counts.
    block_v: Vocab-axis block size. Auto-picked if omitted.
    mesh: Optional mesh to use for sharding.
    act_spec: Optional partition spec for activations.

  Returns:
    Scalar loss, optionally paired with per-target losses and/or metrics.

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

  config = BCEConfig(
      block_v=block_v,
      compute_metrics=return_metrics,
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

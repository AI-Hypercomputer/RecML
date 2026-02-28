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
"""Pointwise splash attention operations for HSTU."""

from collections.abc import Callable
import dataclasses
import enum
import functools

import chex
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu import splash_attention
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask_info
import jax.numpy as jnp
import jaxtyping as jt
import keras
import numpy as np


NUM_LANES = 128
NUM_SUBLANES = 8
# We predefine some useful dimension numbers for dot_general
NN_DIM_NUMBERS = (((1,), (0,)), ((), ()))  # standard matmul
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))  # RHS transposed
TN_DIM_NUMBERS = (((0,), (0,)), ((), ()))  # LHS transposed


class MaskType(enum.StrEnum):
  """Mask type for flash attention."""

  CAUSAL = "causal"
  FULL = "full"


@dataclasses.dataclass(frozen=True, kw_only=True)
class _PointwiseAttentionParams:
  """Parameters for pointwise attention."""

  scale: float
  mask_function: Callable[..., jax.Array] | None

  block_q: int
  block_kv: int
  block_kv_compute: int

  block_q_bwd: int
  block_kv_bwd: int
  block_kv_bwd_compute: int

  interpret: bool


def _next_nonzero(
    h, i, j, data_next_ref, block_mask_ref, m_next_ref, next_i=False
):
  """Gets indices and metadata for next nonzero block."""
  assert (data_next_ref is None) == (block_mask_ref is None)

  if data_next_ref is None and block_mask_ref is None:
    # Handle the case in which we have no masking nor next data information.
    # Simply fetch the next data and apply the mask for every block.
    assert m_next_ref is None
    next_data = i if next_i else j
    return (
        next_data,
        None,  # next mask
        True,  # should run
        False,  # should not mask
    )

  assert data_next_ref.shape == block_mask_ref.shape
  assert m_next_ref is None or data_next_ref.shape[0] == m_next_ref.shape[0]

  # We are working with one head only. Force the head index to 0.
  if data_next_ref.shape[0] == 1:
    h = 0

  # When scalar-memory data is of types smaller than int32, then we have to
  # upcast it back to use it in the kernel.

  to_i32 = lambda x: x.astype(jnp.int32)

  is_nonzero = to_i32(block_mask_ref[h, i, j]) > 0
  if m_next_ref is None:
    should_not_mask = True
    next_m = None
  else:
    should_not_mask = to_i32(block_mask_ref[h, i, j]) != 1
    next_m = to_i32(m_next_ref[h, i, j])
  next_j = to_i32(data_next_ref[h, i, j])
  return next_j, next_m, is_nonzero, should_not_mask


def _apply_mask(
    qk: jax.Array,
    should_not_mask,
    mask_ref,
    q_sequence_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    *,
    k_slice: pl.Slice,
    k_offset: int | jax.Array,
    bq: int,
    k_in_lanes: bool = True,
    mask_function: Callable[..., jax.Array] | None = None,
) -> jax.Array:
  """Applies the attention mask to attention logits."""
  assert mask_ref is None or q_sequence_ref is None
  assert (q_sequence_ref is None) == (mask_function is None)

  masks = []
  if mask_ref is not None:
    if k_in_lanes:
      mask = mask_ref[:, k_slice]
    else:
      mask = mask_ref[k_slice, :]

    snm = jnp.where(should_not_mask, 1, 0)
    masks.append(jnp.bitwise_or(mask, jnp.broadcast_to(snm, mask.shape)) != 0)

  if mask_function is not None:
    # Compute the mask using the given q_sequence indices.
    # KV indices are computed on the fly. This works because we only support Q
    # sequence sharding. If we wanted to compute Q indices too, then we would
    # need to keep into account the current shard along Q sequence.

    if k_in_lanes:
      assert q_sequence_ref.shape == (bq, NUM_LANES)

      k_sequence = k_offset + jax.lax.broadcasted_iota(
          jnp.int32, (bq, k_slice.size), 1
      )

      repeats, rem = divmod(k_slice.size, NUM_LANES)
      assert rem == 0
      q_sequence = pltpu.repeat(
          q_sequence_ref[...], repeats, axis=1
      )  # [bq, k_slice.size]
    else:
      assert q_sequence_ref.shape == (NUM_SUBLANES, bq)

      k_sequence = k_offset + jax.lax.broadcasted_iota(
          jnp.int32, (k_slice.size, bq), 0
      )
      q_sequence = q_sequence_ref[:1, :]  # [1, bq]
      q_sequence = jnp.broadcast_to(q_sequence, (k_slice.size, bq))

    assert q_sequence.shape == k_sequence.shape
    computed_mask = mask_function(q_sequence, k_sequence)  # pytype: disable=wrong-arg-count
    if computed_mask.dtype != jnp.dtype(jnp.bool_):
      raise ValueError(
          "Mask function must return a boolean-valued array, but got:"
          f" {computed_mask.dtype}"
      )
    masks.append(computed_mask)

  if q_segment_ids_ref is not None:
    if k_in_lanes:
      kv_ids = kv_segment_ids_ref[:1, k_slice]  # [1, k_slice]
      repeats, rem = divmod(kv_ids.shape[1], NUM_LANES)
      if rem:
        raise NotImplementedError(f"block_kv must be a multiple of {NUM_LANES}")
      q_ids = pltpu.repeat(q_segment_ids_ref[:], repeats, axis=1)  # [bq, bkv]
    else:
      assert bq == q_segment_ids_ref.shape[-1]
      repeats, rem = divmod(bq, NUM_LANES)
      if rem:
        raise NotImplementedError(f"block_q must be a multiple of {NUM_LANES}")
      kv_ids = pltpu.repeat(
          kv_segment_ids_ref[k_slice, :], repeats, axis=1
      )  # [k_slice, bq]
      q_ids = q_segment_ids_ref[:1, :]  # [1, bq]
    masks.append(q_ids == kv_ids)

  if masks:
    mask = functools.reduce(jnp.logical_and, masks)
    qk = jnp.where(mask, qk, 0.0)

  return qk


def _pointwise_splash_attention_fwd_kernel_impl(
    # Prefetched inputs
    data_next_ref,
    block_mask_ref,
    mask_next_ref,
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    mask_ref,
    q_sequence_ref,
    # Outputs
    o_ref,
    *,
    scale: float,
    seq_len: int,
    bq: int,
    bkv: int,
    bkv_compute: int,
    mask_function: Callable[..., jax.Array] | None = None,
):
  """Pointwise splash attention forward kernel implementation."""

  h, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)

  global_kv_index, _, should_run, should_not_mask = _next_nonzero(
      h, i, j, data_next_ref, block_mask_ref, mask_next_ref
  )

  def body(kv_compute_index, _):
    slice_k = pl.ds(kv_compute_index * bkv_compute, bkv_compute)

    q = q_ref[...]
    k = k_ref[slice_k, :]
    qk = jax.lax.dot_general(
        q, k, NT_DIM_NUMBERS, preferred_element_type=jnp.float32
    )
    assert qk.shape == (bq, bkv_compute)

    qk = qk * scale
    qk = qk * jax.lax.logistic(qk) * (1.0 / seq_len)

    qk = _apply_mask(  # pylint: disable=protected-access
        qk,
        should_not_mask=should_not_mask,
        mask_ref=mask_ref,
        q_sequence_ref=q_sequence_ref,
        q_segment_ids_ref=q_segment_ids_ref,
        kv_segment_ids_ref=kv_segment_ids_ref,
        k_slice=slice_k,
        # When the iteration space is shrunk (for local attention for example),
        # the kv_index program_id does not correspond to the actual coordinates
        # of the KV data. Make sure to use the 'unshrunk' index (coming from the
        # data_next array) when computing the mask.
        k_offset=global_kv_index * bkv + kv_compute_index * bkv_compute,
        bq=bq,
        k_in_lanes=True,
        mask_function=mask_function,
    )

    sv_dims = NN_DIM_NUMBERS
    v = v_ref[slice_k, :]

    to_float32 = lambda x: x.astype(jnp.float32)
    v = to_float32(v)
    o_curr = jax.lax.dot_general(qk, v, sv_dims)
    o_ref[:] = o_curr

  @pl.when(should_run)
  def run():  # pylint: disable=unused-variable
    assert bkv % bkv_compute == 0
    num_iters = k_ref.shape[0] // bkv_compute
    jax.lax.fori_loop(0, num_iters, body, None, unroll=True)


def _pointwise_splash_attention_fwd_kernel(
    params: _PointwiseAttentionParams,
    fwd_mask_info: splash_attention_mask_info.MaskInfo,
    q: jt.Float[jt.Array, "N T H"],
    k: jt.Float[jt.Array, "N S H"],
    v: jt.Float[jt.Array, "N S H"],
    segment_ids: splash_attention.SegmentIds | None,
) -> jt.Float[jt.Array, "N T H"]:
  """Pointwise splash attention forward kernel."""
  num_heads, q_seq_len, head_dim = q.shape
  _, kv_seq_len, _ = k.shape
  bq, bkv = params.block_q, params.block_kv
  bkv_compute = params.block_kv_compute

  if bkv % bkv_compute:
    raise ValueError(f"{bkv=} must be a multiple of {bkv_compute=}.")
  if bkv_compute % NUM_LANES:
    raise ValueError(f"{bkv_compute=} must be a multiple of {NUM_LANES}.")

  if segment_ids is not None:
    if segment_ids.q.shape != (q_seq_len,):
      raise ValueError(
          "Invalid shape for q segment_ids: "
          f"{segment_ids.q.shape}. Expected: {(q_seq_len,)}"
      )
    if segment_ids.kv.shape != (kv_seq_len,):
      raise ValueError(
          "Invalid shape for kv segment_ids: "
          f"{segment_ids.kv.shape}. Expected: {(kv_seq_len,)}"
      )

  if fwd_mask_info.data_next is not None:
    grid_width = fwd_mask_info.data_next.shape[-1]
  else:
    grid_width = kv_seq_len // bkv

  grid = (num_heads, q_seq_len // bq, grid_width)

  def kv_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None):
    next_j, *_ = _next_nonzero(
        h, i, j, data_next_ref, block_mask_ref, mask_next_ref
    )
    return h, next_j, 0

  def kv_segment_ids_index_map(
      h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None
  ):
    next_j, *_ = _next_nonzero(
        h, i, j, data_next_ref, block_mask_ref, mask_next_ref
    )
    return 0, next_j

  def m_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None):
    _, next_m, *_ = _next_nonzero(
        h, i, j, data_next_ref, block_mask_ref, mask_next_ref
    )
    return next_m, 0, 0

  q_spec = pl.BlockSpec((None, bq, head_dim), lambda h, i, *_: (h, i, 0))
  k_spec = pl.BlockSpec((None, bkv, head_dim), kv_index_map)
  v_spec = pl.BlockSpec((None, bkv, head_dim), kv_index_map)
  o_spec = pl.BlockSpec((None, bq, head_dim), lambda h, i, *_: (h, i, 0))
  o_shape = jax.ShapeDtypeStruct((num_heads, q_seq_len, head_dim), q.dtype)

  if segment_ids is not None:
    q_seg_spec = pl.BlockSpec((bq, NUM_LANES), lambda h, i, *_: (i, 0))
    kv_seg_spec = pl.BlockSpec((NUM_SUBLANES, bkv), kv_segment_ids_index_map)
    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q, (q_seq_len, NUM_LANES), (0,)
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv, (NUM_SUBLANES, kv_seq_len), (1,)
    )
  else:
    q_seg_spec = kv_seg_spec = None
    q_segment_ids = kv_segment_ids = None

  if fwd_mask_info.partial_mask_blocks is not None:
    m_spec = pl.BlockSpec((None, bq, bkv), m_index_map)
  else:
    m_spec = None

  assert (
      fwd_mask_info.partial_mask_blocks is None
      or fwd_mask_info.q_sequence is None
  )

  if fwd_mask_info.q_sequence is not None:
    q_seq_spec = pl.BlockSpec((bq, NUM_LANES), lambda h, i, *_: (i, 0))
    q_sequence = jax.lax.broadcast_in_dim(
        fwd_mask_info.q_sequence, (q_seq_len, NUM_LANES), (0,)
    )
  else:
    q_seq_spec = None
    q_sequence = None

  kernel_name = "pointwise_splash_attention_fwd_kernel_impl"
  with jax.named_scope(kernel_name):
    out = pl.pallas_call(
        functools.partial(
            _pointwise_splash_attention_fwd_kernel_impl,
            scale=params.scale,
            seq_len=q_seq_len,
            bq=bq,
            bkv=bkv,
            bkv_compute=bkv_compute,
            mask_function=params.mask_function,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[
                q_spec,
                k_spec,
                v_spec,
                q_seg_spec,
                kv_seg_spec,
                m_spec,
                q_seq_spec,
            ],
            out_specs=o_spec,
            grid=grid,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
        ),
        out_shape=o_shape,
        name=kernel_name,
        interpret=params.interpret,
    )(
        fwd_mask_info.data_next,
        fwd_mask_info.block_mask,
        fwd_mask_info.mask_next,
        q,
        k,
        v,
        q_segment_ids,
        kv_segment_ids,
        fwd_mask_info.partial_mask_blocks,
        q_sequence,
    )

  return out


def _pointwise_splash_attention_bwd_kernel_impl(
    # Prefetched inputs
    data_next_ref,
    block_mask_ref,
    mask_next_ref,
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    do_ref,
    mask_ref,
    q_sequence_ref,
    # Outputs
    dq_scratch_ref,
    dk_scratch_ref,
    dv_scratch_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    *,
    scale: float,
    seq_len: int,
    grid_width: int,
    bq: int,
    bkv: int,
    bkv_compute: int,
    mask_function: Callable[..., jax.Array] | None,
):
  """Pointwise splash attention backward kernel implementation."""
  j, h, i = pl.program_id(0), pl.program_id(1), pl.program_id(2)
  should_initialize = i == 0

  @pl.when(should_initialize)
  def _():
    dk_scratch_ref[...] = jnp.zeros_like(dk_scratch_ref)
    dv_scratch_ref[...] = jnp.zeros_like(dv_scratch_ref)

  _, _, should_run, should_not_mask = _next_nonzero(
      h, i, j, data_next_ref, block_mask_ref, mask_next_ref, next_i=True
  )

  def body(i, _):
    slice_k = pl.ds(i * bkv_compute, bkv_compute)
    q = q_ref[...]
    k = k_ref[slice_k, :]
    v = v_ref[slice_k, :]
    do = do_ref[...]

    qk = jax.lax.dot_general(
        k, q, NT_DIM_NUMBERS, preferred_element_type=jnp.float32
    )
    qk = qk * scale
    sigmoid_qk = jax.lax.logistic(qk)

    # It doesn't matter where we apply the mask since it is just zeroing out
    # elements.
    sigmoid_qk = _apply_mask(
        sigmoid_qk,
        should_not_mask,
        mask_ref,
        q_sequence_ref,
        q_segment_ids_ref,
        kv_segment_ids_ref,
        k_slice=slice_k,
        k_offset=j * bkv + i * bkv_compute,
        bq=bq,
        k_in_lanes=False,
        mask_function=mask_function,
    )

    p = qk * sigmoid_qk * (1.0 / seq_len)

    dv = jax.lax.dot(p.astype(do.dtype), do, preferred_element_type=jnp.float32)
    dv_scratch_ref[slice_k, :] = (
        dv.astype(dv_scratch_ref.dtype) + dv_scratch_ref[slice_k, :]
    )

    dp = jax.lax.dot_general(
        v, do, NT_DIM_NUMBERS, preferred_element_type=jnp.float32
    )
    dp = dp * sigmoid_qk * (1 + qk * (1 - sigmoid_qk)) * (1.0 / seq_len)

    dk = jax.lax.dot_general(
        dp.astype(do.dtype),
        q,
        NN_DIM_NUMBERS,
        preferred_element_type=jnp.float32,
    )
    dk = dk * scale
    dk_scratch_ref[slice_k, :] = (
        dk.astype(dk_scratch_ref.dtype) + dk_scratch_ref[slice_k, :]
    )

    if dq_scratch_ref is not None or dq_ref is not None:
      dq = jax.lax.dot_general(
          dp.astype(k.dtype),
          k,
          TN_DIM_NUMBERS,
          preferred_element_type=jnp.float32,
      )
      dq = dq * scale
      if dq_scratch_ref is not None:  # Compute block size != memory block size
        dq_scratch_ref[...] += dq
      else:
        dq_ref[...] = dq.astype(dq_ref.dtype)

  if dq_scratch_ref is not None:
    dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)
  elif dq_scratch_ref is None and dq_ref is not None:
    dq_ref[...] = jnp.zeros_like(dq_ref)

  @pl.when(should_run)
  def _():
    num_iters = k_ref.shape[0] // bkv_compute
    jax.lax.fori_loop(0, num_iters, body, None, unroll=True)

  if dq_scratch_ref is not None:
    dq_ref[...] = dq_scratch_ref[...].astype(dq_ref.dtype)

  should_write = i == grid_width - 1

  @pl.when(should_write)
  def _():
    dk_ref[...] = dk_scratch_ref[...].astype(dk_ref.dtype)
    dv_ref[...] = dv_scratch_ref[...].astype(dv_ref.dtype)

    if dq_scratch_ref is not None:
      dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)
    dk_scratch_ref[...] = jnp.zeros_like(dk_scratch_ref)
    dv_scratch_ref[...] = jnp.zeros_like(dv_scratch_ref)


def _pointwise_splash_attention_bwd_kernel(
    params: _PointwiseAttentionParams,
    bwd_mask_info: splash_attention_mask_info.MaskInfo,
    q: jt.Float[jt.Array, "N T H"],
    k: jt.Float[jt.Array, "N S H"],
    v: jt.Float[jt.Array, "N S H"],
    segment_ids: splash_attention.SegmentIds | None,
    d_out: jt.Float[jt.Array, "N T H"],
) -> tuple[
    jt.Float[jt.Array, "N T H"],
    jt.Float[jt.Array, "N S H"],
    jt.Float[jt.Array, "N S H"],
]:
  """Pointwise splash attention backward kernel."""

  num_heads, q_seq_len, head_dim = q.shape
  _, kv_seq_len, _ = k.shape
  bq, bkv = params.block_q_bwd, params.block_kv_bwd
  bkv_compute = params.block_kv_bwd_compute

  if bq > q_seq_len:
    raise ValueError(f"{bq=} should not be greater than {q_seq_len}.")
  if bkv > kv_seq_len:
    raise ValueError(f"{bkv=} should not be greater than {kv_seq_len}.")
  if bkv_compute > bkv:
    raise ValueError(f"{bkv_compute=} should not be greater than {bkv=}.")
  if bkv % bkv_compute:
    raise ValueError(f"{bkv=} should be a multiple of {bkv_compute=}.")

  if bwd_mask_info.data_next is not None:
    grid_width = bwd_mask_info.data_next.shape[-2]
  else:
    grid_width = q_seq_len // bq

  grid = (kv_seq_len // bkv, num_heads, grid_width)

  def q_index_map(j, h, i, data_next_ref, block_mask_ref, mask_next_ref=None):
    next_i, *_ = _next_nonzero(
        h, i, j, data_next_ref, block_mask_ref, mask_next_ref, next_i=True
    )
    return h, next_i, 0

  def q_segment_ids_index_map(
      j, h, i, data_next_ref, block_mask_ref, mask_next_ref=None
  ):
    next_i, *_ = _next_nonzero(
        h, i, j, data_next_ref, block_mask_ref, mask_next_ref, next_i=True
    )
    return 0, next_i

  def m_index_map(j, h, i, data_next_ref, block_mask_ref, mask_next_ref=None):
    _, next_m, *_ = _next_nonzero(
        h, i, j, data_next_ref, block_mask_ref, mask_next_ref
    )
    return next_m, 0, 0

  q_spec = pl.BlockSpec((None, bq, head_dim), q_index_map)
  k_spec = pl.BlockSpec((None, bkv, head_dim), lambda j, h, *_: (h, j, 0))
  v_spec = pl.BlockSpec((None, bkv, head_dim), lambda j, h, *_: (h, j, 0))
  do_spec = pl.BlockSpec((None, bq, head_dim), q_index_map)
  dq_spec = pl.BlockSpec(
      (None, None, bq, head_dim), lambda j, h, i, *_: (j, h, i, 0)
  )
  dk_spec = pl.BlockSpec((None, bkv, head_dim), lambda j, h, *_: (h, j, 0))
  dv_spec = pl.BlockSpec((None, bkv, head_dim), lambda j, h, *_: (h, j, 0))
  dk_scratch_spec = pl.BlockSpec((bkv, head_dim), lambda *_: (0, 0))
  dv_scratch_spec = pl.BlockSpec((bkv, head_dim), lambda *_: (0, 0))

  dq_shape = jax.ShapeDtypeStruct((kv_seq_len // bkv, *q.shape), q.dtype)
  dk_shape = jax.ShapeDtypeStruct(k.shape, k.dtype)
  dv_shape = jax.ShapeDtypeStruct(v.shape, v.dtype)
  dk_scratch_shape = jax.ShapeDtypeStruct((bkv, head_dim), jnp.float32)
  dv_scratch_shape = jax.ShapeDtypeStruct((bkv, head_dim), jnp.float32)

  if bkv != bkv_compute:
    dq_scratch_spec = pl.BlockSpec((bq, head_dim), lambda *_: (0, 0))
    dq_scratch_shape = jax.ShapeDtypeStruct((bq, head_dim), jnp.float32)
  else:
    dq_scratch_spec = dq_scratch_shape = None

  if segment_ids is not None:
    q_seg_spec = pl.BlockSpec((NUM_SUBLANES, bq), q_segment_ids_index_map)
    kv_seg_spec = pl.BlockSpec((bkv, NUM_LANES), lambda j, *_: (j, 0))
    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q, (NUM_SUBLANES, q_seq_len), (1,)
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv, (kv_seq_len, NUM_LANES), (0,)
    )
  else:
    q_seg_spec = kv_seg_spec = q_segment_ids = kv_segment_ids = None

  if bwd_mask_info.partial_mask_blocks is not None:
    m_spec = pl.BlockSpec((None, bkv, bq), m_index_map)
  else:
    m_spec = None

  if bwd_mask_info.q_sequence is not None:
    q_seq_spec = pl.BlockSpec((NUM_SUBLANES, bq), q_segment_ids_index_map)
    q_sequence = jax.lax.broadcast_in_dim(
        bwd_mask_info.q_sequence, (NUM_SUBLANES, q_seq_len), (1,)
    )
  else:
    q_seq_spec = q_sequence = None

  kernel_name = "pointwise_splash_attention_bwd_kernel_impl"
  with jax.named_scope(kernel_name):
    _, _, _, dq_unreduced, dk, dv = pl.pallas_call(
        functools.partial(
            _pointwise_splash_attention_bwd_kernel_impl,
            scale=params.scale,
            seq_len=q_seq_len,
            grid_width=grid_width,
            bq=bq,
            bkv=bkv,
            bkv_compute=bkv_compute,
            mask_function=params.mask_function,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[
                q_spec,
                k_spec,
                v_spec,
                q_seg_spec,
                kv_seg_spec,
                do_spec,
                m_spec,
                q_seq_spec,
            ],
            out_specs=[
                dq_scratch_spec,
                dk_scratch_spec,
                dv_scratch_spec,
                dq_spec,
                dk_spec,
                dv_spec,
            ],
            grid=grid,
        ),
        out_shape=[
            dq_scratch_shape,
            dk_scratch_shape,
            dv_scratch_shape,
            dq_shape,
            dk_shape,
            dv_shape,
        ],
        # We set all dimensions to arbitrary because:
        # 1) for kv_seq_len, the prefetch schedule assumes no
        #    megacore
        # 2) for heads, we are reducing over heads
        # 3) for q_seq_len, we are reducing over it to compute dkv
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary", "arbitrary"),
        ),
        name=kernel_name,
        interpret=params.interpret,
    )(
        bwd_mask_info.data_next,
        bwd_mask_info.block_mask,
        bwd_mask_info.mask_next,
        q,
        k,
        v,
        q_segment_ids,
        kv_segment_ids,
        d_out,
        bwd_mask_info.partial_mask_blocks,
        q_sequence,
    )
  dq = jnp.sum(dq_unreduced, axis=0)
  return dq, dk, dv


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _pointwise_splash_attention(
    params: _PointwiseAttentionParams,
    fwd_mask_info: splash_attention_mask_info.MaskInfo,
    bwd_mask_info: splash_attention_mask_info.MaskInfo,
    q: jt.Float[jt.Array, "N T H"],
    k: jt.Float[jt.Array, "N S H"],
    v: jt.Float[jt.Array, "N S H"],
    segment_ids: splash_attention.SegmentIds | None,
) -> jt.Float[jt.Array, "N T H"]:
  """Pointwise splash attention kernel with custom VJP."""
  # The forward function does not use the dqkv MaskInfo, it just forwards
  # them to the backward function as residuals. This is a way to communicate
  # arbitrary Arrays to the backward function. Since the MaskInfos are constants
  # there is no overhead in passing them to the backward function as residuals.
  # When sharding computation MaskInfos are partitioned so both the forward
  # and the backward kernels need to work on the relevant slice. If we
  # recomputed the backward MaskInfos in the backward function from the numpy
  # mask then we would not work with the MaskInfo slice relevant to the current
  # device.
  del bwd_mask_info
  return _pointwise_splash_attention_fwd_kernel(
      params, fwd_mask_info, q, k, v, segment_ids
  )


def _pointwise_splash_attention_fwd(
    params: _PointwiseAttentionParams,
    fwd_mask_info: splash_attention_mask_info.MaskInfo,
    bwd_mask_info: splash_attention_mask_info.MaskInfo,
    q: jt.Float[jt.Array, "N T H"],
    k: jt.Float[jt.Array, "N S H"],
    v: jt.Float[jt.Array, "N S H"],
    segment_ids: splash_attention.SegmentIds | None,
) -> tuple[
    jt.Float[jt.Array, "N T H"],
    tuple[
        jt.Float[jt.Array, "N T H"],
        jt.Float[jt.Array, "N S H"],
        jt.Float[jt.Array, "N S H"],
        splash_attention.SegmentIds | None,
        splash_attention_mask_info.MaskInfo,
    ],
]:
  """Pointwise splash attention forward implementation."""

  out = _pointwise_splash_attention_fwd_kernel(
      params, fwd_mask_info, q, k, v, segment_ids
  )
  return out, (q, k, v, segment_ids, bwd_mask_info)


def _pointwise_splash_attention_bwd(
    params: _PointwiseAttentionParams,
    res: tuple[
        jt.Float[jt.Array, "N T H"],
        jt.Float[jt.Array, "N S H"],
        jt.Float[jt.Array, "N S H"],
        splash_attention.SegmentIds | None,
        splash_attention_mask_info.MaskInfo,
    ],
    d_out: jt.Float[jt.Array, "N T H"],
) -> tuple[
    None,
    None,
    jt.Float[jt.Array, "N T H"],
    jt.Float[jt.Array, "N S H"],
    jt.Float[jt.Array, "N S H"],
    None,
]:
  """Pointwise splash attention backward implementation."""
  q, k, v, segment_ids, bwd_mask_info = res
  dq, dk, dv = _pointwise_splash_attention_bwd_kernel(
      params, bwd_mask_info, q, k, v, segment_ids, d_out
  )
  return None, None, dq, dk, dv, None


_pointwise_splash_attention.defvjp(
    _pointwise_splash_attention_fwd, _pointwise_splash_attention_bwd
)


@functools.partial(
    jax.jit,
    static_argnames=[
        "mask_type",
        "sliding_window_size",
        "qkv_axis_names",
        "segment_id_axis_names",
        "scale",
        "block_q",
        "block_kv",
        "block_kv_compute",
        "block_q_dkv",
        "block_kv_dkv",
        "block_kv_dkv_compute",
        "head_shards",
        "q_seq_shards",
        "downcast_smem_data",
        "interpret",
    ],
)
def pointwise_splash_attention(
    mask_type: MaskType | np.ndarray,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    q_segment_ids: jax.Array | None = None,
    kv_segment_ids: jax.Array | None = None,
    sliding_window_size: int | None = None,
    qkv_axis_names: tuple[str | None, ...] = (),
    segment_id_axis_names: tuple[str | None, ...] = (),
    scale: float | None = None,
    block_q: int | None = None,
    block_kv: int | None = None,
    block_kv_compute: int | None = None,
    block_q_dkv: int | None = None,
    block_kv_dkv: int | None = None,
    block_kv_dkv_compute: int | None = None,
    head_shards: int = 1,
    q_seq_shards: int = 1,
    downcast_smem_data: bool = True,
    interpret: bool = False,
) -> jax.Array:
  """Computes block-sparse pointwise HSTU attention as in [1].

  While the authors used Jagged tensors in PyTorch to achieve 'fully raggified'
  attention computations, we substitute this with block sparsity on TPU.

  NOTE: This currently does not support adding a relative attention bias to the
  attention logits.

  [1] https://arxiv.org/abs/2402.17152

  Args:
    mask_type: The type of mask to scalar prefetch in the op. This will
      determine the mask that is fetched to SMEM on TPU and used for dynamic
      block fetching in the kernel. Can be a MaskType enum or a numpy array of
      shape (q_len, kv_len) with dtype bool.
    q: Query tensor of shape [batch, q_length, num_heads, head_dim].
    k: Key tensor of shape [batch, kv_length, num_heads, head_dim].
    v: Value tensor of shape [batch, kv_length, num_heads, head_dim].
    q_segment_ids: Query segment ids tensor of shape [batch, q_length].  These
      are used to determine the positions to ignore in the sequence when
      computing the softmax, i.e. PAD tokens other sequences.
    kv_segment_ids: Key, value segment ids tensor of shape [batch, kv_length].
      These are used to determine the positions to ignore in the sequence when
      computing the softmax, i.e. PAD tokens other sequences.
    sliding_window_size: Optional sliding window size for the attention kernel.
      If specified, the kernel will compute local attention over sliding windows
      of this size.
    qkv_axis_names: Axis names for partitioning query, key, and value tensors.
    segment_id_axis_names: Axis names for partitioning segment id tensors.
    scale: Scaling factor for the attention logits. If None, the scale defaults
      to as 1 / (query_sequence_length * (head_dim ** 0.5)).
    block_q: Block size for the query tensor.
    block_kv: Block size for the key and value tensors.
    block_kv_compute: Block size for the key and value tensors when computing
      the attention logits.
    block_q_dkv: Block size for the query tensor when computing the attention
      logits and the key and value tensors when computing the attention
      activations.
    block_kv_dkv: Block size for the key and value tensors when computing the
      attention activations.
    block_kv_dkv_compute: Block size for the key and value tensors when
      computing the attention logits and activations.
    head_shards: Number of head shards to use for the attention kernel.
    q_seq_shards: Number of query sequence shards to use for the attention
      kernel.
    downcast_smem_data: Whether to downcast data in SMEM to bfloat16.
    interpret: Whether to run the kernel in interpret mode.

  Returns:
    Output tensor of shape [batch, q_length, num_heads, v_head_dim].
  """
  bs, q_len, num_heads, head_dim = q.shape
  _, kv_len, _, _ = v.shape
  chex.assert_shape(q, (bs, q_len, num_heads, head_dim))
  chex.assert_shape(k, (bs, kv_len, num_heads, head_dim))
  chex.assert_shape(v, (bs, kv_len, num_heads, head_dim))

  scale = scale if scale is not None else 1.0 / (head_dim**0.5)
  block_q = block_q if block_q is not None else min(512, q_len)
  block_kv = block_kv if block_kv is not None else min(512, kv_len)
  block_kv_compute = (
      block_kv_compute if block_kv_compute is not None else block_kv
  )
  block_q_dkv = block_q_dkv if block_q_dkv is not None else min(512, q_len)
  block_kv_dkv = block_kv_dkv if block_kv_dkv is not None else min(512, kv_len)
  block_kv_dkv_compute = (
      block_kv_dkv_compute if block_kv_dkv_compute is not None else block_kv_dkv
  )

  # Transpose to ('batch', 'heads', 'length', 'head_dim')
  query = jnp.transpose(q, axes=(0, 2, 1, 3))
  key = jnp.transpose(k, axes=(0, 2, 1, 3))
  value = jnp.transpose(v, axes=(0, 2, 1, 3))

  if isinstance(mask_type, np.ndarray):
    if mask_type.shape != (q_len, kv_len):
      raise ValueError(
          f"Mask shape {mask_type.shape} does not match query and key length"
          f" ({q_len}, {kv_len})."
      )
    mask = splash_attention.NumpyMask(mask_type)
  elif mask_type == MaskType.CAUSAL:
    mask = splash_attention.CausalMask((q_len, kv_len))
  elif mask_type == MaskType.FULL:
    mask = splash_attention.FullMask((q_len, kv_len))
  else:
    raise ValueError(f"Unsupported mask type: '{mask_type}'.")

  if sliding_window_size is not None:
    mask &= splash_attention.LocalMask(
        (q_len, kv_len),
        window_size=(sliding_window_size, sliding_window_size),
        offset=0,
    )

  multi_head_mask = splash_attention.MultiHeadMask(
      [mask for _ in range(num_heads)]
  )
  fwd_mask_info, mask_function_fwd = splash_attention_mask_info.process_mask(
      multi_head_mask,
      (block_q, block_kv),
      downcast_smem_data=downcast_smem_data,
      head_shards=head_shards,
      q_seq_shards=q_seq_shards,
  )
  fwd_mask_info = jax.tree.map(jnp.array, fwd_mask_info)

  bwd_mask_info, mask_function_dkv = (
      splash_attention_mask_info.process_mask_dkv(
          multi_head_mask,
          (block_q_dkv, block_kv_dkv),
          downcast_smem_data=downcast_smem_data,
          head_shards=head_shards,
          q_seq_shards=q_seq_shards,
          shrink_grid=False,
      )
  )
  assert (mask_function_fwd is None) == (mask_function_dkv is None)
  bwd_mask_info = jax.tree.map(jnp.array, bwd_mask_info)

  segment_ids = None
  if q_segment_ids is not None and kv_segment_ids is not None:
    segment_ids = splash_attention.SegmentIds(q_segment_ids, kv_segment_ids)
  elif q_segment_ids is not None or kv_segment_ids is not None:
    raise ValueError(
        "Both or neither of `q_segment_ids` and `kv_segment_ids` must be"
        " provided."
    )

  if (global_abstract_mesh := jax.sharding.get_abstract_mesh()).shape_tuple:
    abstract_mesh = global_abstract_mesh
  elif (distribution := keras.distribution.distribution()) is not None:
    device_mesh: keras.distribution.DeviceMesh = distribution.device_mesh
    abstract_mesh = jax.sharding.AbstractMesh(
        axis_sizes=tuple(device_mesh.shape),
        axis_names=tuple(device_mesh.axis_names),
    )
  else:
    abstract_mesh = None

  def _kernel_wrapper(
      query: jax.Array,
      key: jax.Array,
      value: jax.Array,
      segment_ids: splash_attention.SegmentIds | None,
  ) -> jax.Array:
    params = _PointwiseAttentionParams(
        scale=scale,
        mask_function=mask_function_fwd,
        block_q=block_q,
        block_kv=block_kv,
        block_kv_compute=block_kv_compute,
        block_q_bwd=block_q_dkv,
        block_kv_bwd=block_kv_dkv,
        block_kv_bwd_compute=block_kv_dkv_compute,
        interpret=interpret,
    )
    return jax.vmap(
        functools.partial(
            _pointwise_splash_attention,
            params,
            fwd_mask_info,
            bwd_mask_info,
        )
    )(query, key, value, segment_ids=segment_ids)

  if abstract_mesh is not None and abstract_mesh.shape_tuple:
    if not qkv_axis_names:
      qkv_axis_names = (
          abstract_mesh.axis_names[0],  # batch dimension
          None,  # heads dimension
          None,  # length dimension
          None,  # hidden dimension
      )
    if not segment_id_axis_names:
      segment_id_axis_names = (
          abstract_mesh.axis_names[0],  # batch dimension
          None,  # length dimension
      )

    qkv_pspecs = jax.sharding.PartitionSpec(*qkv_axis_names)
    segment_id_pspecs = jax.sharding.PartitionSpec(*segment_id_axis_names)
    x = jax.shard_map(
        _kernel_wrapper,
        mesh=abstract_mesh,
        in_specs=(qkv_pspecs, qkv_pspecs, qkv_pspecs, segment_id_pspecs),
        out_specs=qkv_pspecs,
        check_vma=False,  # Pallas kernels do not support `check_vma`.
    )(query, key, value, segment_ids)
  else:
    x = _kernel_wrapper(query, key, value, segment_ids)

  x = jnp.transpose(x, axes=(0, 2, 1, 3))
  chex.assert_shape(x, (bs, q_len, num_heads, head_dim))
  return x

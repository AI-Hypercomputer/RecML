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
"""Mini benchmarks for binary_cross_entropy_ops.

These benchmarks run hundreds of steps on a large vocabulary, so they are
considerably slower than the correctness tests in
`binary_cross_entropy_ops_test.py`. The corresponding BUILD target is tagged
`manual` so that it does not slow down presubmit TAP and only runs in
post-submit.
"""

import time

from absl import logging
from absl.testing import absltest
import jax
import jax.numpy as jnp
import keras
from recml.core.ops import binary_cross_entropy_ops
from recml.core.ops import binary_focal_cross_entropy_ops

_BATCH = 32
_SEQ_LEN = 32
_HIDDEN_DIM = 128
_VOCAB_SIZE = 100_000
_NUM_LABELS = 8
_NUM_STEPS = 200
_BLOCK_V = 4096


def _make_inputs() -> tuple[jax.Array, jax.Array, jax.Array]:
  """Returns random (activations, embeddings, targets) benchmark inputs."""
  key_act, key_emb, key_tgt = jax.random.split(jax.random.PRNGKey(42), 3)

  activations = jax.random.normal(key_act, (_BATCH, _SEQ_LEN, _HIDDEN_DIM))
  embeddings = jax.random.normal(key_emb, (_VOCAB_SIZE, _HIDDEN_DIM))
  targets = jax.random.randint(
      key_tgt, (_BATCH, _SEQ_LEN, _NUM_LABELS), 0, _VOCAB_SIZE
  )
  return activations, embeddings, targets


def _benchmark(name: str, grad_fn, activations, embeddings) -> float:
  """Compiles `grad_fn` and returns its average step time in seconds."""
  logging.info('Compiling %s...', name)
  t0 = time.time()
  compiled = grad_fn.lower(activations, embeddings).compile()
  mem = compiled.memory_analysis()
  grad_fn(activations, embeddings)[0].block_until_ready()
  temp_mb = (mem.temp_size_in_bytes / (1024 * 1024)) if mem else -1.0
  logging.info(
      '%s compiled in %.2f s (temp HBM: %.2f MiB)',
      name,
      time.time() - t0,
      temp_mb,
  )

  logging.info('Benchmarking %s (%d steps)...', name, _NUM_STEPS)
  t0 = time.time()
  for _ in range(_NUM_STEPS):
    g_act, _ = grad_fn(activations, embeddings)
    g_act.block_until_ready()
  step_time = (time.time() - t0) / _NUM_STEPS
  logging.info(
      '%s step time: %.4f ms (temp HBM: %.2f MiB)',
      name,
      step_time * 1000,
      temp_mb,
  )
  return step_time


class BinaryCrossEntropyOpsBenchmarkTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Skipping TPU benchmark.')

  def test_mini_benchmark_keras(self):
    activations, embeddings, targets = _make_inputs()

    # Convert targets to multi-hot for Keras.
    one_hot = jax.nn.one_hot(targets, _VOCAB_SIZE, axis=-1)  # (B, N, L, V)
    multi_hot = jnp.max(one_hot, axis=-2)  # (B, N, V)

    def run_keras(act, emb):
      logits = jnp.matmul(act, emb.T)  # (B, N, V)
      loss_per_token = keras.losses.binary_crossentropy(
          multi_hot, logits, from_logits=True
      )
      return jnp.mean(loss_per_token)

    grad_keras = jax.jit(jax.grad(run_keras, argnums=(0, 1)))
    _benchmark('Keras', grad_keras, activations, embeddings)

  def test_mini_benchmark_cut_bce(self):
    activations, embeddings, targets = _make_inputs()

    def run_cut(act, emb, bv=_BLOCK_V):
      return binary_cross_entropy_ops.cut_binary_cross_entropy(
          act, emb, targets, block_v=bv, optimize_large_l=True
      )

    grad_cut = jax.jit(jax.grad(run_cut, argnums=(0, 1)))
    _benchmark(
        f'Cut (block_v={_BLOCK_V})',
        grad_cut,
        activations,
        embeddings,
    )

  def test_mini_benchmark_unified_balanced_block_v_and_focal(self):
    activations, embeddings, targets = _make_inputs()
    lane = binary_cross_entropy_ops._pallas_lane()
    padded_d = binary_cross_entropy_ops.pl.align_to(_HIDDEN_DIM, lane)
    vmem_budget = binary_cross_entropy_ops._pallas_vmem_budget()
    raw_max_bv = binary_cross_entropy_ops._max_safe_block_v(
        vmem_budget, padded_d, lane, embeddings.dtype
    )
    resolved_bv = binary_cross_entropy_ops._resolve_block_v(
        vocab=_VOCAB_SIZE,
        hidden=_HIDDEN_DIM,
        local_n=_BATCH * _SEQ_LEN,
        block_n=lane,
        dtype=embeddings.dtype,
        user_block_v=None,
        use_pallas=True,
        force_bwd_block_v=False,
    )
    logging.info(
        'Device=%s, VMEM budget=%.1f MiB, raw max_safe_block_v=%d, '
        'unified balanced block_v=%d',
        jax.devices()[0].device_kind,
        vmem_budget / (1024 * 1024),
        raw_max_bv,
        resolved_bv,
    )

    # 1. Unified balanced block_v on V=100,000 vs legacy raw max_safe_block_v
    def run_cut_balanced(act, emb):
      return binary_cross_entropy_ops.cut_binary_cross_entropy(
          act, emb, targets, block_v=None
      )

    def run_cut_legacy_raw(act, emb):
      return binary_cross_entropy_ops.cut_binary_cross_entropy(
          act, emb, targets, block_v=raw_max_bv, force_bwd_block_v=True
      )

    grad_balanced = jax.jit(jax.grad(run_cut_balanced, argnums=(0, 1)))
    grad_legacy = jax.jit(jax.grad(run_cut_legacy_raw, argnums=(0, 1)))
    t_bal = _benchmark(
        f'CutBCE unified balanced (V=100k, block_v={resolved_bv})',
        grad_balanced,
        activations,
        embeddings,
    )
    t_leg = _benchmark(
        f'CutBCE legacy raw ceiling (V=100k, block_v={raw_max_bv})',
        grad_legacy,
        activations,
        embeddings,
    )

    # 2. 128-aligned vocab V=64,000 where _balanced_block_v finds exact divisor
    v_aligned = 64_000
    emb_64k = embeddings[:v_aligned]
    tgt_64k = jnp.mod(targets, v_aligned)
    bal_64k = binary_cross_entropy_ops._resolve_block_v(
        vocab=v_aligned,
        hidden=_HIDDEN_DIM,
        local_n=_BATCH * _SEQ_LEN,
        block_n=lane,
        dtype=embeddings.dtype,
    )

    def run_64k_balanced(act, emb):
      return binary_cross_entropy_ops.cut_binary_cross_entropy(
          act, emb, tgt_64k, block_v=None
      )

    def run_64k_legacy_raw(act, emb):
      return binary_cross_entropy_ops.cut_binary_cross_entropy(
          act, emb, tgt_64k, block_v=raw_max_bv, force_bwd_block_v=True
      )

    t_64k_bal = _benchmark(
        f'CutBCE 64k exact-divisor balanced (block_v={bal_64k}, 0 pad)',
        jax.jit(jax.grad(run_64k_balanced, argnums=(0, 1))),
        activations,
        emb_64k,
    )
    t_64k_leg = _benchmark(
        f'CutBCE 64k legacy raw ceiling (block_v={raw_max_bv}, padded)',
        jax.jit(jax.grad(run_64k_legacy_raw, argnums=(0, 1))),
        activations,
        emb_64k,
    )
    logging.info(
        'Summary: V=100k balanced=%.4f ms vs raw=%.4f ms; '
        'V=64k balanced=%.4f ms vs raw=%.4f ms',
        t_bal * 1000,
        t_leg * 1000,
        t_64k_bal * 1000,
        t_64k_leg * 1000,
    )

    # 3. Focal BCE with unified balanced block_v and 2D target loop
    def run_focal_auto(act, emb):
      return binary_focal_cross_entropy_ops.cut_binary_focal_cross_entropy(
          act, emb, targets, block_v=None
      )

    grad_focal = jax.jit(jax.grad(run_focal_auto, argnums=(0, 1)))
    _benchmark('CutFocalBCE auto unified', grad_focal, activations, embeddings)


if __name__ == '__main__':
  absltest.main()

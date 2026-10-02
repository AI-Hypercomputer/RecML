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
from absl.testing import parameterized
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


def _make_inputs(
    num_labels: int = _NUM_LABELS,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Returns random (activations, embeddings, targets) benchmark inputs."""
  key_act, key_emb, key_tgt = jax.random.split(jax.random.PRNGKey(42), 3)

  activations = jax.random.normal(key_act, (_BATCH, _SEQ_LEN, _HIDDEN_DIM))
  embeddings = jax.random.normal(key_emb, (_VOCAB_SIZE, _HIDDEN_DIM))
  targets = jax.random.randint(
      key_tgt, (_BATCH, _SEQ_LEN, num_labels), 0, _VOCAB_SIZE
  )
  return activations, embeddings, targets


def _benchmark(name: str, grad_fn, activations, embeddings) -> float:
  """Compiles `grad_fn` and returns its average step time in seconds."""
  logging.info('Compiling %s...', name)
  t0 = time.time()
  grad_fn(activations, embeddings)[0].block_until_ready()
  logging.info('%s compiled in %.2f s', name, time.time() - t0)

  logging.info('Benchmarking %s (%d steps)...', name, _NUM_STEPS)
  t0 = time.time()
  for _ in range(_NUM_STEPS):
    g_act, _ = grad_fn(activations, embeddings)
    g_act.block_until_ready()
  step_time = (time.time() - t0) / _NUM_STEPS
  msg = f'{name} step time: {step_time * 1000:.4f} ms'
  logging.info('%s', msg)
  print(msg, flush=True)
  return step_time


class BinaryCrossEntropyOpsBenchmarkTest(parameterized.TestCase):

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
          act, emb, targets, block_v=bv
      )

    grad_cut = jax.jit(jax.grad(run_cut, argnums=(0, 1)))
    _benchmark(
        f'Cut (block_v={_BLOCK_V})',
        grad_cut,
        activations,
        embeddings,
    )

  @parameterized.named_parameters(
      ('L8', 8),
      ('L16', 16),
      ('L32', 32),
      ('L64', 64),
  )
  def test_mini_benchmark_cut_bce_across_l(self, num_labels: int):
    activations, embeddings, targets = _make_inputs(num_labels=num_labels)

    def run_cut_current(act, emb):
      return binary_cross_entropy_ops.cut_binary_cross_entropy(
          act, emb, targets, block_v=_BLOCK_V, optimize_large_l=False
      )

    def run_cut_flag_true(act, emb):
      return binary_cross_entropy_ops.cut_binary_cross_entropy(
          act, emb, targets, block_v=_BLOCK_V, optimize_large_l=True
      )

    t_cur = _benchmark(
        f'CutBCE L={num_labels} (optimize_large_l=False)',
        jax.jit(jax.grad(run_cut_current, argnums=(0, 1))),
        activations,
        embeddings,
    )
    t_opt = _benchmark(
        f'CutBCE L={num_labels} (optimize_large_l=True)',
        jax.jit(jax.grad(run_cut_flag_true, argnums=(0, 1))),
        activations,
        embeddings,
    )

    def run_focal_current(act, emb):
      return binary_focal_cross_entropy_ops.cut_binary_focal_cross_entropy(
          act,
          emb,
          targets,
          block_v=_BLOCK_V,
          gamma=2.0,
          alpha=0.25,
          apply_class_balancing=True,
          optimize_large_l=False,
      )

    def run_focal_flag_true(act, emb):
      return binary_focal_cross_entropy_ops.cut_binary_focal_cross_entropy(
          act,
          emb,
          targets,
          block_v=_BLOCK_V,
          gamma=2.0,
          alpha=0.25,
          apply_class_balancing=True,
          optimize_large_l=True,
      )

    t_focal_cur = _benchmark(
        f'CutFocalBCE L={num_labels} (optimize_large_l=False)',
        jax.jit(jax.grad(run_focal_current, argnums=(0, 1))),
        activations,
        embeddings,
    )
    t_focal_opt = _benchmark(
        f'CutFocalBCE L={num_labels} (optimize_large_l=True)',
        jax.jit(jax.grad(run_focal_flag_true, argnums=(0, 1))),
        activations,
        embeddings,
    )

    summary = (
        f'[SUMMARY L={num_labels}] CutBCE: cur={t_cur*1000:.3f}ms,'
        f' opt={t_opt*1000:.3f}ms | CutFocalBCE: cur={t_focal_cur*1000:.3f}ms,'
        f' opt={t_focal_opt*1000:.3f}ms'
    )
    logging.info('%s', summary)
    print(summary, flush=True)


if __name__ == '__main__':
  absltest.main()

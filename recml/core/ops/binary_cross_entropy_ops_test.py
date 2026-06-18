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
"""Tests for binary_cross_entropy_ops."""

import time
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import keras
import numpy as np
from recml.core.ops import binary_cross_entropy_ops


def _naive_bce(activations, embeddings, targets, weights=None):
  """Naive implementation that materializes the full logits matrix."""
  vocab_size = embeddings.shape[0]
  logits = jnp.matmul(activations, embeddings.T)  # (B, N, V)

  # targets: (B, N, L) -> multi_hot: (B, N, V)
  one_hot = jax.nn.one_hot(targets, vocab_size, axis=-1)  # (B, N, L, V)
  multi_hot = jnp.max(one_hot, axis=-2)  # (B, N, V)

  # Compute stable BCE loss per class
  # Loss = max(x, 0) - x * y + log(1 + exp(-|x|))
  losses = (
      jnp.maximum(logits, 0.0)
      - logits * multi_hot
      + jnp.log1p(jnp.exp(-jnp.abs(logits)))
  )
  loss_per_target = jnp.mean(losses, axis=-1)  # (B, N)

  if weights is not None:
    loss_per_target = loss_per_target * weights
    weight_sum = jnp.sum(weights)
  else:
    weight_sum = np.prod(targets.shape[:-1])

  loss = jnp.sum(loss_per_target) / (weight_sum + 1e-8)
  return loss, loss_per_target


class BinaryCrossEntropyOpsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.devices()[0].platform == 'tpu':

      vmem = pltpu.get_tpu_info().vmem_capacity_bytes
      logging.info(
          'JETS_DEBUG: VMEM capacity: %d bytes (%.2f MB)',
          vmem,
          vmem / 1024 / 1024,
      )

  def test_get_sharding(self):
    class ObjWithSharding:
      sharding = 'dummy_sharding_1'

    class ObjWithAvalSharding:

      class Aval:
        sharding = 'dummy_sharding_2'

      aval = Aval()

    class ObjWithAvalWithoutSharding:

      class Aval:
        pass

      aval = Aval()

    class ObjWithNoSharding:
      pass

    self.assertEqual(
        binary_cross_entropy_ops._get_sharding(ObjWithSharding()),
        'dummy_sharding_1',
    )
    self.assertEqual(
        binary_cross_entropy_ops._get_sharding(ObjWithAvalSharding()),
        'dummy_sharding_2',
    )
    self.assertIsNone(
        binary_cross_entropy_ops._get_sharding(ObjWithAvalWithoutSharding())
    )
    self.assertIsNone(
        binary_cross_entropy_ops._get_sharding(ObjWithNoSharding())
    )

  @parameterized.named_parameters(
      ('standard', 2, 256, 128, 1024, 4, 256),
      ('unaligned_seq_len', 2, 130, 128, 1024, 4, 256),
      ('unaligned_vocab', 2, 256, 128, 1000, 4, 256),
      ('single_label', 2, 256, 128, 1024, 1, 256),
      ('small_block_v', 2, 256, 128, 1024, 4, 128),
  )
  def test_cut_bce_correctness(
      self, batch, seq_len, hidden_dim, vocab_size, num_labels, block_v
  ):
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Skipping TPU test.')

    key = jax.random.PRNGKey(0)
    key_act, key_emb, key_tgt = jax.random.split(key, 3)

    activations = jax.random.normal(key_act, (batch, seq_len, hidden_dim))
    embeddings = jax.random.normal(key_emb, (vocab_size, hidden_dim))
    targets = jax.random.randint(
        key_tgt, (batch, seq_len, num_labels), 0, vocab_size
    )

    # naive BCE
    def run_naive(act, emb):
      loss, _ = _naive_bce(act, emb, targets)
      return loss

    grad_naive_fn = jax.jit(jax.grad(run_naive, argnums=(0, 1)))
    loss_naive = run_naive(activations, embeddings)
    g_act_naive, g_emb_naive = grad_naive_fn(activations, embeddings)

    # cut BCE
    def run_cut(act, emb):
      return binary_cross_entropy_ops.cut_binary_cross_entropy(
          act, emb, targets, block_v=block_v
      )

    grad_cut_fn = jax.jit(jax.grad(run_cut, argnums=(0, 1)))
    loss_cut = run_cut(activations, embeddings)
    g_act_cut, g_emb_cut = grad_cut_fn(activations, embeddings)

    # Compare
    np.testing.assert_allclose(loss_cut, loss_naive, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(g_act_cut, g_act_naive, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(g_emb_cut, g_emb_naive, atol=1e-4, rtol=1e-4)

  @parameterized.named_parameters(
      ('4d_act_4d_tgt', (2, 2, 128, 64), (2, 2, 128, 4)),
      ('4d_act_3d_tgt', (3, 2, 128, 64), (2, 128, 4)),
  )
  def test_cut_bce_4d_correctness(self, act_shape, tgt_shape):
    vocab_size, block_v = 512, 256
    hidden_dim = act_shape[-1]

    key = jax.random.PRNGKey(42)
    key_act, key_emb, key_tgt = jax.random.split(key, 3)

    activations = jax.random.normal(key_act, act_shape)
    embeddings = jax.random.normal(key_emb, (vocab_size, hidden_dim))
    targets = jax.random.randint(key_tgt, tgt_shape, 0, vocab_size)

    def run_naive(act, emb):
      loss, _ = _naive_bce(act, emb, targets)
      return loss

    def run_cut(act, emb):
      return binary_cross_entropy_ops.cut_binary_cross_entropy(
          act, emb, targets, block_v=block_v
      )

    loss_naive = run_naive(activations, embeddings)
    loss_cut = run_cut(activations, embeddings)
    np.testing.assert_allclose(loss_cut, loss_naive, rtol=1e-3, atol=1e-3)

    # Test backward grad in 4D (exercises _bce_bwd_loop_fallback)
    grad_naive_fn = jax.jit(jax.grad(run_naive, argnums=(0, 1)))
    grad_cut_fn = jax.jit(jax.grad(run_cut, argnums=(0, 1)))

    g_act_naive, g_emb_naive = grad_naive_fn(activations, embeddings)
    g_act_cut, g_emb_cut = grad_cut_fn(activations, embeddings)
    np.testing.assert_allclose(g_act_cut, g_act_naive, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(g_emb_cut, g_emb_naive, atol=1e-4, rtol=1e-4)

  def test_cut_bce_with_sharded_embeddings(self):
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Skipping TPU test.')

    batch, seq_len, hidden_dim, vocab_size, num_labels = 2, 128, 128, 512, 4
    key = jax.random.PRNGKey(0)
    key_act, key_emb, key_tgt = jax.random.split(key, 3)

    activations = jax.random.normal(key_act, (batch, seq_len, hidden_dim))
    embeddings = jax.random.normal(key_emb, (vocab_size, hidden_dim))
    targets = jax.random.randint(
        key_tgt, (batch, seq_len, num_labels), 0, vocab_size
    )

    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices), ('devices',))
    act_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('devices', None, None)
    )
    emb_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('devices', None)
    )
    activations_sharded = jax.device_put(activations, act_sharding)
    embeddings_sharded = jax.device_put(embeddings, emb_sharding)

    with mock.patch.object(
        jax.lax,
        'with_sharding_constraint',
        wraps=jax.lax.with_sharding_constraint,
    ) as mock_fwd_constraint:
      binary_cross_entropy_ops.cut_binary_cross_entropy(
          activations_sharded,
          embeddings_sharded,
          targets,
          block_v=256,
      )
      self.assertEqual(mock_fwd_constraint.call_count, 2)

    def run_cut(act, emb):
      return binary_cross_entropy_ops.cut_binary_cross_entropy(
          act, emb, targets, block_v=256
      )

    with mock.patch.object(
        jax.lax,
        'with_sharding_constraint',
        wraps=jax.lax.with_sharding_constraint,
    ) as mock_sharding_constraint:
      grad_fn = jax.jit(jax.grad(run_cut, argnums=(0, 1)))
      g_act, g_emb = grad_fn(activations_sharded, embeddings_sharded)
      self.assertTrue(mock_sharding_constraint.called)
    self.assertIsNotNone(g_act)
    self.assertIsNotNone(g_emb)

  def test_cut_bce_correctness_large_sequence(self):
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Skipping TPU test.')

    # Test with sequence length larger than chunk_n to trigger scan loop
    batch, seq_len, hidden_dim, vocab_size, num_labels = 2, 2048, 128, 512, 4
    block_v = 256

    key = jax.random.PRNGKey(42)
    key_act, key_emb, key_tgt = jax.random.split(key, 3)

    activations = jax.random.normal(key_act, (batch, seq_len, hidden_dim))
    embeddings = jax.random.normal(key_emb, (vocab_size, hidden_dim))
    targets = jax.random.randint(
        key_tgt, (batch, seq_len, num_labels), 0, vocab_size
    )

    # naive BCE
    def run_naive(act, emb):
      loss, _ = _naive_bce(act, emb, targets)
      return loss

    grad_naive_fn = jax.jit(jax.grad(run_naive, argnums=(0, 1)))
    loss_naive = run_naive(activations, embeddings)
    g_act_naive, g_emb_naive = grad_naive_fn(activations, embeddings)

    # cut BCE
    def run_cut(act, emb):
      return binary_cross_entropy_ops.cut_binary_cross_entropy(
          act, emb, targets, block_v=block_v
      )

    grad_cut_fn = jax.jit(jax.grad(run_cut, argnums=(0, 1)))
    loss_cut = run_cut(activations, embeddings)
    g_act_cut, g_emb_cut = grad_cut_fn(activations, embeddings)

    # Compare
    np.testing.assert_allclose(loss_cut, loss_naive, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(g_act_cut, g_act_naive, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(g_emb_cut, g_emb_naive, atol=1e-4, rtol=1e-4)

  def test_cut_bce_vs_keras(self):
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Skipping TPU test.')

    batch, seq_len, hidden_dim, vocab_size, num_labels = 2, 128, 64, 512, 4
    block_v = 256

    key = jax.random.PRNGKey(42)
    key_act, key_emb, key_tgt = jax.random.split(key, 3)

    activations = jax.random.normal(key_act, (batch, seq_len, hidden_dim))
    embeddings = jax.random.normal(key_emb, (vocab_size, hidden_dim))
    targets = jax.random.randint(
        key_tgt, (batch, seq_len, num_labels), 0, vocab_size
    )

    # Convert targets to multi-hot for Keras
    one_hot = jax.nn.one_hot(targets, vocab_size, axis=-1)  # (B, N, L, V)
    multi_hot = jnp.max(one_hot, axis=-2)  # (B, N, V)

    # Keras BCE version
    def run_keras(act, emb):
      logits = jnp.matmul(act, emb.T)  # (B, N, V)
      loss_per_token = keras.losses.binary_crossentropy(
          multi_hot, logits, from_logits=True
      )
      return jnp.mean(loss_per_token)

    grad_keras_fn = jax.jit(jax.grad(run_keras, argnums=(0, 1)))
    loss_keras = run_keras(activations, embeddings)
    g_act_keras, g_emb_keras = grad_keras_fn(activations, embeddings)

    # Cut BCE version
    def run_cut(act, emb):
      return binary_cross_entropy_ops.cut_binary_cross_entropy(
          act, emb, targets, block_v=block_v
      )

    grad_cut_fn = jax.jit(jax.grad(run_cut, argnums=(0, 1)))
    loss_cut = run_cut(activations, embeddings)
    g_act_cut, g_emb_cut = grad_cut_fn(activations, embeddings)

    # Compare
    np.testing.assert_allclose(loss_cut, loss_keras, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(g_act_cut, g_act_keras, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(g_emb_cut, g_emb_keras, atol=1e-4, rtol=1e-4)

  def test_cut_bce_pure_jax_vs_pallas_equivalence(self):
    """Verifies that pure JAX and Pallas branches yield identical forward and backward results."""
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Skipping TPU test.')

    batch, seq_len, hidden_dim, vocab_size, num_labels = 2, 128, 128, 1024, 4
    block_v = 256

    key = jax.random.PRNGKey(123)
    key_act, key_emb, key_tgt = jax.random.split(key, 3)

    activations = jax.random.normal(key_act, (batch, seq_len, hidden_dim))
    embeddings = jax.random.normal(key_emb, (vocab_size, hidden_dim))
    # Mix valid target indices with negative padding tokens (-1)
    targets = jax.random.randint(
        key_tgt, (batch, seq_len, num_labels), -1, vocab_size
    )

    def run_cut(act, emb, pure_jax):
      return binary_cross_entropy_ops.cut_binary_cross_entropy(
          act, emb, targets, block_v=block_v, pure_jax=pure_jax
      )

    # Forward comparison
    loss_pure_jax = run_cut(activations, embeddings, pure_jax=True)
    loss_pallas = run_cut(activations, embeddings, pure_jax=False)
    np.testing.assert_allclose(loss_pure_jax, loss_pallas, atol=1e-6, rtol=1e-6)

    # Backward comparison
    grad_pure_fn = jax.jit(
        jax.grad(lambda a, e: run_cut(a, e, True), argnums=(0, 1))
    )
    grad_pallas_fn = jax.jit(
        jax.grad(lambda a, e: run_cut(a, e, False), argnums=(0, 1))
    )

    g_act_pure, g_emb_pure = grad_pure_fn(activations, embeddings)
    g_act_pallas, g_emb_pallas = grad_pallas_fn(activations, embeddings)

    np.testing.assert_allclose(g_act_pure, g_act_pallas, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(g_emb_pure, g_emb_pallas, atol=1e-5, rtol=1e-5)

  def test_mini_benchmark_keras(self):
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Skipping TPU test.')

    b, m, d, v, l = 32, 32, 128, 100000, 8

    key = jax.random.PRNGKey(42)
    key_act, key_emb, key_tgt = jax.random.split(key, 3)

    activations = jax.random.normal(key_act, (b, m, d))
    embeddings = jax.random.normal(key_emb, (v, d))
    targets = jax.random.randint(key_tgt, (b, m, l), 0, v)

    # Convert targets to multi-hot for Keras
    one_hot = jax.nn.one_hot(targets, v, axis=-1)  # (B, M, L, V)
    multi_hot = jnp.max(one_hot, axis=-2)  # (B, M, V)

    # --- Keras version ---
    def run_keras(act, emb):
      logits = jnp.matmul(act, emb.T)  # (B, M, V)
      loss_per_token = keras.losses.binary_crossentropy(
          multi_hot, logits, from_logits=True
      )
      return jnp.mean(loss_per_token)

    grad_keras = jax.jit(jax.grad(run_keras, argnums=(0, 1)))
    logging.info('Compiling Keras...')
    t0 = time.time()
    grad_keras(activations, embeddings)[0].block_until_ready()
    logging.info('Keras compiled in %.2f s', time.time() - t0)

    num_steps = 200
    logging.info('Benchmarking Keras (%d steps)...', num_steps)
    t0 = time.time()
    for _ in range(num_steps):
      g_act_keras, _ = grad_keras(activations, embeddings)
      g_act_keras.block_until_ready()
    t_keras = (time.time() - t0) / num_steps
    logging.info('Keras step time: %.4f ms', t_keras * 1000)

  def test_mini_benchmark_cut_bce(self):
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Skipping TPU test.')

    b, m, d, v, l = 32, 32, 128, 100000, 8

    key = jax.random.PRNGKey(42)
    key_act, key_emb, key_tgt = jax.random.split(key, 3)

    activations = jax.random.normal(key_act, (b, m, d))
    embeddings = jax.random.normal(key_emb, (v, d))
    targets = jax.random.randint(key_tgt, (b, m, l), 0, v)
    num_steps = 200

    # --- Cut version ---
    block_v = 4096

    def run_cut(act, emb, bv=block_v):
      return binary_cross_entropy_ops.cut_binary_cross_entropy(
          act, emb, targets, block_v=bv
      )

    grad_cut = jax.jit(jax.grad(run_cut, argnums=(0, 1)))
    logging.info('Compiling cut (block_v=%d)...', block_v)
    t0 = time.time()
    grad_cut(activations, embeddings)[0].block_until_ready()
    logging.info(
        'Cut (block_v=%d) compiled in %.2f s', block_v, time.time() - t0
    )

    logging.info(
        'Benchmarking cut (block_v=%d) (%d steps)...', block_v, num_steps
    )
    t0 = time.time()
    for _ in range(num_steps):
      g_act_cut, _ = grad_cut(activations, embeddings)
      g_act_cut.block_until_ready()
    t_cut = (time.time() - t0) / num_steps
    logging.info('Cut (block_v=%d) step time: %.4f ms ', block_v, t_cut * 1000)

  def test_cut_bce_metrics(self):
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Skipping TPU test.')

    batch, seq_len, hidden_dim, vocab_size, num_labels = 2, 64, 32, 128, 2
    key = jax.random.PRNGKey(1)
    activations = jax.random.normal(key, (batch, seq_len, hidden_dim))
    embeddings = jax.random.normal(key, (vocab_size, hidden_dim))
    targets = jax.random.randint(
        key, (batch, seq_len, num_labels), 0, vocab_size
    )

    loss, tp, fp, fn, tn = binary_cross_entropy_ops.cut_binary_cross_entropy(
        activations,
        embeddings,
        targets,
        return_metrics=True,
    )
    self.assertIsNotNone(loss)
    self.assertIsNotNone(tp)
    self.assertIsNotNone(fp)
    self.assertIsNotNone(fn)
    self.assertIsNotNone(tn)

  def test_cut_bce_block_v_capped_at_vocab(self):
    activations = jnp.ones((2, 64, 32))
    embeddings = jnp.ones((200, 32))
    targets = jnp.zeros((2, 64, 2), dtype=jnp.int32)

    with mock.patch.object(
        binary_cross_entropy_ops,
        '_cut_binary_cross_entropy',
        wraps=binary_cross_entropy_ops._cut_binary_cross_entropy,
    ) as mock_fn:
      binary_cross_entropy_ops.cut_binary_cross_entropy(
          activations, embeddings, targets, block_v=1000
      )
      config = mock_fn.call_args[0][0]
      self.assertEqual(config.block_v, 200)

  def test_pallas_vmem_budget(self):
    binary_cross_entropy_ops.set_hyperparams(16.0, 0.8, 512)
    budget = binary_cross_entropy_ops._pallas_vmem_budget()
    if any(d.platform == 'tpu' for d in jax.devices()):
      expected = max(
          16 * 1024 * 1024,
          pltpu.get_tpu_info().vmem_capacity_bytes - 16 * 1024 * 1024,
      )
      self.assertEqual(budget, expected)
    else:
      self.assertEqual(budget, 16 * 1024 * 1024)

  def test_max_safe_block_v(self):
    val = binary_cross_entropy_ops._max_safe_block_v(16 * 1024 * 1024, 256)
    self.assertEqual(val, 512)

  def test_pallas_interpret(self):
    is_interpret = binary_cross_entropy_ops._pallas_interpret()
    has_tpu = any(d.platform == 'tpu' for d in jax.devices())
    self.assertEqual(is_interpret, not has_tpu)

  def test_pallas_lane(self):
    lane = binary_cross_entropy_ops._pallas_lane()
    if any(d.platform == 'tpu' for d in jax.devices()):
      self.assertEqual(lane, pltpu.get_tpu_info().num_lanes)
    else:
      self.assertEqual(lane, 128)

  def test_check_vocab_replicated_in_d(self):
    with self.assertRaises(NotImplementedError):
      binary_cross_entropy_ops._check_vocab_replicated_in_d(
          jax.sharding.PartitionSpec('devices', 'devices')
      )
    binary_cross_entropy_ops._check_vocab_replicated_in_d(
        jax.sharding.PartitionSpec('devices')
    )

  def test_auto_block_v(self):
    num_devices = jax.device_count()
    self.assertEqual(
        binary_cross_entropy_ops._auto_block_v(3277 * num_devices, 10000), 2432
    )
    self.assertEqual(
        binary_cross_entropy_ops._auto_block_v(4096 * num_devices, 10000), 2048
    )
    self.assertEqual(
        binary_cross_entropy_ops._auto_block_v(1000000 * num_devices, 10000),
        256,
    )
    self.assertEqual(
        binary_cross_entropy_ops._auto_block_v(100 * num_devices, 10000), 8192
    )
    self.assertEqual(
        binary_cross_entropy_ops._auto_block_v(1024 * num_devices, 500), 500
    )


if __name__ == '__main__':
  absltest.main()

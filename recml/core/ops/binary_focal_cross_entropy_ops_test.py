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
"""Tests for binary_focal_cross_entropy_ops."""

from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import keras
import numpy as np
from recml.core.ops import binary_focal_cross_entropy_ops


def _naive_focal_bce(
    activations,
    embeddings,
    targets,
    weights=None,
    gamma=2.0,
    alpha=0.25,
    apply_class_balancing=False,
):
  """Naive implementation that materializes the full logits matrix for focal loss."""
  vocab_size = embeddings.shape[0]
  logits = jnp.matmul(activations, embeddings.T)  # (B, N, V)

  # targets: (B, N, L) -> multi_hot: (B, N, V)
  one_hot = jax.nn.one_hot(targets, vocab_size, axis=-1)  # (B, N, L, V)
  multi_hot = jnp.max(one_hot, axis=-2)  # (B, N, V)

  probs = jax.nn.sigmoid(logits)
  p_t = multi_hot * probs + (1.0 - multi_hot) * (1.0 - probs)
  focal_factor = jnp.power(1.0 - p_t, gamma)

  # Compute stable BCE loss per class
  # Loss = max(x, 0) - x * y + log(1 + exp(-|x|))
  bce_losses = (
      jnp.maximum(logits, 0.0)
      - logits * multi_hot
      + jnp.log1p(jnp.exp(-jnp.abs(logits)))
  )
  if gamma == 0.0:
    losses = bce_losses
  else:
    losses = focal_factor * bce_losses
  if apply_class_balancing:
    weight = multi_hot * alpha + (1.0 - multi_hot) * (1.0 - alpha)
    losses = weight * losses

  loss_per_target = jnp.mean(losses, axis=-1)

  if weights is not None:
    loss_per_target = loss_per_target * weights
    weight_sum = jnp.sum(weights)
  else:
    weight_sum = np.prod(targets.shape[:-1])

  loss = jnp.sum(loss_per_target) / (weight_sum + 1e-7)
  return loss, loss_per_target


class BinaryFocalCrossEntropyOpsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.devices()[0].platform == 'tpu':
      vmem = pltpu.get_tpu_info().vmem_capacity_bytes
      logging.info(
          'JETS_DEBUG: VMEM capacity: %d bytes (%.2f MB)',
          vmem,
          vmem / 1024 / 1024,
      )

  @parameterized.named_parameters(
      ('standard_gamma2', 2, 256, 128, 1024, 4, 256, 2.0, 0.25, True),
      ('no_balancing', 2, 256, 128, 1024, 4, 256, 2.0, 0.25, False),
      ('gamma0', 2, 256, 128, 1024, 4, 256, 0.0, 0.25, True),
      ('unaligned_vocab', 2, 256, 128, 1000, 4, 256, 2.0, 0.25, True),
      ('unaligned_seq_len', 2, 130, 128, 1024, 4, 256, 1.5, 0.25, True),
      ('single_label', 2, 256, 128, 1024, 1, 256, 2.0, 0.25, True),
  )
  def test_cut_focal_bce_correctness(
      self,
      batch,
      seq_len,
      hidden_dim,
      vocab_size,
      num_labels,
      block_v,
      gamma,
      alpha,
      apply_class_balancing,
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

    # naive Focal BCE
    def run_naive(act, emb):
      loss, _ = _naive_focal_bce(
          act,
          emb,
          targets,
          gamma=gamma,
          alpha=alpha,
          apply_class_balancing=apply_class_balancing,
      )
      return loss

    grad_naive_fn = jax.jit(jax.grad(run_naive, argnums=(0, 1)))
    loss_naive = run_naive(activations, embeddings)
    g_act_naive, g_emb_naive = grad_naive_fn(activations, embeddings)

    # cut Focal BCE
    def run_cut(act, emb):
      return binary_focal_cross_entropy_ops.cut_binary_focal_cross_entropy(
          act,
          emb,
          targets,
          block_v=block_v,
          gamma=gamma,
          alpha=alpha,
          apply_class_balancing=apply_class_balancing,
      )

    grad_cut_fn = jax.jit(jax.grad(run_cut, argnums=(0, 1)))
    loss_cut = run_cut(activations, embeddings)
    g_act_cut, g_emb_cut = grad_cut_fn(activations, embeddings)

    # Compare
    np.testing.assert_allclose(loss_cut, loss_naive, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(g_act_cut, g_act_naive, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(g_emb_cut, g_emb_naive, atol=1e-4, rtol=1e-4)

  @parameterized.named_parameters(
      ('4d_act_4d_tgt', (2, 2, 64, 32), (2, 2, 64, 2)),
      ('4d_act_3d_tgt', (3, 2, 64, 32), (2, 64, 2)),
  )
  def test_cut_focal_bce_4d(self, act_shape, tgt_shape):
    vocab_size, block_v = 256, 128
    hidden_dim = act_shape[-1]
    key = jax.random.PRNGKey(42)
    key_act, key_emb, key_tgt = jax.random.split(key, 3)

    activations = jax.random.normal(key_act, act_shape)
    embeddings = jax.random.normal(key_emb, (vocab_size, hidden_dim))
    targets = jax.random.randint(key_tgt, tgt_shape, 0, vocab_size)

    def run_naive(act, emb):
      loss, _ = _naive_focal_bce(act, emb, targets)
      return loss

    def run_cut(act, emb):
      return binary_focal_cross_entropy_ops.cut_binary_focal_cross_entropy(
          act, emb, targets, block_v=block_v
      )

    grad_naive_fn = jax.jit(jax.grad(run_naive, argnums=(0, 1)))
    loss_naive = run_naive(activations, embeddings)
    g_act_naive, g_emb_naive = grad_naive_fn(activations, embeddings)

    grad_cut_fn = jax.jit(jax.grad(run_cut, argnums=(0, 1)))
    loss_cut = run_cut(activations, embeddings)
    g_act_cut, g_emb_cut = grad_cut_fn(activations, embeddings)

    np.testing.assert_allclose(loss_cut, loss_naive, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(g_act_cut, g_act_naive, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(g_emb_cut, g_emb_naive, atol=1e-3, rtol=1e-3)

  def test_focal_bce_v_blocks_exact(self):
    activations = jnp.ones((2, 64, 32))
    embeddings = jnp.ones((250, 32))
    targets = jnp.zeros((2, 64, 2), dtype=jnp.int32)
    block_v = 100

    # vocab = 250, block_v = 100 -> v_blocks = 3 (ceil(250/100) = 3)
    with mock.patch.object(jax.lax, 'scan', wraps=jax.lax.scan) as mock_scan:
      binary_focal_cross_entropy_ops.cut_binary_focal_cross_entropy(
          activations, embeddings, targets, block_v=block_v
      )
      self.assertEqual(mock_scan.call_args[0][2].shape[0], 3)

  def test_cut_focal_bce_sharded(self):
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Skipping TPU test.')

    batch, seq_len, hidden_dim, vocab_size, num_labels = 2, 64, 32, 256, 2
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
      binary_focal_cross_entropy_ops.cut_binary_focal_cross_entropy(
          activations_sharded,
          embeddings_sharded,
          targets,
          block_v=128,
      )
      self.assertEqual(mock_fwd_constraint.call_count, 2)

    def run_cut(act, emb):
      return binary_focal_cross_entropy_ops.cut_binary_focal_cross_entropy(
          act, emb, targets, block_v=128
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

  def test_cut_focal_bce_correctness_large_sequence(self):
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Skipping TPU test.')

    batch, seq_len, hidden_dim, vocab_size, num_labels = 2, 2048, 128, 512, 4
    block_v = 256

    key = jax.random.PRNGKey(42)
    key_act, key_emb, key_tgt = jax.random.split(key, 3)

    activations = jax.random.normal(key_act, (batch, seq_len, hidden_dim))
    embeddings = jax.random.normal(key_emb, (vocab_size, hidden_dim))
    targets = jax.random.randint(
        key_tgt, (batch, seq_len, num_labels), 0, vocab_size
    )

    def run_naive(act, emb):
      loss, _ = _naive_focal_bce(act, emb, targets)
      return loss

    grad_naive_fn = jax.jit(jax.grad(run_naive, argnums=(0, 1)))
    loss_naive = run_naive(activations, embeddings)
    g_act_naive, g_emb_naive = grad_naive_fn(activations, embeddings)

    def run_cut(act, emb):
      return binary_focal_cross_entropy_ops.cut_binary_focal_cross_entropy(
          act, emb, targets, block_v=block_v
      )

    grad_cut_fn = jax.jit(jax.grad(run_cut, argnums=(0, 1)))
    loss_cut = run_cut(activations, embeddings)
    g_act_cut, g_emb_cut = grad_cut_fn(activations, embeddings)

    np.testing.assert_allclose(loss_cut, loss_naive, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(g_act_cut, g_act_naive, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(g_emb_cut, g_emb_naive, atol=1e-4, rtol=1e-4)

  def test_cut_focal_bce_metrics(self):
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Skipping TPU test.')

    batch, seq_len, hidden_dim, vocab_size, num_labels = 2, 64, 32, 128, 2
    key = jax.random.PRNGKey(1)
    activations = jax.random.normal(key, (batch, seq_len, hidden_dim))
    embeddings = jax.random.normal(key, (vocab_size, hidden_dim))
    targets = jax.random.randint(
        key, (batch, seq_len, num_labels), 0, vocab_size
    )

    loss, tp, fp, fn, tn = (
        binary_focal_cross_entropy_ops.cut_binary_focal_cross_entropy(
            activations,
            embeddings,
            targets,
            return_metrics=True,
            gamma=2.0,
            alpha=0.25,
            apply_class_balancing=True,
        )
    )
    self.assertIsNotNone(loss)
    self.assertIsNotNone(tp)
    self.assertIsNotNone(fp)
    self.assertIsNotNone(fn)
    self.assertIsNotNone(tn)

  @parameterized.named_parameters(
      ('standard_gamma2', 2, 64, 32, 128, 2, 64, 2.0, 0.25, True),
      ('no_balancing', 2, 64, 32, 128, 2, 64, 2.0, 0.25, False),
  )
  def test_cut_focal_bce_vs_keras(
      self,
      batch,
      seq_len,
      hidden_dim,
      vocab_size,
      num_labels,
      block_v,
      gamma,
      alpha,
      apply_class_balancing,
  ):
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Skipping TPU test.')

    key = jax.random.PRNGKey(200)
    key_act, key_emb, key_tgt = jax.random.split(key, 3)

    activations = jax.random.normal(key_act, (batch, seq_len, hidden_dim))
    embeddings = jax.random.normal(key_emb, (vocab_size, hidden_dim))
    targets = jax.random.randint(
        key_tgt, (batch, seq_len, num_labels), 0, vocab_size
    )

    def run_keras(act, emb):
      logits = jnp.matmul(act, emb.T)
      one_hot = jax.nn.one_hot(targets, vocab_size, axis=-1)
      multi_hot = jnp.max(one_hot, axis=-2)
      loss_fn = keras.losses.BinaryFocalCrossentropy(
          from_logits=True,
          gamma=gamma,
          alpha=alpha,
          apply_class_balancing=apply_class_balancing,
      )
      return jnp.mean(loss_fn(multi_hot, logits))

    grad_keras_fn = jax.jit(jax.grad(run_keras, argnums=(0, 1)))
    loss_keras = run_keras(activations, embeddings)
    g_act_keras, g_emb_keras = grad_keras_fn(activations, embeddings)

    def run_cut(act, emb):
      return binary_focal_cross_entropy_ops.cut_binary_focal_cross_entropy(
          act,
          emb,
          targets,
          block_v=block_v,
          gamma=gamma,
          alpha=alpha,
          apply_class_balancing=apply_class_balancing,
      )

    grad_cut_fn = jax.jit(jax.grad(run_cut, argnums=(0, 1)))
    loss_cut = run_cut(activations, embeddings)
    g_act_cut, g_emb_cut = grad_cut_fn(activations, embeddings)

    np.testing.assert_allclose(loss_cut, loss_keras, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(g_act_cut, g_act_keras, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(g_emb_cut, g_emb_keras, atol=1e-2, rtol=1e-2)

  def test_check_vocab_replicated_in_d(self):
    with self.assertRaises(NotImplementedError):
      binary_focal_cross_entropy_ops._check_vocab_replicated_in_d(
          jax.sharding.PartitionSpec('devices', 'devices')
      )

  def test_cut_focal_bce_pure_jax_vs_pallas_equivalence(self):
    """Verifies that pure JAX and Pallas branches yield identical forward and backward results."""
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Skipping TPU test.')

    batch, seq_len, hidden_dim, vocab_size, num_labels = 2, 128, 128, 1024, 4
    block_v = 256

    key = jax.random.PRNGKey(456)
    key_act, key_emb, key_tgt = jax.random.split(key, 3)

    activations = jax.random.normal(key_act, (batch, seq_len, hidden_dim))
    embeddings = jax.random.normal(key_emb, (vocab_size, hidden_dim))
    # Mix valid target indices with negative padding tokens (-1)
    targets = jax.random.randint(
        key_tgt, (batch, seq_len, num_labels), -1, vocab_size
    )

    def run_cut(act, emb, pure_jax):
      return binary_focal_cross_entropy_ops.cut_binary_focal_cross_entropy(
          act,
          emb,
          targets,
          block_v=block_v,
          gamma=2.0,
          alpha=0.25,
          apply_class_balancing=True,
          pure_jax=pure_jax,
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


if __name__ == '__main__':
  absltest.main()

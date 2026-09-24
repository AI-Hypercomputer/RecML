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

_BYTES_IN_MB = 1024 * 1024

P = jax.sharding.PartitionSpec


def _naive_focal_bce(
    activations,
    embeddings,
    targets,
    weights=None,
    gamma=2.0,
    alpha=0.25,
    apply_class_balancing=False,
):
  """Naive implementation that materializes full logits matrix for focal loss.

  Ground truth for the correctness tests below. Do not replace with
  `keras.losses.binary_focal_crossentropy` which calls
  `keras.ops.binary_crossentropy(..., from_logits=False)`.

  Both compute `w * (1 - p_t) ** gamma * BCE(x, y)` with the same `w` and
  `p_t`. Only `BCE` differs, for a logit `x`, target `y`, `p = sigmoid(x)`:

      Keras: -y * log(p) - (1 - y) * log(1 - p)    # probability space
      Here:  max(x, 0) - x * y + log1p(exp(-|x|))  # logit space

  These are equal over the reals, but in float32 they diverge because
  `sigmoid(x)` saturates: above `x ~= 16.6` it rounds to exactly 1.0, so
  `log(1 - p)` loses all information about `x`, and Keras's clip to
  `[1e-7, 1 - 1e-7]` then caps the loss. The logit form never
  materializes `p` and stays exact, so the two disagree precisely on
  confidently wrong predictions, whose true loss exceeds that cap.

  Args:
    activations: The activations from the model.
    embeddings: The embeddings from the model.
    targets: The targets from the dataset.
    weights: The weights for each target.
    gamma: The gamma parameter for the focal loss.
    alpha: The alpha parameter for the focal loss.
    apply_class_balancing: Whether to apply class balancing.

  Returns:
    A tuple of (loss, loss_per_target).
  """
  vocab_size = embeddings.shape[0]
  logits = jnp.matmul(activations, embeddings.T)  # (B, N, V)

  # targets: (B, N, L) -> multi_hot: (B, N, V)
  one_hot = jax.nn.one_hot(targets, vocab_size, axis=-1)  # (B, N, L, V)
  multi_hot = jnp.max(one_hot, axis=-2)  # (B, N, V)

  # `1 - p_t = sigmoid(-s * x)` with `s = 2y - 1`, so the focal factor is
  # formed in log space. Differentiating `jnp.power(1 - p_t, gamma)` directly
  # gives `gamma * (1 - p_t) ** (gamma - 1) * 0 = inf * 0 = nan` for
  # `gamma < 1` (including `gamma == 0`) wherever float32 sigmoid saturates to
  # `1 - p_t == 0`.
  signs = 2.0 * multi_hot - 1.0
  focal_factor = jnp.exp(gamma * jax.nn.log_sigmoid(-signs * logits))

  # Compute stable BCE loss per class
  # Loss = max(x, 0) - x * y + log(1 + exp(-|x|))
  bce_losses = (
      jnp.maximum(logits, 0.0)
      - logits * multi_hot
      + jnp.log1p(jnp.exp(-jnp.abs(logits)))
  )
  losses = focal_factor * bce_losses
  if apply_class_balancing:
    weight = multi_hot * alpha + (1.0 - multi_hot) * (1.0 - alpha)
    losses = weight * losses

  loss_per_target = jnp.mean(losses, axis=-1)

  if weights is not None:
    loss_per_target = loss_per_target * weights
    weight_sum = jnp.sum(weights)
  else:
    weight_sum = np.prod(activations.shape[:-1])

  loss = jnp.sum(loss_per_target) / (weight_sum + 1e-7)
  return loss, loss_per_target


class BinaryFocalCrossEntropyOpsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.devices()[0].platform == 'tpu':
      vmem = pltpu.get_tpu_info().vmem_capacity_bytes
      logging.info(
          'VMEM capacity: %d bytes (%.2f MB)',
          vmem,
          vmem / _BYTES_IN_MB,
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
      ('4d_act_4d_tgt_pallas', (2, 2, 64, 32), (2, 2, 64, 2), True),
      ('4d_act_3d_tgt_pallas', (3, 2, 64, 32), (2, 64, 2), True),
      ('4d_act_4d_tgt_pure_jax', (2, 2, 64, 32), (2, 2, 64, 2), False),
      ('4d_act_3d_tgt_pure_jax', (3, 2, 64, 32), (2, 64, 2), False),
  )
  def test_cut_focal_bce_4d(self, act_shape, tgt_shape, use_pallas):
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
          act, emb, targets, block_v=block_v, use_pallas=use_pallas
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

  # --- `block_v` plumbing. ---
  # `_auto_block_v` / `_local_tokens` are re-exported from
  # `binary_cross_entropy_ops` and unit-tested there; these two tests only
  # cover this op's own call site and clamp.

  def test_cut_focal_bce_auto_block_v(self):
    """`block_v=None` must go through `_local_tokens` / `_auto_block_v`."""
    vocab_size = 512
    activations = jnp.ones((2, 64, 32))
    embeddings = jnp.ones((vocab_size, 32))
    targets = jnp.zeros((2, 64, 2), dtype=jnp.int32)

    with mock.patch.object(
        binary_focal_cross_entropy_ops,
        '_cut_binary_focal_cross_entropy',
        wraps=binary_focal_cross_entropy_ops._cut_binary_focal_cross_entropy,
    ) as mock_fn:
      binary_focal_cross_entropy_ops.cut_binary_focal_cross_entropy(
          activations, embeddings, targets
      )
      config = mock_fn.call_args[0][0]

    # 128 tokens leave room for the whole vocab in the logits budget of every
    # chip, so the auto-picked block size is the (clamped) vocab size.
    self.assertEqual(config.block_v, vocab_size)
    self.assertEqual(
        config.block_v,
        binary_focal_cross_entropy_ops._auto_block_v(
            binary_focal_cross_entropy_ops._local_tokens(activations),
            vocab_size,
        ),
    )

  def test_cut_focal_bce_block_v_capped_at_vocab(self):
    vocab_size = 200
    activations = jnp.ones((2, 64, 32))
    embeddings = jnp.ones((vocab_size, 32))
    targets = jnp.zeros((2, 64, 2), dtype=jnp.int32)

    with mock.patch.object(
        binary_focal_cross_entropy_ops,
        '_cut_binary_focal_cross_entropy',
        wraps=binary_focal_cross_entropy_ops._cut_binary_focal_cross_entropy,
    ) as mock_fn:
      binary_focal_cross_entropy_ops.cut_binary_focal_cross_entropy(
          activations, embeddings, targets, block_v=1000
      )
      config = mock_fn.call_args[0][0]

    self.assertEqual(config.block_v, vocab_size)

  # --- Target id edge cases. ---
  # The focal op matches target ids in its own `_focal_bce_fwd_chunk` and in
  # its own Pallas backward kernel, so these are not covered by the plain BCE
  # tests.

  def test_cut_focal_bce_ignores_out_of_range_target_ids(self):
    """Ids outside `[0, V)` carry no label; that is how padding is spelled."""
    vocab_size, block_v = 128, 64
    key_act, key_emb = jax.random.split(jax.random.PRNGKey(7))
    activations = jax.random.normal(key_act, (2, 64, 32))
    embeddings = jax.random.normal(key_emb, (vocab_size, 32))

    # One real label, plus a padding slot spelled two different ways.
    real = jnp.full((2, 64, 1), 5, dtype=jnp.int32)
    negative_padded = jnp.concatenate([real, jnp.full_like(real, -1)], axis=-1)
    above_vocab_padded = jnp.concatenate(
        [real, jnp.full_like(real, vocab_size)], axis=-1
    )

    def run_naive(act, emb):
      loss, _ = _naive_focal_bce(act, emb, real)
      return loss

    loss_reference = run_naive(activations, embeddings)
    g_act_reference, g_emb_reference = jax.jit(
        jax.grad(run_naive, argnums=(0, 1))
    )(activations, embeddings)

    for targets in (negative_padded, above_vocab_padded):

      def run_cut(act, emb, targets=targets):
        return binary_focal_cross_entropy_ops.cut_binary_focal_cross_entropy(
            act, emb, targets, block_v=block_v
        )

      loss = run_cut(activations, embeddings)
      g_act, g_emb = jax.jit(jax.grad(run_cut, argnums=(0, 1)))(
          activations, embeddings
      )

      np.testing.assert_allclose(loss, loss_reference, atol=1e-5, rtol=1e-5)
      np.testing.assert_allclose(g_act, g_act_reference, atol=1e-4, rtol=1e-4)
      np.testing.assert_allclose(g_emb, g_emb_reference, atol=1e-4, rtol=1e-4)

  def test_cut_focal_bce_duplicate_target_ids(self):
    """A label repeated across the L axis must not be counted twice."""
    vocab_size, block_v = 128, 64
    key_act, key_emb = jax.random.split(jax.random.PRNGKey(11))
    activations = jax.random.normal(key_act, (2, 64, 32))
    embeddings = jax.random.normal(key_emb, (vocab_size, 32))

    single = jnp.full((2, 64, 1), 5, dtype=jnp.int32)
    duplicated = jnp.full((2, 64, 3), 5, dtype=jnp.int32)

    def run_cut(act, emb, targets):
      return binary_focal_cross_entropy_ops.cut_binary_focal_cross_entropy(
          act, emb, targets, block_v=block_v
      )

    loss_single = run_cut(activations, embeddings, single)
    loss_duplicated = run_cut(activations, embeddings, duplicated)
    g_single = jax.jit(
        jax.grad(lambda a, e: run_cut(a, e, single), argnums=(0, 1))
    )(activations, embeddings)
    g_duplicated = jax.jit(
        jax.grad(lambda a, e: run_cut(a, e, duplicated), argnums=(0, 1))
    )(activations, embeddings)

    np.testing.assert_allclose(
        loss_duplicated, loss_single, atol=1e-6, rtol=1e-6
    )
    for g_dup, g_ref in zip(g_duplicated, g_single):
      np.testing.assert_allclose(g_dup, g_ref, atol=1e-6, rtol=1e-6)

  @parameterized.named_parameters(
      ('gamma2', 2.0, 0.25, False),
      ('gamma0', 0.0, 0.25, False),
      ('gamma2_class_balanced', 2.0, 0.25, True),
  )
  def test_cut_focal_bce_zero_logits(self, gamma, alpha, apply_class_balancing):
    """Zero logits give an analytic loss of `0.5 ** gamma * log(2)` per class."""
    vocab_size, block_v = 128, 64
    activations = jnp.zeros((2, 8, 16))
    embeddings = jnp.ones((vocab_size, 16))
    # A valid id, a duplicate of it, and both flavours of out-of-range padding,
    # i.e. exactly one positive class per position.
    targets = jnp.tile(
        jnp.array([[5, 5, -1, vocab_size]], dtype=jnp.int32), (2, 8, 1)
    )

    loss = binary_focal_cross_entropy_ops.cut_binary_focal_cross_entropy(
        activations,
        embeddings,
        targets,
        block_v=block_v,
        gamma=gamma,
        alpha=alpha,
        apply_class_balancing=apply_class_balancing,
    )

    # At zero logits p_t = 0.5 for positives and negatives alike, so every
    # class contributes `(1 - 0.5) ** gamma * log(2)`.
    expected = 0.5**gamma * np.log(2.0)
    if apply_class_balancing:
      # The class weights only differ for the single positive class, and the
      # loss is averaged over the vocab.
      expected *= (alpha + (vocab_size - 1) * (1.0 - alpha)) / vocab_size
    np.testing.assert_allclose(loss, expected, atol=1e-6, rtol=1e-6)

  def test_cut_focal_bce_with_sharded_embeddings(self):
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
        mesh, P('devices', None, None)
    )
    emb_sharding = jax.sharding.NamedSharding(
        mesh, P('devices', None)
    )
    activations_sharded = jax.device_put(activations, act_sharding)
    embeddings_sharded = jax.device_put(embeddings, emb_sharding)

    # The embeddings must be replicated before the chunked matmul so that the
    # scan over vocab blocks does not trigger a collective per block. Assert on
    # the sharding the inner op actually receives rather than on how many
    # `with_sharding_constraint` calls were made. The activations' hidden dim
    # is replicated one level down, in the shared `_replicate_hidden_dim`,
    # which is covered in `binary_cross_entropy_ops_test`.
    with mock.patch.object(
        binary_focal_cross_entropy_ops,
        '_cut_binary_focal_cross_entropy',
        wraps=binary_focal_cross_entropy_ops._cut_binary_focal_cross_entropy,
    ) as mock_fn:
      loss_cut = binary_focal_cross_entropy_ops.cut_binary_focal_cross_entropy(
          activations_sharded,
          embeddings_sharded,
          targets,
          block_v=128,
      )
      _, inner_activations, inner_embeddings, _ = mock_fn.call_args[0]

    self.assertEqual(
        inner_embeddings.sharding.spec, P()
    )
    self.assertTrue(inner_embeddings.sharding.is_fully_replicated)
    # The token axes keep their sharding; only the vocab axis is gathered.
    self.assertEqual(
        inner_activations.sharding.spec,
        P('devices', None, None),
    )

    # Sharding must not change the answer.
    loss_naive, _ = _naive_focal_bce(activations, embeddings, targets)
    np.testing.assert_allclose(loss_cut, loss_naive, atol=1e-5, rtol=1e-5)

    def run_cut(act, emb):
      return binary_focal_cross_entropy_ops.cut_binary_focal_cross_entropy(
          act, emb, targets, block_v=128
      )

    # Same for the backward pass: assert the gradients, not that a sharding
    # constraint was called. Here the shardings are inferred from the input
    # arrays; the explicit mesh / act_spec / emb_spec path is covered by
    # `test_cut_focal_bce_sharded_backward`.
    grad_fn = jax.jit(jax.grad(run_cut, argnums=(0, 1)))
    g_act, g_emb = grad_fn(activations_sharded, embeddings_sharded)
    g_act_ref, g_emb_ref = grad_fn(activations, embeddings)

    np.testing.assert_allclose(g_act, g_act_ref, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(g_emb, g_emb_ref, atol=1e-5, rtol=1e-5)

  @parameterized.named_parameters(
      ('pure_jax', False),
      ('pallas', True),
  )
  def test_cut_focal_bce_sharded_backward(self, use_pallas):
    """Covers the backward pass when embeddings are vocab-sharded.

    `block_v` is chosen against the *global* vocab, but under
    `_focal_bce_bwd_sharded`'s shard_map each shard only holds `vocab_size /
    num_devices` rows, so the local `block_v` must be clamped and the loss
    normalization must still use the global vocab size.
    """
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Skipping TPU test.')
    if jax.device_count() < 2:
      self.skipTest(
          'Needs >= 2 devices to make the local vocab smaller than block_v; '
          f'got {jax.device_count()}.'
      )

    batch, seq_len, hidden_dim, vocab_size, num_labels = 2, 128, 128, 512, 4
    # Equal to the global vocab, so every shard is strictly smaller.
    block_v = vocab_size

    key_act, key_emb, key_tgt = jax.random.split(jax.random.PRNGKey(0), 3)
    activations = jax.random.normal(key_act, (batch, seq_len, hidden_dim))
    embeddings = jax.random.normal(key_emb, (vocab_size, hidden_dim))
    targets = jax.random.randint(
        key_tgt, (batch, seq_len, num_labels), 0, vocab_size
    )

    # Separate mesh axes: reusing one axis for both batch and vocab sharding
    # makes `_focal_bce_bwd_sharded`'s dp_axes/psum bookkeeping incoherent.
    num_vocab_shards = jax.device_count()
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape(1, num_vocab_shards),
        ('data', 'model'),
    )
    act_spec = P('data', None, None)
    emb_spec = P('model', None)

    def run_cut(act, emb, **kwargs):
      return binary_focal_cross_entropy_ops.cut_binary_focal_cross_entropy(
          act, emb, targets, block_v=block_v, use_pallas=use_pallas, **kwargs
      )

    # Passing mesh/act_spec/emb_spec is what forces `_focal_bce_bwd_sharded`
    # down the shard_map branch, where each shard sees only `vocab_size /
    # num_devices` embedding rows.
    grad_sharded_fn = jax.jit(
        jax.grad(
            lambda a, e: run_cut(
                a, e, mesh=mesh, act_spec=act_spec, emb_spec=emb_spec
            ),
            argnums=(0, 1),
        )
    )
    grad_ref_fn = jax.jit(jax.grad(run_cut, argnums=(0, 1)))

    g_act_sharded, g_emb_sharded = grad_sharded_fn(
        jax.device_put(activations, jax.sharding.NamedSharding(mesh, act_spec)),
        jax.device_put(embeddings, jax.sharding.NamedSharding(mesh, emb_spec)),
    )
    g_act_ref, g_emb_ref = grad_ref_fn(activations, embeddings)

    # Catches two vocab-sharding regressions: without the `block_v` clamp the
    # chunk slice asks for more rows than a shard holds, and without the
    # `global_vocab` normalization the gradients come back scaled by the number
    # of vocab shards.
    np.testing.assert_allclose(g_act_sharded, g_act_ref, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(g_emb_sharded, g_emb_ref, atol=1e-5, rtol=1e-5)

  @parameterized.named_parameters(
      ('hidden_sharded_empty_emb_spec', P('data', None, 'model'), P()),
      ('short_act_spec_none_emb_spec', P('data'), None),
  )
  def test_cut_focal_bce_sharded_backward_hidden_dim_and_short_specs(
      self, act_spec, emb_spec
  ):
    """Covers hidden-dim activation sharding, short act_spec, and None/P() emb_spec."""
    batch, seq_len, hidden_dim, vocab_size, num_labels = 2, 64, 128, 256, 4
    block_v = 128

    key_act, key_emb, key_tgt = jax.random.split(jax.random.PRNGKey(7), 3)
    activations = jax.random.normal(key_act, (batch, seq_len, hidden_dim))
    embeddings = jax.random.normal(key_emb, (vocab_size, hidden_dim))
    targets = jax.random.randint(
        key_tgt, (batch, seq_len, num_labels), 0, vocab_size
    )

    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape(jax.device_count(), 1),
        ('data', 'model'),
    )

    def run_cut(act, emb, **kwargs):
      return binary_focal_cross_entropy_ops.cut_binary_focal_cross_entropy(
          act, emb, targets, block_v=block_v, use_pallas=False, **kwargs
      )

    grad_sharded_fn = jax.jit(
        jax.grad(
            lambda a, e: run_cut(
                a, e, mesh=mesh, act_spec=act_spec, emb_spec=emb_spec
            ),
            argnums=(0, 1),
        )
    )
    grad_ref_fn = jax.jit(jax.grad(run_cut, argnums=(0, 1)))

    emb_put_spec = emb_spec if emb_spec is not None else P()
    g_act_sharded, g_emb_sharded = grad_sharded_fn(
        jax.device_put(activations, jax.sharding.NamedSharding(mesh, act_spec)),
        jax.device_put(
            embeddings, jax.sharding.NamedSharding(mesh, emb_put_spec)
        ),
    )
    g_act_ref, g_emb_ref = grad_ref_fn(activations, embeddings)

    np.testing.assert_allclose(g_act_sharded, g_act_ref, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(g_emb_sharded, g_emb_ref, atol=1e-5, rtol=1e-5)

  @parameterized.named_parameters(
      ('exact_multiple_of_chunk_n', 128),
      ('unaligned_padded_to_chunk_n', 130),
  )
  def test_cut_focal_bce_correctness_large_sequence(self, seq_len):
    # Force chunk_n=128 so `n = batch * seq_len` (256 or 260) exceeds `chunk_n`
    # on every platform and exercises both the exact-multiple (`padded_n == n`)
    # and padded (`padded_n > n`) branches of `_focal_bce_bwd_pallas_chunked_n`.
    batch, hidden_dim, vocab_size, num_labels = 2, 128, 256, 4
    block_v = 128

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

    with mock.patch.object(
        binary_focal_cross_entropy_ops, '_max_safe_chunk_n', return_value=128
    ):
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
      ('gamma0p5_pallas', 0.5, True),
      ('gamma0p5_pure_jax', 0.5, False),
      ('gamma0p25_balanced_pallas', 0.25, True),
      ('gamma1_boundary_pallas', 1.0, True),
  )
  def test_cut_focal_bce_fractional_gamma(self, gamma, use_pallas):
    """Gradients must be exact for `0 < gamma < 1`.

    The derivative contains `(1 - p_t) ** (gamma - 1)`, a negative power for
    fractional `gamma`; clamping that exponent at 0 silently drops the second
    term of the gradient while leaving the forward loss correct.

    Args:
      gamma: Fractional focusing parameter.
      use_pallas: Whether to use the Pallas backward kernel.
    """
    vocab_size, block_v = 256, 128
    key_act, key_emb, key_tgt = jax.random.split(jax.random.PRNGKey(3), 3)
    activations = jax.random.normal(key_act, (2, 64, 32))
    embeddings = jax.random.normal(key_emb, (vocab_size, 32))
    targets = jax.random.randint(key_tgt, (2, 64, 2), 0, vocab_size)
    apply_class_balancing = gamma == 0.25

    def run_naive(act, emb):
      loss, _ = _naive_focal_bce(
          act,
          emb,
          targets,
          gamma=gamma,
          apply_class_balancing=apply_class_balancing,
      )
      return loss

    def run_cut(act, emb):
      return binary_focal_cross_entropy_ops.cut_binary_focal_cross_entropy(
          act,
          emb,
          targets,
          block_v=block_v,
          gamma=gamma,
          apply_class_balancing=apply_class_balancing,
          use_pallas=use_pallas,
      )

    g_act_naive, g_emb_naive = jax.jit(jax.grad(run_naive, argnums=(0, 1)))(
        activations, embeddings
    )
    g_act_cut, g_emb_cut = jax.jit(jax.grad(run_cut, argnums=(0, 1)))(
        activations, embeddings
    )

    np.testing.assert_allclose(
        run_cut(activations, embeddings),
        run_naive(activations, embeddings),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(g_act_cut, g_act_naive, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(g_emb_cut, g_emb_naive, atol=1e-4, rtol=1e-4)

  @parameterized.named_parameters(
      ('pallas', True),
      ('pure_jax', False),
  )
  def test_cut_focal_bce_weighted_metrics_and_backward(self, use_pallas):
    """Covers `weights` with `return_metrics` and backprop through metrics."""
    vocab_size, block_v = 128, 64
    key_act, key_emb, key_tgt, key_w = jax.random.split(
        jax.random.PRNGKey(5), 4
    )
    activations = jax.random.normal(key_act, (2, 16, 32))
    embeddings = jax.random.normal(key_emb, (vocab_size, 32))
    targets = jax.random.randint(key_tgt, (2, 16, 2), 0, vocab_size)
    weights = jax.random.bernoulli(key_w, 0.7, (2, 16)).astype(jnp.float32)

    def run_cut(act, emb):
      loss, losses, tp, fp, fn, tn = (
          binary_focal_cross_entropy_ops.cut_binary_focal_cross_entropy(
              act,
              emb,
              targets,
              weights,
              block_v=block_v,
              return_per_target_losses=True,
              return_metrics=True,
              use_pallas=use_pallas,
          )
      )
      return loss, (losses, tp, fp, fn, tn)

    def run_naive(act, emb):
      return _naive_focal_bce(act, emb, targets, weights=weights)

    (loss, (losses, tp, fp, fn, tn)), (g_act, g_emb) = jax.jit(
        jax.value_and_grad(run_cut, argnums=(0, 1), has_aux=True)
    )(activations, embeddings)
    (loss_ref, losses_ref), (g_act_ref, g_emb_ref) = jax.jit(
        jax.value_and_grad(run_naive, argnums=(0, 1), has_aux=True)
    )(activations, embeddings)

    np.testing.assert_allclose(loss, loss_ref, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(losses, losses_ref, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(g_act, g_act_ref, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(g_emb, g_emb_ref, atol=1e-4, rtol=1e-4)

    # Weighted confusion counts from the materialized logits.
    logits = jnp.matmul(activations, embeddings.T)
    multi_hot = jnp.max(jax.nn.one_hot(targets, vocab_size), axis=-2) > 0
    preds = logits > 0.0
    w = weights[..., None]
    np.testing.assert_allclose(tp, jnp.sum((multi_hot & preds) * w), rtol=1e-6)
    np.testing.assert_allclose(fp, jnp.sum((~multi_hot & preds) * w), rtol=1e-6)
    np.testing.assert_allclose(fn, jnp.sum((multi_hot & ~preds) * w), rtol=1e-6)
    np.testing.assert_allclose(
        tn, jnp.sum((~multi_hot & ~preds) * w), rtol=1e-6
    )

  @parameterized.named_parameters(
      ('4d_tgt', True),
      ('3d_shared_tgt', False),
  )
  def test_cut_focal_bce_sharded_backward_4d_pallas(self, targets_4d):
    """Covers the Pallas group-loop fallback inside the shard_map backward."""
    num_devices = jax.device_count()
    groups, batch, seq_len, hidden_dim = 2, 2 * num_devices, 64, 32
    vocab_size, block_v = 256, 128

    key_act, key_emb, key_tgt = jax.random.split(jax.random.PRNGKey(9), 3)
    activations = jax.random.normal(
        key_act, (groups, batch, seq_len, hidden_dim)
    )
    embeddings = jax.random.normal(key_emb, (vocab_size, hidden_dim))
    tgt_shape = (
        (groups, batch, seq_len, 2) if targets_4d else (batch, seq_len, 2)
    )
    targets = jax.random.randint(key_tgt, tgt_shape, 0, vocab_size)

    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape(num_devices, 1), ('data', 'model')
    )
    act_spec = P(None, 'data')
    emb_spec = P()

    def run_cut(act, emb, **kwargs):
      return binary_focal_cross_entropy_ops.cut_binary_focal_cross_entropy(
          act, emb, targets, block_v=block_v, use_pallas=True, **kwargs
      )

    def run_naive(act, emb):
      loss, _ = _naive_focal_bce(act, emb, targets)
      return loss

    g_act_sharded, g_emb_sharded = jax.jit(
        jax.grad(
            lambda a, e: run_cut(
                a, e, mesh=mesh, act_spec=act_spec, emb_spec=emb_spec
            ),
            argnums=(0, 1),
        )
    )(
        jax.device_put(activations, jax.sharding.NamedSharding(mesh, act_spec)),
        jax.device_put(embeddings, jax.sharding.NamedSharding(mesh, emb_spec)),
    )
    g_act_ref, g_emb_ref = jax.jit(jax.grad(run_naive, argnums=(0, 1)))(
        activations, embeddings
    )

    np.testing.assert_allclose(g_act_sharded, g_act_ref, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(g_emb_sharded, g_emb_ref, atol=1e-4, rtol=1e-4)

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

    # Keras computes BCE from `sigmoid(logits)` clipped to `[1e-7, 1 - 1e-7]`,
    # which caps the loss of confidently wrong logits (|x| >~ 16); see
    # `_naive_focal_bce`. With these inputs that shifts the loss by ~2e-3
    # relative, so a 1e-4 tolerance is not attainable against Keras.
    np.testing.assert_allclose(loss_cut, loss_keras, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(g_act_cut, g_act_keras, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(g_emb_cut, g_emb_keras, atol=1e-2, rtol=1e-2)

  def test_check_vocab_replicated_in_d(self):
    with self.assertRaises(NotImplementedError):
      binary_focal_cross_entropy_ops._check_vocab_replicated_in_d(
          P('devices', 'devices')
      )

  def test_cut_focal_bce_use_pallas_vs_pure_jax_equivalence(self):
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

    def run_cut(act, emb, use_pallas):
      return binary_focal_cross_entropy_ops.cut_binary_focal_cross_entropy(
          act,
          emb,
          targets,
          block_v=block_v,
          gamma=2.0,
          alpha=0.25,
          apply_class_balancing=True,
          use_pallas=use_pallas,
      )

    # Forward comparison
    loss_pure_jax = run_cut(activations, embeddings, use_pallas=False)
    loss_pallas = run_cut(activations, embeddings, use_pallas=True)
    np.testing.assert_allclose(loss_pure_jax, loss_pallas, atol=1e-6, rtol=1e-6)

    # Backward comparison
    grad_pure_fn = jax.jit(
        jax.grad(lambda a, e: run_cut(a, e, False), argnums=(0, 1))
    )
    grad_pallas_fn = jax.jit(
        jax.grad(lambda a, e: run_cut(a, e, True), argnums=(0, 1))
    )

    g_act_pure, g_emb_pure = grad_pure_fn(activations, embeddings)
    g_act_pallas, g_emb_pallas = grad_pallas_fn(activations, embeddings)

    np.testing.assert_allclose(g_act_pure, g_act_pallas, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(g_emb_pure, g_emb_pallas, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()

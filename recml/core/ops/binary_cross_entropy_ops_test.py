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

_BYTES_IN_MB = 1024 * 1024

P = jax.sharding.PartitionSpec


def _keras_bce(activations, embeddings, targets, weights=None):
  """Keras reference implementation that materializes the full logits matrix."""
  vocab_size = embeddings.shape[0]
  logits = jnp.matmul(activations, embeddings.T)  # (B, N, V)

  # targets: (B, N, L) -> multi_hot: (B, N, V)
  one_hot = jax.nn.one_hot(targets, vocab_size, axis=-1)  # (B, N, L, V)
  multi_hot = jnp.max(one_hot, axis=-2)  # (B, N, V)
  # Keras requires identical shapes, so broadcast targets over any leading dims
  # of the activations.
  multi_hot = jnp.broadcast_to(multi_hot, logits.shape)

  # Keras averages the per-class BCE over the last (vocab) axis.
  loss_per_target = keras.losses.binary_crossentropy(
      multi_hot, logits, from_logits=True
  )  # (B, N)

  if weights is not None:
    loss_per_target = loss_per_target * weights
    weight_sum = jnp.sum(weights)
  else:
    weight_sum = np.prod(activations.shape[:-1])

  loss = jnp.sum(loss_per_target) / (weight_sum + 1e-8)
  return loss, loss_per_target


class BinaryCrossEntropyOpsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.devices()[0].platform == 'tpu':

      vmem = pltpu.get_tpu_info().vmem_capacity_bytes
      logging.info(
          'VMEM capacity: %d bytes (%.2f MiB)',
          vmem,
          vmem / _BYTES_IN_MB,
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

  def test_get_mxu_size(self):
    """Fails on new TPU generations if _get_mxu_size is not explicitly extended."""
    if jax.devices()[0].platform != 'tpu':
      self.assertEqual(binary_cross_entropy_ops._get_mxu_size(), 128)
    else:
      mxu_size = binary_cross_entropy_ops._get_mxu_size()
      self.assertIn(mxu_size, (128, 256))

  # --- `block_v` selection: per-device token estimate. ---
  # These are direct unit tests on `_local_tokens` / `_auto_block_v`: the
  # quantity under test is just an integer, and `jax.sharding.AbstractMesh`
  # describes meshes bigger than the single chip a unit test is given, so the
  # multi-way sharding cases are deterministic on every platform.

  def test_local_tokens_falls_back_to_device_count(self):
    activations = jnp.ones((2, 64, 32))
    # A plain array only carries a `SingleDeviceSharding`, so the shard count
    # is unknowable and we fall back to the device count.
    self.assertEqual(
        binary_cross_entropy_ops._local_tokens(activations),
        max(2 * 64 // jax.device_count(), 1),
    )

  def test_local_tokens_only_divides_by_token_axis_sharding(self):
    # B*N is sharded over 'data' only. Dividing by 'model' as well (which
    # `jax.device_count()` does) would under-estimate the local token count by
    # `d_model`x and therefore over-estimate `block_v` by the same factor.
    mesh = jax.sharding.AbstractMesh((4, 2), ('data', 'model'))
    activations = jnp.ones((8, 64, 32))
    self.assertEqual(
        binary_cross_entropy_ops._local_tokens(
            activations, mesh, P('data', None, None)
        ),
        8 * 64 // 4,
    )

  def test_local_tokens_ignores_hidden_dim_sharding(self):
    # `_replicate_hidden_dim` gathers D back before the matmul, so the logits
    # tile is [local_tokens, block_v] however D was sharded.
    mesh = jax.sharding.AbstractMesh((4, 2), ('data', 'model'))
    activations = jnp.ones((8, 64, 32))
    self.assertEqual(
        binary_cross_entropy_ops._local_tokens(
            activations, mesh, P('data', None, 'model')
        ),
        8 * 64 // 4,
    )

  def test_local_tokens_with_multiple_axes_over_tokens(self):
    mesh = jax.sharding.AbstractMesh((4, 2), ('data', 'model'))
    activations = jnp.ones((8, 64, 32))
    self.assertEqual(
        binary_cross_entropy_ops._local_tokens(
            activations, mesh, P(('data', 'model'), None, None)
        ),
        8 * 64 // 8,
    )

  def test_local_tokens_replicated_activations(self):
    # Replicated on a non-empty mesh, e.g. an eval job: every device holds all
    # the tokens.
    mesh = jax.sharding.AbstractMesh((8,), ('data',))
    activations = jnp.ones((8, 64, 32))
    for spec in (P(), P(None, None, None)):
      self.assertEqual(
          binary_cross_entropy_ops._local_tokens(activations, mesh, spec),
          8 * 64,
      )

  def test_local_tokens_pure_data_parallel(self):
    # Regression guard: the already-correct DP case must stay correct.
    mesh = jax.sharding.AbstractMesh((8,), ('data',))
    activations = jnp.ones((16, 128, 32))
    self.assertEqual(
        binary_cross_entropy_ops._local_tokens(
            activations, mesh, P('data', None, None)
        ),
        16 * 128 // 8,
    )

  def test_local_tokens_counts_leading_axis_of_4d_activations(self):
    # For `[E, B, N, D]` the vmapped logits tile is `[E, local_tokens,
    # block_v]`, so E has to be counted rather than skipped.
    mesh = jax.sharding.AbstractMesh((4,), ('data',))
    activations = jnp.ones((2, 8, 64, 32))
    for spec in (
        P(None, 'data', None, None),
        P('data', None, None, None),
    ):
      self.assertEqual(
          binary_cross_entropy_ops._local_tokens(activations, mesh, spec),
          2 * 8 * 64 // 4,
      )

  def test_local_tokens_with_spec_shorter_than_rank(self):
    mesh = jax.sharding.AbstractMesh((4,), ('data',))
    activations = jnp.ones((8, 64, 32))
    self.assertEqual(
        binary_cross_entropy_ops._local_tokens(activations, mesh, P('data')),
        8 * 64 // 4,
    )

  def test_local_tokens_unconstrained_axis_falls_back_to_device_count(self):
    mesh = jax.sharding.AbstractMesh((4,), ('data',))
    activations = jnp.ones((8, 64, 32))
    self.assertEqual(
        binary_cross_entropy_ops._local_tokens(
            activations, mesh, P(P.UNCONSTRAINED, None, None)
        ),
        max(8 * 64 // jax.device_count(), 1),
    )

  def test_local_tokens_empty_mesh_falls_back_to_device_count(self):
    activations = jnp.ones((8, 64, 32))
    self.assertEqual(
        binary_cross_entropy_ops._local_tokens(
            activations,
            jax.sharding.AbstractMesh((), ()),
            P('data', None, None),
        ),
        max(8 * 64 // jax.device_count(), 1),
    )

  def test_token_shard_count_replicated_tracer_falls_back(self):
    # Under tracing a replicated spec is indistinguishable from "sharding not
    # decided yet" (GSPMD may still shard the array), so we report `None` and
    # let the caller fall back to the device count.
    mesh = jax.sharding.AbstractMesh((4,), ('data',))
    shard_counts = []

    def f(x):
      shard_counts.append(
          binary_cross_entropy_ops._token_shard_count(x, mesh, P())
      )
      return x

    jax.eval_shape(f, jax.ShapeDtypeStruct((8, 64, 32), jnp.float32))
    self.assertIsNone(shard_counts[0])

  def test_local_tokens_with_real_mesh(self):
    if jax.device_count() < 2:
      self.skipTest(
          f'Needs >= 2 devices to be meaningful; got {jax.device_count()}.'
      )
    # 'data' is 1-way here, so the estimate must stay at the global token count
    # even though `jax.device_count()` is 2+.
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape(1, jax.device_count()),
        ('data', 'model'),
    )
    activations = jax.device_put(
        jnp.ones((8, 64, 32)),
        jax.sharding.NamedSharding(mesh, P('data', None, None)),
    )
    self.assertEqual(
        binary_cross_entropy_ops._local_tokens(activations), 8 * 64
    )

  # --- `block_v` selection: the memory budget itself. ------------------------

  def test_auto_block_v_is_clamped_to_vocab_size(self):
    self.assertEqual(binary_cross_entropy_ops._auto_block_v(1, 256), 256)
    # Vocabularies below one MXU tile are clamped too.
    self.assertEqual(binary_cross_entropy_ops._auto_block_v(1, 64), 64)

  def test_auto_block_v_is_mxu_aligned(self):
    mxu_size = binary_cross_entropy_ops._get_mxu_size()
    block_v = binary_cross_entropy_ops._auto_block_v(1000, 1 << 20)
    self.assertEqual(block_v % mxu_size, 0)
    self.assertGreaterEqual(block_v, mxu_size)

  def test_auto_block_v_floors_at_one_mxu_tile(self):
    self.assertEqual(
        binary_cross_entropy_ops._auto_block_v(1 << 30, 1 << 20),
        binary_cross_entropy_ops._get_mxu_size(),
    )

  def test_auto_block_v_scales_inversely_with_local_tokens(self):
    # The vocab is large enough that neither value is clamped, and both stay
    # well above one MXU tile on every chip.
    vocab_size = 1 << 20
    self.assertEqual(
        binary_cross_entropy_ops._auto_block_v(64, vocab_size),
        8 * binary_cross_entropy_ops._auto_block_v(512, vocab_size),
    )

  def test_auto_block_v_scales_with_dtype_itemsize(self):
    vocab_size = 1 << 20
    self.assertEqual(
        binary_cross_entropy_ops._auto_block_v(512, vocab_size, jnp.bfloat16),
        2 * binary_cross_entropy_ops._auto_block_v(512, vocab_size),
    )

  def test_auto_block_v_handles_zero_tokens(self):
    self.assertEqual(
        binary_cross_entropy_ops._auto_block_v(0, 1 << 20),
        binary_cross_entropy_ops._auto_block_v(1, 1 << 20),
    )

  def test_cut_bce_auto_block_v(self):
    """`block_v=None` must go through `_local_tokens` / `_auto_block_v`."""
    vocab_size = 512
    activations = jnp.ones((2, 64, 32))
    embeddings = jnp.ones((vocab_size, 32))
    targets = jnp.zeros((2, 64, 2), dtype=jnp.int32)

    with mock.patch.object(
        binary_cross_entropy_ops,
        '_cut_binary_cross_entropy',
        wraps=binary_cross_entropy_ops._cut_binary_cross_entropy,
    ) as mock_fn:
      binary_cross_entropy_ops.cut_binary_cross_entropy(
          activations, embeddings, targets
      )
      config = mock_fn.call_args[0][0]

    # 128 tokens leave room for the whole vocab in the logits budget of every
    # chip, so the auto-picked block size is the (clamped) vocab size.
    self.assertEqual(config.block_v, vocab_size)
    self.assertEqual(
        config.block_v,
        binary_cross_entropy_ops._auto_block_v(
            binary_cross_entropy_ops._local_tokens(activations), vocab_size
        ),
    )

  def test_cut_bce_auto_block_v_under_mesh(self):
    """`block_v=None` with an explicit mesh and activation spec."""
    vocab_size, local_tokens = 8192, 64 * 128
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape(1, jax.device_count()),
        ('data', 'model'),
    )
    act_spec = P('data', None, None)
    activations = jax.device_put(
        jnp.ones((64, 128, 32)), jax.sharding.NamedSharding(mesh, act_spec)
    )
    embeddings = jnp.ones((vocab_size, 32))
    targets = jnp.zeros((64, 128, 2), dtype=jnp.int32)

    with mock.patch.object(
        binary_cross_entropy_ops,
        '_cut_binary_cross_entropy',
        wraps=binary_cross_entropy_ops._cut_binary_cross_entropy,
    ) as mock_fn:
      binary_cross_entropy_ops.cut_binary_cross_entropy(
          activations, embeddings, targets, mesh=mesh, act_spec=act_spec
      )
      config = mock_fn.call_args[0][0]

    # 8192 tokens keep the auto-picked block size strictly between one MXU tile
    # and the vocab size on every chip, so a `d_model`-way over-division would
    # change the answer.
    self.assertEqual(
        config.block_v,
        binary_cross_entropy_ops._auto_block_v(local_tokens, vocab_size),
    )
    if jax.device_count() > 1:
      # The tokens are sharded 1-way ('data'), not `jax.device_count()`-way.
      self.assertNotEqual(
          config.block_v,
          binary_cross_entropy_ops._auto_block_v(
              local_tokens // jax.device_count(), vocab_size
          ),
      )

  def test_replicate_hidden_dim(self):
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape(1, jax.device_count()),
        ('data', 'model'),
    )
    activations = jax.device_put(
        jnp.ones((8, 64, 128)),
        jax.sharding.NamedSharding(mesh, P('data', None, 'model')),
    )
    replicated = binary_cross_entropy_ops._replicate_hidden_dim(activations)
    # Only the hidden dim is gathered; the token axes keep their sharding.
    self.assertEqual(replicated.sharding.spec, P('data', None, None))

    # A short PartitionSpec that omits trailing Nones (e.g. P('data') for 3D
    # [B, N, D]) must preserve the leading batch sharding rather than zeroing
    # out its only explicit entry.
    short_spec_activations = jax.device_put(
        jnp.ones((8, 64, 128)),
        jax.sharding.NamedSharding(mesh, P('data')),
    )
    replicated_short = binary_cross_entropy_ops._replicate_hidden_dim(
        short_spec_activations
    )
    self.assertEqual(replicated_short.sharding.spec, P('data', None, None))

  def test_replicate_hidden_dim_is_a_no_op_without_named_sharding(self):
    activations = jnp.ones((8, 64, 128))
    self.assertIs(
        binary_cross_entropy_ops._replicate_hidden_dim(activations),
        activations,
    )

  # --- Target id edge cases. -------------------------------------------------

  def test_cut_bce_ignores_out_of_range_target_ids(self):
    """Ids outside `[0, V)` carry no label; that is how padding is spelled."""
    vocab_size, block_v = 128, 64
    key_act, key_emb = jax.random.split(jax.random.PRNGKey(7))
    activations = jax.random.normal(key_act, (2, 8, 16))
    embeddings = jax.random.normal(key_emb, (vocab_size, 16))

    # One real label, plus a padding slot spelled two different ways.
    real = jnp.full((2, 8, 1), 5, dtype=jnp.int32)
    negative_padded = jnp.concatenate([real, jnp.full_like(real, -1)], axis=-1)
    above_vocab_padded = jnp.concatenate(
        [real, jnp.full_like(real, vocab_size)], axis=-1
    )

    loss_reference, _ = _keras_bce(activations, embeddings, real)
    for targets in (negative_padded, above_vocab_padded):
      loss = binary_cross_entropy_ops.cut_binary_cross_entropy(
          activations, embeddings, targets, block_v=block_v
      )
      np.testing.assert_allclose(loss, loss_reference, atol=1e-5, rtol=1e-5)

  def test_cut_bce_duplicate_target_ids(self):
    """A label repeated across the L axis must not be counted twice."""
    vocab_size, block_v = 128, 64
    key_act, key_emb = jax.random.split(jax.random.PRNGKey(11))
    activations = jax.random.normal(key_act, (2, 8, 16))
    embeddings = jax.random.normal(key_emb, (vocab_size, 16))

    single = jnp.full((2, 8, 1), 5, dtype=jnp.int32)
    duplicated = jnp.full((2, 8, 3), 5, dtype=jnp.int32)

    loss_single = binary_cross_entropy_ops.cut_binary_cross_entropy(
        activations, embeddings, single, block_v=block_v
    )
    loss_duplicated = binary_cross_entropy_ops.cut_binary_cross_entropy(
        activations, embeddings, duplicated, block_v=block_v
    )
    loss_reference, _ = _keras_bce(activations, embeddings, duplicated)

    np.testing.assert_allclose(
        loss_duplicated, loss_single, atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(
        loss_duplicated, loss_reference, atol=1e-5, rtol=1e-5
    )

  def test_cut_bce_zero_logits(self):
    """Zero logits give log(2) per class, whatever the labels are."""
    vocab_size, block_v = 128, 64
    activations = jnp.zeros((2, 8, 16))
    embeddings = jnp.ones((vocab_size, 16))
    # A valid id, a duplicate, and both flavours of out-of-range padding.
    targets = jnp.tile(
        jnp.array([[5, 5, -1, vocab_size]], dtype=jnp.int32), (2, 8, 1)
    )

    loss = binary_cross_entropy_ops.cut_binary_cross_entropy(
        activations, embeddings, targets, block_v=block_v
    )
    np.testing.assert_allclose(loss, np.log(2.0), atol=1e-6, rtol=1e-6)

  @parameterized.named_parameters(
      ('standard', 2, 256, 128, 1024, 4, 256),
      ('unaligned_seq_len', 2, 130, 128, 1024, 4, 256),
      ('unaligned_vocab', 2, 256, 128, 1000, 4, 256),
      ('single_label', 2, 256, 128, 1024, 1, 256),
      ('small_block_v', 2, 256, 128, 1024, 4, 128),
      ('small_hidden_dim', 2, 128, 64, 512, 4, 256),
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

    # Keras BCE
    def run_keras(act, emb):
      loss, _ = _keras_bce(act, emb, targets)
      return loss

    grad_keras_fn = jax.jit(jax.grad(run_keras, argnums=(0, 1)))
    loss_keras = run_keras(activations, embeddings)
    g_act_keras, g_emb_keras = grad_keras_fn(activations, embeddings)

    # cut BCE
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

  @parameterized.named_parameters(
      ('4d_act_4d_tgt_pallas', (2, 2, 128, 64), (2, 2, 128, 4), True),
      ('4d_act_3d_tgt_pallas', (3, 2, 128, 64), (2, 128, 4), True),
      ('4d_act_4d_tgt_pure_jax', (2, 2, 128, 64), (2, 2, 128, 4), False),
      ('4d_act_3d_tgt_pure_jax', (3, 2, 128, 64), (2, 128, 4), False),
  )
  def test_cut_bce_4d_correctness(self, act_shape, tgt_shape, use_pallas):
    vocab_size, block_v = 512, 256
    hidden_dim = act_shape[-1]

    key = jax.random.PRNGKey(42)
    key_act, key_emb, key_tgt = jax.random.split(key, 3)

    activations = jax.random.normal(key_act, act_shape)
    embeddings = jax.random.normal(key_emb, (vocab_size, hidden_dim))
    targets = jax.random.randint(key_tgt, tgt_shape, 0, vocab_size)

    def run_keras(act, emb):
      loss, _ = _keras_bce(act, emb, targets)
      return loss

    def run_cut(act, emb):
      return binary_cross_entropy_ops.cut_binary_cross_entropy(
          act, emb, targets, block_v=block_v, use_pallas=use_pallas
      )

    loss_keras = run_keras(activations, embeddings)
    loss_cut = run_cut(activations, embeddings)
    np.testing.assert_allclose(loss_cut, loss_keras, rtol=1e-3, atol=1e-3)

    # Test backward grad in 4D (exercises _bce_bwd_loop_fallback when
    # use_pallas=True and _bce_bwd_pure_jax's 4D scan branch when False).
    grad_keras_fn = jax.jit(jax.grad(run_keras, argnums=(0, 1)))
    grad_cut_fn = jax.jit(jax.grad(run_cut, argnums=(0, 1)))

    g_act_keras, g_emb_keras = grad_keras_fn(activations, embeddings)
    g_act_cut, g_emb_cut = grad_cut_fn(activations, embeddings)
    np.testing.assert_allclose(g_act_cut, g_act_keras, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(g_emb_cut, g_emb_keras, atol=1e-4, rtol=1e-4)

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
    act_sharding = jax.sharding.NamedSharding(mesh, P('devices', None, None))
    emb_sharding = jax.sharding.NamedSharding(mesh, P('devices', None))
    activations_sharded = jax.device_put(activations, act_sharding)
    embeddings_sharded = jax.device_put(embeddings, emb_sharding)

    # The embeddings must be replicated before the chunked matmul so that the
    # scan over vocab blocks does not trigger a collective per block. Assert on
    # the sharding the inner op actually receives rather than on how many
    # `with_sharding_constraint` calls were made. The activations' hidden dim
    # is replicated one level down, in `_replicate_hidden_dim`, which is
    # covered by `test_replicate_hidden_dim`.
    with mock.patch.object(
        binary_cross_entropy_ops,
        '_cut_binary_cross_entropy',
        wraps=binary_cross_entropy_ops._cut_binary_cross_entropy,
    ) as mock_fn:
      loss_cut = binary_cross_entropy_ops.cut_binary_cross_entropy(
          activations_sharded,
          embeddings_sharded,
          targets,
          block_v=256,
      )
      _, inner_activations, inner_embeddings, _ = mock_fn.call_args[0]

    self.assertEqual(inner_embeddings.sharding.spec, P())
    self.assertTrue(inner_embeddings.sharding.is_fully_replicated)
    # The token axes keep their sharding; only the vocab axis is gathered.
    self.assertEqual(inner_activations.sharding.spec, P('devices', None, None))

    # Sharding must not change the answer.
    loss_keras, _ = _keras_bce(activations, embeddings, targets)
    np.testing.assert_allclose(loss_cut, loss_keras, atol=1e-5, rtol=1e-5)

    def run_cut(act, emb):
      return binary_cross_entropy_ops.cut_binary_cross_entropy(
          act, emb, targets, block_v=256
      )

    # Same for the backward pass: assert the gradients, not that a sharding
    # constraint was called. Here the shardings are inferred from the input
    # arrays; the explicit mesh / act_spec / emb_spec path is covered by
    # `test_cut_bce_sharded_backward`.
    grad_fn = jax.jit(jax.grad(run_cut, argnums=(0, 1)))
    g_act, g_emb = grad_fn(activations_sharded, embeddings_sharded)
    g_act_ref, g_emb_ref = grad_fn(activations, embeddings)

    np.testing.assert_allclose(g_act, g_act_ref, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(g_emb, g_emb_ref, atol=1e-5, rtol=1e-5)

  @parameterized.named_parameters(
      ('pure_jax', False),
      ('pallas', True),
  )
  def test_cut_bce_sharded_backward(self, use_pallas):
    """Covers the backward pass when embeddings are vocab-sharded.

    `block_v` is chosen against the *global* vocab, but under
    `_bce_bwd_sharded`'s shard_map each shard only holds `vocab_size /
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
    # makes `_bce_bwd_sharded`'s dp_axes/psum bookkeeping incoherent.
    num_vocab_shards = jax.device_count()
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape(1, num_vocab_shards),
        ('data', 'model'),
    )
    act_spec = P('data', None, None)
    emb_spec = P('model', None)

    def run_cut(act, emb, **kwargs):
      return binary_cross_entropy_ops.cut_binary_cross_entropy(
          act, emb, targets, block_v=block_v, use_pallas=use_pallas, **kwargs
      )

    # Passing mesh/act_spec/emb_spec is what forces `_bce_bwd_sharded` down the
    # shard_map branch, where each shard sees only `vocab_size / num_devices`
    # embedding rows.
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
  def test_cut_bce_sharded_backward_hidden_dim_and_short_specs(
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
      return binary_cross_entropy_ops.cut_binary_cross_entropy(
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
  def test_cut_bce_correctness_large_sequence(self, seq_len):
    # Force chunk_n=128 so `n = batch * seq_len` (256 or 260) exceeds `chunk_n`
    # on every platform and exercises both the exact-multiple (`padded_n == n`)
    # and padded (`padded_n > n`) branches of `_bce_bwd_pallas_chunked_n`.
    batch, hidden_dim, vocab_size, num_labels = 2, 128, 256, 4
    block_v = 128

    key = jax.random.PRNGKey(42)
    key_act, key_emb, key_tgt = jax.random.split(key, 3)

    activations = jax.random.normal(key_act, (batch, seq_len, hidden_dim))
    embeddings = jax.random.normal(key_emb, (vocab_size, hidden_dim))
    targets = jax.random.randint(
        key_tgt, (batch, seq_len, num_labels), 0, vocab_size
    )

    # Keras BCE
    def run_keras(act, emb):
      loss, _ = _keras_bce(act, emb, targets)
      return loss

    grad_keras_fn = jax.jit(jax.grad(run_keras, argnums=(0, 1)))
    loss_keras = run_keras(activations, embeddings)
    g_act_keras, g_emb_keras = grad_keras_fn(activations, embeddings)

    # cut BCE
    def run_cut(act, emb):
      return binary_cross_entropy_ops.cut_binary_cross_entropy(
          act, emb, targets, block_v=block_v
      )

    with mock.patch.object(
        binary_cross_entropy_ops, '_max_safe_chunk_n', return_value=128
    ):
      grad_cut_fn = jax.jit(jax.grad(run_cut, argnums=(0, 1)))
      loss_cut = run_cut(activations, embeddings)
      g_act_cut, g_emb_cut = grad_cut_fn(activations, embeddings)

    # Compare
    np.testing.assert_allclose(loss_cut, loss_keras, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(g_act_cut, g_act_keras, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(g_emb_cut, g_emb_keras, atol=1e-4, rtol=1e-4)

  def test_cut_bce_use_pallas_vs_pure_jax_equivalence(self):
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

    def run_cut(act, emb, use_pallas):
      return binary_cross_entropy_ops.cut_binary_cross_entropy(
          act, emb, targets, block_v=block_v, use_pallas=use_pallas
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
    budget = binary_cross_entropy_ops._pallas_vmem_budget()
    if any(d.platform == 'tpu' for d in jax.devices()):
      expected = max(
          16 * _BYTES_IN_MB,
          pltpu.get_tpu_info().vmem_capacity_bytes - 16 * _BYTES_IN_MB,
      )
      self.assertEqual(budget, expected)
    else:
      self.assertEqual(budget, 16 * _BYTES_IN_MB)

  def test_pallas_sublane(self):
    sublane = binary_cross_entropy_ops._pallas_sublane()
    if any(d.platform == 'tpu' for d in jax.devices()):
      self.assertEqual(sublane, pltpu.get_tpu_info().num_sublanes)
    else:
      self.assertEqual(sublane, 8)

  def test_vocab_bytes_per_block_v_grows_with_hidden(self):
    small = binary_cross_entropy_ops._vocab_bytes_per_block_v(128, 128)
    large = binary_cross_entropy_ops._vocab_bytes_per_block_v(512, 128)
    self.assertGreater(large, small)

  def test_max_safe_block_v_is_mxu_aligned(self):
    mxu = binary_cross_entropy_ops._get_mxu_size()
    for vmem_mb in (16, 48, 112):
      val = binary_cross_entropy_ops._max_safe_block_v(
          vmem_mb * _BYTES_IN_MB, 256
      )
      self.assertEqual(val % mxu, 0)

  def test_max_safe_block_v_scales_with_vmem(self):
    small_vmem = binary_cross_entropy_ops._max_safe_block_v(
        32 * _BYTES_IN_MB, 256
    )
    medium_vmem = binary_cross_entropy_ops._max_safe_block_v(
        48 * _BYTES_IN_MB, 256
    )
    large_vmem = binary_cross_entropy_ops._max_safe_block_v(
        64 * _BYTES_IN_MB, 256
    )
    xlarge_vmem = binary_cross_entropy_ops._max_safe_block_v(
        128 * _BYTES_IN_MB, 256
    )
    # Crossing 32 MB steps the share up from _SMALL_VMEM_SHARE (0.3) to
    # _VOCAB_VMEM_SHARE (0.5), so a 1.5x VMEM increase (32 -> 48 MB) more than
    # doubles the safe block_v.
    self.assertGreater(medium_vmem, 2 * small_vmem)
    self.assertGreater(large_vmem, medium_vmem)
    self.assertGreater(xlarge_vmem, large_vmem)

  def test_max_safe_block_v_shrinks_as_hidden_grows(self):
    narrow = binary_cross_entropy_ops._max_safe_block_v(112 * _BYTES_IN_MB, 256)
    wide = binary_cross_entropy_ops._max_safe_block_v(112 * _BYTES_IN_MB, 2048)
    self.assertGreater(narrow, wide)

  def test_max_safe_block_v_floors_at_one_mxu_tile(self):
    mxu = binary_cross_entropy_ops._get_mxu_size()
    val = binary_cross_entropy_ops._max_safe_block_v(_BYTES_IN_MB, 8192)
    self.assertEqual(val, mxu)

  def test_max_safe_chunk_n_is_lane_aligned(self):
    lane = binary_cross_entropy_ops._pallas_lane()
    val = binary_cross_entropy_ops._max_safe_chunk_n(
        112 * _BYTES_IN_MB, 128, 4096
    )
    self.assertEqual(val % lane, 0)

  def test_max_safe_chunk_n_scales_with_vmem(self):
    small_vmem = binary_cross_entropy_ops._max_safe_chunk_n(
        32 * _BYTES_IN_MB, 128, 1024
    )
    medium_vmem = binary_cross_entropy_ops._max_safe_chunk_n(
        48 * _BYTES_IN_MB, 128, 1024
    )
    large_vmem = binary_cross_entropy_ops._max_safe_chunk_n(
        64 * _BYTES_IN_MB, 128, 1024
    )
    xlarge_vmem = binary_cross_entropy_ops._max_safe_chunk_n(
        128 * _BYTES_IN_MB, 128, 1024
    )
    # Crossing 32 MB steps the share up from _SMALL_VMEM_SHARE (0.3) to
    # _TARGET_RATIO (0.8), so a 1.5x VMEM increase (32 -> 48 MB) more than
    # doubles the safe chunk_n.
    self.assertGreater(medium_vmem, 2 * small_vmem)
    self.assertGreater(large_vmem, medium_vmem)
    self.assertGreater(xlarge_vmem, large_vmem)

  def test_max_safe_chunk_n_shrinks_as_block_v_grows(self):
    small_vocab = binary_cross_entropy_ops._max_safe_chunk_n(
        112 * _BYTES_IN_MB, 128, 1024
    )
    large_vocab = binary_cross_entropy_ops._max_safe_chunk_n(
        112 * _BYTES_IN_MB, 128, 8192
    )
    self.assertGreater(small_vocab, large_vocab)

  def test_max_safe_chunk_n_floors_at_min_chunk_tiles(self):
    lane = binary_cross_entropy_ops._pallas_lane()
    # A vocabulary block large enough to exhaust the budget on its own.
    val = binary_cross_entropy_ops._max_safe_chunk_n(_BYTES_IN_MB, 1024, 65536)
    self.assertEqual(val, binary_cross_entropy_ops._MIN_CHUNK_TILES * lane)

  def test_balanced_chunk_n_avoids_padding_waste(self):
    # The regression case: running at the 48768 ceiling would process 97536
    # rows for 65536 tokens. Balancing splits it into two exact halves.
    self.assertEqual(
        binary_cross_entropy_ops._balanced_chunk_n(65536, 48768), 32768
    )

  def test_balanced_chunk_n_never_exceeds_ceiling(self):
    for n in (1000, 65536, 100000, 524288):
      for ceiling in (4096, 32768, 48768):
        chunk = binary_cross_entropy_ops._balanced_chunk_n(n, ceiling)
        self.assertLessEqual(chunk, ceiling)

  def test_balanced_chunk_n_still_covers_all_tokens(self):
    for n in (1000, 65536, 100000, 524288):
      for ceiling in (4096, 32768, 48768):
        chunk = binary_cross_entropy_ops._balanced_chunk_n(n, ceiling)
        n_chunks = (n + chunk - 1) // chunk
        self.assertGreaterEqual(n_chunks * chunk, n)

  def test_balanced_chunk_n_uses_no_more_chunks_than_ceiling(self):
    # Balancing must not increase the number of kernel invocations.
    for n in (65536, 100000, 524288):
      for ceiling in (4096, 32768, 48768):
        chunk = binary_cross_entropy_ops._balanced_chunk_n(n, ceiling)
        self.assertEqual((n + chunk - 1) // chunk, (n + ceiling - 1) // ceiling)

  def test_balanced_chunk_n_is_lane_aligned(self):
    lane = binary_cross_entropy_ops._pallas_lane()
    chunk = binary_cross_entropy_ops._balanced_chunk_n(100000, 48768)
    self.assertEqual(chunk % lane, 0)

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
          P('devices', 'devices')
      )
    binary_cross_entropy_ops._check_vocab_replicated_in_d(P('devices'))


if __name__ == '__main__':
  absltest.main()

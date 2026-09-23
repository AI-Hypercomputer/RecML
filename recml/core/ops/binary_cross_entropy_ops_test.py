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

    loss_keras, _ = _keras_bce(activations, embeddings, targets)
    loss_cut = binary_cross_entropy_ops.cut_binary_cross_entropy(
        activations, embeddings, targets, block_v=block_v
    )

    # Compare
    np.testing.assert_allclose(loss_cut, loss_keras, atol=1e-5, rtol=1e-5)

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

    loss_keras, _ = _keras_bce(activations, embeddings, targets)
    loss_cut = binary_cross_entropy_ops.cut_binary_cross_entropy(
        activations, embeddings, targets, block_v=block_v
    )
    np.testing.assert_allclose(loss_cut, loss_keras, rtol=1e-3, atol=1e-3)

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

    loss_keras, _ = _keras_bce(activations, embeddings, targets)
    loss_cut = binary_cross_entropy_ops.cut_binary_cross_entropy(
        activations, embeddings, targets, block_v=block_v
    )

    # Compare
    np.testing.assert_allclose(loss_cut, loss_keras, atol=1e-5, rtol=1e-5)

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

  def test_cut_bce_backward_not_implemented(self):
    activations = jnp.ones((2, 64, 32))
    embeddings = jnp.ones((128, 32))
    targets = jnp.zeros((2, 64, 2), dtype=jnp.int32)

    def run_cut(act, emb):
      return binary_cross_entropy_ops.cut_binary_cross_entropy(
          act, emb, targets, block_v=128
      )

    with self.assertRaises(NotImplementedError):
      jax.grad(run_cut, argnums=(0, 1))(activations, embeddings)


if __name__ == '__main__':
  absltest.main()

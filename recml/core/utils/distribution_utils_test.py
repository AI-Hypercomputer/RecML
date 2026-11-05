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
"""Tests mesh and batch-axis resolution in distribution_utils."""

import re

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import keras
from keras.src.distribution import distribution_lib
import numpy as np
from recml.core.utils import distribution_utils

# Virtual CPU devices for the test meshes. Must be even for the 2D meshes.
_NUM_DEVICES = 8
# Must run before JAX starts.
jax.config.update('jax_num_cpu_devices', _NUM_DEVICES)
_DEVICES = [f'cpu:{i}' for i in range(_NUM_DEVICES)]


class _StubDistribution:
  """Minimal stand-in for a Keras distribution.

  Real hybrid-FSDP distributions live outside this package, so the two
  attributes `distribution_utils` reads are stubbed here instead.
  """

  def __init__(self, device_mesh, batch_dim_name):
    self.device_mesh = device_mesh
    self.batch_dim_name = batch_dim_name


def _abstract_mesh(
    axis_names: tuple[str, ...], axis_sizes: tuple[int, ...]
) -> jax.sharding.AbstractMesh:
  return jax.sharding.AbstractMesh(
      axis_sizes=axis_sizes,
      axis_names=axis_names,
      axis_types=tuple(jax.sharding.AxisType.Auto for _ in axis_names),
  )


def _all_reduced_elements(hlo_text: str, shape: str) -> int:
  """Counts all-reduced arrays of `shape` (e.g. 'f32[24,8]') in HLO text."""
  count = 0
  for line in hlo_text.splitlines():
    match = re.search(r'=\s*(.*?)\s+all-reduce(?:-start)?\(', line)
    if match:
      count += match.group(1).count(shape)
  return count


class DistributionUtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # The active distribution is process-global state; make sure no test
    # leaks into another.
    keras.distribution.set_distribution(None)
    self.addCleanup(keras.distribution.set_distribution, None)

  @parameterized.named_parameters(
      ('none', None, None),
      ('bare_name', 'data', 'data'),
      ('empty_tuple', (), None),
      ('single_element_tuple', ('data',), 'data'),
      ('single_element_list', ['data'], 'data'),
      ('multi_element_tuple', ('replica', 'fsdp'), ('replica', 'fsdp')),
      ('multi_element_list', ['replica', 'fsdp'], ('replica', 'fsdp')),
  )
  def test_normalize_axis(self, axis, expected):
    self.assertEqual(distribution_utils.normalize_axis(axis), expected)

  @parameterized.named_parameters(
      ('single_axis', ('data',), (8,), 'data'),
      ('batch_and_model', ('batch', 'model'), (2, 4), 'batch'),
      ('hybrid_fsdp', ('replica', 'fsdp'), (2, 4), ('replica', 'fsdp')),
      # Regression: a mesh whose only axis is the model axis must still
      # resolve to something rather than an empty tuple.
      ('model_axis_only', ('model',), (8,), 'model'),
  )
  def test_resolve_batch_axis_from_mesh(self, axis_names, axis_sizes, expected):
    mesh = _abstract_mesh(axis_names, axis_sizes)
    self.assertEqual(distribution_utils.resolve_batch_axis(mesh), expected)

  def test_resolve_batch_axis_without_mesh_or_distribution(self):
    self.assertIsNone(distribution_utils.resolve_batch_axis(None))

  def test_resolve_batch_axis_prefers_distribution_over_mesh(self):
    device_mesh = distribution_lib.DeviceMesh(
        (2, _NUM_DEVICES // 2), ('replica', 'fsdp'), _DEVICES
    )
    keras.distribution.set_distribution(
        _StubDistribution(device_mesh, ('replica', 'fsdp'))
    )
    # The mesh alone would resolve to 'batch'; the distribution wins.
    mesh = _abstract_mesh(('batch', 'model'), (2, 4))
    self.assertEqual(
        distribution_utils.resolve_batch_axis(mesh), ('replica', 'fsdp')
    )

  def test_resolve_batch_axis_from_data_parallel_distribution(self):
    device_mesh = distribution_lib.DeviceMesh(
        (_NUM_DEVICES,), ('batch',), _DEVICES
    )
    keras.distribution.set_distribution(
        keras.distribution.DataParallel(device_mesh=device_mesh)
    )
    self.assertEqual(distribution_utils.resolve_batch_axis(None), 'batch')

  def test_get_batch_dim_name_without_distribution(self):
    self.assertIsNone(distribution_utils.get_batch_dim_name())

  def test_get_abstract_mesh_without_distribution_or_mesh(self):
    self.assertIsNone(distribution_utils.get_abstract_mesh())

  def test_get_abstract_mesh_from_global_jax_mesh(self):
    mesh = jax.sharding.Mesh(jax.devices(), axis_names=('data',))
    with jax.set_mesh(mesh):
      abstract_mesh = distribution_utils.get_abstract_mesh()
    self.assertIsNotNone(abstract_mesh)
    self.assertEqual(abstract_mesh.axis_names, ('data',))

  def test_get_abstract_mesh_from_distribution(self):
    device_mesh = distribution_lib.DeviceMesh(
        (2, _NUM_DEVICES // 2), ('replica', 'fsdp'), _DEVICES
    )
    keras.distribution.set_distribution(_StubDistribution(device_mesh, None))

    mesh = distribution_utils.get_abstract_mesh()

    self.assertIsNotNone(mesh)
    self.assertEqual(mesh.axis_names, ('replica', 'fsdp'))
    self.assertEqual(mesh.axis_sizes, (2, _NUM_DEVICES // 2))
    # Regression: `axis_types` must have one entry per axis. Passing a bare
    # `AxisType` raises `ValueError` on a mesh with more than one axis.
    self.assertEqual(
        mesh.axis_types,
        (jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto),
    )


# TODO(jinchi): Add sharded-table (FSDP) tests once fused table gradients
# support FSDP.
class FusedTableGradTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    keras.distribution.set_distribution(None)
    self.addCleanup(keras.distribution.set_distribution, None)

  def test_fuse_table_grads_disabled_without_distribution(self):
    self.assertFalse(distribution_utils.fuse_table_grads_enabled())

  def test_fused_table_grad_data_parallel_opts_in(self):
    device_mesh = distribution_lib.DeviceMesh(
        (_NUM_DEVICES,), ('batch',), _DEVICES
    )
    keras.distribution.set_distribution(
        distribution_utils.FusedTableGradDataParallel(device_mesh=device_mesh)
    )
    self.assertTrue(distribution_utils.fuse_table_grads_enabled())

  def test_fused_table_grad_call_under_data_parallel(self):
    """Several reads of one replicated table, as in GUM."""
    device_mesh = distribution_lib.DeviceMesh(
        (_NUM_DEVICES,), ('batch',), _DEVICES
    )
    keras.distribution.set_distribution(
        distribution_utils.FusedTableGradDataParallel(device_mesh=device_mesh)
    )
    vocab, dim, batch, tokens = 16, 8, _NUM_DEVICES, 3
    rng = np.random.default_rng(0)
    table = keras.Variable(
        rng.normal(size=(vocab, dim)).astype(np.float32), name='table'
    )
    ids = [
        rng.integers(0, vocab, size=(batch, tokens), dtype=np.int32)
        for _ in range(3)
    ]
    h = rng.normal(size=(batch, dim)).astype(np.float32)
    out_weights = rng.normal(size=(batch, 3 * dim + vocab)).astype(np.float32)

    def lookup(full_table, ids):
      return jnp.take(full_table, ids, axis=0).sum(axis=1)

    def tied_logits(full_table, h):
      return h @ full_table.T

    def loss_fn(table_value, h, fuse_grads):
      with keras.StatelessScope(state_mapping=[(table, table_value)]):
        outs = []
        for i in ids:
          if fuse_grads:
            outs.append(
                distribution_utils.fused_table_grad_call(lookup, table, i)
            )
          else:
            outs.append(lookup(keras.ops.convert_to_tensor(table), i))
        if fuse_grads:
          outs.append(
              distribution_utils.fused_table_grad_call(tied_logits, table, h)
          )
        else:
          outs.append(tied_logits(keras.ops.convert_to_tensor(table), h))
      return jnp.sum(jnp.concatenate(outs, axis=-1) * out_weights)

    grad_fn = jax.jit(
        jax.value_and_grad(loss_fn, argnums=(0, 1)), static_argnums=2
    )
    expected_loss, expected_grads = grad_fn(table.value, h, False)
    loss, grads = grad_fn(table.value, h, True)

    np.testing.assert_allclose(loss, expected_loss, rtol=1e-5)
    for grad, expected_grad in zip(grads, expected_grads):
      np.testing.assert_allclose(grad, expected_grad, rtol=1e-5, atol=1e-5)

  def test_fused_table_grad_call_reduces_replicated_table_once(self):
    """All reads of a replicated table share one gradient all-reduce."""
    device_mesh = distribution_lib.DeviceMesh(
        (_NUM_DEVICES,), ('batch',), _DEVICES
    )
    keras.distribution.set_distribution(
        distribution_utils.FusedTableGradDataParallel(device_mesh=device_mesh)
    )
    vocab, dim, batch, tokens = 24, 8, _NUM_DEVICES, 3
    rng = np.random.default_rng(0)
    table = keras.Variable(
        rng.normal(size=(vocab, dim)).astype(np.float32), name='table'
    )
    ids = [
        rng.integers(0, vocab, size=(batch, tokens), dtype=np.int32)
        for _ in range(3)
    ]
    h = rng.normal(size=(batch, dim)).astype(np.float32)

    def lookup(full_table, ids):
      return jnp.take(full_table, ids, axis=0).sum(axis=1)

    def tied_logits(full_table, h):
      return h @ full_table.T

    def loss_fn(table_value, h):
      with keras.StatelessScope(state_mapping=[(table, table_value)]):
        outs = [
            distribution_utils.fused_table_grad_call(lookup, table, i)
            for i in ids
        ]
        outs.append(
            distribution_utils.fused_table_grad_call(tied_logits, table, h)
        )
      return sum(jnp.sum(out) for out in outs)

    hlo = (
        jax.jit(jax.grad(loss_fn, argnums=(0, 1)))
        .lower(table.value, h)
        .compile()
        .as_text()
    )

    # 4 calls read the table, but its gradient is all-reduced once.
    self.assertEqual(_all_reduced_elements(hlo, f'f32[{vocab},{dim}]'), 1)

  def test_fused_table_grad_call_in_separate_traces(self):
    """Each trace makes its own table copies."""
    device_mesh = distribution_lib.DeviceMesh(
        (_NUM_DEVICES,), ('batch',), _DEVICES
    )
    keras.distribution.set_distribution(
        distribution_utils.FusedTableGradDataParallel(device_mesh=device_mesh)
    )
    vocab, dim, batch = 16, 8, _NUM_DEVICES
    rng = np.random.default_rng(0)
    table = keras.Variable(
        rng.normal(size=(vocab, dim)).astype(np.float32), name='table'
    )
    h = rng.normal(size=(batch, dim)).astype(np.float32)

    def tied_logits(full_table, h):
      return h @ full_table.T

    def loss_fn(table_value, h):
      with keras.StatelessScope(state_mapping=[(table, table_value)]):
        logits = distribution_utils.fused_table_grad_call(tied_logits, table, h)
      return jnp.sum(jnp.square(logits))

    # Reusing the first trace's copies would raise an escaped-tracer error.
    loss = jax.jit(loss_fn)(table.value, h)
    table_grad, _ = jax.jit(jax.grad(loss_fn, argnums=(0, 1)))(table.value, h)

    logits = h @ np.asarray(table.value).T
    np.testing.assert_allclose(loss, np.sum(np.square(logits)), rtol=1e-5)
    np.testing.assert_allclose(
        table_grad, 2 * logits.T @ h, rtol=1e-4, atol=1e-4
    )

  @parameterized.named_parameters(('off', False), ('on', True))
  def test_fuse_table_grads_reads_distribution(self, value):
    device_mesh = distribution_lib.DeviceMesh(
        (_NUM_DEVICES,), ('batch',), _DEVICES
    )
    distribution = keras.distribution.DataParallel(device_mesh=device_mesh)
    distribution.fuse_table_grads = value
    keras.distribution.set_distribution(distribution)
    self.assertEqual(distribution_utils.fuse_table_grads_enabled(), value)


if __name__ == '__main__':
  absltest.main()

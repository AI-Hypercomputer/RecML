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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import keras
from keras.src.distribution import distribution_lib
from recml.core.utils import distribution_utils

_DEVICES = [f'cpu:{i}' for i in range(8)]


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
        (2, 4), ('replica', 'fsdp'), _DEVICES
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
    device_mesh = distribution_lib.DeviceMesh((8,), ('batch',), _DEVICES)
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
        (2, 4), ('replica', 'fsdp'), _DEVICES
    )
    keras.distribution.set_distribution(_StubDistribution(device_mesh, None))

    mesh = distribution_utils.get_abstract_mesh()

    self.assertIsNotNone(mesh)
    self.assertEqual(mesh.axis_names, ('replica', 'fsdp'))
    self.assertEqual(mesh.axis_sizes, (2, 4))
    # Regression: `axis_types` must have one entry per axis. Passing a bare
    # `AxisType` raises `ValueError` on a mesh with more than one axis.
    self.assertEqual(
        mesh.axis_types,
        (jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto),
    )


if __name__ == '__main__':
  absltest.main()

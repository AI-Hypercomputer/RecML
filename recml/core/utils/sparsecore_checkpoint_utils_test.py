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
"""Tests for `sparsecore_checkpoint_utils`.

The SparseCore tables are described by `jax_tpu_embedding` `TableSpec`s stacked
for a given number of SparseCores per chip, so the tests run on CPU without
SparseCore. `sparsecore_checkpoint_utils_multidevice_test` runs them on 8
virtual CPU devices.
"""

from collections.abc import Mapping, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import layout as jax_layout
from jax_tpu_embedding.sparsecore.lib.nn import embedding as jte_embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.nn import table_stacking as jte_table_stacking
import numpy as np
from recml.core.utils import sparsecore_checkpoint_utils

# Three tables in one stack (all dims pad to 8) with vocab sizes that pad
# differently for different numbers of SparseCores.
_VOCABS_AND_DIMS = {"t1": (19, 6), "t2": (35, 8), "t3": (10, 4)}


def _table_stack(
    num_sc_per_device: int,
    rotation: int | None = None,
    global_device_count: int | None = None,
) -> Sequence[embedding_spec.TableSpec]:
  """Returns the tables stacked like `keras_rs` (for the current devices)."""
  feature_specs = []
  for name, (vocab, dim) in _VOCABS_AND_DIMS.items():
    table_spec = embedding_spec.TableSpec(
        name=name,
        vocabulary_size=vocab,
        embedding_dim=dim,
        initializer=jax.nn.initializers.zeros,
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
    )
    feature_specs.append(
        embedding_spec.FeatureSpec(
            name=f"f_{name}",
            table_spec=table_spec,
            input_shape=(8, 1),
            output_shape=(8, dim),
        )
    )
  jte_table_stacking.stack_tables(
      feature_specs,
      list(_VOCABS_AND_DIMS),
      global_device_count=global_device_count or jax.device_count(),
      num_sc_per_device=num_sc_per_device,
      rotation=rotation,
  )
  (table_stack,) = jte_table_stacking.get_table_stacks(
      jte_embedding.get_table_specs(feature_specs)
  ).values()
  return table_stack


def _stack(
    table_stack: Sequence[embedding_spec.TableSpec],
    tables: Mapping[str, np.ndarray],
    num_shards: int,
) -> np.ndarray:
  """Returns `tables` stacked and sharded like `table_stack`."""
  stack_name = table_stack[0].stacked_table_spec.stack_name
  stacked = jte_table_stacking.stack_and_shard_tables(
      list(table_stack),
      [tables[spec.name] for spec in table_stack],
      num_shards,
      pad_value=0.0,
  )[stack_name]
  return np.asarray(stacked).reshape(-1, stacked.shape[-1])


def _sparsecore_sharding() -> jax.sharding.NamedSharding:
  mesh = jax.sharding.Mesh(np.array(jax.devices()), ("sparsecore",))
  return jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec("sparsecore")
  )


class SparseCoreCheckpointUtilsTest(parameterized.TestCase):

  @parameterized.parameters(1, 2, 4)
  def test_settings_in_stack_match_stack_tables(self, num_sc_per_device):
    # Recomputed from a stack for another topology.
    table_stack = _table_stack(num_sc_per_device=8)
    expected = {
        spec.name: spec.setting_in_stack
        for spec in _table_stack(num_sc_per_device)
    }
    self.assertEqual(
        sparsecore_checkpoint_utils.settings_in_stack(
            table_stack,
            jax.device_count() * num_sc_per_device,
            num_sc_per_device,
        ),
        expected,
    )

  @parameterized.parameters((1, 4), (2, 4), (4, 2), (4, 4))
  def test_reshard_stacked_array_matches_direct_stacking(
      self, src_num_sc_per_device, dst_num_sc_per_device
  ):
    rng = np.random.default_rng(0)
    tables = {
        name: rng.normal(size=vocab_and_dim).astype(np.float32)
        for name, vocab_and_dim in _VOCABS_AND_DIMS.items()
    }
    src_num_shards = jax.device_count() * src_num_sc_per_device
    dst_num_shards = jax.device_count() * dst_num_sc_per_device
    src_stacked = _stack(
        _table_stack(src_num_sc_per_device), tables, src_num_shards
    )
    dst_table_stack = _table_stack(dst_num_sc_per_device)
    expected = _stack(dst_table_stack, tables, dst_num_shards)
    sharding = _sparsecore_sharding()
    src_array = jax.device_put(
        src_stacked,
        sparsecore_checkpoint_utils.row_sharding(
            sharding, src_stacked.shape[0]
        ),
    )

    actual = sparsecore_checkpoint_utils.reshard_stacked_array(
        src_array,
        dst_table_stack,
        src_num_shards=src_num_shards,
        src_num_sc_per_device=src_num_sc_per_device,
        dst_num_shards=dst_num_shards,
        dst_num_sc_per_device=dst_num_sc_per_device,
        out_sharding=sharding,
    )

    self.assertTrue(actual.sharding.is_equivalent_to(sharding, actual.ndim))
    np.testing.assert_array_equal(np.asarray(actual), expected)
    # Consumed, whether its buffers could be donated or not.
    self.assertTrue(src_array.is_deleted())

  def test_reshard_stacked_array_from_replicated_row_groups(self):
    num_devices = jax.device_count()
    if num_devices < 4:
      self.skipTest("Requires at least 4 devices.")
    tables = {
        name: (
            np.random.default_rng(0)
            .normal(size=vocab_and_dim)
            .astype(np.float32)
        )
        for name, vocab_and_dim in _VOCABS_AND_DIMS.items()
    }
    # Saved on 1 chip with 4 SparseCores, restored on all devices.
    src_num_shards = 4
    src_stacked = _stack(
        _table_stack(num_sc_per_device=4, global_device_count=1),
        tables,
        src_num_shards,
    )
    dst_table_stack = _table_stack(num_sc_per_device=4)
    dst_num_shards = num_devices * 4
    expected = _stack(dst_table_stack, tables, dst_num_shards)
    # Row shards replicated on pairs of devices, like `row_sharding` for row
    # counts that aren't divisible by the device count.
    in_mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape(num_devices // 2, 2),
        ("rows", "replicas"),
    )
    src_array = jax.device_put(
        src_stacked,
        jax.sharding.NamedSharding(in_mesh, jax.sharding.PartitionSpec("rows")),
    )
    out_sharding = _sparsecore_sharding()

    actual = sparsecore_checkpoint_utils.reshard_stacked_array(
        src_array,
        dst_table_stack,
        src_num_shards=src_num_shards,
        src_num_sc_per_device=4,
        dst_num_shards=dst_num_shards,
        dst_num_sc_per_device=4,
        out_sharding=out_sharding,
    )

    self.assertTrue(actual.sharding.is_equivalent_to(out_sharding, actual.ndim))
    np.testing.assert_array_equal(np.asarray(actual), expected)
    # The shards have other sizes, so it can't be donated but is deleted.
    self.assertTrue(src_array.is_deleted())

  def test_reshard_stacked_array_to_format(self):
    table_stack = _table_stack(num_sc_per_device=4)
    num_shards = jax.device_count() * 4
    stacked = np.zeros(
        (table_stack[0].stacked_table_spec.stack_vocab_size, 8), np.float32
    )
    out_format = jax_layout.Format(None, _sparsecore_sharding())

    actual = sparsecore_checkpoint_utils.reshard_stacked_array(
        jax.device_put(stacked, _sparsecore_sharding()),
        table_stack,
        src_num_shards=num_shards,
        src_num_sc_per_device=4,
        dst_num_shards=num_shards,
        dst_num_sc_per_device=4,
        out_sharding=out_format,
    )

    self.assertTrue(
        actual.sharding.is_equivalent_to(_sparsecore_sharding(), actual.ndim)
    )
    np.testing.assert_array_equal(np.asarray(actual), stacked)

  def test_reshard_stacked_array_rejects_unexpected_layout(self):
    # Stacked with another shard rotation than `keras_rs` uses.
    table_stack = _table_stack(num_sc_per_device=4, rotation=1)
    num_shards = jax.device_count() * 4
    stacked = np.zeros(
        (table_stack[0].stacked_table_spec.stack_vocab_size, 8), np.float32
    )
    with self.assertRaisesRegex(
        ValueError, "Unexpected SparseCore stacking layout"
    ):
      sparsecore_checkpoint_utils.reshard_stacked_array(
          jax.numpy.asarray(stacked),
          table_stack,
          src_num_shards=num_shards,
          src_num_sc_per_device=4,
          dst_num_shards=num_shards,
          dst_num_sc_per_device=4,
          out_sharding=_sparsecore_sharding(),
      )

  def test_row_sharding_keeps_divisible_rows(self):
    sharding = _sparsecore_sharding()
    self.assertIs(
        sparsecore_checkpoint_utils.row_sharding(
            sharding, 8 * jax.device_count()
        ),
        sharding,
    )

  def test_row_sharding_splits_uneven_rows_into_groups(self):
    num_devices = jax.device_count()
    if num_devices < 4:
      self.skipTest("Requires at least 4 devices.")
    # E.g. 8 devices and 4 * 3 rows: 4 row shards, each on 2 devices.
    num_rows = (num_devices // 2) * 3
    sharding = sparsecore_checkpoint_utils.row_sharding(
        _sparsecore_sharding(), num_rows
    )
    self.assertEqual(sharding.mesh.devices.shape, (num_devices // 2, 2))
    x = jax.device_put(np.arange(num_rows * 2.0).reshape(num_rows, 2), sharding)
    self.assertEqual(
        {shard.data.shape for shard in x.addressable_shards}, {(3, 2)}
    )


if __name__ == "__main__":
  absltest.main()

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
for a given number of SparseCores per chip, and models hold them in a layer with
the SparseCore state of `keras_rs.layers.DistributedEmbedding`, so the tests run
on CPU without SparseCore. `sparsecore_checkpoint_utils_multidevice_test` runs
them on 8 virtual CPU devices.
"""

from collections.abc import Mapping, Sequence
import os
import shutil
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import layout as jax_layout
from jax_tpu_embedding.sparsecore.lib.nn import embedding as jte_embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.nn import table_stacking as jte_table_stacking
import keras
import numpy as np
from recml.core.utils import keras_utils
from recml.core.utils import sparsecore_checkpoint_utils

# Three tables in one stack (all dims pad to 8) with vocab sizes that pad
# differently for different numbers of SparseCores.
_VOCABS_AND_DIMS = {"t1": (19, 6), "t2": (35, 8), "t3": (10, 4)}


def _stacked_feature_specs(
    num_sc_per_device: int,
    rotation: int | None = None,
) -> list[embedding_spec.FeatureSpec]:
  """Returns features of tables stacked for the devices, like `keras_rs`."""
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
      global_device_count=jax.device_count(),
      num_sc_per_device=num_sc_per_device,
      rotation=rotation,
  )
  return feature_specs


def _table_stack(
    num_sc_per_device: int, rotation: int | None = None
) -> Sequence[embedding_spec.TableSpec]:
  """Returns the tables stacked for the current devices, like `keras_rs`."""
  (table_stack,) = jte_table_stacking.get_table_stacks(
      jte_embedding.get_table_specs(
          _stacked_feature_specs(num_sc_per_device, rotation)
      )
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


class _FakeSparseCoreLayer(keras.layers.Layer):
  """Holds the SparseCore state of a `keras_rs.layers.DistributedEmbedding`.

  `sparsecore_checkpoint_utils` reads the SparseCore lookup config, layout and
  stacked table and slot variables of the layers. This layer has one stack of
  `_VOCABS_AND_DIMS`, stacked like `keras_rs` for `num_sc_per_device`
  SparseCores per chip, and sharded across the devices like the SparseCore
  tables, so that checkpoints can be tested without SparseCore.
  """

  def __init__(self, num_sc_per_device: int, **kwargs):
    super().__init__(**kwargs)
    self._num_sc_per_device = num_sc_per_device
    feature_specs = _stacked_feature_specs(num_sc_per_device)
    (table_stack,) = jte_table_stacking.get_table_stacks(
        jte_embedding.get_table_specs(feature_specs)
    ).values()
    # In a namespace, so that Keras doesn't wrap the lists for tracking.
    self._specs = types.SimpleNamespace(
        feature_specs=feature_specs, table_stack=table_stack
    )

  @property
  def table_stack(self) -> Sequence[embedding_spec.TableSpec]:
    return self._specs.table_stack

  @property
  def num_shards(self) -> int:
    return jax.device_count() * self._num_sc_per_device

  def build(self, input_shape=None):
    del input_shape
    stacked_spec = self.table_stack[0].stacked_table_spec
    shape = (stacked_spec.stack_vocab_size, stacked_spec.stack_embedding_dim)
    sharding = _sparsecore_sharding()
    layout = keras.distribution.TensorLayout(
        ("sparsecore", None),
        keras.distribution.DeviceMesh(
            (jax.device_count(),), ("sparsecore",), jax.devices()
        ),
    )
    # Like `keras_rs`, a JAX `Format` (here without the TPU tiling).
    layout._backend_layout = jax_layout.Format(None, sharding)  # pylint: disable=protected-access
    table = self.add_weight(shape=shape, initializer="zeros", name="table")
    # Trained on SparseCore, so the Keras optimizer has no slots for it.
    table.overwrite_with_gradient = True
    slot = self.add_weight(
        shape=shape, initializer="zeros", name="slot", trainable=False
    )
    for var in (table, slot):
      var._layout = layout  # pylint: disable=protected-access
      var.assign(var.value)
    self._config = types.SimpleNamespace(
        num_sc_per_device=self._num_sc_per_device,
        mesh=sharding.mesh,
        feature_specs=self._specs.feature_specs,
    )
    self._sparsecore_layout = layout
    self._table_and_slot_variables = {
        stacked_spec.stack_name: types.SimpleNamespace(
            table=table, slot=(slot,)
        )
    }
    self._sparsecore_built = True
    self.built = True

  @property
  def stack_variables(self) -> types.SimpleNamespace:
    (emb_vars,) = self._table_and_slot_variables.values()
    return emb_vars

  def set_tables(
      self, tables: Mapping[str, np.ndarray], slots: Mapping[str, np.ndarray]
  ) -> None:
    self.stack_variables.table.assign(
        _stack(self.table_stack, tables, self.num_shards)
    )
    (slot,) = self.stack_variables.slot
    slot.assign(_stack(self.table_stack, slots, self.num_shards))

  def get_tables(self, var: keras.Variable) -> dict[str, np.ndarray]:
    stack_name = self.table_stack[0].stacked_table_spec.stack_name
    tables = jte_table_stacking.unshard_and_unstack_tables(
        list(self.table_stack), {stack_name: var.value}, self.num_shards
    )
    return {
        spec.name: np.asarray(table)
        for spec, table in zip(self.table_stack, tables)
    }


class _TestModel(keras.Model):

  def __init__(self, num_sc_per_device: int | None):
    super().__init__(name="test_model")
    self._num_sc_per_device = num_sc_per_device
    self.sc_layer = (
        _FakeSparseCoreLayer(num_sc_per_device, name="sc_emb")
        if num_sc_per_device
        else None
    )
    self.dense = keras.layers.Dense(2, name="head")

  def build(self, input_shape=None):
    if self.sc_layer is not None:
      self.sc_layer.build()
    self.dense.build((None, 8))
    self.built = True

  def get_config(self):
    # Saved with the checkpoints.
    return {"num_sc_per_device": self._num_sc_per_device}


def _build_model(num_sc_per_device: int | None) -> _TestModel:
  model = _TestModel(num_sc_per_device)
  model.build()
  model.compile(optimizer=keras.optimizers.Adam(1e-3))
  model.optimizer.build(model.trainable_variables)
  return model


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

    actual = sparsecore_checkpoint_utils.reshard_stacked_array(
        jax.device_put(
            src_stacked,
            sparsecore_checkpoint_utils.row_sharding(
                sharding, src_stacked.shape[0]
            ),
        ),
        dst_table_stack,
        src_num_shards=src_num_shards,
        src_num_sc_per_device=src_num_sc_per_device,
        dst_num_shards=dst_num_shards,
        dst_num_sc_per_device=dst_num_sc_per_device,
        out_sharding=sharding,
    )

    self.assertTrue(actual.sharding.is_equivalent_to(sharding, actual.ndim))
    np.testing.assert_array_equal(np.asarray(actual), expected)

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

  @parameterized.parameters(2, 4)
  def test_sparsecore_topology(self, num_sc_per_device):
    model = _build_model(num_sc_per_device)
    self.assertTrue(sparsecore_checkpoint_utils.has_sparsecore_tables(model))
    (state,) = sparsecore_checkpoint_utils.sparsecore_layer_states(model)
    self.assertIs(state[1], model.sc_layer._sparsecore_layout)
    self.assertEqual(
        sparsecore_checkpoint_utils.sparsecore_topology(model),
        {
            model.sc_layer.stack_variables.table.path: {
                "num_shards": jax.device_count() * num_sc_per_device,
                "num_sc_per_device": num_sc_per_device,
            }
        },
    )

  def test_models_without_sparsecore_tables(self):
    model = _build_model(None)
    self.assertFalse(sparsecore_checkpoint_utils.has_sparsecore_tables(model))
    self.assertEqual(sparsecore_checkpoint_utils.sparsecore_topology(model), {})
    self.assertEqual(
        sparsecore_checkpoint_utils.sparsecore_topology_save_args(model), {}
    )

  def test_unbuilt_sparsecore_layers_are_skipped(self):
    model = keras.Sequential([_FakeSparseCoreLayer(2)])
    self.assertEmpty(sparsecore_checkpoint_utils.sparsecore_layer_states(model))

  @parameterized.named_parameters(
      ("v2", keras_utils.KerasOrbaxCheckpointManagerV2),
      ("v3", keras_utils.KerasOrbaxCheckpointManagerV3),
  )
  def test_sparsecore_topology_is_saved_with_the_checkpoint(self, manager_cls):
    for num_sc_per_device in (None, 4):
      with self.subTest(num_sc_per_device=num_sc_per_device):
        model = _build_model(num_sc_per_device)
        ckpt_dir = self.create_tempdir().full_path
        manager = manager_cls(ckpt_dir, choose_store_cell=False)
        manager.save_model_variables(model, epoch=5)
        manager.wait_until_finished()
        manager.close()
        self.assertEqual(
            sparsecore_checkpoint_utils.saved_sparsecore_topology(
                os.path.join(ckpt_dir, "5")
            ),
            sparsecore_checkpoint_utils.sparsecore_topology(model) or None,
        )

  def test_stack_restore_needs_reshard(self):
    def plan(
        src_num_sc_per_device: int,
    ) -> sparsecore_checkpoint_utils.StackRestore:
      return sparsecore_checkpoint_utils.StackRestore(
          variables=(),
          saved_shapes=(),
          table_stack=(),
          layout=None,
          src_num_shards=jax.device_count() * src_num_sc_per_device,
          src_num_sc_per_device=src_num_sc_per_device,
          dst_num_shards=jax.device_count() * 4,
          dst_num_sc_per_device=4,
      )

    self.assertTrue(plan(2).needs_reshard)
    self.assertFalse(plan(4).needs_reshard)

  def test_plan_sparsecore_restore_without_sparsecore_tables(self):
    self.assertEmpty(
        sparsecore_checkpoint_utils.plan_sparsecore_restore(
            _build_model(None), "/unused", {"unused": None}
        )
    )

  @parameterized.product(
      manager_cls=(
          keras_utils.KerasOrbaxCheckpointManagerV2,
          keras_utils.KerasOrbaxCheckpointManagerV3,
      ),
      restore_with_manager=(False, True),
  )
  def test_cross_topology_restore(self, manager_cls, restore_with_manager):
    rng = np.random.default_rng(0)
    tables = {
        name: rng.normal(size=vocab_and_dim).astype(np.float32)
        for name, vocab_and_dim in _VOCABS_AND_DIMS.items()
    }
    slots = {name: table * 2.0 for name, table in tables.items()}
    kernel = np.full((8, 2), 3.5, dtype=np.float32)
    src_model = _build_model(4)
    src_model.sc_layer.set_tables(tables, slots)
    src_model.dense.kernel.assign(kernel)
    ckpt_dir = self.create_tempdir().full_path
    manager = manager_cls(ckpt_dir, choose_store_cell=False)
    manager.save_model_variables(src_model, epoch=5)
    manager.wait_until_finished()

    # Restore on topologies with fewer and more SparseCores (resharded) and on
    # the same one (as is).
    for dst_num_sc_per_device in (2, 4, 8):
      with self.subTest(dst_num_sc_per_device=dst_num_sc_per_device):
        dst_model = _build_model(dst_num_sc_per_device)
        layer = dst_model.sc_layer
        with mock.patch.object(
            sparsecore_checkpoint_utils,
            "reshard_stacked_array",
            wraps=sparsecore_checkpoint_utils.reshard_stacked_array,
        ) as mock_reshard:
          if restore_with_manager:
            manager.restore_model_variables(dst_model, 5)
          else:
            keras_utils.restore_keras_checkpoint(
                ckpt_dir, model=dst_model, epoch=5
            )
        # The table and its slot.
        self.assertEqual(
            mock_reshard.call_count, 2 if dst_num_sc_per_device != 4 else 0
        )
        (slot,) = layer.stack_variables.slot
        for var, expected in (
            (layer.stack_variables.table, tables),
            (slot, slots),
        ):
          self.assertIs(var._layout, layer._sparsecore_layout)
          self.assertEqual(
              var.value.shape,
              (
                  layer.table_stack[0].stacked_table_spec.stack_vocab_size,
                  8,
              ),
          )
          restored = layer.get_tables(var)
          for name, table in expected.items():
            np.testing.assert_allclose(restored[name], table, err_msg=name)
        np.testing.assert_allclose(
            np.asarray(dst_model.dense.kernel.value), kernel
        )
    manager.close()

  def test_restore_without_saved_topology(self):
    tables = {
        name: np.full(vocab_and_dim, i + 1.0, dtype=np.float32)
        for i, (name, vocab_and_dim) in enumerate(_VOCABS_AND_DIMS.items())
    }
    src_model = _build_model(4)
    src_model.sc_layer.set_tables(tables, tables)
    ckpt_dir = self.create_tempdir().full_path
    manager = keras_utils.KerasOrbaxCheckpointManagerV3(
        ckpt_dir, choose_store_cell=False
    )
    manager.save_model_variables(src_model, epoch=5)
    manager.wait_until_finished()
    manager.close()
    # Like checkpoints saved before the SparseCore topology was persisted.
    shutil.rmtree(
        os.path.join(
            ckpt_dir, "5", sparsecore_checkpoint_utils.SPARSECORE_TOPOLOGY_KEY
        )
    )

    with self.subTest("same_topology"):
      # Restored as is.
      dst_model = _build_model(4)
      keras_utils.restore_keras_checkpoint(ckpt_dir, model=dst_model, epoch=5)
      restored = dst_model.sc_layer.get_tables(
          dst_model.sc_layer.stack_variables.table
      )
      for name, table in tables.items():
        np.testing.assert_allclose(restored[name], table, err_msg=name)
    with self.subTest("other_topology"):
      with self.assertRaisesRegex(
          ValueError, "without its SparseCore topology"
      ):
        keras_utils.restore_keras_checkpoint(
            ckpt_dir, model=_build_model(2), epoch=5
        )


if __name__ == "__main__":
  absltest.main()

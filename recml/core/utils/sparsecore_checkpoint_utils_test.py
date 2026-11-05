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
import orbax.checkpoint as ocp
from recml.core.utils import keras_utils
from recml.core.utils import sparsecore_checkpoint_utils

# Three tables in one stack (all dims pad to 8) with vocab sizes that pad
# differently for different numbers of SparseCores.
_VOCABS_AND_DIMS = types.MappingProxyType(
    {"t1": (19, 6), "t2": (35, 8), "t3": (10, 4)}
)


def _stacked_feature_specs(
    num_sc_per_device: int,
    rotation: int | None = None,
    vocabs_and_dims: Mapping[str, tuple[int, int]] = _VOCABS_AND_DIMS,
    global_device_count: int | None = None,
) -> list[embedding_spec.FeatureSpec]:
  """Returns features of tables stacked like `keras_rs` (for the devices)."""
  feature_specs = []
  for name, (vocab, dim) in vocabs_and_dims.items():
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
      list(vocabs_and_dims),
      global_device_count=global_device_count or jax.device_count(),
      num_sc_per_device=num_sc_per_device,
      rotation=rotation,
  )
  return feature_specs


def _table_stack(
    num_sc_per_device: int,
    rotation: int | None = None,
    global_device_count: int | None = None,
) -> Sequence[embedding_spec.TableSpec]:
  """Returns the tables stacked like `keras_rs` (for the current devices)."""
  (table_stack,) = jte_table_stacking.get_table_stacks(
      jte_embedding.get_table_specs(
          _stacked_feature_specs(
              num_sc_per_device,
              rotation,
              global_device_count=global_device_count,
          )
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

  def __init__(
      self,
      num_sc_per_device: int,
      vocabs_and_dims: Mapping[str, tuple[int, int]] = _VOCABS_AND_DIMS,
      has_slots: bool = True,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self._num_sc_per_device = num_sc_per_device
    self._has_slots = has_slots
    self._sparsecore_built = False
    feature_specs = _stacked_feature_specs(
        num_sc_per_device, vocabs_and_dims=vocabs_and_dims
    )
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
    # pylint: disable-next=protected-access
    layout._backend_layout = jax_layout.Format(None, sharding)
    # Like `keras_rs`, the table and its slot are trainable but trained on
    # SparseCore, so the Keras optimizer has no slots for them. When
    # `add_slot_variables=False`, `keras_rs` sets `slot=None`.
    table = self.add_weight(shape=shape, initializer="zeros", name="table")
    slot = (
        self.add_weight(shape=shape, initializer="zeros", name="slot")
        if self._has_slots
        else None
    )
    for var in (table, slot):
      if var is None:
        continue
      var.overwrite_with_gradient = True
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
            table=table, slot=(slot,) if slot is not None else None
        )
    }
    self._sparsecore_built = True
    self.built = True

  @property
  def stack_variables(self) -> types.SimpleNamespace:
    (emb_vars,) = self._table_and_slot_variables.values()
    return emb_vars

  def set_tables(
      self,
      tables: Mapping[str, np.ndarray],
      slots: Mapping[str, np.ndarray] | None = None,
  ) -> None:
    self.stack_variables.table.assign(
        _stack(self.table_stack, tables, self.num_shards)
    )
    if slots is not None:
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

  def __init__(
      self,
      num_sc_per_device: int | None,
      vocabs_and_dims: Mapping[str, tuple[int, int]] = _VOCABS_AND_DIMS,
      sc_layer_name: str = "sc_emb",
      has_slots: bool = True,
  ):
    super().__init__(name="test_model")
    self._num_sc_per_device = num_sc_per_device
    self.sc_layer = None
    if num_sc_per_device:
      self.sc_layer = _FakeSparseCoreLayer(
          num_sc_per_device,
          vocabs_and_dims,
          has_slots=has_slots,
          name=sc_layer_name,
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


def _build_model(
    num_sc_per_device: int | None,
    vocabs_and_dims: Mapping[str, tuple[int, int]] = _VOCABS_AND_DIMS,
    sc_layer_name: str = "sc_emb",
    has_slots: bool = True,
) -> _TestModel:
  model = _TestModel(
      num_sc_per_device, vocabs_and_dims, sc_layer_name, has_slots=has_slots
  )
  model.build()
  model.compile(optimizer=keras.optimizers.Adam(1e-3))
  model.optimizer.build(model.trainable_variables)
  return model


def _random_tables(
    vocabs_and_dims: Mapping[str, tuple[int, int]] = _VOCABS_AND_DIMS,
) -> dict[str, np.ndarray]:
  rng = np.random.default_rng(0)
  return {
      name: rng.normal(size=vocab_and_dim).astype(np.float32)
      for name, vocab_and_dim in vocabs_and_dims.items()
  }


def _save_checkpoint(
    manager_cls,
    ckpt_dir: str,
    model: _TestModel,
) -> None:
  manager = manager_cls(ckpt_dir, choose_store_cell=False)
  manager.save_model_variables(model, epoch=5)
  manager.wait_until_finished()
  manager.close()


def _stack_restore(
    src_num_shards: int, src_num_sc_per_device: int
) -> sparsecore_checkpoint_utils.StackRestore:
  """Returns a plan from the given topology to 4 SparseCores per chip."""
  return sparsecore_checkpoint_utils.StackRestore(
      variables=(),
      saved_shapes=(),
      table_stack=(),
      layout=None,
      src_num_shards=src_num_shards,
      src_num_sc_per_device=src_num_sc_per_device,
      dst_num_shards=jax.device_count() * 4,
      dst_num_sc_per_device=4,
  )


class SparseCoreCheckpointUtilsTest(parameterized.TestCase):

  def test_checkpoint_item_keys_match_keras_utils(self):
    self.assertEqual(
        sparsecore_checkpoint_utils._STATE_CHECKPOINT_KEY,
        keras_utils.STATE_CHECKPOINT_KEY,
    )
    self.assertEqual(
        sparsecore_checkpoint_utils._TRAINABLE_VARIABLES_KEY,
        keras_utils.TRAINABLE_VARIABLES_KEY,
    )
    self.assertEqual(
        sparsecore_checkpoint_utils._SHARED_VARIABLE_MAP_KEY,
        keras_utils.SHARED_VARIABLE_MAP_KEY,
    )
    self.assertEqual(
        sparsecore_checkpoint_utils._CHECKPOINT_VERSION_V2,
        keras_utils.CheckpointVersion.V2,
    )
    self.assertEqual(
        sparsecore_checkpoint_utils._CHECKPOINT_VERSION_V3,
        keras_utils.CheckpointVersion.V3,
    )

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

  @parameterized.parameters(2, 4)
  def test_sparsecore_topology(self, num_sc_per_device):
    model = _build_model(num_sc_per_device)
    self.assertTrue(sparsecore_checkpoint_utils.has_sparsecore_tables(model))
    (state,) = sparsecore_checkpoint_utils.sparsecore_layer_states(model)
    self.assertIs(state.layout, model.sc_layer._sparsecore_layout)
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

  def test_unbuilt_layers_raise(self):
    unbuilt_model = _TestModel(2)
    with self.assertRaisesRegex(ValueError, "must be built"):
      sparsecore_checkpoint_utils.sparsecore_layer_states(unbuilt_model)

    model = _build_model(2)
    model.sc_layer._sparsecore_built = False
    with self.assertRaisesRegex(ValueError, "must be built"):
      sparsecore_checkpoint_utils.sparsecore_layer_states(model)

  @parameterized.parameters(
      "_config", "_sparsecore_layout", "_table_and_slot_variables"
  )
  def test_incomplete_sparsecore_layer_state_raises(self, attr):
    model = _build_model(2)
    setattr(model.sc_layer, attr, None)
    with self.assertRaisesRegex(ValueError, f"is missing .*'{attr}'"):
      sparsecore_checkpoint_utils.sparsecore_layer_states(model)

  def test_duplicate_sparsecore_table_paths_raise(self):
    model = _build_model(2)
    emb_vars = model.sc_layer.stack_variables
    # Like two SparseCore layers with the same names.
    model.sc_layer._table_and_slot_variables = {
        "stack_a": emb_vars,
        "stack_b": emb_vars,
    }
    with self.assertRaisesRegex(ValueError, "Duplicate SparseCore table"):
      sparsecore_checkpoint_utils.sparsecore_topology(model)

  @parameterized.named_parameters(
      ("v2", keras_utils.KerasOrbaxCheckpointManagerV2),
      ("v3", keras_utils.KerasOrbaxCheckpointManagerV3),
  )
  def test_sparsecore_topology_is_saved_with_the_checkpoint(self, manager_cls):
    model = _build_model(4)
    ckpt_dir = self.create_tempdir().full_path
    _save_checkpoint(manager_cls, ckpt_dir, model)
    self.assertEqual(
        sparsecore_checkpoint_utils.saved_sparsecore_topology(
            os.path.join(ckpt_dir, "5")
        ),
        {
            model.sc_layer.stack_variables.table.path: {
                "num_shards": jax.device_count() * 4,
                "num_sc_per_device": 4,
            }
        },
    )

  @parameterized.named_parameters(
      ("v2", keras_utils.KerasOrbaxCheckpointManagerV2),
      ("v3", keras_utils.KerasOrbaxCheckpointManagerV3),
  )
  def test_sparsecore_topology_is_not_saved_without_sparsecore_tables(
      self, manager_cls
  ):
    ckpt_dir = self.create_tempdir().full_path
    _save_checkpoint(manager_cls, ckpt_dir, _build_model(None))
    self.assertIsNone(
        sparsecore_checkpoint_utils.saved_sparsecore_topology(
            os.path.join(ckpt_dir, "5")
        )
    )

  @parameterized.named_parameters(
      ("fewer_sparsecores_per_chip", 1, 2, True),
      ("more_chips", 2, 4, True),
      ("same_topology", 1, 4, False),
  )
  def test_stack_restore_needs_reshard(
      self, chips_factor, src_num_sc_per_device, expected
  ):
    num_chips = jax.device_count() * chips_factor
    plan = _stack_restore(
        src_num_shards=num_chips * src_num_sc_per_device,
        src_num_sc_per_device=src_num_sc_per_device,
    )
    self.assertEqual(plan.needs_reshard, expected)

  def test_plan_sparsecore_restore_without_sparsecore_tables(self):
    self.assertEmpty(
        sparsecore_checkpoint_utils.plan_sparsecore_restore(
            _build_model(None), {"unused": None}, None
        )
    )

  def test_plan_sparsecore_restore_skips_matching_topology_early(self):
    model = _build_model(4)
    table = model.sc_layer.stack_variables.table
    (slot,) = model.sc_layer.stack_variables.slot
    trainable_metadata = {
        table.path: types.SimpleNamespace(shape=table.shape),
        slot.path: types.SimpleNamespace(shape=slot.shape),
    }
    saved_topology = sparsecore_checkpoint_utils.sparsecore_topology(model)
    with (
        mock.patch.object(
            jte_table_stacking, "get_table_stacks", autospec=True
        ) as mock_get_stacks,
        mock.patch.object(
            sparsecore_checkpoint_utils, "_stack_shape", autospec=True
        ) as mock_stack_shape,
    ):
      self.assertEmpty(
          sparsecore_checkpoint_utils.plan_sparsecore_restore(
              model, trainable_metadata, saved_topology
          )
      )
      mock_get_stacks.assert_not_called()
      mock_stack_shape.assert_not_called()

  def test_plan_sparsecore_restore_cross_topology_shape_mismatch_raises(self):
    src_model = _build_model(4)
    dst_model = _build_model(2, {**_VOCABS_AND_DIMS, "t1": (300, 6)})
    dst_table = dst_model.sc_layer.stack_variables.table
    (dst_slot,) = dst_model.sc_layer.stack_variables.slot
    src_shape = src_model.sc_layer.stack_variables.table.shape
    trainable_metadata = {
        dst_table.path: types.SimpleNamespace(shape=src_shape),
        dst_slot.path: types.SimpleNamespace(shape=src_shape),
    }
    saved_topology = {
        dst_table.path: {
            "num_shards": jax.device_count() * 4,
            "num_sc_per_device": 4,
        }
    }
    with self.assertRaisesRegex(ValueError, "tables of the stack changed"):
      sparsecore_checkpoint_utils.plan_sparsecore_restore(
          dst_model, trainable_metadata, saved_topology
      )

  def test_prepare_sparsecore_restore_skips_metadata_on_matching_topology(self):
    ckpt_dir = self.create_tempdir().full_path
    _save_checkpoint(
        keras_utils.KerasOrbaxCheckpointManagerV3, ckpt_dir, _build_model(4)
    )
    step_dir = os.path.join(ckpt_dir, "5")
    with mock.patch.object(
        sparsecore_checkpoint_utils, "saved_trainable_metadata", autospec=True
    ) as mock_saved_metadata:
      self.assertEmpty(
          sparsecore_checkpoint_utils.prepare_sparsecore_restore(
              _build_model(4),
              step_dir,
              version=keras_utils.CheckpointVersion.V3,
          )
      )
      mock_saved_metadata.assert_not_called()

  @parameterized.named_parameters(
      (
          "v2",
          keras_utils.KerasOrbaxCheckpointManagerV2,
          keras_utils.CheckpointVersion.V2,
      ),
      (
          "v3",
          keras_utils.KerasOrbaxCheckpointManagerV3,
          keras_utils.CheckpointVersion.V3,
      ),
  )
  def test_prepare_sparsecore_restore_passes_version(
      self, manager_cls, version
  ):
    ckpt_dir = self.create_tempdir().full_path
    _save_checkpoint(manager_cls, ckpt_dir, _build_model(4))
    step_dir = os.path.join(ckpt_dir, "5")
    dst_model = _build_model(2)
    with mock.patch.object(
        sparsecore_checkpoint_utils,
        "saved_trainable_metadata",
        autospec=True,
        return_value=({}, {}),
    ) as mock_saved_metadata:
      self.assertEmpty(
          sparsecore_checkpoint_utils.prepare_sparsecore_restore(
              dst_model, step_dir, version=version
          )
      )
    mock_saved_metadata.assert_called_once_with(
        dst_model, step_dir, version=version
    )

  @parameterized.named_parameters(
      (
          "v2",
          keras_utils.KerasOrbaxCheckpointManagerV2,
          keras_utils.CheckpointVersion.V2,
      ),
      (
          "v3",
          keras_utils.KerasOrbaxCheckpointManagerV3,
          keras_utils.CheckpointVersion.V3,
      ),
  )
  def test_saved_trainable_metadata(self, manager_cls, version):
    model = _build_model(4)
    ckpt_dir = self.create_tempdir().full_path
    _save_checkpoint(manager_cls, ckpt_dir, model)
    trainable_metadata, shared_var_map = (
        sparsecore_checkpoint_utils.saved_trainable_metadata(
            _build_model(2), os.path.join(ckpt_dir, "5"), version=version
        )
    )
    self.assertEqual(
        {path: tuple(meta.shape) for path, meta in trainable_metadata.items()},
        {var.path: tuple(var.shape) for var in model.trainable_variables},
    )
    self.assertEqual(shared_var_map, {})

  @parameterized.named_parameters(
      (
          "v2_as_v3",
          keras_utils.KerasOrbaxCheckpointManagerV2,
          keras_utils.CheckpointVersion.V3,
          "Expected the V3 checkpoint",
      ),
      (
          "v3_as_v2",
          keras_utils.KerasOrbaxCheckpointManagerV3,
          keras_utils.CheckpointVersion.V2,
          "Expected the V2 checkpoint",
      ),
      (
          "v1",
          keras_utils.KerasOrbaxCheckpointManagerV3,
          keras_utils.CheckpointVersion.V1,
          "Unsupported checkpoint version 'v1'",
      ),
  )
  def test_saved_trainable_metadata_with_wrong_version_raises(
      self, manager_cls, version, expected_error_regex
  ):
    ckpt_dir = self.create_tempdir().full_path
    _save_checkpoint(manager_cls, ckpt_dir, _build_model(4))
    with self.assertRaisesRegex(ValueError, expected_error_regex):
      sparsecore_checkpoint_utils.saved_trainable_metadata(
          _build_model(2), os.path.join(ckpt_dir, "5"), version=version
      )

  def test_saved_trainable_metadata_v2_variable_count_mismatch_raises(self):
    ckpt_dir = self.create_tempdir().full_path
    _save_checkpoint(
        keras_utils.KerasOrbaxCheckpointManagerV2, ckpt_dir, _build_model(4)
    )
    with self.assertRaisesRegex(ValueError, "restored by variable order"):
      sparsecore_checkpoint_utils.saved_trainable_metadata(
          _build_model(None),
          os.path.join(ckpt_dir, "5"),
          version=keras_utils.CheckpointVersion.V2,
      )

  def test_saved_trainable_metadata_without_state_raises(self):
    ckpt_dir = self.create_tempdir().full_path
    _save_checkpoint(
        keras_utils.KerasOrbaxCheckpointManagerV3, ckpt_dir, _build_model(4)
    )
    step_dir = os.path.join(ckpt_dir, "5")
    shutil.rmtree(
        os.path.join(
            step_dir, sparsecore_checkpoint_utils._STATE_CHECKPOINT_KEY
        )
    )
    with self.assertRaisesRegex(ValueError, "has no saved trainable variables"):
      sparsecore_checkpoint_utils.saved_trainable_metadata(
          _build_model(2), step_dir, version=keras_utils.CheckpointVersion.V3
      )

  @parameterized.product(
      manager_cls=(
          keras_utils.KerasOrbaxCheckpointManagerV2,
          keras_utils.KerasOrbaxCheckpointManagerV3,
      ),
      restore_with_manager=(False, True),
      # Fewer and more SparseCores (resharded) and the same ones (as is).
      dst_num_sc_per_device=(2, 4, 8),
  )
  def test_cross_topology_restore(
      self, manager_cls, restore_with_manager, dst_num_sc_per_device
  ):
    tables = _random_tables()
    slots = {name: table * 2.0 for name, table in tables.items()}
    kernel = np.full((8, 2), 3.5, dtype=np.float32)
    src_model = _build_model(4)
    src_model.sc_layer.set_tables(tables, slots)
    src_model.dense.kernel.assign(kernel)
    ckpt_dir = self.create_tempdir().full_path
    _save_checkpoint(manager_cls, ckpt_dir, src_model)
    dst_model = _build_model(dst_num_sc_per_device)

    if restore_with_manager:
      manager = manager_cls(ckpt_dir, choose_store_cell=False)
      manager.restore_model_variables(dst_model, 5)
      manager.close()
    else:
      keras_utils.restore_keras_checkpoint(ckpt_dir, model=dst_model, epoch=5)

    layer = dst_model.sc_layer
    (slot,) = layer.stack_variables.slot
    stack_shape = (layer.table_stack[0].stacked_table_spec.stack_vocab_size, 8)
    for var, expected in ((layer.stack_variables.table, tables), (slot, slots)):
      self.assertIs(var._layout, layer._sparsecore_layout)
      self.assertEqual(var.value.shape, stack_shape)
      restored = layer.get_tables(var)
      for name, table in expected.items():
        np.testing.assert_allclose(restored[name], table, err_msg=name)
    np.testing.assert_allclose(np.asarray(dst_model.dense.kernel.value), kernel)

  def test_cross_topology_restore_with_same_stack_shape(self):
    # Vocab sizes padded to the same stack size for 2 and 4 SparseCores per
    # chip, so only the row layout differs between the topologies.
    vocabs_and_dims = {
        "t1": (64 * jax.device_count(), 8),
        "t2": (128 * jax.device_count(), 8),
    }
    tables = _random_tables(vocabs_and_dims)
    src_model = _build_model(4, vocabs_and_dims)
    src_model.sc_layer.set_tables(tables, tables)
    ckpt_dir = self.create_tempdir().full_path
    _save_checkpoint(
        keras_utils.KerasOrbaxCheckpointManagerV3, ckpt_dir, src_model
    )
    dst_model = _build_model(2, vocabs_and_dims)
    self.assertEqual(
        dst_model.sc_layer.stack_variables.table.shape,
        src_model.sc_layer.stack_variables.table.shape,
    )

    keras_utils.restore_keras_checkpoint(ckpt_dir, model=dst_model, epoch=5)

    restored = dst_model.sc_layer.get_tables(
        dst_model.sc_layer.stack_variables.table
    )
    for name, table in tables.items():
      np.testing.assert_allclose(restored[name], table, err_msg=name)

  @parameterized.named_parameters(
      ("same_topology", 4, "not compatible with the stored"),
      ("other_topology", 2, "tables of the stack changed"),
  )
  def test_restore_with_changed_tables_raises(
      self, dst_num_sc_per_device, expected_error_regex
  ):
    ckpt_dir = self.create_tempdir().full_path
    _save_checkpoint(
        keras_utils.KerasOrbaxCheckpointManagerV3, ckpt_dir, _build_model(4)
    )
    changed = {**_VOCABS_AND_DIMS, "t1": (300, 6)}
    with self.assertRaisesRegex(ValueError, expected_error_regex):
      keras_utils.restore_keras_checkpoint(
          ckpt_dir, model=_build_model(dst_num_sc_per_device, changed), epoch=5
      )

  def test_restore_without_topology_of_a_stack_raises(self):
    ckpt_dir = self.create_tempdir().full_path
    _save_checkpoint(
        keras_utils.KerasOrbaxCheckpointManagerV2, ckpt_dir, _build_model(4)
    )
    # V2 checkpoints are matched by variable order, so the renamed table is
    # restored, but it has no saved topology.
    dst_model = _build_model(2, sc_layer_name="renamed_sc_emb")
    with self.assertRaisesRegex(ValueError, "has no saved SparseCore topology"):
      keras_utils.restore_keras_checkpoint(ckpt_dir, model=dst_model, epoch=5)

  @parameterized.named_parameters(
      ("restore_keras_checkpoint", False),
      ("manager", True),
  )
  def test_failed_restore_leaves_the_model_intact(self, restore_with_manager):
    ckpt_dir = self.create_tempdir().full_path
    _save_checkpoint(
        keras_utils.KerasOrbaxCheckpointManagerV3, ckpt_dir, _build_model(4)
    )
    tables = _random_tables()
    dst_model = _build_model(2)
    dst_model.sc_layer.set_tables(tables, tables)
    table = dst_model.sc_layer.stack_variables.table
    with (
        mock.patch.object(
            keras_utils,
            "_prepare_v3_restore",
            autospec=True,
            side_effect=ValueError("Invalid checkpoint"),
        ),
        self.assertRaisesRegex(ValueError, "Invalid checkpoint"),
    ):
      if restore_with_manager:
        manager = keras_utils.KerasOrbaxCheckpointManagerV3(
            ckpt_dir, choose_store_cell=False
        )
        try:
          manager.restore_model_variables(dst_model, 5)
        finally:
          manager.close()
      else:
        keras_utils.restore_keras_checkpoint(ckpt_dir, model=dst_model, epoch=5)

    self.assertIs(table._layout, dst_model.sc_layer._sparsecore_layout)
    self.assertFalse(dst_model.dense.kernel.value.is_deleted())
    restored = dst_model.sc_layer.get_tables(table)
    for name, expected in tables.items():
      np.testing.assert_allclose(restored[name], expected, err_msg=name)

  def test_unreadable_checkpoint_metadata_raises(self):
    ckpt_dir = self.create_tempdir().full_path
    _save_checkpoint(
        keras_utils.KerasOrbaxCheckpointManagerV2, ckpt_dir, _build_model(4)
    )
    with (
        mock.patch.object(
            ocp.handlers.PyTreeCheckpointHandler,
            "metadata",
            autospec=True,
            side_effect=OSError("Unreadable"),
        ),
        self.assertRaisesRegex(ValueError, "Failed to read the metadata"),
    ):
      keras_utils.restore_keras_checkpoint(
          ckpt_dir, model=_build_model(2), epoch=5
      )

  def test_restore_without_saved_topology(self):
    tables = {
        name: np.full(vocab_and_dim, i + 1.0, dtype=np.float32)
        for i, (name, vocab_and_dim) in enumerate(_VOCABS_AND_DIMS.items())
    }
    src_model = _build_model(4)
    src_model.sc_layer.set_tables(tables, tables)
    ckpt_dir = self.create_tempdir().full_path
    _save_checkpoint(
        keras_utils.KerasOrbaxCheckpointManagerV3, ckpt_dir, src_model
    )
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

  def test_restore_checkpoint_without_sparsecore_tables(self):
    # Like the checkpoints of models without SparseCore tables: no stacked
    # SparseCore variables and no `sparsecore_topology` item.
    src_model = _build_model(None)
    kernel = np.full((8, 2), 3.5, dtype=np.float32)
    src_model.dense.kernel.assign(kernel)
    ckpt_dir = self.create_tempdir().full_path
    _save_checkpoint(
        keras_utils.KerasOrbaxCheckpointManagerV3, ckpt_dir, src_model
    )

    with self.subTest("full_restore_into_sparsecore_model"):
      # The SparseCore restore is not triggered (no plan, no SparseCore
      # warning); the regular V3 restore raises for the missing tables.
      with (
          mock.patch.object(
              sparsecore_checkpoint_utils, "_plan_stack_restore", autospec=True
          ) as mock_plan,
          mock.patch.object(
              sparsecore_checkpoint_utils.logging, "warning", autospec=True
          ) as mock_warning,
          self.assertRaisesRegex(ValueError, "Missing paths in checkpoint"),
      ):
        keras_utils.restore_keras_checkpoint(
            ckpt_dir, model=_build_model(4), epoch=5
        )
      mock_plan.assert_not_called()
      self.assertEmpty([
          call
          for call in mock_warning.call_args_list
          if "SparseCore" in str(call.args[0])
      ])
    with self.subTest("partial_restore_of_dense_weights"):
      dst_model = _build_model(4)
      dst_kernel = dst_model.dense.kernel
      partial_vars = {
          keras_utils.TRAINABLE_VARIABLES_KEY: {dst_kernel.path: dst_kernel}
      }
      restored_state = keras_utils.restore_partial_checkpoint(
          ckpt_dir, partial_vars, epoch=5
      )
      np.testing.assert_allclose(
          np.asarray(
              restored_state[keras_utils.TRAINABLE_VARIABLES_KEY][
                  dst_kernel.path
              ]
          ),
          kernel,
      )

  @parameterized.named_parameters(
      ("cross_topology", 2),
      ("same_topology", 4),
  )
  def test_restore_with_shared_sparsecore_layer_alias(
      self, dst_num_sc_per_device
  ):
    class _Tower(keras.layers.Layer):

      def __init__(self, sc_layer, **kwargs):
        super().__init__(**kwargs)
        self.sc_layer = sc_layer

      def build(self, input_shape=None):
        if not self.sc_layer.built:
          self.sc_layer.build()
        self.built = True

    class _SharedModel(keras.Model):

      def __init__(self, num_sc_per_device, share_across_towers):
        super().__init__(name="shared_model")
        self._num_sc_per_device = num_sc_per_device
        sc_layer = _FakeSparseCoreLayer(num_sc_per_device, name="sc_emb")
        self.tower_a = (
            _Tower(sc_layer, name="tower_a") if share_across_towers else None
        )
        self.tower_b = _Tower(sc_layer, name="tower_b")
        self.dense = keras.layers.Dense(2, name="head")

      def build(self, input_shape=None):
        if self.tower_a is not None:
          self.tower_a.build()
        self.tower_b.build()
        self.dense.build((None, 8))
        self.built = True

      def get_config(self):
        return {"num_sc_per_device": self._num_sc_per_device}

    tables = _random_tables()
    slots = {name: table * 2.0 for name, table in tables.items()}
    src_model = _SharedModel(4, share_across_towers=True)
    src_model.build()
    src_model.compile(optimizer=keras.optimizers.Adam(1e-3))
    src_model.optimizer.build(src_model.trainable_variables)
    src_model.tower_a.sc_layer.set_tables(tables, slots)
    ckpt_dir = self.create_tempdir().full_path
    _save_checkpoint(
        keras_utils.KerasOrbaxCheckpointManagerV3, ckpt_dir, src_model
    )

    # `src_model` saved `tower_a/sc_emb/table` as the canonical path in
    # `sparsecore_topology` and `tower_b/sc_emb/table` in `shared_variable_map`.
    dst_model = _SharedModel(dst_num_sc_per_device, share_across_towers=False)
    dst_model.build()
    dst_model.compile(optimizer=keras.optimizers.Adam(1e-3))
    dst_model.optimizer.build(dst_model.trainable_variables)

    keras_utils.restore_keras_checkpoint(ckpt_dir, model=dst_model, epoch=5)

    layer = dst_model.tower_b.sc_layer
    (slot,) = layer.stack_variables.slot
    for var, expected in ((layer.stack_variables.table, tables), (slot, slots)):
      self.assertIs(var._layout, layer._sparsecore_layout)
      restored = layer.get_tables(var)
      for name, table in expected.items():
        np.testing.assert_allclose(restored[name], table, err_msg=name)

  def test_cross_topology_restore_without_slot_variables(self):
    tables = _random_tables()
    src_model = _build_model(4, has_slots=False)
    self.assertIsNone(src_model.sc_layer.stack_variables.slot)
    src_model.sc_layer.set_tables(tables)
    ckpt_dir = self.create_tempdir().full_path
    _save_checkpoint(
        keras_utils.KerasOrbaxCheckpointManagerV3, ckpt_dir, src_model
    )
    dst_model = _build_model(2, has_slots=False)
    self.assertIsNone(dst_model.sc_layer.stack_variables.slot)

    keras_utils.restore_keras_checkpoint(ckpt_dir, model=dst_model, epoch=5)

    table = dst_model.sc_layer.stack_variables.table
    self.assertIs(table._layout, dst_model.sc_layer._sparsecore_layout)
    restored = dst_model.sc_layer.get_tables(table)
    for name, expected in tables.items():
      np.testing.assert_allclose(restored[name], expected, err_msg=name)


if __name__ == "__main__":
  absltest.main()

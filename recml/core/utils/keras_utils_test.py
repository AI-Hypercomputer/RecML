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
"""Tests or utilities."""

from collections.abc import Sequence
import getpass
import json
import os
from typing import Any

from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import keras
import keras_hub
import numpy as np
from recml.core.utils import keras_utils


def _create_dummy_inputs() -> dict[str, jax.Array]:
  k1, k2, k3, k4 = jax.random.split(jax.random.key(42), 4)
  return {
      "token_ids": jax.random.randint(k1, (64, 128), minval=0, maxval=2048),
      "segment_ids": jax.random.randint(k2, (64, 128), minval=0, maxval=8),
      "padding_mask": jax.random.uniform(k3, (64, 128)),
      "mask_positions": jax.random.randint(k4, (64, 20), minval=0, maxval=32),
  }


def _create_model(input_shapes: Sequence[int]) -> keras.Model:
  model = keras_hub.models.BertMaskedLM(
      backbone=keras_hub.models.BertBackbone(
          vocabulary_size=2048,
          num_layers=4,
          num_heads=8,
          hidden_dim=32,
          intermediate_dim=64,
          max_sequence_length=128,
          num_segments=8,
          dropout=0.1,
      )
  )
  optimizer = keras.optimizers.Adam(
      learning_rate=keras.optimizers.schedules.PolynomialDecay(
          initial_learning_rate=0.1,
          decay_steps=100,
          end_learning_rate=0.01,
          power=1.0,
      )
  )
  loss = keras.losses.SparseCategoricalCrossentropy()
  metrics = [keras.metrics.SparseCategoricalAccuracy()]
  model.compile(optimizer, loss, weighted_metrics=metrics)
  model.build(input_shapes)
  optimizer.build(model.trainable_variables)
  return model


@keras.saving.register_keras_serializable(package="Recml")
class MyNonTrainableLayer(keras.layers.Layer):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.non_trainable_weight = self.add_weight(
        shape=(10,), initializer="ones", trainable=False, name="weight"
    )

  def call(self, x):
    return x


@keras.saving.register_keras_serializable(package="Recml")
class MyTestModel(keras.Model):

  def __init__(self, layer_name, **kwargs):
    super().__init__(**kwargs)
    self.layer_name = layer_name
    self.my_layer = MyNonTrainableLayer(name=layer_name)
    self.dense = keras.layers.Dense(5, name="my_trainable_dense")
    self.direct_non_trainable = self.add_weight(
        shape=(5,),
        initializer="ones",
        trainable=False,
        name="direct_non_trainable",
    )

  def call(self, x):
    x = self.my_layer(x)
    x = self.dense(x)
    return x

  def get_config(self):
    config = super().get_config()
    config.update({"layer_name": self.layer_name})
    return config


class KerasUtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Workaround to make `create_tempdir` work with pytest.
    if not flags.FLAGS.is_parsed():
      flags.FLAGS.mark_as_parsed()

  @parameterized.named_parameters(
      {
          "testcase_name": "single_core",
          "data_parallel": False,
          "restore_with_checkpointer": True,
      },
      {
          "testcase_name": "data_parallel",
          "data_parallel": True,
          "restore_with_checkpointer": True,
      },
      {
          "testcase_name": "restore_without_checkpointer_single_core",
          "data_parallel": False,
          "restore_with_checkpointer": False,
      },
      {
          "testcase_name": "restore_without_checkpointer_data_parallel",
          "data_parallel": True,
          "restore_with_checkpointer": False,
      },
  )
  def test_keras_orbax_checkpointer_v2(
      self, data_parallel: bool, restore_with_checkpointer: bool
  ):
    if data_parallel:
      keras.distribution.set_distribution(keras.distribution.DataParallel())
    else:
      keras.distribution.set_distribution(None)

    checkpoint_dir = self.create_tempdir().full_path
    checkpoint_manager = keras_utils.KerasOrbaxCheckpointManagerV2(
        checkpoint_dir, max_to_keep=5
    )
    dummy_inputs = _create_dummy_inputs()

    bert_pretrainer = _create_model(jax.tree.map(jnp.shape, dummy_inputs))
    state = (
        [v.value for v in bert_pretrainer.trainable_variables],
        [v.value for v in bert_pretrainer.non_trainable_variables],
        [v.value for v in bert_pretrainer.optimizer.variables],
    )
    checkpoint_manager.save_model_variables(bert_pretrainer, 0)
    checkpoint_manager.wait_until_finished()

    preds = bert_pretrainer(dummy_inputs)

    bert_pretrainer = _create_model(jax.tree.map(jnp.shape, dummy_inputs))
    if restore_with_checkpointer:
      checkpoint_manager.restore_model_variables(bert_pretrainer, 0)
    else:
      keras_utils.restore_keras_checkpoint(
          checkpoint_dir, model=bert_pretrainer, restore_optimizer_vars=True
      )

    checkpoint_manager.close()

    restored_state = (
        [v.value for v in bert_pretrainer.trainable_variables],
        [v.value for v in bert_pretrainer.non_trainable_variables],
        [v.value for v in bert_pretrainer.optimizer.variables],
    )
    preds_after_restoration = bert_pretrainer(dummy_inputs)

    keras.tree.assert_same_structure(state, restored_state)
    for expected, observed in zip(
        jax.tree.flatten(state)[0], jax.tree.flatten(restored_state)[0]
    ):
      # Ensures the objects are different but the values are the same.
      self.assertNotEqual(id(expected), id(observed))
      self.assertEqual(expected.shape, observed.shape)
      self.assertEqual(expected.dtype, observed.dtype)
      self.assertEqual(expected.sharding, observed.sharding)
      np.testing.assert_allclose(observed, expected)

    # Ensures predictions are identical.
    np.testing.assert_allclose(preds, preds_after_restoration)

  @parameterized.named_parameters(
      {
          "testcase_name": "single_core",
          "data_parallel": False,
          "restore_with_checkpointer": True,
      },
      {
          "testcase_name": "data_parallel",
          "data_parallel": True,
          "restore_with_checkpointer": True,
      },
      {
          "testcase_name": "restore_without_checkpointer_single_core",
          "data_parallel": False,
          "restore_with_checkpointer": False,
      },
      {
          "testcase_name": "restore_without_checkpointer_data_parallel",
          "data_parallel": True,
          "restore_with_checkpointer": False,
      },
  )
  def test_keras_orbax_checkpointer_v3(
      self, data_parallel: bool, restore_with_checkpointer: bool
  ):
    if data_parallel:
      keras.distribution.set_distribution(keras.distribution.DataParallel())
    else:
      keras.distribution.set_distribution(None)

    checkpoint_dir = self.create_tempdir().full_path
    checkpoint_manager = keras_utils.KerasOrbaxCheckpointManagerV3(
        checkpoint_dir, max_to_keep=5
    )
    dummy_inputs = _create_dummy_inputs()

    bert_pretrainer = _create_model(jax.tree.map(jnp.shape, dummy_inputs))

    state = (
        [v.value for v in bert_pretrainer.trainable_variables],
        [v.value for v in bert_pretrainer.non_trainable_variables],
        [v.value for v in bert_pretrainer.optimizer.variables],
    )
    checkpoint_manager.save_model_variables(bert_pretrainer, 0)
    checkpoint_manager.wait_until_finished()

    preds = bert_pretrainer(dummy_inputs)

    bert_pretrainer = _create_model(jax.tree.map(jnp.shape, dummy_inputs))
    if restore_with_checkpointer:
      checkpoint_manager.restore_model_variables(bert_pretrainer, 0)
    else:
      keras_utils.restore_keras_checkpoint(
          checkpoint_dir, model=bert_pretrainer, restore_optimizer_vars=True
      )

    checkpoint_manager.close()

    restored_state = (
        [v.value for v in bert_pretrainer.trainable_variables],
        [v.value for v in bert_pretrainer.non_trainable_variables],
        [v.value for v in bert_pretrainer.optimizer.variables],
    )
    preds_after_restoration = bert_pretrainer(dummy_inputs)

    keras.tree.assert_same_structure(state, restored_state)
    for expected, observed in zip(
        jax.tree.flatten(state)[0], jax.tree.flatten(restored_state)[0]
    ):
      # Ensures the objects are different but the values are the same.
      self.assertNotEqual(id(expected), id(observed))
      self.assertEqual(expected.shape, observed.shape)
      self.assertEqual(expected.dtype, observed.dtype)
      self.assertEqual(expected.sharding, observed.sharding)
      np.testing.assert_allclose(observed, expected)

    # Ensures predictions are identical.
    np.testing.assert_allclose(preds, preds_after_restoration)

  def test_restore_keras_checkpoint(self):
    dummy_inputs = _create_dummy_inputs()
    bert_pretrainer = _create_model(jax.tree.map(jnp.shape, dummy_inputs))
    preds = bert_pretrainer(dummy_inputs)

    checkpoint_dir = self.create_tempdir().full_path
    checkpoint_manager = keras_utils.KerasOrbaxCheckpointManagerV2(
        checkpoint_dir
    )
    checkpoint_manager.save_model_variables(bert_pretrainer, epoch=1)
    checkpoint_manager.close()

    restored_model = keras_utils.restore_keras_checkpoint(checkpoint_dir)
    preds_after_restoration = restored_model(dummy_inputs)

    for expected, observed in zip(
        [v.value for v in bert_pretrainer.variables],
        [v.value for v in restored_model.variables],
    ):
      # Ensures the objects are different but the values are the same.
      self.assertNotEqual(id(expected), id(observed))
      self.assertEqual(expected.shape, observed.shape)
      self.assertEqual(expected.dtype, observed.dtype)
      self.assertEqual(expected.sharding, observed.sharding)
      np.testing.assert_allclose(observed, expected)

    self.assertDictEqual(
        bert_pretrainer.get_config(), restored_model.get_config()
    )
    np.testing.assert_allclose(preds, preds_after_restoration)

  def test_restore_keras_checkpoint_v3(self):
    dummy_inputs = _create_dummy_inputs()
    bert_pretrainer = _create_model(jax.tree.map(jnp.shape, dummy_inputs))
    preds = bert_pretrainer(dummy_inputs)

    checkpoint_dir = self.create_tempdir().full_path
    checkpoint_manager = keras_utils.KerasOrbaxCheckpointManagerV3(
        checkpoint_dir
    )
    checkpoint_manager.save_model_variables(bert_pretrainer, epoch=1)
    checkpoint_manager.close()

    restored_model = keras_utils.restore_keras_checkpoint(checkpoint_dir)
    preds_after_restoration = restored_model(dummy_inputs)

    for expected, observed in zip(
        [v.value for v in bert_pretrainer.variables],
        [v.value for v in restored_model.variables],
    ):
      self.assertNotEqual(id(expected), id(observed))
      self.assertEqual(expected.shape, observed.shape)
      self.assertEqual(expected.dtype, observed.dtype)
      self.assertEqual(expected.sharding, observed.sharding)
      np.testing.assert_allclose(observed, expected)

    self.assertDictEqual(
        bert_pretrainer.get_config(), restored_model.get_config()
    )
    np.testing.assert_allclose(preds, preds_after_restoration)

  def test_restore_shape_mismatch_fails(self):
    dummy_inputs = _create_dummy_inputs()
    bert_pretrainer = _create_model(jax.tree.map(jnp.shape, dummy_inputs))

    checkpoint_dir = self.create_tempdir().full_path
    checkpoint_manager = keras_utils.KerasOrbaxCheckpointManagerV3(
        checkpoint_dir, max_to_keep=5
    )
    checkpoint_manager.save_model_variables(bert_pretrainer, 0)
    checkpoint_manager.wait_until_finished()

    # Create a model with a different intermediate_dim, causing shape mismatches
    different_pretrainer = keras_hub.models.BertMaskedLM(
        backbone=keras_hub.models.BertBackbone(
            vocabulary_size=2048,
            num_layers=4,
            num_heads=8,
            hidden_dim=32,
            intermediate_dim=128,  # Different shape!
            max_sequence_length=128,
            num_segments=8,
            dropout=0.1,
        )
    )
    optimizer = keras.optimizers.Adam()
    different_pretrainer.compile(optimizer=optimizer)
    different_pretrainer.build(jax.tree.map(jnp.shape, dummy_inputs))
    different_pretrainer.optimizer.build(
        different_pretrainer.trainable_variables
    )

    with self.assertRaises(ValueError):
      checkpoint_manager.restore_model_variables(different_pretrainer, 0)

    checkpoint_manager.close()

  def test_restore_renamed_non_trainable_and_optimizer_variables(self):
    checkpoint_dir = self.create_tempdir().full_path
    checkpoint_manager = keras_utils.KerasOrbaxCheckpointManagerV3(
        checkpoint_dir, max_to_keep=5
    )

    # Save side: model with my_layer, optimizer named adam_save
    model_save = MyTestModel(layer_name="my_layer", name="my_test_model")
    model_save(np.zeros((1, 10)))
    optimizer_save = keras.optimizers.Adam(name="adam_save")
    model_save.compile(optimizer=optimizer_save)
    optimizer_save.build(model_save.trainable_variables)

    # Initialize non-trainable and optimizer slot values to distinct values
    for i, var in enumerate(model_save.non_trainable_variables):
      var.assign(np.ones(var.shape) * (i + 7.0))
    for i, var in enumerate(model_save.optimizer.variables):
      var.assign(np.ones(var.shape) * (i + 9.0))

    checkpoint_manager.save_model_variables(model_save, 0)
    checkpoint_manager.wait_until_finished()

    # Restore side: model with my_layer, optimizer named adam_restore
    model_restore = MyTestModel(layer_name="my_layer", name="my_test_model")
    model_restore(np.zeros((1, 10)))
    optimizer_restore = keras.optimizers.Adam(name="adam_restore")
    model_restore.compile(optimizer=optimizer_restore)
    optimizer_restore.build(model_restore.trainable_variables)

    # Initialize to zeros
    for var in model_restore.non_trainable_variables:
      var.assign(np.zeros(var.shape))
    for var in model_restore.optimizer.variables:
      var.assign(np.zeros(var.shape))

    # Restore should succeed and map optimizer variables by index ordering
    checkpoint_manager.restore_model_variables(model_restore, 0)
    checkpoint_manager.close()

    # Verify non-trainable variables were restored
    for i, var in enumerate(model_restore.non_trainable_variables):
      np.testing.assert_allclose(var.value, np.ones(var.shape) * (i + 7.0))

    # Verify optimizer variables were restored
    for i, var in enumerate(model_restore.optimizer.variables):
      np.testing.assert_allclose(var.value, np.ones(var.shape) * (i + 9.0))

  def test_load_keras_model_config(self):
    dummy_inputs = _create_dummy_inputs()
    bert_pretrainer = _create_model(jax.tree.map(jnp.shape, dummy_inputs))
    config = keras.utils.serialize_keras_object(bert_pretrainer)
    config = json.loads(json.dumps(config))  # Converts tuples to lists.

    checkpoint_dir = self.create_tempdir().full_path
    checkpoint_manager = keras_utils.KerasOrbaxCheckpointManagerV2(
        checkpoint_dir
    )
    checkpoint_manager.save_model_variables(bert_pretrainer, epoch=1)
    checkpoint_manager.close()

    self.assertEqual(
        config, keras_utils.load_keras_model_config(checkpoint_dir, epoch=1)
    )

  def test_restore_keras_checkpoint_flat_path_matches_predictions(self):
    dummy_inputs = _create_dummy_inputs()
    bert_pretrainer = _create_model(jax.tree.map(jnp.shape, dummy_inputs))
    preds = bert_pretrainer(dummy_inputs)
    checkpoint_dir = self.create_tempdir().full_path
    checkpoint_manager = keras_utils.KerasOrbaxCheckpointManagerV2(
        checkpoint_dir
    )
    checkpoint_manager.save_model_variables(bert_pretrainer, epoch=1)
    checkpoint_manager.close()
    flat_checkpoint_dir = os.path.join(checkpoint_dir, "1")

    restored_model = keras_utils.restore_keras_checkpoint(flat_checkpoint_dir)
    preds_after_restoration = restored_model(dummy_inputs)

    np.testing.assert_allclose(preds, preds_after_restoration)

  def test_load_keras_model_config_flat_path_returns_same_config(self):
    dummy_inputs = _create_dummy_inputs()
    bert_pretrainer = _create_model(jax.tree.map(jnp.shape, dummy_inputs))
    config = keras.utils.serialize_keras_object(bert_pretrainer)
    config = json.loads(json.dumps(config))
    checkpoint_dir = self.create_tempdir().full_path
    checkpoint_manager = keras_utils.KerasOrbaxCheckpointManagerV2(
        checkpoint_dir
    )
    checkpoint_manager.save_model_variables(bert_pretrainer, epoch=1)
    checkpoint_manager.close()
    flat_checkpoint_dir = os.path.join(checkpoint_dir, "1")

    restored_config = keras_utils.load_keras_model_config(flat_checkpoint_dir)

    self.assertEqual(config, restored_config)

  @parameterized.named_parameters(
      {
          "testcase_name": "nested_with_epoch",
          "test_path_suffix": "",
          "input_epoch": 42,
          "expected_epoch": 42,
      },
      {
          "testcase_name": "nested_without_epoch",
          "test_path_suffix": "",
          "input_epoch": None,
          "expected_epoch": 42,
      },
      {
          "testcase_name": "flat_path",
          "test_path_suffix": "42",
          "input_epoch": None,
          "expected_epoch": None,
      },
  )
  def test_resolve_orbax_checkpoint_path_success(
      self,
      test_path_suffix,
      input_epoch,
      expected_epoch,
  ):
    dummy_inputs = _create_dummy_inputs()
    bert_pretrainer = _create_model(jax.tree.map(jnp.shape, dummy_inputs))
    checkpoint_dir = self.create_tempdir().full_path
    checkpoint_manager = keras_utils.KerasOrbaxCheckpointManagerV2(
        checkpoint_dir
    )
    checkpoint_manager.save_model_variables(bert_pretrainer, epoch=42)
    checkpoint_manager.close()
    test_path = os.path.join(checkpoint_dir, test_path_suffix)

    resolved_path, resolved_epoch = keras_utils.resolve_orbax_checkpoint_path(
        test_path, epoch=input_epoch
    )

    self.assertEqual(resolved_epoch, expected_epoch)
    self.assertEqual(resolved_path, os.path.join(checkpoint_dir, "42"))

  def test_resolve_orbax_checkpoint_path_missing_step(self):
    dummy_inputs = _create_dummy_inputs()
    bert_pretrainer = _create_model(jax.tree.map(jnp.shape, dummy_inputs))
    test_dir = self.create_tempdir().full_path
    checkpoint_manager = keras_utils.KerasOrbaxCheckpointManagerV2(test_dir)
    checkpoint_manager.save_model_variables(bert_pretrainer, epoch=42)
    checkpoint_manager.close()

    with self.assertRaisesRegex(ValueError, "Step 99 not found"):
      keras_utils.resolve_orbax_checkpoint_path(test_dir, epoch=99)

  def test_resolve_orbax_checkpoint_path_empty_dir(self):
    test_dir = self.create_tempdir().full_path

    with self.assertRaisesRegex(FileNotFoundError, "No checkpoints found"):
      keras_utils.resolve_orbax_checkpoint_path(test_dir, epoch=None)

  @parameterized.named_parameters(
      {
          "testcase_name": "single_core",
          "data_parallel": False,
          "restore_with_checkpointer": True,
      },
      {
          "testcase_name": "data_parallel",
          "data_parallel": True,
          "restore_with_checkpointer": True,
      },
      {
          "testcase_name": "restore_without_checkpointer_data_parallel",
          "data_parallel": True,
          "restore_with_checkpointer": False,
      },
      {
          "testcase_name": "restore_without_checkpointer_single_core",
          "data_parallel": False,
          "restore_with_checkpointer": False,
      },
  )
  def test_keras_orbax_checkpointer(
      self, data_parallel: bool, restore_with_checkpointer: bool
  ):
    if data_parallel:
      keras.distribution.set_distribution(keras.distribution.DataParallel())
    else:
      keras.distribution.set_distribution(None)

    checkpoint_dir = self.create_tempdir().full_path
    checkpoint_manager = keras_utils.KerasOrbaxCheckpointManager(
        checkpoint_dir, max_to_keep=5
    )
    dummy_inputs = _create_dummy_inputs()

    bert_pretrainer = _create_model(jax.tree.map(jnp.shape, dummy_inputs))
    state = (
        [v.value for v in bert_pretrainer.trainable_variables],
        [v.value for v in bert_pretrainer.non_trainable_variables],
        [v.value for v in bert_pretrainer.optimizer.variables],
    )
    checkpoint_manager.save_model_variables(bert_pretrainer, epoch=1)
    checkpoint_manager.wait_until_finished()
    preds = bert_pretrainer(dummy_inputs)

    bert_pretrainer = _create_model(jax.tree.map(jnp.shape, dummy_inputs))
    if restore_with_checkpointer:
      checkpoint_manager.restore_model_variables(bert_pretrainer, epoch=1)
    else:
      keras_utils.restore_keras_model(bert_pretrainer, checkpoint_dir)

    checkpoint_manager.close()

    restored_state = (
        [v.value for v in bert_pretrainer.trainable_variables],
        [v.value for v in bert_pretrainer.non_trainable_variables],
        [v.value for v in bert_pretrainer.optimizer.variables],
    )
    preds_after_restoration = bert_pretrainer(dummy_inputs)

    # Ensures the objects are different but the values are the same.
    def _close(a: jax.Array, b: jax.Array):
      return bool(np.array(jnp.allclose(a, b))) and id(a) != id(b)

    for x in jax.tree.leaves(jax.tree.map(_close, state, restored_state)):
      self.assertTrue(x)

    # Ensures predictions are identical.
    self.assertTrue(_close(preds, preds_after_restoration))

  def test_restore_keras_model_error_cases(self):
    dummy_inputs = _create_dummy_inputs()
    bert_pretrainer = _create_model(jax.tree.map(jnp.shape, dummy_inputs))

    checkpoint_dir = self.create_tempdir().full_path
    checkpointer = keras_utils.KerasOrbaxCheckpointManager(checkpoint_dir)
    checkpointer.save_model_variables(bert_pretrainer, epoch=2)
    checkpointer.wait_until_finished()
    with self.assertRaises(ValueError):
      keras_utils.restore_keras_model(bert_pretrainer, checkpoint_dir, step=0)

    with self.assertRaises(FileNotFoundError):
      keras_utils.restore_keras_model(bert_pretrainer, "not_found_dir")

  def test_restore_keras_model_fails_on_v3_checkpoint(self):
    dummy_inputs = _create_dummy_inputs()
    bert_pretrainer = _create_model(jax.tree.map(jnp.shape, dummy_inputs))

    checkpoint_dir = self.create_tempdir().full_path
    checkpoint_manager = keras_utils.KerasOrbaxCheckpointManagerV3(
        checkpoint_dir
    )
    checkpoint_manager.save_model_variables(bert_pretrainer, epoch=1)
    checkpoint_manager.wait_until_finished()

    with self.assertRaisesRegex(
        ValueError, "is in V2/V3 format.*restore_keras_checkpoint"
    ):
      keras_utils.restore_keras_model(bert_pretrainer, checkpoint_dir, step=1)

    checkpoint_manager.close()

  def test_restore_keras_model_fails_on_v2_checkpoint(self):
    dummy_inputs = _create_dummy_inputs()
    bert_pretrainer = _create_model(jax.tree.map(jnp.shape, dummy_inputs))

    checkpoint_dir = self.create_tempdir().full_path
    checkpoint_manager = keras_utils.KerasOrbaxCheckpointManagerV2(
        checkpoint_dir
    )
    checkpoint_manager.save_model_variables(bert_pretrainer, epoch=1)
    checkpoint_manager.wait_until_finished()

    with self.assertRaisesRegex(
        ValueError, "is in V2/V3 format.*restore_keras_checkpoint"
    ):
      keras_utils.restore_keras_model(bert_pretrainer, checkpoint_dir, step=1)

    checkpoint_manager.close()

  def test_restore_keras_checkpoint_fails_on_v1_checkpoint(self):
    dummy_inputs = _create_dummy_inputs()
    bert_pretrainer = _create_model(jax.tree.map(jnp.shape, dummy_inputs))

    checkpoint_dir = self.create_tempdir().full_path
    checkpoint_manager = keras_utils.KerasOrbaxCheckpointManager(checkpoint_dir)
    checkpoint_manager.save_model_variables(bert_pretrainer, epoch=1)
    checkpoint_manager.wait_until_finished()

    # We expect it to fail because restore_keras_checkpoint expects V2/V3
    # structure. We want it to fail loudly and suggest using
    # restore_keras_model.
    with self.assertRaisesRegex(
        ValueError, "is in V1 format.*restore_keras_model"
    ):
      keras_utils.restore_keras_checkpoint(
          checkpoint_dir, model=bert_pretrainer
      )

    checkpoint_manager.close()

  @parameterized.named_parameters(
      {
          "testcase_name": "restore_with_checkpointer",
          "restore_with_checkpointer": True,
      },
      {
          "testcase_name": "restore_without_checkpointer",
          "restore_with_checkpointer": False,
      },
  )
  def test_metrics_variables_checkpointing(
      self, restore_with_checkpointer: bool
  ):
    checkpoint_dir = self.create_tempdir().full_path
    checkpointer = keras_utils.KerasOrbaxCheckpointManager(checkpoint_dir)
    epoch = 1
    dummy_inputs = _create_dummy_inputs()

    source_bert_pretrainer = _create_model(
        jax.tree.map(jnp.shape, dummy_inputs)
    )
    source_state = source_bert_pretrainer._get_jax_state(  # pylint: disable=protected-access
        trainable_variables=True,
        non_trainable_variables=True,
        optimizer_variables=True,
        metrics_variables=True,
    )
    checkpointer.save(step=epoch, items=source_state)
    checkpointer.wait_until_finished()

    target_bert_pretrainer = _create_model(
        jax.tree.map(jnp.shape, dummy_inputs)
    )
    if restore_with_checkpointer:
      checkpointer.restore_model_variables(target_bert_pretrainer, epoch)
    else:
      keras_utils.restore_keras_model(target_bert_pretrainer, checkpoint_dir)

    self.assertGreater(target_bert_pretrainer.count_params(), 0)
    self.assertLen(
        target_bert_pretrainer.layers, len(source_bert_pretrainer.layers)
    )
    for l1, l2 in zip(
        target_bert_pretrainer.layers, source_bert_pretrainer.layers
    ):
      for w1, w2 in zip(l1.weights, l2.weights):
        np.testing.assert_almost_equal(
            keras.ops.convert_to_numpy(w1.value),
            keras.ops.convert_to_numpy(w2.value),
        )
        self.assertSequenceEqual(w1.dtype, w2.dtype)

  @parameterized.named_parameters(
      {
          "testcase_name": "restore_all_variables",
          "restore_optimizer_vars": True,
          "restore_steps": True,
          "restore_iterations": True,
          "expected_learning_rate": 0.01,
          "expected_iterations": 100,
          "expected_initial_epoch": 2,
          "legacy_format": True,
      },
      {
          "testcase_name": "restore_without_optimizer_vars",
          "restore_optimizer_vars": False,
          "restore_steps": True,
          "restore_iterations": True,
          "expected_learning_rate": 0.1,
          "expected_iterations": 0,
          "expected_initial_epoch": 2,
          "legacy_format": True,
      },
      {
          "testcase_name": "restore_without_steps",
          "restore_optimizer_vars": True,
          "restore_steps": False,
          "restore_iterations": True,
          "expected_learning_rate": 0.01,
          "expected_iterations": 100,
          "expected_initial_epoch": None,
          "legacy_format": True,
      },
      {
          "testcase_name": "restore_without_iterations",
          "restore_optimizer_vars": True,
          "restore_steps": True,
          "restore_iterations": False,
          "expected_learning_rate": 0.1,
          "expected_iterations": 0,
          "expected_initial_epoch": 2,
          "legacy_format": True,
      },
      {
          "testcase_name": "restore_only_model_variables",
          "restore_optimizer_vars": False,
          "restore_steps": False,
          "restore_iterations": False,
          "expected_learning_rate": 0.1,
          "expected_iterations": 0,
          "expected_initial_epoch": None,
          "legacy_format": True,
      },
      {
          "testcase_name": "restore_all_variables_with_new_format",
          "restore_optimizer_vars": True,
          "restore_steps": True,
          "restore_iterations": True,
          "expected_learning_rate": 0.01,
          "expected_iterations": 100,
          "expected_initial_epoch": 2,
          "legacy_format": False,
      },
      {
          "testcase_name": "restore_only_model_variables_with_new_format",
          "restore_optimizer_vars": False,
          "restore_steps": False,
          "restore_iterations": False,
          "expected_learning_rate": 0.1,
          "expected_iterations": 0,
          "expected_initial_epoch": None,
          "legacy_format": False,
      },
  )
  def test_restore_keras_model_with_different_options(
      self,
      restore_optimizer_vars: bool,
      restore_steps: bool,
      restore_iterations: bool,
      expected_learning_rate: float,
      expected_iterations: int,
      expected_initial_epoch: int | None,
      legacy_format: bool,
  ):
    checkpoint_dir = self.create_tempdir().full_path
    if legacy_format:
      checkpointer = keras_utils.KerasOrbaxCheckpointManager(checkpoint_dir)
    else:
      checkpointer = keras_utils.KerasOrbaxCheckpointManagerV2(checkpoint_dir)
    epoch = 1
    dummy_inputs = _create_dummy_inputs()
    source_bert_pretrainer = _create_model(
        jax.tree.map(jnp.shape, dummy_inputs)
    )
    source_bert_pretrainer.optimizer.iterations.assign(100)
    source_state = source_bert_pretrainer._get_jax_state(  # pylint: disable=protected-access
        trainable_variables=True,
        non_trainable_variables=True,
        optimizer_variables=True,
    )
    if legacy_format:
      checkpointer.save(step=epoch, items=source_state)
    else:
      checkpointer.save_model_variables(source_bert_pretrainer, epoch=epoch)
    checkpointer.wait_until_finished()

    target_bert_pretrainer = _create_model(
        jax.tree.map(jnp.shape, dummy_inputs)
    )
    if legacy_format:
      keras_utils.restore_keras_model(
          target_bert_pretrainer,
          checkpoint_dir,
          restore_optimizer_vars=restore_optimizer_vars,
          restore_steps=restore_steps,
          restore_iterations=restore_iterations,
      )
    else:
      keras_utils.restore_keras_checkpoint(
          checkpoint_dir,
          model=target_bert_pretrainer,
          restore_optimizer_vars=restore_optimizer_vars,
          restore_model_epoch=restore_steps,
          restore_iterations=restore_iterations,
      )

    self.assertEqual(
        target_bert_pretrainer.optimizer.iterations.value, expected_iterations
    )
    self.assertEqual(
        target_bert_pretrainer.optimizer.learning_rate,
        expected_learning_rate,
    )
    self.assertEqual(
        target_bert_pretrainer._initial_epoch, expected_initial_epoch
    )

  @parameterized.named_parameters(
      {
          "testcase_name": "KerasOrbaxCheckpointManager",
          "checkpoint_manager_cls": keras_utils.KerasOrbaxCheckpointManager,
      },
      {
          "testcase_name": "KerasOrbaxCheckpointManagerV2",
          "checkpoint_manager_cls": keras_utils.KerasOrbaxCheckpointManagerV2,
      },
  )
  def test_epoch_orbax_checkpoint_and_restore_callback_saves_at_epoch_0(
      self, checkpoint_manager_cls
  ):
    checkpoint_dir = self.create_tempdir().full_path
    checkpoint_manager = checkpoint_manager_cls(
        checkpoint_dir, max_to_keep=5
    )
    dummy_inputs = _create_dummy_inputs()
    model = _create_model(jax.tree.map(jnp.shape, dummy_inputs))
    callback = keras_utils.EpochOrbaxCheckpointAndRestoreCallback(
        checkpoint_manager
    )
    callback.set_model(model)

    self.assertIsNone(checkpoint_manager.latest_step())

    callback.on_train_begin()
    checkpoint_manager.wait_until_finished()
    self.assertTrue(os.path.exists(os.path.join(checkpoint_dir, "0")))


class KerasOrbaxCheckpointUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.checkpoint_dir = self.create_tempdir().full_path

  def test_is_v3_checkpoint_true(self):
    manager = keras_utils.KerasOrbaxCheckpointManagerV3(
        checkpoint_dir=self.checkpoint_dir,
        max_to_keep=1,
        save_interval_epochs=1,
    )
    model = keras.Sequential([keras.layers.Dense(1)])
    model.compile(optimizer="adam")
    model.build((1, 1))
    model.optimizer.build(model.trainable_variables)

    manager.save_model_variables(model, epoch=1)
    manager.wait_until_finished()

    self.assertTrue(keras_utils.is_v3_checkpoint(self.checkpoint_dir))
    self.assertTrue(keras_utils.is_v3_checkpoint(self.checkpoint_dir, epoch=1))

  def test_is_v3_checkpoint_false_for_v1(self):
    manager = keras_utils.KerasOrbaxCheckpointManager(
        checkpoint_dir=self.checkpoint_dir,
        max_to_keep=1,
        save_interval_epochs=1,
    )
    model = keras.Sequential([keras.layers.Dense(1)])
    model.compile(optimizer="adam")
    model.build((1, 1))
    model.optimizer.build(model.trainable_variables)

    manager.save_model_variables(model, epoch=1)
    manager.wait_until_finished()

    self.assertFalse(keras_utils.is_v3_checkpoint(self.checkpoint_dir))

  def test_is_v3_checkpoint_false_for_missing(self):
    with self.assertRaises(FileNotFoundError):
      keras_utils.is_v3_checkpoint(self.checkpoint_dir)
    with self.assertRaises(FileNotFoundError):
      keras_utils.is_v3_checkpoint(self.checkpoint_dir, epoch=1)

  def test_restore_partial_checkpoint_success(self):
    manager = keras_utils.KerasOrbaxCheckpointManagerV3(
        checkpoint_dir=self.checkpoint_dir,
        max_to_keep=1,
        save_interval_epochs=1,
    )
    model = keras.Sequential(
        [
            keras.layers.Dense(2, name="dense1"),
            keras.layers.Dense(1, name="dense2"),
        ],
        name="my_model",
    )
    model.compile(optimizer="adam")
    model.build((1, 1))
    model.optimizer.build(model.trainable_variables)

    model.layers[0].kernel.assign([[1.0, 2.0]])
    model.layers[1].kernel.assign([[3.0], [4.0]])

    manager.save_model_variables(model, epoch=1)
    manager.wait_until_finished()

    new_model = keras.Sequential(
        [
            keras.layers.Dense(2, name="dense1"),
            keras.layers.Dense(1, name="dense2"),
        ],
        name="my_model",
    )
    new_model.build((1, 1))

    new_model.layers[0].kernel.assign([[0.0, 0.0]])
    new_model.layers[1].kernel.assign([[0.0], [0.0]])

    partial_vars = {
        keras_utils.TRAINABLE_VARIABLES_KEY: {
            new_model.layers[0].kernel.path: new_model.layers[0].kernel,
            new_model.layers[0].bias.path: new_model.layers[0].bias,
        }
    }

    restored_state = keras_utils.restore_partial_checkpoint(
        self.checkpoint_dir, partial_vars, epoch=1
    )

    for key, var_dict in partial_vars.items():
      for path, var in var_dict.items():
        var.assign(restored_state[key][path])

    np.testing.assert_allclose(new_model.layers[0].kernel.value, [[1.0, 2.0]])
    np.testing.assert_allclose(new_model.layers[1].kernel.value, [[0.0], [0.0]])

  def test_restore_partial_checkpoint_non_overlapping_architectures(self):
    manager = keras_utils.KerasOrbaxCheckpointManagerV3(
        checkpoint_dir=self.checkpoint_dir,
        max_to_keep=1,
        save_interval_epochs=1,
    )
    # Model in checkpoint has dense1 and dense2
    model = keras.Sequential(
        [
            keras.layers.Dense(2, name="dense1"),
            keras.layers.Dense(1, name="dense2"),
        ],
        name="my_model",
    )
    model.compile(optimizer="adam")
    model.build((1, 1))
    model.optimizer.build(model.trainable_variables)

    model.layers[0].kernel.assign([[1.0, 2.0]])
    model.layers[1].kernel.assign([[3.0], [4.0]])

    manager.save_model_variables(model, epoch=1)
    manager.wait_until_finished()

    # New model has dense1 and dense3 (dense2 is missing, dense3 is new)
    new_model = keras.Sequential(
        [
            keras.layers.Dense(2, name="dense1"),
            keras.layers.Dense(3, name="dense3"),
        ],
        name="my_model",
    )
    new_model.build((1, 1))

    new_model.layers[0].kernel.assign([[0.0, 0.0]])
    new_model.layers[1].kernel.assign([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    # We only restore dense1 from the checkpoint
    partial_vars = {
        keras_utils.TRAINABLE_VARIABLES_KEY: {
            new_model.layers[0].kernel.path: new_model.layers[0].kernel,
            new_model.layers[0].bias.path: new_model.layers[0].bias,
        }
    }

    restored_state = keras_utils.restore_partial_checkpoint(
        self.checkpoint_dir, partial_vars, epoch=1
    )

    for key, var_dict in partial_vars.items():
      for path, var in var_dict.items():
        var.assign(restored_state[key][path])

    # dense1 should be restored
    np.testing.assert_allclose(new_model.layers[0].kernel.value, [[1.0, 2.0]])
    # dense3 should remain unchanged (zeros)
    np.testing.assert_allclose(
        new_model.layers[1].kernel.value, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )

  def test_restore_partial_checkpoint_nested_layer_mismatch(self):
    manager = keras_utils.KerasOrbaxCheckpointManagerV3(
        checkpoint_dir=self.checkpoint_dir,
        max_to_keep=1,
        save_interval_epochs=1,
    )

    class ParentLayer(keras.layers.Layer):

      def __init__(self, nested_name, **kwargs):
        super().__init__(**kwargs)
        self.nested = keras.layers.Dense(2, name=nested_name)

      def call(self, x):
        return self.nested(x)

    # Model A has nested layer named "dense_a" under "parent"
    model_a = keras.Sequential(
        [ParentLayer(nested_name="dense_a", name="parent")], name="model"
    )
    model_a.compile(optimizer="adam")
    model_a.build((1, 1))
    model_a.optimizer.build(model_a.trainable_variables)
    manager.save_model_variables(model_a, epoch=1)
    manager.wait_until_finished()

    # Model B has nested layer named "dense_b" under "parent"
    model_b = keras.Sequential(
        [ParentLayer(nested_name="dense_b", name="parent")], name="model"
    )
    model_b.build((1, 1))

    # Try to restore model_b variables (which have paths like
    # 'parent/dense_b/kernel')
    partial_vars = {
        keras_utils.TRAINABLE_VARIABLES_KEY: {
            model_b.layers[0].nested.kernel.path: (
                model_b.layers[0].nested.kernel
            ),
        }
    }

    # This should fail because 'parent/dense_b/kernel' is not in the checkpoint
    with self.assertRaisesRegex(ValueError, "Missing paths in checkpoint"):
      keras_utils.restore_partial_checkpoint(
          self.checkpoint_dir, partial_vars, epoch=1
      )

  def test_restore_partial_checkpoint_invalid_keys(self):
    manager = keras_utils.KerasOrbaxCheckpointManagerV3(
        checkpoint_dir=self.checkpoint_dir,
        max_to_keep=1,
        save_interval_epochs=1,
    )
    model = keras.Sequential([keras.layers.Dense(1)])
    model.compile(optimizer="adam")
    model.build((1, 1))
    model.optimizer.build(model.trainable_variables)

    manager.save_model_variables(model, epoch=1)
    manager.wait_until_finished()

    partial_vars = {
        keras_utils.NON_TRAINABLE_VARIABLES_KEY: {
            "some_path": model.trainable_variables[0]
        }
    }
    with self.assertRaisesRegex(
        ValueError,
        "Partial restoration is only supported for trainable variables",
    ):
      keras_utils.restore_partial_checkpoint(
          self.checkpoint_dir, partial_vars, epoch=1
      )

    partial_vars = {
        keras_utils.OPTIMIZER_VARIABLES_KEY: {
            "some_path": model.trainable_variables[0]
        }
    }
    with self.assertRaisesRegex(
        ValueError,
        "Partial restoration is only supported for trainable variables",
    ):
      keras_utils.restore_partial_checkpoint(
          self.checkpoint_dir, partial_vars, epoch=1
      )

  def test_restore_optimizer_mismatch_fails(self):
    manager = keras_utils.KerasOrbaxCheckpointManagerV3(
        checkpoint_dir=self.checkpoint_dir,
        max_to_keep=1,
        save_interval_epochs=1,
    )
    # Save side: model with SGD optimizer
    model_sgd = keras.Sequential(
        [keras.layers.Dense(1, name="dense")], name="my_model"
    )
    model_sgd.compile(optimizer="sgd")
    model_sgd.build((1, 1))
    model_sgd.optimizer.build(model_sgd.trainable_variables)

    manager.save_model_variables(model_sgd, epoch=1)
    manager.wait_until_finished()

    # Restore side: model with Adam optimizer
    model_adam = keras.Sequential(
        [keras.layers.Dense(1, name="dense")], name="my_model"
    )
    model_adam.compile(optimizer="adam")
    model_adam.build((1, 1))
    model_adam.optimizer.build(model_adam.trainable_variables)

    # Attempting to restore should fail because Adam has more variables than SGD
    # and they cannot be matched.
    with self.assertRaisesRegex(
        ValueError, "Failed to restore optimizer variables"
    ):
      keras_utils.restore_keras_checkpoint(
          self.checkpoint_dir,
          model=model_adam,
          restore_optimizer_vars=True,
          epoch=1,
      )
    manager.close()

  def test_restore_partial_checkpoint_fails_on_v2(self):
    manager = keras_utils.KerasOrbaxCheckpointManagerV2(
        checkpoint_dir=self.checkpoint_dir
    )
    model = keras.Sequential([keras.layers.Dense(1)])
    model.compile(optimizer="adam")
    model.build((1, 1))
    model.optimizer.build(model.trainable_variables)
    manager.save_model_variables(model, epoch=1)
    manager.wait_until_finished()

    partial_vars = {
        keras_utils.TRAINABLE_VARIABLES_KEY: {
            model.layers[0].kernel.path: model.layers[0].kernel
        }
    }
    with self.assertRaisesRegex(
        ValueError, "restore_partial_checkpoint only supports V3"
    ):
      keras_utils.restore_partial_checkpoint(
          self.checkpoint_dir, partial_vars, epoch=1
      )
    manager.close()

  def test_restore_non_trainable_mismatch_fails(self):
    manager = keras_utils.KerasOrbaxCheckpointManagerV3(
        checkpoint_dir=self.checkpoint_dir,
        max_to_keep=1,
        save_interval_epochs=1,
    )
    # Save side: model WITHOUT non-trainable variables (Dense only)
    model_a = keras.Sequential(
        [keras.layers.Dense(1, name="dense")], name="my_model"
    )
    model_a.compile(optimizer="sgd")
    model_a.build((1, 1))
    model_a.optimizer.build(model_a.trainable_variables)

    manager.save_model_variables(model_a, epoch=1)
    manager.wait_until_finished()

    # Restore side: model WITH non-trainable variables (BatchNormalization)
    # BN adds moving_mean and moving_variance (non-trainable)
    # scale=False, center=False to avoid adding trainable vars (gamma, beta)
    model_b = keras.Sequential(
        [
            keras.layers.BatchNormalization(
                scale=False, center=False, name="bn"
            ),
            keras.layers.Dense(1, name="dense"),
        ],
        name="my_model",
    )
    model_b.compile(optimizer="sgd")
    model_b.build((1, 1))
    model_b.optimizer.build(model_b.trainable_variables)

    # Restore should fail because BN variables are missing from checkpoint
    with self.assertRaises(ValueError):
      keras_utils.restore_keras_checkpoint(
          self.checkpoint_dir,
          model=model_b,
          restore_optimizer_vars=False,
          epoch=1,
      )
    manager.close()


if __name__ == "__main__":
  absltest.main()

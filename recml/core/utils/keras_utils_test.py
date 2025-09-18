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
import json

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

    self.assertDictEqual(
        config, keras_utils.load_keras_model_config(checkpoint_dir, epoch=1)
    )

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
      },
      {
          "testcase_name": "restore_without_optimizer_vars",
          "restore_optimizer_vars": False,
          "restore_steps": True,
          "restore_iterations": True,
          "expected_learning_rate": 0.1,
          "expected_iterations": 0,
          "expected_initial_epoch": 2,
      },
      {
          "testcase_name": "restore_without_steps",
          "restore_optimizer_vars": True,
          "restore_steps": False,
          "restore_iterations": True,
          "expected_learning_rate": 0.01,
          "expected_iterations": 100,
          "expected_initial_epoch": None,
      },
      {
          "testcase_name": "restore_without_iterations",
          "restore_optimizer_vars": True,
          "restore_steps": True,
          "restore_iterations": False,
          "expected_learning_rate": 0.1,
          "expected_iterations": 0,
          "expected_initial_epoch": 2,
      },
      {
          "testcase_name": "restore_only_model_variables",
          "restore_optimizer_vars": False,
          "restore_steps": False,
          "restore_iterations": False,
          "expected_learning_rate": 0.1,
          "expected_iterations": 0,
          "expected_initial_epoch": None,
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
  ):
    checkpoint_dir = self.create_tempdir().full_path
    checkpointer = keras_utils.KerasOrbaxCheckpointManager(checkpoint_dir)
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
    checkpointer.save(step=epoch, items=source_state)
    checkpointer.wait_until_finished()

    target_bert_pretrainer = _create_model(
        jax.tree.map(jnp.shape, dummy_inputs)
    )
    keras_utils.restore_keras_model(
        target_bert_pretrainer,
        checkpoint_dir,
        restore_optimizer_vars=restore_optimizer_vars,
        restore_steps=restore_steps,
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


if __name__ == "__main__":
  absltest.main()

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

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import keras
import keras_hub
import numpy as np
from recml.core.utils import keras_utils


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
  optimizer = keras.optimizers.Adam(learning_rate=0.1)
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
          "testcase_name": "restore_without_checkpointer_data_parallel",
          "data_parallel": True,
          "restore_with_checkpointer": False,
      },
      {
          "testcase_name": "restore_without_checkpointer_model_parallel",
          "data_parallel": False,
          "restore_with_checkpointer": False,
      },
  )
  def test_keras_orbax_checkpointer(
      self, data_parallel: bool, restore_with_checkpointer: bool
  ):
    if data_parallel:
      keras.distribution.set_distribution(keras.distribution.DataParallel())
    checkpoint_dir = self.create_tempdir().full_path
    checkpointer = keras_utils.KerasOrbaxCheckpointManager(
        checkpoint_dir, max_to_keep=5
    )
    epoch = 1
    dummy_inputs = {
        "token_ids": jax.random.randint(
            jax.random.key(0), (64, 128), minval=0, maxval=50_000
        ),
        "segment_ids": jax.random.randint(
            jax.random.key(0), (64, 128), minval=0, maxval=7
        ),
        "padding_mask": jax.random.uniform(jax.random.key(0), (64, 128)),
        "mask_positions": jax.random.randint(
            jax.random.key(0), (64, 20), minval=0, maxval=128
        ),
    }

    def _create_model(input_shapes: Sequence[int]) -> keras.Model:
      model = keras_hub.models.BertMaskedLM(
          backbone=keras_hub.models.BertBackbone(
              vocabulary_size=50_000,
              num_layers=10,
              num_heads=8,
              hidden_dim=256,
              intermediate_dim=3072,
              max_sequence_length=128,
              num_segments=7,
              dropout=0.1,
          )
      )
      optimizer = keras.optimizers.Adam(learning_rate=0.1)
      loss = keras.losses.SparseCategoricalCrossentropy()
      metrics = [keras.metrics.SparseCategoricalAccuracy()]
      model.compile(optimizer, loss, weighted_metrics=metrics)
      model.build(input_shapes)
      optimizer.build(model.trainable_variables)
      return model

    bert_pretrainer = _create_model(jax.tree.map(jnp.shape, dummy_inputs))
    state = (
        [v.value for v in bert_pretrainer.trainable_variables],
        [v.value for v in bert_pretrainer.non_trainable_variables],
        [v.value for v in bert_pretrainer.optimizer.variables],
    )
    checkpointer.save_model_variables(bert_pretrainer, epoch)
    preds = bert_pretrainer(dummy_inputs)

    bert_pretrainer = _create_model(jax.tree.map(jnp.shape, dummy_inputs))
    if restore_with_checkpointer:
      checkpointer.restore_model_variables(bert_pretrainer, epoch)
    else:
      keras_utils.restore_keras_model(bert_pretrainer, checkpoint_dir)
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
    checkpoint_dir = self.create_tempdir().full_path
    checkpointer = keras_utils.KerasOrbaxCheckpointManager(checkpoint_dir)
    epoch = 2
    dummy_inputs = {
        "token_ids": jax.random.randint(
            jax.random.key(0), (64, 128), minval=0, maxval=50_000
        ),
        "segment_ids": jax.random.randint(
            jax.random.key(0), (64, 128), minval=0, maxval=7
        ),
        "padding_mask": jax.random.uniform(jax.random.key(0), (64, 128)),
        "mask_positions": jax.random.randint(
            jax.random.key(0), (64, 20), minval=0, maxval=128
        ),
    }

    bert_pretrainer = _create_model(jax.tree.map(jnp.shape, dummy_inputs))
    checkpointer.save_model_variables(bert_pretrainer, epoch)
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
    dummy_inputs = {
        "token_ids": jax.random.randint(
            jax.random.key(0), (64, 128), minval=0, maxval=50_000
        ),
        "segment_ids": jax.random.randint(
            jax.random.key(0), (64, 128), minval=0, maxval=7
        ),
        "padding_mask": jax.random.uniform(jax.random.key(0), (64, 128)),
        "mask_positions": jax.random.randint(
            jax.random.key(0), (64, 20), minval=0, maxval=128
        ),
    }

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


if __name__ == "__main__":
  absltest.main()

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
"""Tests for Jax training library."""

import os
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import keras
from recml.core.training import core
from recml.core.training import keras_trainer
import tensorflow as tf


class _KerasTask(keras_trainer.KerasTask):

  def create_dataset(
      self, training: bool, eval_name: str | None = None
  ) -> tf.data.Dataset:
    if eval_name:
      self.last_eval_name = eval_name

    def _map_fn(x: int):
      return (tf.cast(x, tf.float32), 0.1 * tf.cast(x, tf.float32) + 3)

    return tf.data.Dataset.range(1000).map(_map_fn).batch(2)

  def create_model(self) -> keras.Model:
    inputs = keras.Input(shape=(1,), dtype=tf.float32)
    outputs = keras.layers.Dense(
        1, kernel_initializer=keras.initializers.constant(-1.0)
    )(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adagrad(0.1),
        loss=keras.losses.MeanSquaredError(),
    )
    return model


class KerasTrainerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Workaround to make `create_tempdir` work with pytest.
    if not flags.FLAGS.is_parsed():
      flags.FLAGS.mark_as_parsed()

  @parameterized.named_parameters(
      {"testcase_name": "train", "mode": core.Trainer.Mode.TRAIN},
      {"testcase_name": "eval", "mode": core.Trainer.Mode.EVAL},
      {
          "testcase_name": "train_and_eval",
          "mode": core.Trainer.Mode.TRAIN_AND_EVAL,
      },
      {
          "testcase_name": "continuous_eval_",
          "mode": core.Trainer.Mode.CONTINUOUS_EVAL,
      },
      {
          "testcase_name": "train_and_eval_legacy_checkpoint_format",
          "mode": core.Trainer.Mode.TRAIN_AND_EVAL,
          "legacy_checkpoint_format": True,
      },
      {
          "testcase_name": "continuous_eval_legacy_checkpoint_format",
          "mode": core.Trainer.Mode.CONTINUOUS_EVAL,
          "legacy_checkpoint_format": True,
      },
  )
  def test_keras_task_and_trainer(
      self, mode: str, legacy_checkpoint_format: bool = False
  ):
    if keras.backend.backend() == "jax":
      distribution = keras.distribution.DataParallel()
    else:
      distribution = None
      if mode == core.Trainer.Mode.CONTINUOUS_EVAL:
        self.skipTest("Continuous eval is only supported on the Jax backend.")

    trainer = keras_trainer.KerasTrainer(
        distribution=distribution,
        train_steps=5,
        steps_per_eval=3,
        steps_per_loop=2,
        model_dir=self.create_tempdir().full_path,
        continuous_eval_timeout=5,
        legacy_checkpoint_format=legacy_checkpoint_format,
    )
    experiment = core.Experiment(_KerasTask(), trainer)

    if mode == core.Trainer.Mode.CONTINUOUS_EVAL:
      # Produce one checkpoint so there is something to evaluate.
      core.run_experiment(experiment, core.Trainer.Mode.TRAIN)

    history = core.run_experiment(experiment, mode)

    if (
        mode in [core.Trainer.Mode.TRAIN, core.Trainer.Mode.TRAIN_AND_EVAL]
        and keras.backend.backend() == "jax"
    ):
      self.assertEqual(history.history["num_params/trainable"][0], 2)

  def test_eval_name(self):
    if keras.backend.backend() != "jax":
      self.skipTest(
          "`EpochSummaryCallback` and `eval_name` are only supported on the Jax"
          " backend."
      )

    model_dir = self.create_tempdir().full_path
    eval_name = "custom_eval"

    trainer = keras_trainer.KerasTrainer(
        model_dir=model_dir,
        steps_per_eval=1,
        distribution=keras.distribution.DataParallel(),
    )
    trainer.set_job_info(
        core.JobInfo(eval_name, core.Trainer.Mode.EVAL).to_string()
    )
    experiment = core.Experiment(_KerasTask(), trainer)

    # Run evaluation
    core.run_experiment(experiment, core.Trainer.Mode.EVAL)

    # Check if log directory exists
    expected_log_dir = os.path.join(model_dir, "logs", eval_name)
    self.assertTrue(os.path.exists(expected_log_dir))
    self.assertEqual(experiment.task.last_eval_name, eval_name)


if __name__ == "__main__":
  absltest.main()

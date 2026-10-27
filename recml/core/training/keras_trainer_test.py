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
from recml.core.utils import keras_utils
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


class _MarkerCallback(keras_utils.MetricsCallback):
  """Reports a fixed metric at the end of each training epoch and eval pass."""

  def on_epoch_end(self, epoch, logs=None):
    self._record(logs, "custom/marker", 7.0)

  def on_test_end(self, logs=None):
    self._record(logs, "custom/marker", 7.0)


class KerasTrainerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    keras.backend.clear_session()
    # Workaround to make `create_tempdir` work with pytest.
    if not flags.FLAGS.is_parsed():
      flags.FLAGS.mark_as_parsed()

  @parameterized.named_parameters(
      {"testcase_name": "train", "mode": core.Trainer.Mode.TRAIN},
      {
          "testcase_name": "train_with_callbacks",
          "mode": core.Trainer.Mode.TRAIN,
          "with_train_callbacks": True,
      },
      {"testcase_name": "eval", "mode": core.Trainer.Mode.EVAL},
      {
          "testcase_name": "train_and_eval_v3",
          "mode": core.Trainer.Mode.TRAIN_AND_EVAL,
          "checkpoint_version": "v3",
      },
      {
          "testcase_name": "train_and_eval_with_callbacks",
          "mode": core.Trainer.Mode.TRAIN_AND_EVAL,
          "with_train_callbacks": True,
      },
      {
          "testcase_name": "train_and_eval_v2",
          "mode": core.Trainer.Mode.TRAIN_AND_EVAL,
          "checkpoint_version": "v2",
      },
      {
          "testcase_name": "train_and_eval_v1",
          "mode": core.Trainer.Mode.TRAIN_AND_EVAL,
          "checkpoint_version": "v1",
      },
      {
          "testcase_name": "continuous_eval_v3",
          "mode": core.Trainer.Mode.CONTINUOUS_EVAL,
          "checkpoint_version": "v3",
      },
      {
          "testcase_name": "continuous_eval_v2",
          "mode": core.Trainer.Mode.CONTINUOUS_EVAL,
          "checkpoint_version": "v2",
      },
      {
          "testcase_name": "continuous_eval_v1",
          "mode": core.Trainer.Mode.CONTINUOUS_EVAL,
          "checkpoint_version": "v1",
      },
  )
  def test_keras_task_and_trainer(
      self,
      mode: str,
      checkpoint_version: str = "v3",
      with_train_callbacks: bool = False,
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
        checkpoint_version=checkpoint_version,
        train_callbacks=[_MarkerCallback()] if with_train_callbacks else (),
    )
    experiment = core.Experiment(_KerasTask(), trainer)

    if mode == core.Trainer.Mode.CONTINUOUS_EVAL:
      # Produce one checkpoint so there is something to evaluate.
      core.run_experiment(experiment, core.Trainer.Mode.TRAIN)
      keras.backend.clear_session()

    history = core.run_experiment(experiment, mode)

    if (
        mode in [core.Trainer.Mode.TRAIN, core.Trainer.Mode.TRAIN_AND_EVAL]
        and keras.backend.backend() == "jax"
    ):
      self.assertEqual(history.history["num_params/trainable"][0], 2)
    if with_train_callbacks:
      self.assertEqual(
          history.history["custom/marker"],
          [7.0] * len(history.history["loss"]),
      )

  def _run_eval_with_callbacks(self, model_dir: str, mode: core.Trainer.Mode):
    if keras.backend.backend() != "jax":
      self.skipTest("`EpochSummaryCallback` is only used on the Jax backend.")
    trainer = keras_trainer.KerasTrainer(
        distribution=keras.distribution.DataParallel(),
        train_steps=2,
        steps_per_eval=1,
        steps_per_loop=2,
        model_dir=model_dir,
        continuous_eval_timeout=1,
        eval_callbacks=[_MarkerCallback()],
    )
    with mock.patch.object(
        keras_utils.EpochSummaryCallback, "on_epoch_end", autospec=True
    ) as on_epoch_end:
      core.run_experiment(core.Experiment(_KerasTask(), trainer), mode)

    val_logs = on_epoch_end.call_args.args[2]
    self.assertEqual(val_logs["val_custom/marker"], 7.0)
    self.assertIn("val_loss", val_logs)
    self.assertIn("val_steps_per_second", val_logs)

  def test_eval_metrics_callbacks_are_written_in_eval(self):
    self._run_eval_with_callbacks(
        self.create_tempdir().full_path, core.Trainer.Mode.EVAL
    )

  def test_eval_metrics_callbacks_are_written_in_continuous_eval(self):
    if keras.backend.backend() != "jax":
      self.skipTest("`EpochSummaryCallback` is only used on the Jax backend.")
    model_dir = self.create_tempdir().full_path
    # Produce one checkpoint so there is something to evaluate.
    train_trainer = keras_trainer.KerasTrainer(
        distribution=keras.distribution.DataParallel(),
        train_steps=2,
        steps_per_eval=1,
        steps_per_loop=2,
        model_dir=model_dir,
    )
    core.run_experiment(
        core.Experiment(_KerasTask(), train_trainer), core.Trainer.Mode.TRAIN
    )
    keras.backend.clear_session()

    self._run_eval_with_callbacks(model_dir, core.Trainer.Mode.CONTINUOUS_EVAL)

  def test_val_logs_warns_when_metrics_callback_overwrites_key(self):
    trainer = keras_trainer.KerasTrainer(
        model_dir=self.create_tempdir().full_path
    )
    callback = _MarkerCallback()
    callback.on_test_end()

    with self.assertLogs(level="WARNING") as log:
      val_logs = trainer._val_logs(
          {"custom/marker": 1.0}, [callback], steps_per_second=2.0
      )

    self.assertIn("overwrites existing evaluation log key", log.output[0])
    self.assertEqual(val_logs["val_custom/marker"], 7.0)

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

  def test_legacy_checkpoint_format_deprecated(self):
    model_dir = self.create_tempdir().full_path

    # Test legacy_checkpoint_format=True -> V1
    with self.assertLogs(level="WARNING") as log:
      trainer = keras_trainer.KerasTrainer(
          model_dir=model_dir,
          legacy_checkpoint_format=True,
      )
      self.assertEqual(
          trainer._checkpoint_version, keras_utils.CheckpointVersion.V1
      )
      self.assertIn("legacy_checkpoint_format is deprecated", log.output[0])

    # Test legacy_checkpoint_format=False -> V2
    with self.assertLogs(level="WARNING") as log:
      trainer = keras_trainer.KerasTrainer(
          model_dir=model_dir,
          legacy_checkpoint_format=False,
      )
      self.assertEqual(
          trainer._checkpoint_version, keras_utils.CheckpointVersion.V2
      )
      self.assertIn("legacy_checkpoint_format is deprecated", log.output[0])


if __name__ == "__main__":
  absltest.main()

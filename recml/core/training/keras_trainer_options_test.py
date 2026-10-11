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
"""Tests for the opt-in training loop options of `KerasTrainer`."""

import os
import types
from unittest import mock

from absl.testing import absltest
import jax
import keras
from keras.src.utils import python_utils
import numpy as np
from recml.core.training import batch_callbacks
from recml.core.training import device_prefetch
from recml.core.training import keras_trainer
from recml.core.utils import keras_utils


def _trainer(tmpdir, **overrides):
  kwargs = {
      "model_dir": tmpdir,
      "train_steps": 10,
      "steps_per_loop": 5,
      "steps_per_eval": 2,
  }
  kwargs.update(overrides)
  return keras_trainer.KerasTrainer(**kwargs)


class _DummyModel:
  """Records what `fit` received; enough of `keras.Model` for `train`."""

  def __init__(self):
    self.train_function = None
    self.seen_in_fit = []
    self.batches_in_fit = []

  def make_train_function(self, force=False):
    # Like Keras: sets `train_function` once and returns None.
    if self.train_function is not None and not force:
      return
    self.train_function = lambda state, it: (next(it), state)

  def fit(self, dataset, epochs, steps_per_epoch, callbacks, initial_epoch):
    del epochs, initial_epoch
    self.make_train_function()
    fn = self.train_function
    self.seen_in_fit.append((fn, callbacks))
    it = iter(dataset)
    for _ in range(steps_per_epoch):
      batch, _ = fn(None, it)
      self.batches_in_fit.append(batch)
    return "fit_history"

  def summary(self, print_fn=None):
    del print_fn


def _task(model, exported):
  return types.SimpleNamespace(
      create_dataset=lambda training: list(range(100)),
      create_model=lambda: model,
      export_model=lambda m, model_dir: exported.append((m, model_dir)),
  )


class KerasTrainerOptionsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._tmpdir = self.create_tempdir().full_path

  def test_rejects_negative_prefetch_depth(self):
    with self.assertRaises(ValueError):
      _trainer(self._tmpdir, device_prefetch_depth=-1)

  def test_default_leaves_the_model_unchanged(self):
    trainer = _trainer(self._tmpdir)
    model = _DummyModel()
    orig_make = model.make_train_function
    exported = []

    history = trainer.train(_task(model, exported))

    self.assertEqual(history, "fit_history")
    fn_in_fit, callbacks_in_fit = model.seen_in_fit[0]
    self.assertNotIsInstance(
        fn_in_fit, device_prefetch.PrefetchingTrainFunction
    )
    self.assertIs(fn_in_fit, model.train_function)
    self.assertIs(callbacks_in_fit, trainer.train_callbacks)
    self.assertEqual(model.make_train_function, orig_make)
    self.assertEqual(model.batches_in_fit, [0, 1, 2, 3, 4])
    self.assertEqual(exported, [(model, self._tmpdir)])

  def test_prefetch_wraps_the_model_only_during_train(self):
    trainer = _trainer(self._tmpdir, device_prefetch_depth=2)
    model = _DummyModel()
    orig_make = model.make_train_function
    exported = []

    history = trainer.train(_task(model, exported))

    self.assertEqual(history, "fit_history")
    fn_in_fit, callbacks_in_fit = model.seen_in_fit[0]
    self.assertIsInstance(fn_in_fit, device_prefetch.PrefetchingTrainFunction)
    self.assertEqual(model.batches_in_fit, [0, 1, 2, 3, 4])
    # The base callback list is passed unchanged.
    self.assertIs(callbacks_in_fit, trainer.train_callbacks)
    self.assertEqual(model.make_train_function, orig_make)
    self.assertNotIn("make_train_function", model.__dict__)
    self.assertIsNone(model.train_function)
    self.assertEqual(exported, [(model, self._tmpdir)])

  def test_prefetch_skips_export_on_non_chief_hosts(self):
    trainer = _trainer(self._tmpdir, device_prefetch_depth=2)
    model = _DummyModel()
    exported = []

    with mock.patch.object(jax, "process_index", return_value=1):
      history = trainer.train(_task(model, exported))

    self.assertIsNone(history)
    self.assertLen(model.seen_in_fit, 1)
    self.assertEmpty(exported)

  def test_train_and_evaluate_warns_that_prefetch_is_not_applied(self):
    trainer = _trainer(self._tmpdir, device_prefetch_depth=2)
    task = mock.Mock()
    task.create_dataset.side_effect = RuntimeError("stop")
    with self.assertLogs(level="WARNING") as logs:
      with self.assertRaisesRegex(RuntimeError, "stop"):
        trainer.train_and_evaluate(task)
    self.assertIn("device_prefetch_depth", logs.output[0])

  def test_train_and_evaluate_warns_that_skip_batch_logs_is_not_applied(self):
    trainer = _trainer(self._tmpdir, skip_batch_logs=True)
    task = mock.Mock()
    task.create_dataset.side_effect = RuntimeError("stop")
    with self.assertLogs(level="WARNING") as logs:
      with self.assertRaisesRegex(RuntimeError, "stop"):
        trainer.train_and_evaluate(task)
    self.assertIn("skip_batch_logs", logs.output[0])

  def test_skip_batch_logs_uses_the_zero_sync_list(self):
    trainer = _trainer(self._tmpdir, skip_batch_logs=True)
    model = _DummyModel()

    trainer.train(_task(model, []))

    fn_in_fit, callbacks_in_fit = model.seen_in_fit[0]
    self.assertNotIsInstance(
        fn_in_fit, device_prefetch.PrefetchingTrainFunction
    )
    self.assertIsInstance(
        callbacks_in_fit, batch_callbacks.ZeroSyncCallbackList
    )
    summaries = [
        cbk
        for cbk in callbacks_in_fit.callbacks
        if isinstance(cbk, keras_utils.EpochSummaryCallback)
    ]
    self.assertLen(summaries, 1)
    self.assertIsInstance(
        summaries[0], batch_callbacks.EpochSummaryNoBatchHooks
    )
    self.assertEqual(
        callbacks_in_fit.params, {"verbose": 0, "epochs": 3, "steps": 5}
    )
    self.assertIs(callbacks_in_fit._model, model)  # pylint: disable=protected-access
    self.assertIsNotNone(callbacks_in_fit._history)  # pylint: disable=protected-access

  def test_skip_batch_logs_with_prefetch(self):
    trainer = _trainer(
        self._tmpdir, device_prefetch_depth=2, skip_batch_logs=True
    )
    model = _DummyModel()

    trainer.train(_task(model, []))

    fn_in_fit, callbacks_in_fit = model.seen_in_fit[0]
    self.assertIsInstance(fn_in_fit, device_prefetch.PrefetchingTrainFunction)
    self.assertIsInstance(
        callbacks_in_fit, batch_callbacks.ZeroSyncCallbackList
    )
    self.assertEqual(model.batches_in_fit, [0, 1, 2, 3, 4])

  def test_skip_batch_logs_end_to_end_real_keras_model(self):
    inputs = keras.Input(shape=(1,), dtype="float32")
    outputs = keras.layers.Dense(
        1, kernel_initializer=keras.initializers.constant(1.0)
    )(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.SGD(0.01),
        loss=keras.losses.MeanSquaredError(),
    )
    x = np.ones((2, 1), dtype=np.float32)
    y = np.ones((2, 1), dtype=np.float32)
    batches = [(x, y)] * 10
    exported = []
    trainer = _trainer(
        self._tmpdir,
        train_steps=4,
        steps_per_loop=2,
        skip_batch_logs=True,
        distribution=keras.distribution.DataParallel(),
    )
    task = types.SimpleNamespace(
        create_dataset=lambda training: (b for b in batches),
        create_model=lambda: model,
        export_model=lambda m, model_dir: exported.append((m, model_dir)),
    )

    with mock.patch.object(
        python_utils, "pythonify_logs", wraps=python_utils.pythonify_logs
    ) as pythonify:
      history = trainer.train(task)

    # 2 epochs x 2 steps/epoch: pythonify_logs runs only at train/epoch
    # boundaries (on_train_begin, 2x on_epoch_begin, 2x on_epoch_end,
    # on_train_end = 6 calls total; only the 2 on_epoch_end + 1 on_train_end
    # calls receive non-None logs), and never during the 4 training batches.
    self.assertEqual(pythonify.call_count, 6)
    non_empty_log_calls = [
        c for c in pythonify.call_args_list if c.args and c.args[0] is not None
    ]
    self.assertLen(non_empty_log_calls, 3)
    self.assertEqual(history.history["num_params/trainable"], [2, 2])
    self.assertLen(history.history["loss"], 2)
    self.assertTrue(os.listdir(os.path.join(self._tmpdir, "logs", "train")))
    self.assertEqual(exported, [(model, self._tmpdir)])


if __name__ == "__main__":
  absltest.main()

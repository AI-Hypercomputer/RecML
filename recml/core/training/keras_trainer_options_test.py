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

import types
from unittest import mock

from absl.testing import absltest
import jax
from recml.core.training import device_prefetch
from recml.core.training import keras_trainer


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
    with self.assertRaisesRegex(
        ValueError, "device_prefetch_depth must be >= 0"
    ):
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


if __name__ == "__main__":
  absltest.main()

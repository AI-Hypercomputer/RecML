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
"""Tests that batch_callbacks skip per-step host work but keep needed logs."""

from unittest import mock

from absl.testing import absltest
import keras
from keras.src.utils import python_utils
from recml.core.training import batch_callbacks
from recml.core.training import keras_trainer
from recml.core.utils import keras_utils


def _train_callbacks(tmpdir):
  return keras_trainer.KerasTrainer(
      model_dir=tmpdir, train_steps=10, steps_per_loop=5
  ).train_callbacks


class BatchCallbacksTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._tmpdir = self.create_tempdir().full_path

  def test_without_batch_hooks_removes_the_summary_batch_hooks(self):
    callbacks = batch_callbacks.without_batch_hooks(
        _train_callbacks(self._tmpdir)
    )
    summary = next(
        cbk
        for cbk in callbacks
        if isinstance(cbk, keras_utils.EpochSummaryCallback)
    )
    self.assertIsInstance(summary, batch_callbacks.EpochSummaryNoBatchHooks)
    self.assertFalse(
        batch_callbacks._overrides_hook(summary, "on_train_batch_begin")  # pylint: disable=protected-access
    )
    self.assertFalse(
        batch_callbacks._overrides_hook(summary, "on_train_batch_end")  # pylint: disable=protected-access
    )
    self.assertFalse(summary.requires_batch_logs)

  def test_zero_sync_list_skips_progbar_and_pythonify_logs(self):
    cb_list = batch_callbacks.ZeroSyncCallbackList(
        batch_callbacks.without_batch_hooks(_train_callbacks(self._tmpdir)),
        add_history=True,
        add_progbar=True,
        verbose=1,
    )
    self.assertIsNone(cb_list._progbar)  # pylint: disable=protected-access
    self.assertFalse(cb_list._async_train)  # pylint: disable=protected-access
    with mock.patch.object(
        python_utils, "pythonify_logs", wraps=python_utils.pythonify_logs
    ) as pythonify:
      cb_list.on_train_batch_begin(0, logs={"loss": 1.0})
      cb_list.on_train_batch_end(0, logs={"loss": 1.0})
    pythonify.assert_not_called()

  def test_zero_sync_list_keeps_logs_for_a_callback_that_needs_them(self):
    seen_logs = []

    class _NeedsLogsCallback(keras.callbacks.Callback):

      def on_train_batch_end(self, batch, logs=None):
        del batch
        seen_logs.append(logs)

    cb_list = batch_callbacks.ZeroSyncCallbackList(
        [_NeedsLogsCallback()],
        add_history=True,
        add_progbar=True,
        verbose=1,
    )
    self.assertIsNone(cb_list._progbar)  # pylint: disable=protected-access
    with mock.patch.object(
        python_utils, "pythonify_logs", wraps=python_utils.pythonify_logs
    ) as pythonify:
      cb_list.on_train_batch_end(0, logs={"loss": 1.5})

    pythonify.assert_called()
    self.assertEqual(seen_logs, [{"loss": 1.5}])

  def test_without_batch_hooks_preserves_summary_callback_attributes(self):
    sentinel_writer = object()
    original = keras_utils.EpochSummaryCallback(
        log_dir=self._tmpdir,
        steps_per_epoch=42,
        write_steps_per_second=False,
        eval_subdir="custom_eval",
        enable_xm_measurements=False,
    )
    original._xm_measurements_writer = sentinel_writer  # pylint: disable=protected-access

    upgraded = batch_callbacks.without_batch_hooks([original])[0]

    self.assertIsInstance(upgraded, batch_callbacks.EpochSummaryNoBatchHooks)
    self.assertEqual(upgraded.log_dir, self._tmpdir)
    self.assertEqual(upgraded._steps_per_epoch, 42)  # pylint: disable=protected-access
    self.assertFalse(upgraded.write_steps_per_second)
    self.assertEqual(upgraded._eval_subdir, "custom_eval")  # pylint: disable=protected-access
    self.assertIs(upgraded._xm_measurements_writer, sentinel_writer)  # pylint: disable=protected-access

  def test_zero_sync_list_calls_opted_out_hooks_without_logs(self):
    seen_begin = []
    seen_end = []

    class _OptedOutCallback(keras.callbacks.Callback):
      requires_batch_logs = False

      def on_train_batch_begin(self, batch, logs=None):
        seen_begin.append((batch, logs))

      def on_train_batch_end(self, batch, logs=None):
        seen_end.append((batch, logs))

    cb_list = batch_callbacks.ZeroSyncCallbackList(
        [_OptedOutCallback()],
        add_history=True,
        add_progbar=True,
        verbose=1,
    )
    self.assertFalse(cb_list._needs_batch_logs)  # pylint: disable=protected-access
    self.assertFalse(cb_list._async_train)  # pylint: disable=protected-access
    with mock.patch.object(
        python_utils, "pythonify_logs", wraps=python_utils.pythonify_logs
    ) as pythonify:
      cb_list.on_train_batch_begin(3, logs={"loss": 1.0})
      cb_list.on_train_batch_end(3, logs={"loss": 1.0})

    pythonify.assert_not_called()
    self.assertEqual(seen_begin, [(3, None)])
    self.assertEqual(seen_end, [(3, None)])

  def test_zero_sync_list_detects_legacy_on_batch_end_and_lambda_callback(self):
    seen_legacy = []
    seen_lambda = []

    class _LegacyBatchEndCallback(keras.callbacks.Callback):

      def on_batch_end(self, batch, logs=None):
        seen_legacy.append((batch, logs))

    lambda_cb = keras.callbacks.LambdaCallback(
        on_train_batch_end=lambda batch, logs: seen_lambda.append((batch, logs))
    )
    with self.assertLogs(level="INFO") as info_logs:
      cb_list = batch_callbacks.ZeroSyncCallbackList(
          [_LegacyBatchEndCallback(), lambda_cb],
          add_history=True,
      )
    self.assertTrue(cb_list._needs_batch_logs)  # pylint: disable=protected-access
    self.assertTrue(
        any(
            "_LegacyBatchEndCallback" in msg and "LambdaCallback" in msg
            for msg in info_logs.output
        )
    )
    cb_list.on_train_batch_end(2, logs={"loss": 2.5})
    self.assertEqual(seen_legacy, [(2, {"loss": 2.5})])
    self.assertEqual(seen_lambda, [(2, {"loss": 2.5})])

  def test_without_batch_hooks_keeps_summary_subclasses(self):

    class _CustomSummary(keras_utils.EpochSummaryCallback):
      pass

    custom = _CustomSummary(log_dir=self._tmpdir, steps_per_epoch=1)
    self.assertEqual(batch_callbacks.without_batch_hooks([custom]), [custom])


if __name__ == "__main__":
  absltest.main()

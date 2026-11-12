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
from recml.core.training import zero_copy_batches
from recml.core.utils import keras_utils
import tensorflow as tf


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
    if isinstance(dataset, zero_copy_batches._ZeroCopyAdapter):
      it = dataset.get_jax_iterator()
    else:
      it = iter(dataset)
    for _ in range(steps_per_epoch):
      batch, _ = fn(None, it)
      self.batches_in_fit.append(batch)
    return "fit_history"

  def summary(self, print_fn=None):
    del print_fn


class _StepModel(_DummyModel):
  """A `_DummyModel` whose train function runs `train_step`, like Keras."""

  built = True

  def __init__(self):
    super().__init__()
    self.datasets_in_fit = []

  def train_step(self, state, data):
    return data, state

  def _symbolic_build(self, iterator=None, data_batch=None):
    del iterator, data_batch

  def make_train_function(self, force=False):
    if self.train_function is not None and not force:
      return
    train_step = self.train_step
    self.train_function = lambda state, it: train_step(state, next(it))

  def fit(self, dataset, **kwargs):
    self.datasets_in_fit.append(dataset)
    return super().fit(dataset, **kwargs)


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

  def test_rejects_negative_fold_state_max_leaf_size(self):
    with self.assertRaisesRegex(
        ValueError, "fold_state_max_leaf_size must be >= 0"
    ):
      _trainer(self._tmpdir, fold_state_max_leaf_size=-1)

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

  def test_fold_batch_leaves_folds_the_dataset_and_unfolds_in_the_step(self):
    features = {
        "a": np.arange(40, dtype=np.float32).reshape([20, 2]),
        "b": -np.arange(60, dtype=np.float32).reshape([20, 3]),
    }
    dataset = tf.data.Dataset.from_tensor_slices(features).batch(
        2, drop_remainder=True
    )
    model = _StepModel()
    task = types.SimpleNamespace(
        create_dataset=lambda training: dataset,
        create_model=lambda: model,
        export_model=lambda m, model_dir: None,
    )

    _trainer(self._tmpdir, fold_batch_leaves=True).train(task)

    # `fit` got 1 folded array per batch instead of the 2 leaves.
    self.assertLen(model.datasets_in_fit, 1)
    dataset_in_fit = model.datasets_in_fit[0]
    folded_specs, other_specs = dataset_in_fit.element_spec
    self.assertLen(folded_specs, 1)
    self.assertEmpty(other_specs)
    # The train step got the original batches.
    self.assertLen(model.batches_in_fit, 5)
    for step, batch in enumerate(model.batches_in_fit):
      self.assertCountEqual(batch, ["a", "b"])
      for key, value in batch.items():
        np.testing.assert_array_equal(
            value, features[key][2 * step : 2 * step + 2]
        )
    self.assertNotIn("train_step", vars(model))

  def test_rejects_fold_batch_leaves_on_other_backends(self):
    with mock.patch.object(keras.backend, "backend", return_value="tensorflow"):
      with self.assertRaisesRegex(ValueError, "needs the JAX backend"):
        _trainer(self._tmpdir, fold_batch_leaves=True)

  def test_fold_batch_leaves_with_compiled_loss_and_metrics(self):
    rng = np.random.default_rng(0)
    x = rng.random((20, 4)).astype(np.float32)
    y = rng.random((20, 1)).astype(np.float32)

    def build_model():
      keras.utils.set_random_seed(0)
      inputs = keras.Input(shape=(4,), dtype="float32")
      model = keras.Model(inputs=inputs, outputs=keras.layers.Dense(1)(inputs))
      # The compiled loss and metrics stay unbuilt until the first batch.
      model.compile(optimizer="sgd", loss="mse", metrics=["mae"])
      return model

    def train(model_dir, **options):
      model = build_model()
      os.makedirs(model_dir)
      history = _trainer(
          model_dir, train_steps=10, steps_per_loop=5, **options
      ).train(
          types.SimpleNamespace(
              create_dataset=lambda training: tf.data.Dataset.from_tensor_slices(
                  (x, y)
              )
              .batch(2, drop_remainder=True)
              .repeat(),
              create_model=lambda: model,
              export_model=lambda m, model_dir: None,
          )
      )
      return model, history.history

    want_model, want_history = train(os.path.join(self._tmpdir, "plain"))
    model, history = train(
        os.path.join(self._tmpdir, "folded"), fold_batch_leaves=True
    )

    for actual, want in zip(
        model.get_weights(), want_model.get_weights(), strict=True
    ):
      np.testing.assert_allclose(actual, want, rtol=1e-6)
    self.assertCountEqual(history, want_history)
    for key, values in want_history.items():
      np.testing.assert_allclose(history[key], values, rtol=1e-6, err_msg=key)

  def test_train_and_evaluate_warns_that_folding_is_not_applied(self):
    trainer = _trainer(self._tmpdir, fold_batch_leaves=True)
    task = mock.Mock()
    task.create_dataset.side_effect = RuntimeError("stop")
    with self.assertLogs(level="WARNING") as logs:
      with self.assertRaisesRegex(RuntimeError, "stop"):
        trainer.train_and_evaluate(task)
    self.assertIn("fold_batch_leaves", logs.output[0])

  def test_zero_copy_batches_wraps_the_dataset_in_a_zero_copy_adapter(self):
    features = np.arange(20, dtype=np.float32).reshape([10, 2])
    dataset = tf.data.Dataset.from_tensor_slices(features).batch(
        2, drop_remainder=True
    )
    model = _StepModel()
    task = types.SimpleNamespace(
        create_dataset=lambda training: dataset,
        create_model=lambda: model,
        export_model=lambda m, model_dir: None,
    )

    _trainer(self._tmpdir, zero_copy_batches=True).train(task)

    self.assertLen(model.datasets_in_fit, 1)
    self.assertIsInstance(
        model.datasets_in_fit[0], zero_copy_batches._ZeroCopyAdapter
    )
    self.assertLen(model.batches_in_fit, 5)
    self.assertFalse(model.batches_in_fit[0].flags.owndata)

  def test_zero_copy_batches_with_fold_batch_leaves(self):
    features = {
        "a": np.arange(40, dtype=np.float32).reshape([20, 2]),
        "b": -np.arange(60, dtype=np.float32).reshape([20, 3]),
    }
    dataset = tf.data.Dataset.from_tensor_slices(features).batch(
        2, drop_remainder=True
    )
    model = _StepModel()
    task = types.SimpleNamespace(
        create_dataset=lambda training: dataset,
        create_model=lambda: model,
        export_model=lambda m, model_dir: None,
    )

    _trainer(
        self._tmpdir, fold_batch_leaves=True, zero_copy_batches=True
    ).train(task)

    self.assertLen(model.datasets_in_fit, 1)
    self.assertIsInstance(
        model.datasets_in_fit[0], zero_copy_batches._ZeroCopyAdapter
    )
    self.assertLen(model.batches_in_fit, 5)
    for step, batch in enumerate(model.batches_in_fit):
      self.assertCountEqual(batch, ["a", "b"])
      for key, value in batch.items():
        np.testing.assert_array_equal(
            value, features[key][2 * step : 2 * step + 2]
        )

  def test_train_and_evaluate_warns_that_zero_copy_batches_is_not_applied(self):
    trainer = _trainer(self._tmpdir, zero_copy_batches=True)
    task = mock.Mock()
    task.create_dataset.side_effect = RuntimeError("stop")
    with self.assertLogs(level="WARNING") as logs:
      with self.assertRaisesRegex(RuntimeError, "stop"):
        trainer.train_and_evaluate(task)
    self.assertIn("zero_copy_batches", logs.output[0])

  def test_train_and_evaluate_warns_that_fold_state_is_not_applied(self):
    trainer = _trainer(self._tmpdir, fold_state_max_leaf_size=64)
    task = mock.Mock()
    task.create_dataset.side_effect = RuntimeError("stop")
    with self.assertLogs(level="WARNING") as logs:
      with self.assertRaisesRegex(RuntimeError, "stop"):
        trainer.train_and_evaluate(task)
    self.assertIn("fold_state_max_leaf_size", logs.output[0])

  def test_fold_state_max_leaf_size_folds_the_state_during_train(self):
    rng = np.random.default_rng(0)
    x = rng.random((8, 4)).astype(np.float32)
    y = rng.random((8, 1)).astype(np.float32)
    inputs = keras.Input(shape=(4,), dtype="float32")
    model = keras.Model(inputs, keras.layers.Dense(1)(inputs))
    model.compile(optimizer="sgd", loss="mse")
    model.optimizer.build(model.trainable_variables)
    task = types.SimpleNamespace(
        create_dataset=lambda training: tf.data.Dataset.from_tensor_slices(
            (x, y)
        )
        .batch(2, drop_remainder=True)
        .repeat(),
        create_model=lambda: model,
        export_model=lambda m, model_dir: None,
    )

    with self.assertLogs(level="INFO") as logs:
      _trainer(
          self._tmpdir,
          train_steps=2,
          steps_per_loop=1,
          fold_state_max_leaf_size=64,
      ).train(task)

    self.assertTrue(
        any("Folding the training state" in line for line in logs.output)
    )

  def test_all_host_speed_options_end_to_end_real_keras_model(self):
    rng = np.random.default_rng(0)
    x = rng.random((40, 4)).astype(np.float32)
    y = rng.random((40, 1)).astype(np.float32)

    def build_model():
      keras.utils.set_random_seed(0)
      inp = keras.Input(shape=(4,), dtype="float32")
      out = keras.layers.Dense(1)(keras.layers.Dense(4, activation="relu")(inp))
      model = keras.Model(inputs=inp, outputs=out)
      model.compile(
          optimizer=keras.optimizers.Adam(0.01),
          loss=keras.losses.MeanSquaredError(),
          metrics=["mae", "mse"],
      )
      model.evaluate(x[:2], y[:2], verbose=0)
      model.optimizer.build(model.trainable_variables)
      return model

    def make_dataset():
      return (
          tf.data.Dataset.from_tensor_slices((x, y))
          .batch(2, drop_remainder=True)
          .repeat()
      )

    ref_model = build_model()
    ref_dir = os.path.join(self._tmpdir, "ref")
    os.makedirs(ref_dir)
    ref_history = _trainer(
        ref_dir,
        train_steps=10,
        steps_per_loop=5,
        distribution=keras.distribution.DataParallel(),
    ).train(
        types.SimpleNamespace(
            create_dataset=lambda training: make_dataset(),
            create_model=lambda: ref_model,
            export_model=lambda m, model_dir: None,
        )
    )

    opt_model = build_model()
    opt_dir = os.path.join(self._tmpdir, "opt")
    os.makedirs(opt_dir)
    opt_history = _trainer(
        opt_dir,
        train_steps=10,
        steps_per_loop=5,
        device_prefetch_depth=1,
        skip_batch_logs=True,
        fold_batch_leaves=True,
        zero_copy_batches=True,
        fold_state_max_leaf_size=64,
        distribution=keras.distribution.DataParallel(),
    ).train(
        types.SimpleNamespace(
            create_dataset=lambda training: make_dataset(),
            create_model=lambda: opt_model,
            export_model=lambda m, model_dir: None,
        )
    )

    for actual, want in zip(opt_model.get_weights(), ref_model.get_weights()):
      np.testing.assert_allclose(actual, want, rtol=1e-6, atol=1e-7)
    for key, values in ref_history.history.items():
      np.testing.assert_allclose(
          opt_history.history[key], values, rtol=1e-6, err_msg=key
      )

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
    batches = [(x, y)] * 25
    exported = []
    trainer = _trainer(
        self._tmpdir,
        train_steps=20,
        steps_per_loop=10,
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

    # 2 epochs x 10 steps/epoch (20 batches): `pythonify_logs` runs only at
    # train/epoch boundaries and does not scale with steps (< 20 calls vs 46
    # when per-step log conversion is active).
    self.assertLess(pythonify.call_count, 20)
    self.assertEqual(history.history["num_params/trainable"], [2, 2])
    self.assertLen(history.history["loss"], 2)
    self.assertTrue(os.listdir(os.path.join(self._tmpdir, "logs", "train")))
    self.assertEqual(exported, [(model, self._tmpdir)])


if __name__ == "__main__":
  absltest.main()

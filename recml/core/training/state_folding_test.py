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
import threading

from absl.testing import absltest
import jax
import jax.numpy as jnp
import keras
from keras.src.utils import python_utils
import numpy as np
from recml.core.training import state_folding


def _data() -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(0)
  x = rng.random([32, 8]).astype(np.float32)
  y = rng.random([32, 1]).astype(np.float32)
  return x, y


class _ExtraLogModel(keras.Model):
  """Adds a non-metric scalar key to `train_step` logs."""

  def __init__(self):
    super().__init__()
    self._dense1 = keras.layers.Dense(4, activation="relu")
    self._bn = keras.layers.BatchNormalization()
    self._dense2 = keras.layers.Dense(1)

  def call(self, x, training=False):
    return self._dense2(self._bn(self._dense1(x), training=training))

  def train_step(self, state, data):
    logs, state = super().train_step(state, data)
    logs["extra_scalar"] = jnp.float32(7.5)
    return logs, state


class _NonScalarLogModel(_ExtraLogModel):
  """Adds a 3-element log, which turns log folding off."""

  def train_step(self, state, data):
    logs, state = super().train_step(state, data)
    logs["per_class"] = jnp.arange(3, dtype=jnp.float32)
    return logs, state


def _model(
    build: bool = True, model_cls: type[keras.Model] = _ExtraLogModel
) -> keras.Model:
  keras.utils.set_random_seed(0)
  model = model_cls()
  model.compile(
      optimizer=keras.optimizers.Adam(0.01),
      loss="mse",
      metrics=["mae", "mse", keras.metrics.RootMeanSquaredError()],
  )
  if build:
    model.build((None, 8))
  return model


class StateFoldingTest(absltest.TestCase):

  def _fit(
      self,
      max_leaf_size: int,
      steps_per_execution: int = 1,
      build: bool = True,
  ) -> tuple[keras.Model, dict[str, list[float]], list[int]]:
    x, y = _data()
    model = _model(build=build)
    model.steps_per_execution = steps_per_execution
    num_args = []
    with state_folding.model_state_folding_scope(model, max_leaf_size):
      model.make_train_function()
      train_function = model.train_function

      def counting_train_function(state, iterator):
        num_args.append(len(jax.tree.leaves(state)))
        return train_function(state, iterator)

      model.train_function = counting_train_function
      history = model.fit(
          x, y, batch_size=8, epochs=3, shuffle=False, verbose=0
      )
    return model, history.history, num_args

  def _assert_same_training(self, model, history, expected, expected_history):
    self.assertNotEmpty(model.get_weights())
    for actual, want in zip(
        model.get_weights(), expected.get_weights(), strict=True
    ):
      np.testing.assert_allclose(actual, want, rtol=1e-6, atol=1e-7)
    self.assertCountEqual(history, expected_history)
    for key, values in expected_history.items():
      np.testing.assert_allclose(history[key], values, rtol=1e-6, err_msg=key)

  def test_same_weights_and_logs(self):
    expected, expected_history, expected_args = self._fit(max_leaf_size=0)
    model, history, num_args = self._fit(max_leaf_size=16)

    self.assertLess(max(num_args), min(expected_args))
    self.assertLen(num_args, len(expected_args))
    self._assert_same_training(model, history, expected, expected_history)
    for actual, want in zip(
        model.optimizer.variables, expected.optimizer.variables, strict=True
    ):
      np.testing.assert_allclose(
          np.asarray(actual), np.asarray(want), rtol=1e-6, atol=1e-7
      )
    self.assertEqual(history["extra_scalar"], [7.5, 7.5, 7.5])

  def test_same_weights_with_steps_per_execution(self):
    expected, expected_history, _ = self._fit(
        max_leaf_size=0, steps_per_execution=2
    )
    model, history, _ = self._fit(max_leaf_size=16, steps_per_execution=2)

    self._assert_same_training(model, history, expected, expected_history)

  def test_unbuilt_model_folds(self):
    expected, expected_history, expected_args = self._fit(
        max_leaf_size=0, build=False
    )
    model, history, num_args = self._fit(max_leaf_size=16, build=False)

    self.assertLess(max(num_args), min(expected_args))
    self._assert_same_training(model, history, expected, expected_history)

  def test_validation_data_under_distribution_works(self):
    x, y = _data()
    distribution = keras.distribution.DataParallel()
    with distribution.scope():
      expected = _model()
      want_history = expected.fit(
          x,
          y,
          validation_data=(x[:8], y[:8]),
          batch_size=8,
          epochs=2,
          shuffle=False,
          verbose=0,
      ).history

      model = _model()
      with state_folding.model_state_folding_scope(model, max_leaf_size=16):
        got_history = model.fit(
            x,
            y,
            validation_data=(x[:8], y[:8]),
            batch_size=8,
            epochs=2,
            shuffle=False,
            verbose=0,
        ).history

    self.assertCountEqual(got_history, want_history)
    for key, values in want_history.items():
      np.testing.assert_allclose(
          got_history[key], values, rtol=1e-6, err_msg=key
      )

  def test_sharded_small_leaves_pass_through(self):
    devices = jax.devices()
    self.assertLen(devices, 4)
    mesh = jax.sharding.Mesh(np.array(devices), ("model",))
    sharded = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("model")
    )
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    leaves = [
        jax.device_put(jnp.zeros([4], jnp.float32), sharded),
        jax.device_put(jnp.zeros([4], jnp.float32), replicated),
        jax.device_put(jnp.zeros([], jnp.float32), replicated),
    ]

    layout = state_folding._ListLayout(leaves, max_leaf_size=16)

    # The sharded leaf is small but passes through, keeping its sharding.
    self.assertEqual(layout.passthrough_leaf_indices, [0])
    self.assertEqual(layout.vector_leaf_indices, [[1, 2]])

  def test_same_weights_under_model_parallel(self):
    x, y = _data()
    mesh = keras.distribution.DeviceMesh(
        (2, 2), ["batch", "model"], devices=keras.distribution.list_devices()
    )
    layout_map = keras.distribution.LayoutMap(mesh)
    layout_map[".*dense/kernel"] = (None, "model")
    # Small but sharded: passes through.
    layout_map[".*dense/bias"] = ("model",)
    distribution = keras.distribution.ModelParallel(
        layout_map=layout_map, batch_dim_name="batch"
    )
    results = []
    for max_leaf_size in (0, 16):
      with distribution.scope():
        keras.utils.set_random_seed(0)
        inputs = keras.Input((8,))
        hidden = keras.layers.Dense(4, name="dense")(inputs)
        model = keras.Model(inputs, keras.layers.Dense(1)(hidden))
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        with state_folding.model_state_folding_scope(model, max_leaf_size):
          history = model.fit(
              x, y, batch_size=8, epochs=2, shuffle=False, verbose=0
          ).history
      results.append((model, history))

    expected, expected_history = results[0]
    model, history = results[1]
    self._assert_same_training(model, history, expected, expected_history)
    for variable, want in zip(model.weights, expected.weights, strict=True):
      self.assertEqual(
          variable.value.sharding, want.value.sharding, msg=variable.path
      )

  def test_restores_the_model_on_exit(self):
    model, _, _ = self._fit(max_leaf_size=16)

    for name in (
        "train_step",
        "make_train_function",
        "state_sync",
        "_get_jax_state",
        "_get_state_sharding_spec",
    ):
      self.assertNotIn(name, vars(model), msg=name)
    x, y = _data()
    model.fit(x, y, batch_size=8, epochs=1, verbose=0)
    model.evaluate(x, y, batch_size=8, verbose=0)

    custom_train_step = lambda state, data: _ExtraLogModel.train_step(
        model, state, data
    )
    model.train_step = custom_train_step
    with state_folding.model_state_folding_scope(model, max_leaf_size=16):
      self.assertIsNot(model.train_step, custom_train_step)
    self.assertIs(model.train_step, custom_train_step)

  def test_state_layout_preserves_replicated_named_shardings(self):
    devices = jax.devices()
    self.assertLen(devices, 4)
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape(2, 2), ("batch", "model")
    )
    replicated_2d = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(None, None)
    )
    replicated_0d = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )
    state = [[
        jax.device_put(jnp.ones([2, 2], jnp.float32), replicated_2d),
        jax.device_put(jnp.array(3.0, jnp.float32), replicated_0d),
    ]]

    layout = state_folding._StateLayout(state, max_leaf_size=4)
    folded = layout.fold(state)
    unfolded = layout.unfold(folded)

    self.assertEqual(folded[0][0].sharding, replicated_0d)
    self.assertEqual(unfolded[0][0].sharding, replicated_2d)
    self.assertEqual(unfolded[0][1].sharding, replicated_0d)
    np.testing.assert_array_equal(unfolded[0][0], np.ones([2, 2], np.float32))
    np.testing.assert_array_equal(unfolded[0][1], 3.0)

  def test_zero_max_leaf_size_leaves_the_model_untouched(self):
    model = _model()

    with state_folding.model_state_folding_scope(model, max_leaf_size=0):
      self.assertNotIn("train_step", vars(model))

  def test_list_layout_folds_small_leaves_by_dtype(self):
    leaves = [
        jnp.zeros([3], jnp.int32),  # 3 elements: too large, passes through.
        jnp.zeros([], jnp.float32),
        jnp.zeros([], jnp.int32),
        jnp.zeros([1, 1], jnp.float32),
        jnp.zeros([2], jnp.int32),
    ]

    layout = state_folding._ListLayout(leaves, max_leaf_size=2)

    self.assertEqual(layout.passthrough_leaf_indices, [0])
    self.assertEqual(layout.vector_leaf_indices, [[1, 3], [2, 4]])
    self.assertEqual(layout.folded_list_length, 3)
    float_leaves, int_leaves = layout.split_vectors([
        jnp.array([5.0, 6.0], jnp.float32),
        jnp.array([7, 8, 9], jnp.int32),
    ])
    np.testing.assert_array_equal(float_leaves[0], np.float32(5.0))
    np.testing.assert_array_equal(float_leaves[1], [[6.0]])
    np.testing.assert_array_equal(int_leaves[0], np.int32(7))
    np.testing.assert_array_equal(int_leaves[1], [8, 9])

  def test_folded_logs_defers_read_until_values_accessed(self):
    logs = state_folding._FoldedLogs(
        {"float32": ["loss", "mae"]},
        {"float32": jnp.array([1.25, 0.5], dtype=jnp.float32)},
    )
    self.assertEqual(set(logs), {"loss", "mae"})
    self.assertIn("loss", logs)
    self.assertIsNotNone(logs._vectors)

    copied = dict(logs)
    self.assertIsNone(logs._vectors)
    self.assertEqual(copied, {"loss": np.float32(1.25), "mae": np.float32(0.5)})

  def test_folded_logs_survive_concurrent_reads(self):
    slow = jax.jit(
        lambda m: (
            jnp.stack([jnp.sum(m @ m), jnp.sum(m)]),
            jnp.stack([jnp.sum(m @ m).astype(jnp.int32), jnp.int32(2)]),
        )
    )
    m = jnp.ones((1024, 1024), jnp.float32) * 1e-3
    errors = []
    worker_results = []

    def read(logs: state_folding._FoldedLogs) -> None:
      try:
        worker_results.append(python_utils.pythonify_logs(logs))
      except Exception as e:  # pylint: disable=broad-exception-caught
        errors.append(e)

    for _ in range(100):
      del worker_results[:]
      float_vector, int_vector = slow(m)
      logs = state_folding._FoldedLogs(
          {"float32": ["a", "b"], "int32": ["c", "d"]},
          {"float32": float_vector, "int32": int_vector},
      )
      worker = threading.Thread(target=read, args=(logs,))
      worker.start()
      main_result = None
      try:
        main_result = dict(logs)  # What `JAXTrainer.fit` does at epoch end.
      except Exception as e:  # pylint: disable=broad-exception-caught
        errors.append(e)
      worker.join()
      self.assertIsNotNone(main_result)
      self.assertLen(worker_results, 1)
      for key in ("a", "b", "c", "d"):
        self.assertIsNotNone(main_result[key])
        self.assertIsNotNone(worker_results[0][key])
      self.assertEqual(main_result["d"], np.int32(2))
      self.assertEqual(worker_results[0]["d"], 2)
    self.assertEmpty(errors)

  def test_fold_logs_empty_and_multiple_dtypes(self):
    folding = state_folding._Folding(_model(), max_leaf_size=16, fold_logs=True)

    self.assertEqual(folding._fold_logs({}), {})
    self.assertEqual(folding._log_names_by_dtype, {})

    folded = folding._fold_logs({
        "loss": jnp.array(1.0, dtype=jnp.float32),
        "mae": jnp.array(0.5, dtype=jnp.bfloat16),
        "step": jnp.array(100, dtype=jnp.int32),
    })
    self.assertCountEqual(folded, ["float32", "bfloat16", "int32"])
    self.assertEqual(folded["float32"].shape, (1,))
    self.assertEqual(folded["bfloat16"].shape, (1,))
    self.assertEqual(folded["int32"].shape, (1,))

  def test_folded_logs_read_methods_read_values_first(self):
    def folded_logs():
      return state_folding._FoldedLogs(
          {"float32": ["loss", "mae"], "int32": ["step"]},
          {
              "float32": jnp.array([1.25, 0.5], dtype=jnp.float32),
              "int32": jnp.array([10], dtype=jnp.int32),
          },
      )

    self.assertEqual(folded_logs().get("loss"), np.float32(1.25))
    self.assertEqual(folded_logs().get("missing", 0.0), 0.0)
    self.assertIn(("step", np.int32(10)), list(folded_logs().items()))
    self.assertIn(np.float32(0.5), list(folded_logs().values()))
    self.assertEqual(
        folded_logs().copy(),
        {
            "loss": np.float32(1.25),
            "mae": np.float32(0.5),
            "step": np.int32(10),
        },
    )
    self.assertIn("1.25", repr(folded_logs()))

  def test_non_scalar_log_disables_log_folding_with_a_warning(self):
    folding = state_folding._Folding(_model(), max_leaf_size=16, fold_logs=True)
    logs = {"loss": jnp.float32(1.0), "per_class": jnp.zeros([3])}

    with self.assertLogs(level="WARNING") as captured:
      folded = folding._fold_logs(logs)

    self.assertIs(folded, logs)
    self.assertIsNone(folding._log_names_by_dtype)
    self.assertIn("per_class", captured.output[0])

  def test_nested_dict_log_disables_log_folding_with_a_warning(self):
    folding = state_folding._Folding(_model(), max_leaf_size=16, fold_logs=True)
    logs = {"loss": jnp.float32(1.0), "nested": {"a": jnp.float32(2.0)}}

    with self.assertLogs(level="WARNING") as captured:
      folded = folding._fold_logs(logs)

    self.assertIs(folded, logs)
    self.assertIn("nested", captured.output[0])

  def test_folded_logs_mutation_and_comparison_read_values_first(self):
    def folded_logs():
      return state_folding._FoldedLogs(
          {"float32": ["loss", "mae"]},
          {"float32": jnp.array([1.25, 0.5], dtype=jnp.float32)},
      )

    self.assertEqual(folded_logs(), {"loss": 1.25, "mae": 0.5})
    self.assertNotEqual(folded_logs(), {"loss": None, "mae": None})
    self.assertEqual(folded_logs().pop("loss"), np.float32(1.25))
    self.assertEqual(folded_logs().setdefault("mae"), np.float32(0.5))
    logs = folded_logs()
    logs["loss"] = 5.0
    self.assertEqual(logs["loss"], 5.0)
    logs = folded_logs()
    logs.update(mae=7.0)
    self.assertEqual(dict(logs), {"loss": 1.25, "mae": 7.0})
    self.assertEqual(folded_logs().popitem(), ("mae", np.float32(0.5)))
    logs = folded_logs()
    del logs["mae"]
    self.assertEqual(logs["loss"], np.float32(1.25))
    self.assertNotIn("mae", logs)

  def test_fit_folds_the_step_logs(self):
    x, y = _data()
    model = _model()
    log_types = []
    with state_folding.model_state_folding_scope(
        model, max_leaf_size=16, fold_logs=True
    ):
      model.make_train_function()
      train_function = model.train_function

      def recording_train_function(state, iterator):
        logs, state = train_function(state, iterator)
        log_types.append(type(logs))
        return logs, state

      model.train_function = recording_train_function
      model.fit(x, y, batch_size=8, epochs=1, shuffle=False, verbose=0)

    self.assertEqual(set(log_types), {state_folding._FoldedLogs})

  def test_fit_without_fold_logs_returns_plain_logs(self):
    x, y = _data()
    model = _model()
    log_types = []
    with state_folding.model_state_folding_scope(model, max_leaf_size=16):
      model.make_train_function()
      train_function = model.train_function

      def recording_train_function(state, iterator):
        logs, state = train_function(state, iterator)
        log_types.append(type(logs))
        return logs, state

      model.train_function = recording_train_function
      model.fit(x, y, batch_size=8, epochs=1, shuffle=False, verbose=0)

    self.assertNotIn(state_folding._FoldedLogs, set(log_types))

  def test_non_scalar_log_passes_all_logs_through_fit(self):
    histories = []
    for max_leaf_size in (0, 16):
      x, y = _data()
      model = _model(model_cls=_NonScalarLogModel)
      with state_folding.model_state_folding_scope(
          model, max_leaf_size, fold_logs=True
      ):
        histories.append(
            model.fit(
                x, y, batch_size=8, epochs=2, shuffle=False, verbose=0
            ).history
        )

    want = histories[0]
    got = histories[1]
    self.assertCountEqual(got, want)
    for key, values in want.items():
      np.testing.assert_allclose(got[key], values, rtol=1e-6, err_msg=key)


if __name__ == "__main__":
  absltest.main()

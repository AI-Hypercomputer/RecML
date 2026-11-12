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
from absl.testing import absltest
import jax
import jax.numpy as jnp
import keras
import numpy as np
from recml.core.training import state_folding


def _data() -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(0)
  x = rng.random([32, 8]).astype(np.float32)
  y = rng.random([32, 1]).astype(np.float32)
  return x, y


class _ExtraLogModel(keras.Model):
  """Adds a non-metric scalar key to `train_step` logs, like GUM."""

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


def _model(build: bool = True) -> keras.Model:
  keras.utils.set_random_seed(0)
  model = _ExtraLogModel()
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


if __name__ == "__main__":
  absltest.main()

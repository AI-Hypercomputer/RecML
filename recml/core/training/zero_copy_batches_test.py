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
from typing import Any

from absl.testing import absltest
import jax.experimental.sparse as jax_sparse
import keras
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters import tf_dataset_adapter
import numpy as np
from recml.core.training import zero_copy_batches
import tensorflow as tf


def _dataset(drop_remainder: bool = True, size: int = 64) -> tf.data.Dataset:
  x = np.arange(size * 3, dtype=np.float32).reshape(size, 3) / 100.0
  y = (x.sum(axis=1, keepdims=True) > 1.0).astype(np.float32)
  ids = np.arange(size, dtype=np.int64).reshape(size, 1)
  return tf.data.Dataset.from_tensor_slices(({"x": x, "ids": ids}, y)).batch(
      8, drop_remainder=drop_remainder
  )


def _leaves(batches: list[Any]) -> list[Any]:
  """Flattens batches, including Keras' unstacked remainders, into leaves."""
  leaves = []
  for batch in batches:
    if isinstance(batch, data_adapter_utils.PartialBatchList):
      for part in batch:
        leaves.extend(keras.tree.flatten(part))
    else:
      leaves.extend(keras.tree.flatten(batch))
  return leaves


def _model() -> keras.Model:
  keras.utils.set_random_seed(0)
  x = keras.Input((3,), name="x")
  ids = keras.Input((1,), name="ids", dtype="int32")
  h = keras.layers.Concatenate()([x, keras.ops.cast(ids, "float32") / 64.0])
  model = keras.Model({"x": x, "ids": ids}, keras.layers.Dense(1)(h))
  model.compile(optimizer=keras.optimizers.SGD(0.1), loss="mse")
  return model


class ZeroCopyBatchesTest(absltest.TestCase):

  def test_yields_views_with_the_same_values(self):
    dataset = _dataset()
    adapter = zero_copy_batches.zero_copy_adapter(dataset)

    batches = list(adapter.get_jax_iterator())

    expected = list(dataset.as_numpy_iterator())
    self.assertLen(batches, len(expected))
    for (features, labels), (want_features, want_labels) in zip(
        batches, expected
    ):
      for got, want in (
          (features["x"], want_features["x"]),
          (features["ids"], want_features["ids"]),
          (labels, want_labels),
      ):
        self.assertIsInstance(got, np.ndarray)
        self.assertFalse(got.flags.owndata)
        self.assertEqual(got.dtype, want.dtype)
        np.testing.assert_array_equal(got, want)

  def test_matches_the_keras_tf_dataset_adapter(self):
    # 68 examples leave a remainder batch; bfloat16 takes Keras' conversion.
    for drop_remainder in (True, False):
      dataset = _dataset(drop_remainder, size=68).map(
          lambda features, labels: (
              {**features, "x_bf16": tf.cast(features["x"], tf.bfloat16)},
              labels,
          )
      )
      for super_batch in (None, 2, 3):
        with self.subTest(
            drop_remainder=drop_remainder, super_batch=super_batch
        ):
          keras_adapter = tf_dataset_adapter.TFDatasetAdapter(dataset)
          want = list(keras_adapter.get_jax_iterator(super_batch))
          adapter = zero_copy_batches.zero_copy_adapter(dataset)
          got = list(adapter.get_jax_iterator(super_batch))

          self.assertEqual([type(b) for b in got], [type(b) for b in want])
          got_leaves, want_leaves = _leaves(got), _leaves(want)
          self.assertLen(got_leaves, len(want_leaves))
          for got_leaf, want_leaf in zip(got_leaves, want_leaves, strict=True):
            self.assertIs(type(got_leaf), type(want_leaf))
            self.assertEqual(got_leaf.dtype, want_leaf.dtype)
            np.testing.assert_array_equal(got_leaf, want_leaf)

  def test_sparse_tensor_converts_to_jax_sparse(self):
    sparse = tf.SparseTensor(
        indices=[[0, 1], [1, 2]], values=[3.0, 4.0], dense_shape=[2, 4]
    )
    dataset = tf.data.Dataset.from_tensors(sparse)
    adapter = zero_copy_batches.zero_copy_adapter(dataset)

    [got] = list(adapter.get_jax_iterator())

    self.assertIsInstance(got, jax_sparse.BCOO)
    np.testing.assert_array_equal(
        got.todense(), np.array([[0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0]])
    )

  def test_fit_matches_fit_on_the_dataset(self):
    reference = _model()
    want = reference.fit(_dataset(), epochs=2, verbose=0).history["loss"]

    model = _model()
    got = model.fit(
        zero_copy_batches.zero_copy_adapter(_dataset()), epochs=2, verbose=0
    ).history["loss"]

    self.assertEqual(got, want)
    for variable, want_variable in zip(model.weights, reference.weights):
      np.testing.assert_array_equal(
          keras.ops.convert_to_numpy(variable),
          keras.ops.convert_to_numpy(want_variable),
      )

  def test_rejects_non_tf_dataset(self):
    with self.assertRaisesRegex(TypeError, "tf.data.Dataset"):
      zero_copy_batches.zero_copy_adapter([1, 2, 3])


if __name__ == "__main__":
  absltest.main()

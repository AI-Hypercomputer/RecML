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
import keras
import numpy as np
from recml.core.training import batch_folding
import tensorflow as tf

_BATCH_SIZE = 4
_NUM_BATCHES = 3


def _features() -> dict[str, np.ndarray]:
  rng = np.random.default_rng(0)
  n = _BATCH_SIZE * _NUM_BATCHES
  return {
      "ids": rng.integers(0, 10, [n, 3]).astype(np.int32),
      "segments": rng.integers(0, 10, [n, 2, 2]).astype(np.int32),
      "values": rng.random([n, 5]).astype(np.float32),
      "weight": rng.random([n]).astype(np.float32),
      # The only bool leaf: it has no group to fold with.
      "present": rng.random([n]) > 0.5,
  }


def _labels() -> np.ndarray:
  n = _BATCH_SIZE * _NUM_BATCHES
  return np.random.default_rng(1).random([n, 1]).astype(np.float32)


def _dataset() -> tf.data.Dataset:
  return tf.data.Dataset.from_tensor_slices((_features(), _labels())).batch(
      _BATCH_SIZE, drop_remainder=True
  )


def _assert_trees_equal(actual, expected):
  keras.tree.assert_same_structure(actual, expected)
  for a, e in zip(keras.tree.flatten(actual), keras.tree.flatten(expected)):
    a, e = np.asarray(a), np.asarray(e)
    np.testing.assert_array_equal(a, e)
    np.testing.assert_equal(a.dtype, e.dtype)


class _Model(keras.Model):
  """Regression on the features; its loss needs every leaf."""

  def __init__(self):
    super().__init__()
    self._dense = keras.layers.Dense(1)

  def call(self, inputs):
    ids = keras.ops.cast(inputs["ids"], "float32")
    segments = keras.ops.cast(inputs["segments"], "float32")
    x = keras.ops.concatenate(
        [ids, keras.ops.reshape(segments, (-1, 4)), inputs["values"]], axis=1
    )
    present = keras.ops.cast(inputs["present"], "float32")
    return self._dense(x) * (inputs["weight"] * present)[:, None]

  def compute_loss(
      self, x=None, y=None, y_pred=None, sample_weight=None, training=True
  ):
    del x, sample_weight, training
    return keras.ops.mean(keras.ops.square(y_pred - y))


def _built_model() -> _Model:
  keras.utils.set_random_seed(0)
  model = _Model()
  model.compile(optimizer=keras.optimizers.SGD(0.1))
  model(next(iter(_dataset()))[0])
  return model


class FoldDatasetTest(absltest.TestCase):

  def test_docstring_example(self):
    dataset = tf.data.Dataset.from_tensors({
        "ids": np.array([[1, 2], [3, 4]], np.int32),
        "mask": np.array([[[1, 0]], [[0, 1]]], np.int32),
        "present": np.array([True, False]),
        "weight": np.array([0.5, 2.0], np.float32),
    })

    folded_dataset, unfold = batch_folding.fold_dataset(dataset)

    folded = next(folded_dataset.as_numpy_iterator())
    expected_folded = (
        (np.array([[1, 2, 1, 0], [3, 4, 0, 1]], np.int32),),
        (np.array([True, False]), np.array([0.5, 2.0], np.float32)),
    )
    _assert_trees_equal(folded, expected_folded)
    _assert_trees_equal(unfold(folded), next(dataset.as_numpy_iterator()))

  def test_folds_leaves_by_dtype_and_leading_dimension(self):
    folded_dataset, _ = batch_folding.fold_dataset(_dataset())

    folded_arrays, other_leaves = folded_dataset.element_spec
    # int32: ids (3) + segments (4). float32: values (5) + weight (1) +
    # labels (1).
    self.assertCountEqual(
        [(spec.dtype, tuple(spec.shape)) for spec in folded_arrays],
        [(tf.int32, (_BATCH_SIZE, 7)), (tf.float32, (_BATCH_SIZE, 7))],
    )
    self.assertEqual(
        [(spec.dtype, tuple(spec.shape)) for spec in other_leaves],
        [(tf.bool, (_BATCH_SIZE,))],
    )

  def test_unfold_restores_the_elements(self):
    dataset = _dataset()
    folded_dataset, unfold = batch_folding.fold_dataset(dataset)

    num_elements = 0
    for element, folded in zip(
        dataset.as_numpy_iterator(), folded_dataset.as_numpy_iterator()
    ):
      _assert_trees_equal(unfold(folded), element)
      num_elements += 1
    self.assertEqual(num_elements, _NUM_BATCHES)

  def test_unfold_accepts_another_leading_dimension(self):
    dataset = _dataset()
    folded_dataset, unfold = batch_folding.fold_dataset(dataset)
    elements = list(dataset.as_numpy_iterator())
    folded = list(folded_dataset.as_numpy_iterator())

    # On several hosts, the arrays of a step hold the batches of all hosts.
    joined = keras.tree.map_structure(
        lambda *leaves: np.concatenate(leaves, axis=0), *folded
    )

    _assert_trees_equal(
        unfold(joined),
        keras.tree.map_structure(
            lambda *leaves: np.concatenate(leaves, axis=0), *elements
        ),
    )

  def test_folds_a_dict_element(self):
    dataset = tf.data.Dataset.from_tensor_slices(_features()).batch(
        _BATCH_SIZE, drop_remainder=True
    )
    folded_dataset, unfold = batch_folding.fold_dataset(dataset)

    for element, folded in zip(
        dataset.as_numpy_iterator(),
        folded_dataset.as_numpy_iterator(),
        strict=True,
    ):
      _assert_trees_equal(unfold(folded), element)

  def test_ragged_leaves_pass_through(self):
    features = _features()
    dataset = tf.data.Dataset.from_tensor_slices({
        "ids": features["ids"],
        "segments": features["segments"],
        "ragged": tf.ragged.constant(
            [[1, 2], [3]] * (_BATCH_SIZE * _NUM_BATCHES // 2), dtype=tf.int32
        ),
    }).batch(_BATCH_SIZE, drop_remainder=True)

    folded_dataset, unfold = batch_folding.fold_dataset(dataset)

    for element, folded in zip(dataset, folded_dataset, strict=True):
      folded_arrays, other_leaves = folded
      self.assertLen(folded_arrays, 1)
      [ragged] = other_leaves
      restored = unfold(folded)
      self.assertIs(restored["ragged"], ragged)
      self.assertEqual(ragged.to_list(), element["ragged"].to_list())
      np.testing.assert_array_equal(restored["ids"], element["ids"])
      np.testing.assert_array_equal(restored["segments"], element["segments"])

  def test_one_leaf_per_dtype_is_not_folded(self):
    features = _features()
    dataset = tf.data.Dataset.from_tensor_slices(
        {"ids": features["ids"], "values": features["values"]}
    ).batch(_BATCH_SIZE, drop_remainder=True)

    folded_dataset, unfold = batch_folding.fold_dataset(dataset)

    self.assertIs(folded_dataset, dataset)
    self.assertIsNone(unfold)

  def test_folds_leaves_with_an_unknown_leading_dimension(self):
    # 10 rows batched by 4 without `drop_remainder` yield sizes [4, 4, 2].
    features = {k: v[:10] for k, v in _features().items()}
    labels = _labels()[:10]
    dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(
        _BATCH_SIZE
    )

    folded_dataset, unfold = batch_folding.fold_dataset(dataset)

    folded_arrays, other_leaves = folded_dataset.element_spec
    self.assertCountEqual(
        [(spec.dtype, tuple(spec.shape)) for spec in folded_arrays],
        [(tf.int32, (None, 7)), (tf.float32, (None, 7))],
    )
    self.assertEqual(
        [(spec.dtype, tuple(spec.shape)) for spec in other_leaves],
        [(tf.bool, (None,))],
    )
    num_elements = 0
    for element, folded in zip(
        dataset.as_numpy_iterator(), folded_dataset.as_numpy_iterator()
    ):
      _assert_trees_equal(unfold(folded), element)
      num_elements += 1
    self.assertEqual(num_elements, 3)

  def test_unknown_trailing_dimension_passes_through(self):
    dataset = tf.data.Dataset.from_generator(
        lambda: [
            {"a": np.zeros([2, 3], np.int32), "b": np.zeros([2, 4], np.int32)}
        ],
        output_signature={
            "a": tf.TensorSpec(shape=(2, None), dtype=tf.int32),
            "b": tf.TensorSpec(shape=(2, None), dtype=tf.int32),
        },
    )

    folded_dataset, unfold = batch_folding.fold_dataset(dataset)

    self.assertIs(folded_dataset, dataset)
    self.assertIsNone(unfold)

  def test_rejects_other_dataset_types(self):
    with self.assertRaisesRegex(TypeError, "tf.data.Dataset"):
      batch_folding.fold_dataset([1, 2, 3])


class ModelUnfoldScopeTest(absltest.TestCase):

  def test_training_on_folded_batches_gives_the_same_weights(self):
    dataset = _dataset()
    expected = _built_model()
    expected.fit(dataset, epochs=2, verbose=0)
    model = _built_model()
    folded_dataset, unfold = batch_folding.fold_dataset(dataset)

    with batch_folding.model_unfold_scope(model, unfold):
      model.fit(folded_dataset, epochs=2, verbose=0)

    self.assertNotEmpty(model.trainable_weights)
    for actual, want in zip(model.get_weights(), expected.get_weights()):
      np.testing.assert_allclose(actual, want, rtol=1e-6)
    initial = _built_model()
    self.assertFalse(
        np.allclose(model.get_weights()[0], initial.get_weights()[0])
    )

  def test_restores_the_train_step_on_exit(self):
    model = _built_model()
    _, unfold = batch_folding.fold_dataset(_dataset())

    with batch_folding.model_unfold_scope(model, unfold):
      self.assertIn("train_step", vars(model))
      self.assertEqual(model.train_step.__name__, "train_step")

    self.assertNotIn("train_step", vars(model))
    self.assertIsNone(model.train_function)
    # The model trains on plain batches again.
    model.fit(_dataset(), epochs=1, verbose=0)

  def test_no_unfold_leaves_the_model_untouched(self):
    model = _built_model()

    with batch_folding.model_unfold_scope(model, None):
      self.assertNotIn("train_step", vars(model))

  def test_rejects_an_unbuilt_model(self):
    model = _Model()
    _, unfold = batch_folding.fold_dataset(_dataset())

    with self.assertRaisesRegex(ValueError, "must be built"):
      with batch_folding.model_unfold_scope(model, unfold):
        pass


if __name__ == "__main__":
  absltest.main()

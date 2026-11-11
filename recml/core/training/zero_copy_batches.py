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
"""Hands `tf.data` batches to the Keras JAX training loop without a copy.

Keras converts each leaf of a `tf.data` batch with `np.array`, which copies the
leaf on the Python thread that feeds the device. For a batch of tens of MB this
copy takes longer than the transfer to the device.

`zero_copy_adapter` wraps a dataset in a Keras data adapter that passes NumPy
views of the dense integer, floating-point and boolean `tf.Tensor` leaves
instead. Other leaves, including bfloat16 and complex tensors, are converted
(and copied) as Keras converts them. A view keeps its tensor alive, and
`tf.data` never writes to a tensor that it has produced, so the views are safe
to read for as long as they exist. The views are read-only.
"""

from collections.abc import Iterator
from typing import Any

import jax.numpy as jnp
import keras
from keras.src.backend.tensorflow.ops import core as keras_tf_core
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters import tf_dataset_adapter
import numpy as np
import tensorflow as tf


def _to_jax_or_numpy(x: Any) -> Any:
  """Returns a zero-copy NumPy view of a dense tensor, or Keras' conversion."""
  if isinstance(x, tf.SparseTensor):
    return data_adapter_utils.tf_sparse_to_jax_sparse(x)
  if (
      isinstance(x, tf.Tensor)
      and (x.dtype.is_integer or x.dtype.is_floating or x.dtype.is_bool)
      and x.dtype != tf.bfloat16
  ):
    return np.asarray(x)
  return keras_tf_core.convert_to_numpy(x)


class _ZeroCopyAdapter(tf_dataset_adapter.TFDatasetAdapter):
  """A `TFDatasetAdapter` that does not copy dense leaves for JAX."""

  def get_jax_iterator(self, super_batch: int | None = None) -> Iterator[Any]:
    dataset = self._dataset
    if super_batch and self.batch_size is not None:
      dataset = dataset.batch(super_batch).prefetch(tf.data.AUTOTUNE)
    iterator = (
        keras.tree.map_structure(_to_jax_or_numpy, batch, none_is_leaf=False)
        for batch in dataset
    )
    if super_batch and self.batch_size is None:
      iterator = data_adapter_utils.super_batch_iterator(
          iterator, super_batch, stack_fn=jnp.stack
      )
    yield from iterator


def zero_copy_adapter(
    dataset: tf.data.Dataset | tf.distribute.DistributedDataset,
) -> _ZeroCopyAdapter:
  """Returns a Keras data adapter for `dataset` that does not copy batches.

  Pass the result to `model.fit` in place of `dataset`. Dense integer,
  floating-point and boolean tensors become read-only NumPy views. Other
  leaves, including bfloat16 and complex tensors, are converted (and copied)
  as Keras converts them.

  Args:
    dataset: A `tf.data.Dataset` or `tf.distribute.DistributedDataset`.

  Returns:
    A Keras data adapter that yields NumPy views of the dense leaves on the JAX
    backend and behaves as the Keras `tf.data` adapter otherwise.

  Raises:
    TypeError: If `dataset` is not a `tf.data.Dataset` or `DistributedDataset`.
  """
  if not isinstance(
      dataset, (tf.data.Dataset, tf.distribute.DistributedDataset)
  ):
    raise TypeError(
        f"zero_copy_batches needs a `tf.data.Dataset`, got {type(dataset)}."
    )
  return _ZeroCopyAdapter(
      dataset, distribution=keras.distribution.distribution()
  )

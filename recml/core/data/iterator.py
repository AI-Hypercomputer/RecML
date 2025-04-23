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
"""Data loading and preprocessing for feeding Jax models."""

from collections.abc import Callable
import os
from typing import Any

import clu.data as clu_data
from etils import epath
import numpy as np
import tensorflow as tf


DatasetIterator = clu_data.DatasetIterator


class TFDatasetIterator(clu_data.DatasetIterator):
  """An iterator for TF Datasets that supports postprocessing."""

  def __init__(
      self,
      dataset: tf.data.Dataset,
      postprocessor: Callable[..., Any] | None = None,
      checkpoint: bool = False,
  ):
    """Initializes the iterator.

    Args:
      dataset: The TF Dataset to iterate over.
      postprocessor: An optional postprocessor to apply to each batch. This is
        useful for sending embedded ID features to a separate accelerator.
      checkpoint: Whether to checkpoint the iterator state.
    """
    self._dataset = dataset
    self._iterator = iter(dataset)
    self._postprocessor = postprocessor
    self._prefetched_batch = None
    self._element_spec = None
    self._checkpoint = None
    if checkpoint:
      self._checkpoint = tf.train.Checkpoint(ds=self._iterator)

  def __next__(self) -> clu_data.Element:
    """Returns the next batch."""
    if self._prefetched_batch is not None:
      batch = self._prefetched_batch
      self._prefetched_batch = None
    else:
      batch = next(self._iterator)
      if self._postprocessor is not None:
        batch = self._postprocessor(batch)

    def _maybe_to_numpy(
        x: tf.Tensor | tf.SparseTensor | tf.RaggedTensor | np.ndarray,
    ) -> np.ndarray | tf.SparseTensor | tf.RaggedTensor:
      if isinstance(x, (tf.SparseTensor, tf.RaggedTensor, np.ndarray)):
        return x
      if hasattr(x, "_numpy"):
        numpy = x._numpy()  # pylint: disable=protected-access
      else:
        numpy = x.numpy()
      if isinstance(numpy, np.ndarray):
        # `numpy` shares the same underlying buffer as the `x` Tensor.
        # Tensors are expected to be immutable, so we disable writes.
        numpy.setflags(write=False)
      return numpy

    return tf.nest.map_structure(_maybe_to_numpy, batch)

  @property
  def element_spec(self) -> clu_data.ElementSpec:
    if self._element_spec is not None:
      return self._element_spec

    batch = next(self._iterator)
    if self._postprocessor is not None:
      batch = self._postprocessor(batch)

    self._prefetched_batch = batch

    def _to_element_spec(
        x: tf.Tensor | tf.SparseTensor | tf.RaggedTensor | np.ndarray,
    ) -> clu_data.ArraySpec:
      if isinstance(x, tf.SparseTensor):
        return clu_data.ArraySpec(
            dtype=x.dtype.as_numpy_dtype,
            shape=tuple(x.shape[0], *[None for _ in x.shape[1:]]),
        )
      if isinstance(x, tf.RaggedTensor):
        return clu_data.ArraySpec(
            dtype=x.dtype.as_numpy_dtype,  # pylint: disable=attribute-error
            shape=tuple(x.shape.as_list()),  # pylint: disable=attribute-error
        )
      if isinstance(x, tf.Tensor):
        return clu_data.ArraySpec(
            dtype=x.dtype.as_numpy_dtype, shape=tuple(x.shape.as_list())
        )
      return clu_data.ArraySpec(dtype=x.dtype, shape=tuple(x.shape))

    element_spec = tf.nest.map_structure(_to_element_spec, batch)
    self._element_spec = element_spec
    return element_spec

  def reset(self):
    self._iterator = iter(self._dataset)
    if self._checkpoint is not None:
      self._checkpoint = tf.train.Checkpoint(ds=self._iterator)

  def save(self, filename: epath.Path):
    if self._checkpoint is not None:
      self._checkpoint.write(os.fspath(filename))

  def restore(self, filename: epath.Path):
    if self._checkpoint is not None:
      self._checkpoint.read(os.fspath(filename)).assert_consumed()

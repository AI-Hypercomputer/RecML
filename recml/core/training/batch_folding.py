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
"""Folds the dense leaves of training batches into a few arrays.

The Keras JAX training loop converts and copies every leaf of a batch to the
device one by one, in Python. With many leaves per batch, as in packed sequence
training, this holds the GIL for tens of milliseconds per step, and the thread
that starts the training steps needs the GIL too.

`fold_dataset` concatenates the dense leaves of each batch in `tf.data`, one
array per dtype and leading dimension, so that only a few arrays cross to the
device. `model_unfold_scope` makes the jitted train step of a model split them
back, so the model sees the original batch.

Terms:
  leaf: one tensor of a batch, in the order of `keras.tree.flatten`.
  leaf group: the leaves that fold into one array. They share dtype and leading
    dimension, and there are 2 or more of them.
  folded array: the concatenation of a leaf group, of shape
    `[leading_dim, total_width]`, where the width of a leaf is the product of
    its dimensions after the first.
"""

import collections
from collections.abc import Callable, Iterator, Sequence
import contextlib
import dataclasses
import functools
import math
from typing import Any

from absl import logging
import keras
import tensorflow as tf


@dataclasses.dataclass(frozen=True)
class _LeafGroups:
  """Which leaves of a batch fold into which array, and how to undo it.

  Attributes:
    structure: The element structure of the dataset before folding.
    specs: The `tf.TensorSpec` (or other spec) of each leaf, in flatten order.
    folded_leaf_groups: For each folded array, the indices of its leaves in the
      order of their columns.
    passthrough_leaf_indices: The indices of the leaves that pass through.
  """

  structure: Any
  specs: Sequence[Any]
  folded_leaf_groups: Sequence[Sequence[int]]
  passthrough_leaf_indices: Sequence[int]

  def leaf_width(self, leaf_index: int) -> int:
    """Returns the number of columns of a leaf in its folded array."""
    return math.prod(self.specs[leaf_index].shape[1:])

  # All control flow here is over static Python specs, not tensors.
  @tf.autograph.experimental.do_not_convert
  def fold(self, *element: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Returns `(folded_arrays, passthrough_leaves)` of a dataset element."""
    # `Dataset.map` passes the components of a tuple element as arguments.
    if not isinstance(self.structure, tuple):
      [element] = element
    leaves = keras.tree.flatten(element)
    folded_arrays = []
    for leaf_group in self.folded_leaf_groups:
      columns = [  # Each entry has shape [leading_dim, leaf_width].
          tf.reshape(leaves[i], [-1, self.leaf_width(i)]) for i in leaf_group
      ]
      folded_arrays.append(tf.concat(columns, axis=1))
    passthrough_leaves = [leaves[i] for i in self.passthrough_leaf_indices]
    return tuple(folded_arrays), tuple(passthrough_leaves)

  def unfold(self, element: Any) -> Any:
    """Returns the original element of a `(folded_arrays, ...)` element.

    The leading dimension of the arrays may differ from the one in `specs`, as
    when the arrays hold the batches of several hosts.

    Args:
      element: `(folded_arrays, passthrough_leaves)` as returned by `fold`.

    Returns:
      The element in `structure`.
    """
    folded_arrays, passthrough_leaves = element
    leaves = [None] * len(self.specs)
    for folded_array, leaf_group in zip(
        folded_arrays, self.folded_leaf_groups, strict=True
    ):
      leading_dim = folded_array.shape[0]
      first_column = 0
      for i in leaf_group:
        last_column = first_column + self.leaf_width(i)
        columns = folded_array[:, first_column:last_column]
        leaf_shape = (leading_dim, *self.specs[i].shape[1:])
        leaves[i] = keras.ops.reshape(columns, leaf_shape)
        first_column = last_column
    for i, leaf in zip(
        self.passthrough_leaf_indices, passthrough_leaves, strict=True
    ):
      leaves[i] = leaf
    return keras.tree.pack_sequence_as(self.structure, leaves)


def fold_dataset(
    dataset: tf.data.Dataset,
) -> tuple[tf.data.Dataset, Callable[[Any], Any] | None]:
  """Folds the dense leaves of the elements of `dataset`.

  Leaves of rank 1 or more whose dimensions after the first are all known are
  grouped by dtype and static leading dimension. Each group of 2 or more leaves
  becomes one array of shape `[leading_dim, total_width]`. All other leaves pass
  through unchanged. Leaves with an unknown leading dimension form their own
  group; `tf.concat` fails if their leading dimensions differ at run time.

  For example, a batch of 2 rows (keys flatten in sorted order)

    {"ids": int32 [[1, 2], [3, 4]],            # shape [2, 2], width 2
     "mask": int32 [[[1, 0]], [[0, 1]]],       # shape [2, 1, 2], width 2
     "present": bool [True, False],            # only bool leaf
     "weight": float32 [0.5, 2.0]}             # only float32 leaf

  folds into one int32 array, with "ids" in columns 0-1 and "mask" in columns
  2-3, while "present" and "weight" have no leaf to fold with:

    ((int32 [[1, 2, 1, 0], [3, 4, 0, 1]],), (present, weight))

  Args:
    dataset: A batched dataset.

  Returns:
    `(folded_dataset, unfold)`. The elements of `folded_dataset` are
    `(folded_arrays, passthrough_leaves)` tuples. `unfold` maps such an element
    back to the structure of the elements of `dataset`; it accepts arrays with
    a different leading dimension, as on multiple hosts, and runs under `jit`.
    If no leaves can be folded, returns `(dataset, None)`.

  Raises:
    TypeError: If `dataset` is not a `tf.data.Dataset`.
  """
  if not isinstance(dataset, tf.data.Dataset):
    raise TypeError(
        f"Folding batches needs a `tf.data.Dataset`, got {type(dataset)}."
    )
  structure = dataset.element_spec
  specs = keras.tree.flatten(structure)
  leaf_indices_by_dtype_and_leading_dim = collections.defaultdict(list)
  for i, spec in enumerate(specs):
    if (
        isinstance(spec, tf.TensorSpec)
        and spec.shape.rank
        and spec.shape[1:].is_fully_defined()
    ):
      key = (spec.dtype, spec.shape[0])
      leaf_indices_by_dtype_and_leading_dim[key].append(i)
  folded_leaf_groups = [
      leaf_indices
      for leaf_indices in leaf_indices_by_dtype_and_leading_dim.values()
      if len(leaf_indices) > 1
  ]
  if not folded_leaf_groups:
    return dataset, None
  folded_leaf_indices = set().union(*folded_leaf_groups)
  leaf_groups = _LeafGroups(
      structure=structure,
      specs=specs,
      folded_leaf_groups=folded_leaf_groups,
      passthrough_leaf_indices=[
          i for i in range(len(specs)) if i not in folded_leaf_indices
      ],
  )
  logging.info(
      "Folding %d of %d batch leaves into %d arrays.",
      len(folded_leaf_indices),
      len(specs),
      len(folded_leaf_groups),
  )
  return dataset.map(leaf_groups.fold), leaf_groups.unfold


@contextlib.contextmanager
def model_unfold_scope(
    model: keras.Model, unfold: Callable[[Any], Any] | None
) -> Iterator[None]:
  """Makes the train step of `model` unfold its batches.

  `model.fit` must run on the folded dataset inside this context. The model,
  its compiled loss and metrics, and its optimizer must be built (for example
  with `model._symbolic_build(data_batch=unfolded_dataset.element_spec)`):
  Keras builds unbuilt parts from the first batch, which is folded.

  Args:
    model: A built Keras model on the JAX backend.
    unfold: The function returned by `fold_dataset`. `None` leaves `model`
      untouched.

  Yields:
    None.

  Raises:
    ValueError: If `unfold` is set and `model` is not built.
  """
  if unfold is None:
    yield
    return
  if not model.built:
    raise ValueError("The model must be built to train on folded batches.")
  had_instance_train_step = "train_step" in vars(model)
  train_step = model.train_step

  # A 2-line closure over `train_step` and `unfold` reads more simply than a
  # class with the same two fields.
  @functools.wraps(train_step)
  def unfolded_train_step(state, data):
    return train_step(state, unfold(data))

  model.train_step = unfolded_train_step
  model.train_function = None
  try:
    yield
  finally:
    if had_instance_train_step:
      model.train_step = train_step
    else:
      del model.train_step
    model.train_function = None

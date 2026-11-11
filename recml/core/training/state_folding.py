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
"""Folds small replicated leaves of the Keras JAX training state into vectors.

The Keras JAX training loop passes every variable of the model, the optimizer,
and the metrics to the jitted train step as its own array, and receives one
array back for each, plus one array per scalar log. The host pays per-array
overhead on every step: Python argument parsing, per-chip buffer handles and
output buffer allocation, and Python array wrapper construction/destruction.
Models with hundreds of metrics pass thousands of small arrays this way.

`model_state_folding_scope` concatenates the small fully-replicated arrays of
each of the 4 state lists into one 1-D vector per dtype. Inside the jitted
train step, the vectors are split and reshaped back before calling the model's
`train_step`, and the updated state is folded back on return. Large or sharded
arrays pass through unchanged.

Terms:
  leaf: one array of a state list.
  small leaf: a fully replicated leaf with at most `max_leaf_size` elements.
  vector: the 1-D concatenation of the raveled small leaves of one dtype.
  passthrough leaf: a leaf that is not in a vector (large, sharded, or the
    only small leaf of its dtype).
  folded list: the passthrough leaves in their order, then the vectors.
"""

from collections.abc import Iterator, Sequence
import contextlib
from typing import Any

from absl import logging
import jax
from jax import flatten_util
import keras
import numpy as np

_STATE_KEYS = (
    "trainable_variables",
    "non_trainable_variables",
    "optimizer_variables",
    "metrics_variables",
)


@contextlib.contextmanager
def _override_attribute(obj: Any, name: str, value: Any) -> Iterator[None]:
  """Sets `obj.<name>` to `value` and restores the instance attribute on exit.

  If `obj` had no instance attribute `name` before (only a class attribute or
  method), the instance attribute is deleted on exit.

  Args:
    obj: The object to change.
    name: The attribute name.
    value: The value to set while the context is active.

  Yields:
    None.
  """
  had_instance_attribute = name in vars(obj)
  saved_value = vars(obj).get(name)
  setattr(obj, name, value)
  try:
    yield
  finally:
    if had_instance_attribute:
      setattr(obj, name, saved_value)
    else:
      delattr(obj, name)


def _replicated_vector_sharding(
    leaf_sharding: jax.sharding.Sharding,
) -> jax.sharding.Sharding:
  """Returns a replicated sharding on the devices of a small leaf."""
  if isinstance(leaf_sharding, jax.sharding.NamedSharding):
    return jax.sharding.NamedSharding(
        leaf_sharding.mesh, jax.sharding.PartitionSpec()
    )
  return leaf_sharding


class _ListLayout:
  """How one state list is folded.

  For example, with `max_leaf_size=2` the list

    [w: float32[2, 2], b: float32[2], c: float32[], n: int32[]]

  has small leaves b, c (float32) and n (int32). w has 4 elements, so it
  passes through. n is the only small int32 leaf, so it passes through too.
  The folded list is

    [w, n, float32[3] vector = concat(ravel(b), ravel(c))]

  and `passthrough_leaf_indices == [0, 3]`, `vector_leaf_indices == [[1, 2]]`.
  """

  def __init__(self, leaves: Sequence[jax.Array], max_leaf_size: int):
    self.num_leaves = len(leaves)
    self.leaf_shardings = [leaf.sharding for leaf in leaves]
    small_leaf_indices_by_dtype: dict[np.dtype[Any], list[int]] = {}
    for i, leaf in enumerate(leaves):
      if leaf.size <= max_leaf_size and leaf.sharding.is_fully_replicated:
        dtype = np.dtype(leaf.dtype)
        small_leaf_indices_by_dtype.setdefault(dtype, []).append(i)
    # For each vector, the indices of its leaves in the order of their entries.
    self.vector_leaf_indices = [
        leaf_indices
        for leaf_indices in small_leaf_indices_by_dtype.values()
        if len(leaf_indices) > 1
    ]
    leaf_indices_in_vectors = set().union(*self.vector_leaf_indices)
    self.passthrough_leaf_indices = [
        i for i in range(self.num_leaves) if i not in leaf_indices_in_vectors
    ]
    self.vector_shardings = [
        _replicated_vector_sharding(self.leaf_shardings[leaf_indices[0]])
        for leaf_indices in self.vector_leaf_indices
    ]
    # `unravel` splits a vector back into its leaves with their shapes.
    self._unravel_vectors = [
        flatten_util.ravel_pytree(vector_leaves)[1]
        for vector_leaves in self.leaves_by_vector(leaves)
    ]

  @property
  def folded_list_length(self) -> int:
    """Returns the length of the folded list."""
    return len(self.passthrough_leaf_indices) + len(self.vector_leaf_indices)

  def leaves_by_vector(self, leaves: Sequence[Any]) -> list[list[Any]]:
    """Returns, for each vector, the entries of `leaves` that go into it."""
    return [
        [leaves[i] for i in leaf_indices]
        for leaf_indices in self.vector_leaf_indices
    ]

  def concat_vectors(
      self, leaves_by_vector: Sequence[Sequence[jax.Array]]
  ) -> list[jax.Array]:
    """Returns one raveled 1-D vector per entry of `leaves_by_vector`."""
    return [
        flatten_util.ravel_pytree(list(vector_leaves))[0]
        for vector_leaves in leaves_by_vector
    ]

  def split_vectors(
      self, vectors: Sequence[jax.Array]
  ) -> list[list[jax.Array]]:
    """Returns the leaves of each vector, reshaped back. Inverts the above."""
    return [
        unravel(vector)
        for unravel, vector in zip(self._unravel_vectors, vectors)
    ]

  def fold_list(
      self, leaves: Sequence[Any], vectors: Sequence[Any]
  ) -> list[Any]:
    """Returns the folded list of `leaves`, given its `vectors`."""
    return [leaves[i] for i in self.passthrough_leaf_indices] + list(vectors)

  def vectors_in_folded_list(self, folded_list: Sequence[Any]) -> list[Any]:
    """Returns the vectors at the end of a folded list."""
    return list(folded_list[len(self.passthrough_leaf_indices) :])

  def unfold_list(
      self,
      folded_list: Sequence[Any],
      leaves_by_vector: Sequence[Sequence[Any]],
  ) -> list[Any]:
    """Returns the unfolded leaves, given the split vectors of `folded_list`."""
    leaves = [None] * self.num_leaves
    for i, leaf in zip(self.passthrough_leaf_indices, folded_list):
      leaves[i] = leaf
    for vector_leaves, leaf_indices in zip(
        leaves_by_vector, self.vector_leaf_indices
    ):
      for i, leaf in zip(leaf_indices, vector_leaves):
        leaves[i] = leaf
    return leaves


class _StateLayout:
  """How the 4 lists of the training state are folded."""

  def __init__(self, state: Sequence[Sequence[jax.Array]], max_leaf_size: int):
    self.list_layouts = [_ListLayout(leaves, max_leaf_size) for leaves in state]
    # `fold` and `unfold` pass only the small leaves and vectors through these:
    # passing the large leaves to `jit` outside a trace would copy them.
    self._concat_vectors_jit = jax.jit(
        self._concat_all_vectors,
        out_shardings=[layout.vector_shardings for layout in self.list_layouts],
    )
    self._split_vectors_jit = jax.jit(
        self._split_all_vectors,
        out_shardings=[
            layout.leaves_by_vector(layout.leaf_shardings)
            for layout in self.list_layouts
        ],
    )
    logging.info(
        "Folding the training state from %s to %s arrays.",
        [layout.num_leaves for layout in self.list_layouts],
        [layout.folded_list_length for layout in self.list_layouts],
    )

  def _concat_all_vectors(
      self, leaves_by_vector_by_list: Sequence[Sequence[Sequence[jax.Array]]]
  ) -> list[list[jax.Array]]:
    return [
        layout.concat_vectors(leaves_by_vector)
        for layout, leaves_by_vector in zip(
            self.list_layouts, leaves_by_vector_by_list
        )
    ]

  def _split_all_vectors(
      self, vectors_by_list: Sequence[Sequence[jax.Array]]
  ) -> list[list[Any]]:
    return [
        layout.split_vectors(vectors)
        for layout, vectors in zip(self.list_layouts, vectors_by_list)
    ]

  def fold(
      self, state: Sequence[Sequence[jax.Array]]
  ) -> tuple[list[jax.Array], ...]:
    """Returns `state` with its small replicated leaves folded into vectors."""
    leaves_by_vector_by_list = [
        layout.leaves_by_vector(leaves)
        for layout, leaves in zip(self.list_layouts, state)
    ]
    vectors_by_list = self._concat_vectors_jit(leaves_by_vector_by_list)
    return tuple(
        layout.fold_list(leaves, vectors)
        for layout, leaves, vectors in zip(
            self.list_layouts, state, vectors_by_list
        )
    )

  def unfold(
      self, folded: Sequence[Sequence[jax.Array]]
  ) -> tuple[list[jax.Array], ...]:
    """Returns the unfolded `state` of `folded`."""
    vectors_by_list = [
        layout.vectors_in_folded_list(folded_list)
        for layout, folded_list in zip(self.list_layouts, folded)
    ]
    leaves_by_vector_by_list = self._split_vectors_jit(vectors_by_list)
    return tuple(
        layout.unfold_list(folded_list, leaves_by_vector)
        for layout, folded_list, leaves_by_vector in zip(
            self.list_layouts, folded, leaves_by_vector_by_list
        )
    )

  def fold_shardings(
      self, shardings: Sequence[Sequence[jax.sharding.Sharding]]
  ) -> tuple[list[jax.sharding.Sharding], ...]:
    """Returns the shardings of the folded state."""
    return tuple(
        layout.fold_list(list_shardings, layout.vector_shardings)
        for layout, list_shardings in zip(self.list_layouts, shardings)
    )


class _Folding:
  """The folded train step of a model and the state conversions around it."""

  def __init__(self, model: keras.Model, max_leaf_size: int):
    self._model = model
    self._max_leaf_size = max_leaf_size
    self._layout: _StateLayout | None = None
    self.original_train_step = model.train_step
    self.original_make_train_function = model.make_train_function
    self.original_state_sync = model.state_sync
    # pylint: disable=protected-access
    self.original_get_jax_state = model._get_jax_state
    self.original_get_state_sharding_spec = model._get_state_sharding_spec
    # pylint: enable=protected-access

  def _get_layout(self, state: Any = None) -> _StateLayout:
    """Returns the layout, built on first use from the unfolded `state`."""
    if self._layout is None:
      if state is None:
        state = self.original_get_jax_state(
            trainable_variables=True,
            non_trainable_variables=True,
            optimizer_variables=True,
            metrics_variables=True,
        )
      self._layout = _StateLayout(state, self._max_leaf_size)
    return self._layout

  def folded_train_step(
      self, state: Sequence[Sequence[jax.Array]], data: Any
  ) -> tuple[Any, tuple[list[jax.Array], ...]]:
    layout = self._get_layout()
    logs, state = self.original_train_step(layout.unfold(state), data)
    return logs, layout.fold(state)

  def folded_make_train_function(self, force: bool = False) -> None:
    """Builds `model.train_function` with folded state shardings."""
    if self._model.train_function is not None and not force:
      return
    # Only the train function sees folded shardings; `evaluate` and
    # `validation_data` keep the unfolded ones.
    with _override_attribute(
        self._model,
        "_get_state_sharding_spec",
        self.folded_get_state_sharding_spec,
    ):
      self.original_make_train_function(force=force)

  def folded_get_jax_state(self, **kwargs: bool) -> Any:
    state = self.original_get_jax_state(**kwargs)
    # `fit` and `train_on_batch` ask for all 4 lists. `evaluate`, `predict` and
    # the checkpoint helpers ask for fewer and keep getting unfolded state.
    if not all(kwargs.get(key, False) for key in _STATE_KEYS):
      return state
    return self._get_layout(state).fold(state)

  def folded_get_state_sharding_spec(
      self,
  ) -> tuple[list[jax.sharding.Sharding], ...]:
    unfolded_shardings = self.original_get_state_sharding_spec()
    return self._get_layout().fold_shardings(unfolded_shardings)

  def folded_state_sync(self) -> None:
    """Unfolds `_jax_state` before syncing back to the model variables."""
    model = self._model
    jax_state = getattr(model, "_jax_state", None)
    if (
        jax_state
        and not model._jax_state_synced  # pylint: disable=protected-access
        and all(jax_state.get(key) is not None for key in _STATE_KEYS)
    ):
      state = [jax_state[key] for key in _STATE_KEYS]
      unfolded = self._get_layout().unfold(state)
      model._jax_state = dict(  # pylint: disable=protected-access
          zip(_STATE_KEYS, unfolded)
      )
    self.original_state_sync()


@contextlib.contextmanager
def model_state_folding_scope(
    model: keras.Model, max_leaf_size: int
) -> Iterator[None]:
  """Makes `model.fit` pass small replicated arrays of the state folded.

  Args:
    model: A Keras model on the JAX backend.
    max_leaf_size: Arrays with at most this many elements are folded. `0` leaves
      `model` untouched.

  Yields:
    None.
  """
  if max_leaf_size == 0:
    yield
    return
  folding = _Folding(model, max_leaf_size)
  model.train_function = None
  try:
    with (
        _override_attribute(model, "train_step", folding.folded_train_step),
        _override_attribute(
            model, "make_train_function", folding.folded_make_train_function
        ),
        _override_attribute(model, "state_sync", folding.folded_state_sync),
        _override_attribute(
            model, "_get_jax_state", folding.folded_get_jax_state
        ),
    ):
      yield
  finally:
    model.train_function = None

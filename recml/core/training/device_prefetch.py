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
"""Background-thread input prefetch for the Keras JAX training loop.

Keras' `JAXEpochIterator._one_batch_ahead_iterator` pulls the next batch
(`IteratorGetNext` plus `jax.device_put`) on the main Python thread, between
two `train_step` dispatches, and keeps only one batch in flight. Any stall in
the input pipeline (e.g. a slow `IteratorGetNext`) then leaves the accelerator
idle, and with many hosts in lockstep one host's stall delays every chip.

`threaded_prefetch` and `PrefetchingTrainFunction` move the pull and the
host-to-device copy onto one worker thread that keeps up to `depth`
device-resident batches ahead, so a stall on the input side only costs
something once those batches run out. `depth=1` keeps as many batches alive
as Keras does (2: one in the step, one ahead).
"""

import collections
from collections.abc import Iterator
import concurrent.futures
import contextlib
import functools
import inspect
from typing import Any, Callable, TypeVar

import keras

T = TypeVar("T")


def threaded_prefetch(iterator: Iterator[T], depth: int) -> Iterator[T]:
  """Yields `iterator`'s items, pulled up to `depth` ahead on one thread.

  One worker thread keeps `depth` pulls in flight, so at most `depth`
  produced-but-unconsumed items exist. With a single worker the items keep
  their order, and an exception raised by `iterator` is re-raised in the
  consumer at the position it occurred. The worker sees the caller's Keras
  distribution, which Keras keeps in a thread-local. Closing the returned
  generator cancels the queued pulls and waits for the one in progress.

  Args:
    iterator: Source iterator; only ever advanced by the worker thread.
    depth: Maximum number of produced-but-unconsumed items. Must be >= 1.

  Yields:
    The items of `iterator`, in order.

  Raises:
    ValueError: If `depth < 1`.
  """
  if depth < 1:
    raise ValueError(f"depth must be >= 1, got {depth}.")
  executor = concurrent.futures.ThreadPoolExecutor(
      max_workers=1,
      thread_name_prefix="device_prefetch",
      initializer=keras.distribution.set_distribution,
      initargs=(keras.distribution.distribution(),),
  )
  try:
    pulls = collections.deque(
        executor.submit(next, iterator) for _ in range(depth)
    )
    while True:
      try:
        item = pulls.popleft().result()
      except StopIteration:
        return
      pulls.append(executor.submit(next, iterator))
      yield item
  finally:
    executor.shutdown(cancel_futures=True)


def cache_distribution_data_layouts(
    distribution: (
        keras.distribution.DataParallel
        | keras.distribution.ModelParallel
        | None
    ),
) -> None:
  """Memoizes `distribution.get_data_layout` by `tuple(data_shape)`.

  In Keras 3, `TensorLayout.backend_layout` already caches its constructed
  `jax.sharding.NamedSharding` on the `TensorLayout` instance, but
  `DataParallel.get_data_layout` and `ModelParallel.get_data_layout` allocate a
  new `TensorLayout` on every call (~150 leaves per batch in GUM). Memoizing the
  public `distribution.get_data_layout` method by shape tuple reuses the same
  `TensorLayout` (and its cached `NamedSharding`) across leaves and steps.

  The memoized method stays installed on `distribution` for its lifetime; this
  is safe because the built-in layouts depend only on the shape.

  Args:
    distribution: A `keras.distribution.DataParallel` or
      `keras.distribution.ModelParallel` instance, or `None`.
  """
  if distribution is None or getattr(
      distribution, "_data_layout_cached", False
  ):
    return
  cached = functools.cache(distribution.get_data_layout)
  distribution.get_data_layout = lambda data_shape: cached(tuple(data_shape))
  setattr(distribution, "_data_layout_cached", True)


def _unwrap_one_batch_ahead_iterator(iterator: Iterator[T]) -> Iterator[T]:
  """Unwraps Keras' unstarted `_one_batch_ahead_iterator` generator if present.

  On non-CPU backends, `JAXEpochIterator._get_iterator()` wraps its
  `_distributed_iterator()` in `_one_batch_ahead_iterator(iterator)`. Because
  that generator has not yet been advanced when passed to `train_function`,
  unwrapping its inner `iterator` avoids holding an extra device batch inside
  `_one_batch_ahead_iterator`'s local frame while `threaded_prefetch` holds
  `depth` batches on its worker thread.

  Args:
    iterator: Iterator passed to `model.train_function(state, iterator)`.

  Returns:
    The inner `_distributed_iterator` if `iterator` is an unstarted
    `_one_batch_ahead_iterator` generator, otherwise `iterator` unchanged.
  """
  if (
      inspect.isgenerator(iterator)
      and inspect.getgeneratorstate(iterator) == inspect.GEN_CREATED
      and getattr(iterator.gi_code, "co_name", None)
      == "_one_batch_ahead_iterator"
      and iterator.gi_frame is not None
  ):
    inner = iterator.gi_frame.f_locals.get("iterator")
    if inner is not None:
      iterator.close()
      return inner
  return iterator


class PrefetchingTrainFunction:
  """Wraps `model.train_function` to prefetch device batches on a thread.

  In `JAXTrainer.fit()`, `for begin_step, end_step, iterator in epoch_iterator:`
  yields the active epoch iterator without advancing it; the only consumer of
  `iterator` is `model.train_function(state, iterator)`. Wrapping
  `model.train_function` moves batch pulls (`IteratorGetNext`) and
  host-to-device transfers (`_distribute_data`) onto `threaded_prefetch` using
  public `keras.distribution` APIs.
  """

  def __init__(
      self,
      train_function: Callable[[Any, Iterator[Any]], Any],
      depth: int,
  ):
    """Initializes the wrapper.

    Args:
      train_function: The compiled `model.train_function(state, iterator)`
        callable to wrap.
      depth: Maximum number of prefetched device batches kept in flight. Must
        be >= 1.

    Raises:
      ValueError: If `depth < 1`.
    """
    if depth < 1:
      raise ValueError(f"depth must be >= 1, got {depth}.")
    self._train_function = train_function
    self._depth = depth
    self._source_iterator: Iterator[Any] | None = None
    self._prefetched_iterator: Iterator[Any] | None = None

  def _start_prefetch(self, iterator: Iterator[Any]) -> Iterator[Any]:
    """Starts a prefetch thread for `iterator`."""
    self.close()
    self._source_iterator = iterator
    self._prefetched_iterator = threaded_prefetch(
        _unwrap_one_batch_ahead_iterator(iterator), self._depth
    )
    return self._prefetched_iterator

  def __call__(self, state: Any, iterator: Iterator[Any]) -> Any:
    """Runs one training step or `steps_per_execution` block.

    Args:
      state: JAX training state passed by `JAXTrainer.fit`.
      iterator: Epoch batch iterator passed by `JAXTrainer.fit`.

    Returns:
      The step outputs returned by the wrapped `train_function`.
    """
    if (
        iterator is not self._source_iterator
        or self._prefetched_iterator is None
    ):
      prefetched = self._start_prefetch(iterator)
    else:
      prefetched = self._prefetched_iterator
    return self._train_function(state, prefetched)

  def close(self) -> None:
    """Stops the active prefetch generator and drops its pending batches."""
    if self._prefetched_iterator is not None:
      close_fn = getattr(self._prefetched_iterator, "close", None)
      if callable(close_fn):
        close_fn()
      self._prefetched_iterator = None
    self._source_iterator = None


@contextlib.contextmanager
def model_prefetch_scope(model: Any, depth: int) -> Iterator[None]:
  """Configures `model` to prefetch up to `depth` device batches on a thread.

  Wraps `model.make_train_function` on the `model` instance so that the compiled
  `model.train_function` pulls batches from `threaded_prefetch(iterator, depth)`
  and caches `distribution.get_data_layout` results by shape (see
  `cache_distribution_data_layouts`; the cache outlives the context). When
  exiting the context, any active prefetch generator is closed and `model`'s
  original `make_train_function` / `train_function` attributes are restored.

  Args:
    model: A `keras.Model` instance.
    depth: Number of device batches kept in flight on the worker thread. `0`
      leaves `model` untouched.

  Yields:
    None.

  Raises:
    ValueError: If `depth < 0`.
  """
  if depth < 0:
    raise ValueError(f"depth must be >= 0, got {depth}.")
  if depth == 0:
    yield
    return

  cache_distribution_data_layouts(keras.distribution.distribution())
  had_instance_make = "make_train_function" in getattr(model, "__dict__", {})
  orig_make_train_function = model.make_train_function
  orig_train_function = getattr(model, "train_function", None)
  active_wrappers: list[PrefetchingTrainFunction] = []

  def _make_prefetching_train_function(*args: Any, **kwargs: Any) -> None:
    # Like Keras' own `make_train_function`, this sets `model.train_function`
    # and returns None.
    orig_make_train_function(*args, **kwargs)
    current_fn = model.train_function
    if current_fn is not None and not isinstance(
        current_fn, PrefetchingTrainFunction
    ):
      wrapper = PrefetchingTrainFunction(current_fn, depth=depth)
      active_wrappers.append(wrapper)
      model.train_function = wrapper

  # Patch `make_train_function` rather than calling it before `yield`:
  # `JAXTrainer.fit()` builds lazy model/optimizer/metric variables first, and
  # `make_train_function()` needs those variables built to compute shardings.
  model.make_train_function = _make_prefetching_train_function
  try:
    yield
  finally:
    for wrapper in active_wrappers:
      wrapper.close()
    if had_instance_make:
      model.make_train_function = orig_make_train_function
    elif "make_train_function" in getattr(model, "__dict__", {}):
      delattr(model, "make_train_function")
    model.train_function = orig_train_function

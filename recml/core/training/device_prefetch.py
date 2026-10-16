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
host-to-device copy onto a daemon thread that keeps up to `depth`
device-resident batches queued, so a stall on the input side only costs
something once the queue runs dry.
"""

from collections.abc import Iterator
import contextlib
import inspect
import queue
import threading
from typing import Any, Callable, TypeVar

T = TypeVar("T")

_POLL_SECONDS = 0.1


class _Done:
  pass


class _Failure:

  def __init__(self, exc: BaseException):
    self.exc = exc


def threaded_prefetch(
    iterator: Iterator[T],
    depth: int,
    thread_initializer: Callable[[], None] | None = None,
) -> Iterator[T]:
  """Yields `iterator`'s items, produced up to `depth` ahead on a thread.

  Order is preserved. An exception raised by `iterator` is re-raised in the
  consumer at the position it occurred. Closing the returned generator stops
  the producer thread and immediately drains any queued items from the buffer.

  Args:
    iterator: Source iterator; only ever advanced by the producer thread.
    depth: Maximum number of produced-but-unconsumed items. Must be >= 1.
    thread_initializer: Optional callable run on the producer thread before the
      first item is pulled, e.g. to copy thread-local state from the caller's
      thread. Its exceptions are re-raised in the consumer.

  Yields:
    The items of `iterator`, in order.

  Raises:
    ValueError: If `depth < 1`.
  """
  if depth < 1:
    raise ValueError(f"depth must be >= 1, got {depth}.")
  buf: queue.Queue[Any] = queue.Queue(maxsize=depth)
  stop = threading.Event()

  def put(item: Any) -> bool:
    while not stop.is_set():
      try:
        buf.put(item, timeout=_POLL_SECONDS)
        return True
      except queue.Full:
        continue
    return False

  def produce() -> None:
    try:
      if thread_initializer is not None:
        thread_initializer()
      for item in iterator:
        if stop.is_set() or not put(item):
          return
      put(_Done())
    except BaseException as exc:  # pylint: disable=broad-exception-caught
      put(_Failure(exc))
    finally:
      close_fn = getattr(iterator, "close", None)
      if callable(close_fn):
        try:
          close_fn()
        except BaseException:  # pylint: disable=broad-exception-caught
          pass

  thread = threading.Thread(target=produce, name="device_prefetch", daemon=True)
  thread.start()
  try:
    while True:
      item = buf.get()
      if isinstance(item, _Done):
        return
      if isinstance(item, _Failure):
        raise item.exc
      yield item
  finally:
    stop.set()
    while True:
      try:
        buf.get_nowait()
      except queue.Empty:
        break


def cache_distribution_data_layouts(distribution: Any) -> None:
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
    distribution: A `keras.distribution.Distribution` instance, or `None`.
  """
  if distribution is None or getattr(
      distribution, "_data_layout_cached", False
  ):
    return
  orig_get_data_layout = getattr(distribution, "get_data_layout", None)
  if not callable(orig_get_data_layout):
    return

  layout_cache: dict[tuple[int, ...], Any] = {}

  def _cached_get_data_layout(data_shape: Any) -> Any:
    key = tuple(data_shape)
    layout = layout_cache.get(key)
    if layout is None and key not in layout_cache:
      layout = orig_get_data_layout(data_shape)
      layout_cache[key] = layout
    return layout

  distribution.get_data_layout = _cached_get_data_layout
  setattr(distribution, "_data_layout_cached", True)


def _unwrap_one_batch_ahead_iterator(iterator: Iterator[T]) -> Iterator[T]:
  """Unwraps Keras' unstarted `_one_batch_ahead_iterator` generator if present.

  On non-CPU backends, `JAXEpochIterator._get_iterator()` wraps its
  `_distributed_iterator()` in `_one_batch_ahead_iterator(iterator)`. Because
  that generator has not yet been advanced when passed to `train_function`,
  unwrapping its inner `iterator` avoids holding an extra device batch inside
  `_one_batch_ahead_iterator`'s local frame while `threaded_prefetch` buffers
  `depth` batches on the producer thread.

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
  """Wraps `model.train_function` to prefetch device batches on a daemon thread.

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
    if depth < 1:
      raise ValueError(f"depth must be >= 1, got {depth}.")
    self._train_function = train_function
    self._depth = depth
    self._source_iterator: Iterator[Any] | None = None
    self._prefetched_iterator: Iterator[Any] | None = None

  def _start_prefetch(self, iterator: Iterator[Any]) -> Iterator[Any]:
    """Starts a daemon prefetch thread for `iterator`."""
    import keras  # pylint: disable=g-import-not-at-top

    self.close()
    self._source_iterator = iterator
    target_iterator = _unwrap_one_batch_ahead_iterator(iterator)
    distribution = keras.distribution.distribution()
    cache_distribution_data_layouts(distribution)

    def _init() -> None:
      if distribution is not None:
        keras.distribution.set_distribution(distribution)

    self._prefetched_iterator = threaded_prefetch(
        target_iterator, self._depth, thread_initializer=_init
    )
    return self._prefetched_iterator

  def __call__(self, state: Any, iterator: Iterator[Any]) -> Any:
    if (
        iterator is not self._source_iterator
        or self._prefetched_iterator is None
    ):
      prefetched = self._start_prefetch(iterator)
    else:
      prefetched = self._prefetched_iterator
    return self._train_function(state, prefetched)

  def close(self) -> None:
    """Stops the active prefetch generator and drains queued device batches."""
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
    depth: Number of device batches kept in flight on the producer thread. `0`
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

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
import inspect
import threading
import time
from typing import Any
import weakref

from absl.testing import absltest
import keras
import numpy as np
from recml.core.training import device_prefetch


class _BatchBox:
  """Weak-referenceable stand-in for a device-resident batch PyTree."""


class ThreadedPrefetchTest(absltest.TestCase):

  def test_preserves_order_and_ends(self):
    self.assertEqual(
        list(device_prefetch.threaded_prefetch(iter(range(50)), depth=3)),
        list(range(50)),
    )

  def test_empty_source(self):
    self.assertEqual(
        list(device_prefetch.threaded_prefetch(iter([]), depth=2)), []
    )

  def test_source_runs_on_another_thread(self):
    main = threading.get_ident()
    seen = []

    def src():
      for i in range(3):
        seen.append(threading.get_ident())
        yield i

    list(device_prefetch.threaded_prefetch(src(), depth=1))
    self.assertTrue(all(t != main for t in seen))

  def test_error_reraised_after_earlier_items(self):
    def src():
      yield 1
      yield 2
      raise RuntimeError("boom")

    it = device_prefetch.threaded_prefetch(src(), depth=2)
    self.assertEqual(next(it), 1)
    self.assertEqual(next(it), 2)
    with self.assertRaisesRegex(RuntimeError, "boom"):
      next(it)

  def test_depth_bounds_lookahead(self):
    produced = []

    def src():
      for i in range(100):
        produced.append(i)
        yield i

    it = device_prefetch.threaded_prefetch(src(), depth=2)
    self.assertEqual(next(it), 0)
    time.sleep(0.3)
    # 1 consumed + 2 pulls in flight, the same as Keras at depth=1 plus 1.
    self.assertLessEqual(len(produced), 3)
    it.close()

  def test_close_stops_producer(self):
    def src():
      i = 0
      while True:
        yield i
        i += 1

    before = threading.active_count()
    it = device_prefetch.threaded_prefetch(src(), depth=1)
    next(it)
    it.close()
    deadline = time.time() + 5
    while threading.active_count() > before and time.time() < deadline:
      time.sleep(0.05)
    self.assertLessEqual(threading.active_count(), before)

  def test_close_drains_queued_items_immediately(self):
    refs = []
    ready = threading.Event()

    def src():
      for i in range(10):
        box = _BatchBox()
        refs.append(weakref.ref(box))
        if i == 2:
          ready.set()
        yield box

    it = device_prefetch.threaded_prefetch(src(), depth=2)
    first = next(it)
    del first
    self.assertTrue(ready.wait(timeout=5.0))
    time.sleep(0.05)
    # Items 1 and 2 are done pulls; closing `it` must release them.
    it.close()
    self.assertIsNone(refs[1]())
    self.assertIsNone(refs[2]())

  def test_prefetches_while_consumer_is_busy(self):
    # Deterministic check that the worker prefetches `depth` items while the
    # consumer is mid-step, without wall-clock sleep thresholds.
    prefetched = threading.Event()
    allow_finish = threading.Event()
    produced = []

    def src():
      for i in range(6):
        produced.append(i)
        if i == 3:
          prefetched.set()
          allow_finish.wait(timeout=5.0)
        yield i

    it = device_prefetch.threaded_prefetch(src(), depth=3)
    try:
      self.assertEqual(next(it), 0)
      # While the consumer holds item 0 ("in train_step"), the worker thread
      # pulls items 1, 2, 3 in the background and sets `prefetched`.
      self.assertTrue(prefetched.wait(timeout=5.0))
      self.assertEqual(produced, [0, 1, 2, 3])
      allow_finish.set()
      self.assertEqual(list(it), [1, 2, 3, 4, 5])
    finally:
      allow_finish.set()
      it.close()

  def test_rejects_bad_depth(self):
    with self.assertRaisesRegex(ValueError, "depth must be >= 1"):
      list(device_prefetch.threaded_prefetch(iter([1]), depth=0))
    with self.assertRaisesRegex(ValueError, "depth must be >= 1"):
      device_prefetch.PrefetchingTrainFunction(
          lambda state, it: (next(it), state), depth=0
      )
    with self.assertRaisesRegex(ValueError, "depth must be >= 0"):
      with device_prefetch.model_prefetch_scope(None, -1):
        pass

  def test_prefetching_train_function_pulls_across_steps_and_unwraps_generator(
      self,
  ):
    jax_epoch_iterator = keras.src.backend.jax.trainer.JAXEpochIterator
    seen_threads = []

    def raw_source():
      for x in [7, 8, 9]:
        seen_threads.append(threading.get_ident())
        yield x

    wrapped_gen = jax_epoch_iterator._one_batch_ahead_iterator(
        None, raw_source()
    )
    fn = device_prefetch.PrefetchingTrainFunction(
        lambda state, it: (next(it), state + 1), depth=2
    )
    try:
      out = []
      state = 0
      for _ in range(3):
        val, state = fn(state, wrapped_gen)
        out.append(val)
        # The unstarted Keras generator is closed and its inner iterator used.
        self.assertEqual(
            inspect.getgeneratorstate(wrapped_gen), inspect.GEN_CLOSED
        )
      self.assertEqual(out, [7, 8, 9])
      self.assertEqual(state, 3)
      self.assertTrue(all(t != threading.get_ident() for t in seen_threads))
    finally:
      fn.close()

  def test_unwrap_leaves_other_iterators_unchanged(self):
    def _one_batch_ahead_iterator(iterator):
      yield from iterator

    def other_generator(iterator):
      yield from iterator

    started = _one_batch_ahead_iterator(iter([1, 2]))
    next(started)
    plain = iter([1, 2])
    other = other_generator(iter([1, 2]))
    for it in (started, plain, other):
      self.assertIs(device_prefetch._unwrap_one_batch_ahead_iterator(it), it)
    self.assertEqual(next(other), 1)

  def test_prefetching_train_function_restarts_on_new_iterator(self):
    fn = device_prefetch.PrefetchingTrainFunction(
        lambda state, it: (next(it), state), depth=2
    )
    try:
      self.assertEqual(fn("s", iter([1, 2, 3])), (1, "s"))
      old_prefetched = fn._prefetched_iterator
      self.assertEqual(fn("s", iter([10, 20])), (10, "s"))
      self.assertIsNot(fn._prefetched_iterator, old_prefetched)
      self.assertEqual(
          inspect.getgeneratorstate(old_prefetched), inspect.GEN_CLOSED
      )
    finally:
      fn.close()

  def test_model_prefetch_scope_wraps_and_restores_train_function(self):

    class _DummyModel:
      """Mimics Keras' `make_train_function` contract (returns None)."""

      def __init__(self):
        self.train_function: Any = None

      def make_train_function(self, force=False):
        if self.train_function is not None and not force:
          return
        self.train_function = lambda state, it: (next(it), state)

    model = _DummyModel()
    orig_make = model.make_train_function
    with device_prefetch.model_prefetch_scope(model, depth=0):
      self.assertEqual(model.make_train_function, orig_make)

    with device_prefetch.model_prefetch_scope(model, depth=2):
      self.assertNotEqual(model.make_train_function, orig_make)
      self.assertIsNone(model.make_train_function())
      self.assertIsInstance(
          model.train_function, device_prefetch.PrefetchingTrainFunction
      )
      src = iter([4, 5])
      self.assertEqual(model.train_function("s", src), (4, "s"))
      self.assertEqual(model.train_function("s", src), (5, "s"))

    self.assertEqual(model.make_train_function, orig_make)
    self.assertNotIn("make_train_function", model.__dict__)
    self.assertIsNone(model.train_function)

  def test_model_prefetch_scope_with_a_real_keras_model(self):
    dist = keras.distribution.DataParallel()
    if hasattr(dist, "_auto_shard_dataset"):
      setattr(dist, "_auto_shard_dataset", False)
    with dist.scope():
      # Use an unbuilt Sequential model so `fit()` builds variables lazily
      # before calling the patched `make_train_function` under `DataParallel`.
      model = keras.Sequential([keras.layers.Dense(1)])
      model.compile(optimizer="sgd", loss="mse")
      main = threading.get_ident()
      pull_threads = []

      def batches():
        while True:
          pull_threads.append(threading.get_ident())
          yield np.ones((2, 3), np.float32), np.ones((2, 1), np.float32)

      with device_prefetch.model_prefetch_scope(model, depth=2):
        history = model.fit(batches(), epochs=2, steps_per_epoch=3, verbose=0)
        self.assertIsInstance(
            model.train_function, device_prefetch.PrefetchingTrainFunction
        )

    self.assertTrue(getattr(dist, "_data_layout_cached", False))
    self.assertLen(history.history["loss"], 2)
    # Keras peeks at the first batches on the main thread to infer the specs;
    # the training pulls run on the prefetch thread.
    self.assertTrue(any(t != main for t in pull_threads))
    self.assertNotIn("make_train_function", model.__dict__)
    self.assertIsNone(model.train_function)

  def test_keras_distribution_visible_on_producer_thread(self):
    # Keras stores the distribution in a threading.local; without propagation
    # `_distribute_data` on the producer thread would see None.
    dist = keras.distribution.DataParallel()
    seen = []

    def src():
      seen.append(keras.distribution.distribution())
      yield 1

    fn = device_prefetch.PrefetchingTrainFunction(
        lambda state, it: (next(it), state), depth=2
    )
    try:
      keras.distribution.set_distribution(dist)
      self.assertEqual(fn("s", src()), (1, "s"))
    finally:
      fn.close()
      keras.distribution.set_distribution(None)
    self.assertEqual(seen, [dist])

  def test_cache_distribution_data_layouts_caches_by_shape(self):
    dist = keras.distribution.DataParallel()
    orig_get_data_layout = dist.get_data_layout
    layout_calls = []

    def counting_get_data_layout(data_shape):
      layout_calls.append(tuple(data_shape))
      return orig_get_data_layout(data_shape)

    dist.get_data_layout = counting_get_data_layout
    device_prefetch.cache_distribution_data_layouts(dist)

    layout_a1 = dist.get_data_layout((2, 4))
    layout_b1 = dist.get_data_layout((2, 4))
    layout_c1 = dist.get_data_layout((2, 8, 3))
    layout_a2 = dist.get_data_layout((2, 4))
    layout_c2 = dist.get_data_layout((2, 8, 3))

    self.assertIs(layout_a1, layout_b1)
    self.assertIs(layout_a1, layout_a2)
    self.assertIs(layout_c1, layout_c2)
    self.assertIs(layout_a1.backend_layout, layout_b1.backend_layout)
    # Two distinct shapes ((2, 4) and (2, 8, 3)) across 5 lookups ->
    # underlying get_data_layout is called only twice.
    self.assertEqual(layout_calls, [(2, 4), (2, 8, 3)])


if __name__ == "__main__":
  absltest.main()

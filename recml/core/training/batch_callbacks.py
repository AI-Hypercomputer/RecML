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
"""Keras callback plumbing that keeps per-step host work off the train loop.

`JAXTrainer.fit()` calls `callbacks.on_train_batch_end(step, logs)` on every
step, with `logs` holding every metric as a device-resident `jax.Array`. The
default `keras.callbacks.CallbackList` then

  1. attaches `ProgbarLogger` (the recml trainer does not pass `verbose=0`),
     which formats every metric to stdout on every step, and
  2. runs `python_utils.pythonify_logs(logs)` on a thread pool, which copies
     every metric to the host (`float(np.asarray(v))`) on every step.

With hundreds of metrics this costs over 100 ms of host time per step and
starves the dispatch thread. `EpochSummaryCallback` also calls
`summary.scalar` for every key on every step even with `update_freq="epoch"`;
no summary writer is installed during batches, so these calls record nothing.

`ZeroSyncCallbackList` always drops the progress bar, and skips the per-step
conversion when no active callback needs per-step logs.
`EpochSummaryNoBatchHooks` is an `EpochSummaryCallback` without batch hooks, so
it does not count as such a callback. Epoch-end logging is unchanged.
"""

from collections.abc import Sequence
from typing import Any

from absl import logging
import keras
from recml.core.utils import keras_utils


def _overrides_hook(cb: keras.callbacks.Callback, hook_name: str) -> bool:
  """Returns whether `cb` overrides `keras.callbacks.Callback.<hook_name>`."""
  base_hook = getattr(keras.callbacks.Callback, hook_name, None)
  if hook_name in getattr(cb, "__dict__", {}):
    method = cb.__dict__[hook_name]
    func = getattr(method, "__func__", method)
    return func is not base_hook
  return getattr(type(cb), hook_name, None) is not base_hook


class EpochSummaryNoBatchHooks(keras_utils.EpochSummaryCallback):
  """`EpochSummaryCallback` with the base `Callback` no-op batch hooks.

  `EpochSummaryCallback` subclasses `keras.callbacks.TensorBoard` with
  `update_freq="epoch"`. `TensorBoard.on_train_batch_end` still calls
  `self.summary.scalar` for `batch_steps_per_second` and every key in `logs` on
  every step, but no default summary writer is installed during batches, so
  these calls record nothing. `EpochSummaryCallback.on_epoch_end` computes
  `steps_per_second` from state recorded in `on_epoch_begin`, so the batch
  hooks can be no-ops.
  """

  requires_batch_logs: bool = False
  on_train_batch_begin = keras.callbacks.Callback.on_train_batch_begin
  on_train_batch_end = keras.callbacks.Callback.on_train_batch_end

  @classmethod
  def from_base(
      cls, cbk: keras_utils.EpochSummaryCallback
  ) -> "EpochSummaryNoBatchHooks":
    """Returns a copy of `cbk` as an `EpochSummaryNoBatchHooks`.

    Args:
      cbk: The `EpochSummaryCallback` to copy.

    Returns:
      An `EpochSummaryNoBatchHooks` with the same settings as `cbk`.
    """
    # Forwards all `EpochSummaryCallback.__init__` arguments (`log_dir`,
    # `steps_per_epoch`, `write_steps_per_second`, `eval_subdir`, and
    # `enable_xm_measurements`); update if `EpochSummaryCallback.__init__`
    # gains new parameters.
    upgraded = cls(
        log_dir=cbk.log_dir,
        steps_per_epoch=cbk._steps_per_epoch,  # pylint: disable=protected-access
        write_steps_per_second=cbk.write_steps_per_second,
        eval_subdir=cbk._eval_subdir,  # pylint: disable=protected-access
    )
    return upgraded


def without_batch_hooks(
    callbacks: Sequence[keras.callbacks.Callback],
) -> list[keras.callbacks.Callback]:
  """Returns `callbacks` with `EpochSummaryCallback`s made batch-hook free.

  Args:
    callbacks: The training callbacks.

  Returns:
    A new list where each callback of exact type `EpochSummaryCallback` is
    replaced by an `EpochSummaryNoBatchHooks` copy; other callbacks are kept.
  """
  result = []
  for cbk in callbacks:
    # Exact type: `from_base` would drop a subclass's overrides and state.
    if type(cbk) is keras_utils.EpochSummaryCallback:  # pylint: disable=unidiomatic-typecheck
      result.append(EpochSummaryNoBatchHooks.from_base(cbk))
    else:
      result.append(cbk)
  return result


class ZeroSyncCallbackList(keras.callbacks.CallbackList):
  """`CallbackList` that skips `ProgbarLogger` and per-step `pythonify_logs`.

  The progress bar is always dropped (`verbose=0`). When no active batch
  callback requires `logs` (a callback opts out with
  `requires_batch_logs = False`), intermediate training steps do not convert
  `logs` to Python, and `on_train_batch_begin` / `on_train_batch_end` run
  synchronously on the main thread with `logs=None` (rather than `{}` so an
  opted-out callback that accidentally reads `logs` fails loudly with
  `TypeError`/`AttributeError`). `on_epoch_end` still gets the epoch logs, which
  `JAXTrainer.fit()` computes from the synced metric variables once per epoch. A
  callback that needs per-step logs keeps the default behaviour for the whole
  list.
  """

  def __init__(
      self,
      callbacks: Sequence[keras.callbacks.Callback] | None = None,
      add_history: bool = True,
      add_progbar: bool = False,
      model: keras.Model | None = None,
      **params: Any,
  ):
    """Initializes `ZeroSyncCallbackList`.

    Args:
      callbacks: Sequence of `keras.callbacks.Callback` instances.
      add_history: Whether to attach a `History` callback if not already
        present. Defaults to `True`.
      add_progbar: Accepted for `keras.callbacks.CallbackList` signature
        compatibility and ignored (`ProgbarLogger` is always disabled).
      model: The `keras.Model` being trained.
      **params: Additional parameters forwarded to `CallbackList` (`verbose` is
        always forced to `0` so `ProgbarLogger` is not attached).
    """
    del add_progbar
    params["verbose"] = 0
    super().__init__(
        callbacks=callbacks,
        add_history=add_history,
        add_progbar=False,
        model=model,
        **params,
    )
    self._active_train_batch_begin = [
        cb
        for cb in self.callbacks
        if _overrides_hook(cb, "on_train_batch_begin")
        or _overrides_hook(cb, "on_batch_begin")
    ]
    self._active_train_batch_end = [
        cb
        for cb in self.callbacks
        if _overrides_hook(cb, "on_train_batch_end")
        or _overrides_hook(cb, "on_batch_end")
    ]
    needs_logs = [
        type(cb).__name__
        for cb in (
            *self._active_train_batch_begin,
            *self._active_train_batch_end,
        )
        if getattr(cb, "requires_batch_logs", True)
    ]
    self._needs_batch_logs = bool(needs_logs)
    if self._needs_batch_logs:
      logging.info(
          "skip_batch_logs: per-step logs still converted for %s", needs_logs
      )
    else:
      self._async_train = False

  def on_train_batch_begin(
      self, batch: int, logs: dict[str, Any] | None = None
  ) -> None:
    if self._needs_batch_logs:
      super().on_train_batch_begin(batch, logs=logs)
      return
    for cb in self._active_train_batch_begin:
      cb.on_train_batch_begin(batch, logs=None)

  def on_train_batch_end(
      self, batch: int, logs: dict[str, Any] | None = None
  ) -> None:
    if self._needs_batch_logs:
      super().on_train_batch_end(batch, logs=logs)
      return
    for cb in self._active_train_batch_end:
      cb.on_train_batch_end(batch, logs=None)

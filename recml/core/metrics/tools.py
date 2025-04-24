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
"""Tools for RecML metrics."""

from collections.abc import Mapping
import concurrent.futures
import functools
import os
import re
from typing import Any

from absl import flags
from clu import metric_writers
import clu.metrics as clu_metrics
import jax
from recml.core.metrics import base_metrics


FLAGS = flags.FLAGS


class AsyncMultiWriter(metric_writers.AsyncMultiWriter):
  """A multi writer that logs to a summary writer and a logging writer."""

  def __init__(self, *, log_dir: str, name: str):
    summary_writer = metric_writers.SummaryWriter(
        os.fspath(os.path.join(log_dir, name))
    )
    writers = [summary_writer]

    super().__init__(writers)
    self._summary_writer = summary_writer

  @property
  def summary_writer(self) -> metric_writers.SummaryWriter:
    return self._summary_writer


class MetricAccumulator:
  """A utility for asynchronously accumulating metrics."""

  def __init__(self, writer: AsyncMultiWriter, max_workers: int = 1):
    if not isinstance(writer, AsyncMultiWriter):
      raise ValueError(
          "`summary_writer` must be an instance of AsyncMultiWriter, got"
          f" {type(writer)}."
      )

    self._writer = writer
    self._metrics: list[Mapping[str, clu_metrics.Metric]] = []
    self._scalar_log_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    )
    self._scalar_log_futures: list[concurrent.futures.Future[None]] = []

  def accumulate(
      self, metrics_accum: Mapping[str, clu_metrics.Metric], step: int
  ):
    """Asynchronously accumulates a set of metrics and logs scalars."""
    self._metrics.append(metrics_accum)

    scalar_metrics_accum = {
        k: v
        for k, v in metrics_accum.items()
        if isinstance(v, base_metrics.ScalarMetric)
    }

    self._scalar_log_futures.append(
        self._scalar_log_pool.submit(
            _localize_and_log_scalars,
            # We only want to log per-step scalars via the summary writer.
            # Logging per-step scalars via other writers can be expensive.
            self._writer.summary_writer,
            step,
            scalar_metrics_accum,
        )
    )

  def compute_and_log_scalars(
      self, step: int
  ) -> Mapping[str, base_metrics.Scalar]:
    """Computes the scalars from the accumulated metrics and logs them."""

    if not self._metrics:
      return {}

    for future in self._scalar_log_futures:
      future.result()

    self._scalar_log_futures.clear()

    metrics = functools.reduce(
        merge_metrics, [jax.tree.map(_localize, ms) for ms in self._metrics]
    )
    self._metrics.clear()
    scalars = compute_metrics(metrics)

    # Log only non-reported scalars but return all for tracking in checkpoints.
    non_reported_scalars = {
        k: v
        for k, v in scalars.items()
        if not isinstance(metrics[k], base_metrics.ScalarMetric)
    }
    self._writer.write_scalars(step, non_reported_scalars)
    self._writer.flush()

    return scalars


def compute_metrics(
    metrics: Mapping[str, clu_metrics.Metric | base_metrics.Metric],
) -> Mapping[str, base_metrics.Scalar]:
  """Collects the merged metrics and returns the computed scalars."""
  return {k: m.compute() for k, m in metrics.items()}


def merge_metrics(
    a: Mapping[str, clu_metrics.Metric | base_metrics.Metric],
    b: Mapping[str, clu_metrics.Metric | base_metrics.Metric],
) -> Mapping[str, clu_metrics.Metric]:
  """Merges two mappings of metrics."""
  merged_metrics = {}
  for k in [*a.keys(), *b.keys()]:
    if k in a and k in b:
      merged_metrics[k] = a[k].merge(b[k])
    elif k in a:
      merged_metrics[k] = a[k]
    elif k in b:
      merged_metrics[k] = b[k]
  return merged_metrics


def _localize(x: Any) -> Any:
  """Returns the localized data for an object."""
  x = jax.device_get(x)
  if isinstance(x, jax.Array) and not isinstance(x, jax.core.Tracer):
    return x.addressable_data(0)
  return x


def _localize_and_log_scalars(
    summary_writer: metric_writers.SummaryWriter,
    step: int,
    scalar_metrics: Mapping[str, base_metrics.ScalarMetric],
) -> None:
  """Localizes the metrics from device to host and logs scalars."""
  scalar_metrics = jax.tree.map(_localize, scalar_metrics)
  summary_writer.write_scalars(step, compute_metrics(scalar_metrics))

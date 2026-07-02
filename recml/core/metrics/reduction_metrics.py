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
"""Reduction metrics."""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import Any, Self

import jax
import jax.numpy as jnp
from recml.core.metrics import base_metrics


class ReductionMetric(base_metrics.Metric):
  """A base class for reduction metrics."""

  @classmethod
  def from_model_output(
      cls, values: jax.Array, weights: jax.Array | None = None, **_
  ) -> Self:
    raise NotImplementedError()

  @classmethod
  def from_fun(cls, fun: Callable[..., Any], **kwargs) -> type[Self]:
    """Returns a reduction metric class that is computed from a function."""
    base_cls = cls
    bound_kwargs = kwargs

    class _FromFun(cls):  # pyrefly: ignore[invalid-inheritance]
      """A reduction metric that is computed from a function."""

      @classmethod
      def from_model_output(cls, *args, **kwargs) -> ReductionMetric:
        if "weights" in kwargs:
          weights = kwargs.pop("weights")
        else:
          weights = None

        values = fun(*args, **bound_kwargs, **kwargs)

        return base_cls.from_model_output(values, weights)

    return _FromFun


class Sum(ReductionMetric):
  """Computes the sum of observations over multiple batches."""

  total: jax.Array

  @classmethod
  def from_model_output(
      cls, values: jax.Array, weights: jax.Array | None = None, **_
  ) -> Self:
    values = jnp.asarray(values, dtype=jnp.float32)
    if weights is not None:
      weights = jnp.asarray(weights, dtype=jnp.float32)
      values, weights = _maybe_reshape_or_broadcast(values, weights)
      total = jnp.sum(values * weights)
    else:
      total = jnp.sum(values)

    return cls(total=total)

  def merge(self, other: Self) -> Self:  # pytype: disable=signature-mismatch
    return type(self)(total=self.total + other.total)

  def compute(self) -> base_metrics.Scalar:
    return self.total


class Mean(ReductionMetric):
  """Computes the mean of observations over multiple batches.

  This is done by tracking a total and a count over multiple observations and
  aggregating their mean over multiple batches when `compute` is called.
  """

  total: jax.Array
  count: jax.Array

  @classmethod
  def from_model_output(
      cls, values: jax.Array, weights: jax.Array | None = None, **_
  ) -> Self:
    values = jnp.asarray(values, dtype=jnp.float32)
    if weights is not None:
      weights = jnp.asarray(weights, dtype=jnp.float32)
      values, weights = _maybe_reshape_or_broadcast(values, weights)
      total = jnp.sum(values * weights)
      count = jnp.sum(weights)
    elif values.ndim >= 1:
      total = jnp.sum(values)
      count = jnp.asarray(math.prod(values.shape), jnp.float32)
    else:
      total = values
      count = jnp.ones((), dtype=jnp.float32)

    return cls(total=total, count=count)

  def merge(self, other: Self) -> Self:  # pytype: disable=signature-mismatch
    return type(self)(
        total=self.total + other.total,
        count=self.count + other.count,
    )

  def compute(self) -> base_metrics.Scalar:
    return self.total / self.count


def mean(values: jax.Array, weights: jax.Array | None = None, **_) -> Mean:
  """Computes a mean metric from values and optional weights.

  The resulting metric instance is a reduction metric that will aggregate the
  mean of the values over multiple batches.

  The total and counts are computed as follows:
    weights = broadcast_to(weights, values.shape)
    total = sum(values * weights)
    count = sum(weights)

  Where the output of an aggregated metric is the total / count.

  Example usage:

  ```
  metrics = {
      # Reports the mean accuracy over multiple batches.
      'accuracy': mean(y_true == y_pred),
      # Reports the mean loss over multiple batches.
      'loss': mean(loss),
  }
  ```

  Args:
    values: The values to compute the mean over of shape [D1, ..., DN].
    weights: Optional weights to apply to the values. If provided, the shape of
      the weights must be broadcastable to the shape of the values. If not
      provided, all values will effectively have a weight of 1.0.
    **_: Unused keyword arguments.

  Returns:
    A mean metric accumulation.
  """
  return Mean.from_model_output(values, weights)


def sum(values: jax.Array, weights: jax.Array | None = None, **_) -> Sum:  # pylint: disable=redefined-builtin
  """Computes a sum metric from values and optional weights.

  The sum is computed as follows:
    weights = broadcast_to(weights, values.shape)
    total = sum(values * weights)

  Where total is the output of an aggregated metric.

  Example usage:

  ```
  metrics = {
      # Reports the total number of hits over multiple batches.
      'number_of_hits': sum(y_true == y_pred),
  }
  ```

  Args:
    values: The values to compute the sum over of shape [D1, ..., DN].
    weights: Optional weights to apply to the values. If provided, the shape of
      the weights must be broadcastable to the shape of the values. If not
      provided, all values will effectively have a weight of 1.0.
    **_: Unused keyword arguments.

  Returns:
    A sum metric accumulation.
  """
  return Sum.from_model_output(values, weights)


def _maybe_reshape_or_broadcast(
    values: jax.Array, weights: jax.Array
) -> tuple[jax.Array, jax.Array]:
  """Reshapes or broadcasts arrays to have the same shape or throws an error."""
  # Note that we broadcast the weights explicitly so that the sum of the weights
  # is not performed on the non-broadcasted array.
  if values.shape == weights.shape:
    return values, weights
  elif values.ndim == weights.ndim and all(
      v_d == w_d or w_d == 1 for v_d, w_d in zip(values.shape, weights.shape)
  ):
    return values, jnp.broadcast_to(weights, values.shape)
  elif values.ndim == weights.ndim:
    raise ValueError(
        f"Got incompatible shapes {values.shape} and {weights.shape}."
    )
  elif (
      values.ndim > weights.ndim
      and values.shape[: weights.ndim] == weights.shape
  ):
    weights = jax.lax.expand_dims(
        weights, list(range(weights.ndim, values.ndim))
    )
    return values, jnp.broadcast_to(weights, values.shape)
  elif (
      weights.ndim > values.ndim
      and weights.shape[: values.ndim] == values.shape
  ):
    values = jax.lax.expand_dims(values, list(range(values.ndim, weights.ndim)))
    return values, weights

  raise ValueError(
      "The arrays must have the same shape or the shape of one array must be"
      f" a broadcastable to the other. Got shapes: {values.shape} and"
      f" {weights.shape}."
  )

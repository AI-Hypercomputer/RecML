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
"""Functional metrics inspired by the CLU interface and Keras semantics."""

import abc
from collections.abc import Mapping, Sequence
import math
from typing import Self, dataclass_transform

import clu.metrics as clu_metrics
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

Scalar = float | Sequence[float] | Mapping[str, float] | jax.Array | np.ndarray

# TODO(aahil): Look into why pytype doesn't respect the Self type as a generic
# type. We should not be violating LSP.

# TODO(b/387463777): Consider removing the dependency on CLU metrics longer term
# since it's just an interface.
@dataclass_transform(field_specifiers=(struct.field,))
class Metric(abc.ABC, clu_metrics.Metric, struct.PyTreeNode):
  """PyTree node representing the state of a metric.

  Note: This class follows the same interface as CLU metrics and can be used
  interchangeably.

  There are a few suble differences between subclasses and standard CLU metrics:
  1. Inheriting from this automatically makes the metric a PyTree node.
  2. `mask` has been replaced by `weights` to be consistent with Keras.
  3. Subclasses do not implement methods apart from the ones listed below.
  4. The `localize` method is added to specify how to localize the metric from
     device to host.
  """

  @classmethod
  def from_model_output(cls, *args, **kwargs) -> Self:
    """Creates a metric from observations.

    Args:
      *args: Positional arguments to pass to the metric.
      **kwargs: Keyword arguments to pass to the metric.

    Returns:
      A new instance of the metric.

    NOTE: This metric is always called on the device and should therefore use
    only jax ops.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def merge(self, other: Self) -> Self:
    """Merges two metrics.

    Args:
      other: Another metric of the same type to merge with.

    Returns:
      A new instance of the same class that is the merge of the two.

    NOTE: This method is almost always called on the host which means that it
    should *never* call jax ops - this will implicitly move the state of the
    metric back to the device for computation. It is safest to rely on dunder
    methods via `+` / `-`.
    """

  @abc.abstractmethod
  def compute(self) -> Scalar:  # pyrefly: ignore[bad-override]
    """Computes the value of the metric.

    NOTE: This method is almost always called on the host which means that it
    should *never* call jax ops - this will implicitly move the state of the
    metric back to the device for computation. Use numpy ops instead.
    """

  def localize(self) -> Self:
    """Localizes the metric from device to host.

    Returns:
      A new instance of the same class that is localized, i.e. jax arrays on the
      metric are replaced by numpy arrays.
    """

    def _localize(x):
      x = jax.device_get(x)
      if isinstance(x, jax.Array) and not isinstance(x, jax.core.Tracer):
        return x.addressable_data(0)
      return x

    return jax.tree.map(_localize, self)


class ScalarMetric(Metric):
  """A metric for reporting scalar values without aggregation."""

  value: jax.Array

  @classmethod
  def from_model_output(cls, value: jax.Array | float) -> Self:
    if hasattr(value, "shape") and math.prod(value.shape) != 1:
      raise ValueError(
          f"Scalar metric values must be scalars. Got shape: {value.shape}"
          " instead."
      )
    return cls(value=jnp.squeeze(jnp.asarray(value, dtype=jnp.float32)))

  def merge(self, other: Self) -> Self:
    return other

  def compute(self) -> Scalar:
    return self.value


def scalar(value: float | jax.Array) -> ScalarMetric:
  """Creates a scalar metric from a scalar value at a specific step.

  This is useful for reporting batch metrics during training. When merged with
  other instances, effectively the last value observed is reported.

  Note that using this metric during evaluation will result in multiple values
  being reported for the same step, which is generally undesirable.

  Example usage:

  ```
  state = ...
  metrics = {
      "average_loss": mean(loss),
      "per_batch_loss": scalar(loss),
      "learning_rate": scalar(learning_rate),
  }
  ```

  Args:
    value: The scalar value to report.

  Returns:
    A scalar metric reporting the value.
  """
  return ScalarMetric.from_model_output(value)

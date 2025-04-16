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
"""Optax optimizer factories."""

from collections.abc import Callable
import dataclasses
import re
from typing import Any

import jax
import optax
from recml.core.utils import types


def _default_weight_decay_mask(params: optax.Params) -> optax.Params:
  """Default weight decay mask that only applies to non-1D parameters."""
  return jax.tree.map(lambda p: p.ndim > 1, params)


def _regex_mask(regex: str) -> Callable[[optax.Params], optax.Params]:
  """Returns a mask that applies to parameters matching a regex."""

  def _matches_regex(path: tuple[str, ...], _: Any) -> bool:
    key = '/'.join([jax.tree_util.keystr((k,), simple=True) for k in path])
    return re.fullmatch(regex, key) is not None

  def _mask(params: optax.Params) -> optax.Params:
    return jax.tree.map_with_path(_matches_regex, params)

  return _mask


class OptimizerFactory(types.Factory[optax.GradientTransformation]):
  """Standard optimizer factory for Optax optimizers.

  Attributes:
    learning_rate: The learning rate to use for the optimizer.
    scaling: The gradient scaling transformation to use during optimization.
      Defaults to identity.
    weight_decay: Optional weight decay to apply to variables during
      optimization. Defaults to None.
    grad_clip_norm: Optional gradient clipping norm to limit the maximum
      magnitude of the gradients during optimization. Defaults to None.
    weight_decay_mask: The weight decay mask to use when applying weight decay.
      Defaults applying weight decay to all non-1D parameters.
    freeze_mask: Optional mask to freeze parameters during optimization.
      Defaults to None.

  Example usage:

  ```
  sgd = OptimizerFactory(learning_rate=0.001).make()

  adamw = OptimizerFactory(
      learning_rate=1e-3,
      scale_transform=optax.scale_by_adam(),
      weight_decay=1e-7,
      grad_clip_norm=1.0,
  ).make()
  ```
  """

  learning_rate: optax.ScalarOrSchedule
  scaling: optax.GradientTransformation = dataclasses.field(
      default_factory=optax.identity
  )
  weight_decay: float | None = None
  grad_clip_norm: float | None = None
  weight_decay_mask: str | Callable[[optax.Params], optax.Params] = (
      _default_weight_decay_mask
  )
  freeze_mask: str | Callable[[optax.Params], optax.Params] | None = None

  def make(self) -> optax.GradientTransformation:
    if self.grad_clip_norm is not None:
      apply_clipping = optax.clip_by_global_norm(self.grad_clip_norm)
    else:
      apply_clipping = optax.identity()

    # Tags the learning rate as a stateful hyperparameter so it can be logged.
    lr_scaling = optax.inject_stateful_hyperparams(
        optax.scale_by_learning_rate
    )(learning_rate=self.learning_rate)

    if self.weight_decay is not None:
      if isinstance(self.weight_decay_mask, str):
        mask = _regex_mask(self.weight_decay_mask)
      else:
        mask = self.weight_decay_mask
      weight_decay = optax.add_decayed_weights(self.weight_decay, mask=mask)
    else:
      weight_decay = optax.identity()

    tx = optax.chain(*[
        apply_clipping,
        self.scaling,
        weight_decay,
        lr_scaling,
    ])

    if self.freeze_mask is not None:
      if isinstance(self.freeze_mask, str):
        mask = _regex_mask(self.freeze_mask)
      else:
        mask = self.freeze_mask

      def _param_labels(params: optax.Params) -> optax.Params:
        return jax.tree.map(
            lambda p: 'frozen' if mask(p) else 'trainable', params
        )

      tx = optax.multi_transform(
          transforms={'trainable': tx, 'frozen': optax.set_to_zero()},
          param_labels=_param_labels,
      )
    return tx


class AdamFactory(types.Factory[optax.GradientTransformation]):
  """Adam optimizer factory.

  Attributes:
    learning_rate: The learning rate to use for the optimizer.
    b1: The beta1 coefficient for the Adam optimizer. Defaults to 0.9.
    b2: The beta2 coefficient for the Adam optimizer. Defaults to 0.999.
    eps: The epsilon coefficient for the Adam optimizer. Defaults to 1e-8.
    weight_decay: Optional weight decay to apply to variables during
      optimization. Defaults to None.
    grad_clip_norm: Optional gradient clipping norm to limit the maximum
      magnitude of the gradients during optimization. Defaults to None.
    weight_decay_mask: The weight decay mask to use when applying weight decay.
      Defaults applying weight decay to all non-1D parameters.
    freeze_mask: Optional mask to freeze parameters during optimization.
      Defaults to None.

  Example usage:
  ```
  adam = AdamFactory(learning_rate=1e-3).make()

  adamw = AdamFactory(
      learning_rate=1e-3,
      weight_decay=1e-7,
      grad_clip_norm=1.0,
  ).make()
  ```
  """

  learning_rate: optax.ScalarOrSchedule
  b1: float = 0.9
  b2: float = 0.999
  eps: float = 1e-8
  weight_decay: float | None = None
  grad_clip_norm: float | None = None
  weight_decay_mask: str | Callable[[optax.Params], optax.Params] = (
      _default_weight_decay_mask
  )
  freeze_mask: str | Callable[[optax.Params], optax.Params] | None = None

  def make(self) -> optax.GradientTransformation:
    return OptimizerFactory(
        learning_rate=self.learning_rate,
        scaling=optax.scale_by_adam(b1=self.b1, b2=self.b2, eps=self.eps),
        weight_decay=self.weight_decay,
        grad_clip_norm=self.grad_clip_norm,
        weight_decay_mask=self.weight_decay_mask,
    ).make()


class AdagradFactory(types.Factory[optax.GradientTransformation]):
  """Adagrad optimizer factory.

  Attributes:
    learning_rate: The learning rate to use for the optimizer.
    initial_accumulator_value: The initial accumulator value for the Adagrad
      optimizer. Defaults to 0.1.
    eps: The epsilon coefficient for the Adagrad optimizer. Defaults to 1e-7.
    grad_clip_norm: Optional gradient clipping norm to limit the maximum
      magnitude of the gradients during optimization. Defaults to None.
    freeze_mask: Optional mask to freeze parameters during optimization.
      Defaults to None.

  Example usage:
  ```
  adagrad = AdagradFactory(learning_rate=1e-3).make()
  ```
  """

  learning_rate: optax.ScalarOrSchedule
  initial_accumulator_value: float = 0.1
  eps: float = 1e-7
  grad_clip_norm: float | None = None
  freeze_mask: str | Callable[[optax.Params], optax.Params] | None = None

  def make(self) -> optax.GradientTransformation:
    return OptimizerFactory(
        learning_rate=self.learning_rate,
        scaling=optax.scale_by_rss(
            initial_accumulator_value=self.initial_accumulator_value,
            eps=self.eps,
        ),
        grad_clip_norm=self.grad_clip_norm,
    ).make()

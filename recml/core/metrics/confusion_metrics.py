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
"""Confusion metrics."""

from collections.abc import Sequence
import math
from typing import Any, Self

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from recml.core.metrics import base_metrics

# pylint: disable=redefined-outer-name
EPSILON = 1e-7


class ConfusionMetric(base_metrics.Metric):
  """Computes the confusion matrix of binary predictions and labels.

  Note: For several confusion metrics, only a subset of true positives, true
  negatives, false positives, and false negatives are required. However, by
  sharing the same computation logic across all metrics, we effectively allow
  XLA to dedup multiple computations of the confusion matrix over the same
  inputs and thresholds during it's optimization passes.
  """

  true_positives: jax.Array
  true_negatives: jax.Array
  false_positives: jax.Array
  false_negatives: jax.Array

  @classmethod
  def from_model_output(
      cls,
      y_true: jax.Array,
      y_pred: jax.Array,
      weights: jax.Array | None = None,
      *,
      threshold: float | Sequence[float] = 0.5,
      **_,
  ) -> Self:
    if not isinstance(threshold, Sequence):
      threshold = [threshold]

    tp, tn, fp, fn = estimate_confusion_matrix(
        predictions=y_pred,
        labels=y_true,
        weights=weights,
        thresholds=threshold,
    )
    return cls(
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
    )

  def merge(self, other: Self) -> Self:
    return type(self)(
        true_positives=self.true_positives + other.true_positives,
        true_negatives=self.true_negatives + other.true_negatives,
        false_positives=self.false_positives + other.false_positives,
        false_negatives=self.false_negatives + other.false_negatives,
    )


class Precision(ConfusionMetric):
  """Precision confusion metric."""

  def compute(self) -> float | Sequence[float]:
    precision_ = _np_divide_no_nan(
        self.true_positives, self.true_positives + self.false_positives
    )
    return _maybe_squeeze(precision_)  # pyrefly: ignore[bad-return]


class Recall(ConfusionMetric):
  """Recall confusion metric."""

  def compute(self) -> float | Sequence[float]:
    recall_ = _np_divide_no_nan(
        self.true_positives, self.true_positives + self.false_negatives
    )
    return _maybe_squeeze(recall_)  # pyrefly: ignore[bad-return]


class FBeta(ConfusionMetric):
  """FBeta confusion metric."""

  beta: float

  @classmethod
  def from_model_output(
      cls,
      y_true: jax.Array,
      y_pred: jax.Array,
      weights: jax.Array | None = None,
      *,
      threshold: float | Sequence[float] = 0.5,
      beta: float = 1.0,
      **_,
  ) -> Self:
    if not isinstance(threshold, Sequence):
      threshold = [threshold]

    tp, tn, fp, fn = estimate_confusion_matrix(
        predictions=y_pred,
        labels=y_true,
        weights=weights,
        thresholds=threshold,
    )
    return cls(
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        beta=beta,
    )

  def merge(self, other: Self) -> Self:
    return type(self)(
        true_positives=self.true_positives + other.true_positives,
        true_negatives=self.true_negatives + other.true_negatives,
        false_positives=self.false_positives + other.false_positives,
        false_negatives=self.false_negatives + other.false_negatives,
        beta=self.beta,
    )

  def compute(self) -> float | Sequence[float]:
    precision_ = _np_divide_no_nan(
        self.true_positives, self.true_positives + self.false_positives
    )
    recall_ = _np_divide_no_nan(
        self.true_positives, self.true_positives + self.false_negatives
    )
    return _maybe_squeeze(  # pyrefly: ignore[bad-return]
        _np_divide_no_nan(
            np.multiply(precision_, recall_) * (self.beta + 1.0),
            np.multiply(precision_, self.beta) + recall_,
        )
    )


class AUCPR(ConfusionMetric):
  """Computes the area under the precision-recall curve."""

  @classmethod
  def from_model_output(
      cls,
      y_true: jax.Array,
      y_pred: jax.Array,
      weights: jax.Array | None = None,
      *,
      from_logits: bool = False,
      num_thresholds: int = 200,
      **_,
  ) -> Self:
    if from_logits:
      y_pred = jax.nn.sigmoid(y_pred)

    tp, tn, fp, fn = estimate_confusion_matrix(
        predictions=y_pred,
        labels=y_true,
        weights=weights,
        thresholds=default_thresholds(num_thresholds),  # pyrefly: ignore[bad-argument-type]
    )
    return cls(
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
    )

  def compute(self) -> float:
    """Computes the area under the precision-recall curve.

    Interpolation formula inspired by section 4 of Davis & Goadrich 2006 [1]
    and adapted from Keras's implementation here [2].

    Returns:
      The area under the precision-recall curve.

    References:
      [1] https://pages.cs.wisc.edu/~jdavis/davisgoadrichcamera2.pdf
      [2] keras/src/metrics/confusion_metrics.py
    """

    tp = self.true_positives
    fp = self.false_positives
    fn = self.false_negatives
    num_thresholds = tp.shape[0]

    tp = np.sum(tp, axis=-1) if tp.ndim > 1 else tp
    fp = np.sum(fp, axis=-1) if fp.ndim > 1 else fp
    fn = np.sum(fn, axis=-1) if fn.ndim > 1 else fn

    dtp = tp[: num_thresholds - 1] - tp[1:]

    p = tp + fp
    dp = p[: num_thresholds - 1] - p[1:]
    prec_slope = _np_divide_no_nan(dtp, np.maximum(dp, 0))
    intercept = tp[1:] - np.multiply(prec_slope, p[1:])

    safe_p_ratio = np.where(
        np.logical_and(p[: num_thresholds - 1] > 0, p[1:] > 0),
        _np_divide_no_nan(p[: num_thresholds - 1], np.maximum(p[1:], 0)),
        np.ones_like(p[1:]),
    )

    pr_auc_increment = _np_divide_no_nan(
        prec_slope * (dtp + intercept * np.log(safe_p_ratio)),
        np.maximum(tp[1:] + fn[1:], 0),
    )
    return np.sum(pr_auc_increment)


class AUCROC(AUCPR):
  """Computes the area under the receiver operating characteristic curve."""

  def compute(self) -> float:
    tp_rate = _np_divide_no_nan(
        self.true_positives, self.true_positives + self.false_negatives
    )
    fp_rate = _np_divide_no_nan(
        self.false_positives, self.false_positives + self.true_negatives
    )
    # We negate the integral because the thresholds are in ascending order.
    return -np.trapezoid(tp_rate, fp_rate)


class PrecisionAtRecall(ConfusionMetric):
  """Computes best precision where recall is >= specified value."""

  recall: float

  @classmethod
  def from_model_output(
      cls,
      y_true: jax.Array,
      y_pred: jax.Array,
      weights: jax.Array | None = None,
      *,
      num_thresholds: int = 200,
      recall: float = 0.1,
      **_,
  ) -> Self:

    tp, tn, fp, fn = estimate_confusion_matrix(
        predictions=y_pred,
        labels=y_true,
        weights=weights,
        thresholds=default_thresholds(num_thresholds),  # pyrefly: ignore[bad-argument-type]
    )
    return cls(
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        recall=recall,
    )

  def merge(self, other: Self) -> Self:
    if self.recall != other.recall:
      raise ValueError(
          "Recalls must be the same to merge PrecisionAtRecall metrics."
      )
    return type(self)(
        true_positives=self.true_positives + other.true_positives,
        true_negatives=self.true_negatives + other.true_negatives,
        false_positives=self.false_positives + other.false_positives,
        false_negatives=self.false_negatives + other.false_negatives,
        recall=self.recall,
    )

  def compute(self) -> float:
    """Computes best precision where recall is >= specified value.

    Returns:
      The precision at recall.

    References:
      [1] keras/src/metrics/confusion_metrics.py
      [2] https://github.com/tensorflow/tensorflow/issues/37256
    """

    recalls = _np_divide_no_nan(
        self.true_positives, self.true_positives + self.false_negatives
    )
    precisions = _np_divide_no_nan(
        self.true_positives, self.true_positives + self.false_positives
    )
    index = np.argmin(np.abs(recalls - self.recall))
    return precisions[index]


def precision(
    y_true: jax.Array,
    y_pred: jax.Array,
    weights: jax.Array | None = None,
    *,
    threshold: float | Sequence[float] = 0.5,
    **_,
) -> Precision:
  """Computes precision from observations.

  Args:
    y_true: The true labels of shape [..., num_classes].
    y_pred: The predicted logits of shape [..., num_classes].
    weights: Optional weights of shape [..., num_classes].
    threshold: A scalar threshold to compute the precision at or optionally a
      sequence of thresholds to compute the precision at. If `y_pred` is log
      probabilities then the threshold must be set to 0.0. Defaults to 0.5.
    **_: Unused keyword arguments.

  Returns:
    A precision metric accumulation.
  """
  return Precision.from_model_output(
      y_true, y_pred, weights, threshold=threshold
  )


def recall(
    y_true: jax.Array,
    y_pred: jax.Array,
    weights: jax.Array | None = None,
    *,
    threshold: float | Sequence[float] = 0.5,
    **_,
) -> Recall:
  """Computes recall from observations.

  Args:
    y_true: The true labels of shape [..., num_classes].
    y_pred: The predicted logits of shape [..., num_classes].
    weights: Optional weights of shape [..., num_classes].
    threshold: A scalar threshold to compute the recall at or optionally a
      sequence of thresholds to compute the recall at. If `y_pred` is log
      probabilities then the threshold must be set to 0.0. Defaults to 0.5.
    **_: Unused keyword arguments.

  Returns:
    A recall metric accumulation.
  """
  return Recall.from_model_output(y_true, y_pred, weights, threshold=threshold)


def fbeta_score(
    y_true: jax.Array,
    y_pred: jax.Array,
    weights: jax.Array | None = None,
    *,
    beta: float,
    threshold: float | Sequence[float] = 0.5,
    **_,
) -> FBeta:
  """Computes the F-beta score from observations.

  Args:
    y_true: The true labels of shape [..., num_classes].
    y_pred: The predicted logits of shape [..., num_classes].
    weights: Optional weights of shape [..., num_classes].
    beta: The beta parameter of the F-beta score.s
    threshold: A scalar threshold to compute the F-beta score at or optionally a
      sequence of thresholds to compute the F-beta score at. If `y_pred` is log
      probabilities then the threshold must be set to 0.0. Defaults to 0.5.
    **_: Unused keyword arguments.

  Returns:
    A F-beta score metric accumulation.
  """
  return FBeta.from_model_output(
      y_true, y_pred, weights, beta=beta, threshold=threshold
  )


def f1_score(
    y_true: jax.Array,
    y_pred: jax.Array,
    weights: jax.Array | None = None,
    *,
    threshold: float | Sequence[float] = 0.5,
    **_,
) -> FBeta:
  """Computes the F-1 score from observations.

  Args:
    y_true: The true labels of shape [..., num_classes].
    y_pred: The predicted logits of shape [..., num_classes].
    weights: Optional weights of shape [..., num_classes].
    threshold: A scalar threshold to compute the F-1 score at or optionally a
      sequence of thresholds to compute the F-1 score at. If `y_pred` is log
      probabilities then the threshold must be set to 0.0. Defaults to 0.5.
    **_: Unused keyword arguments.

  Returns:
    A F-1 score metric accumulation.
  """
  return fbeta_score(y_true, y_pred, weights, threshold=threshold, beta=1.0)


def aucpr(
    y_true: jax.Array,
    y_pred: jax.Array,
    weights: jax.Array | None = None,
    *,
    from_logits: bool = False,
    num_thresholds: int = 200,
    **_,
) -> AUCPR:
  """Computes the area under the precision-recall curve from observations.

  Args:
    y_true: The true labels of shape [..., num_classes].
    y_pred: The predicted logits of shape [..., num_classes].
    weights: Optional weights of shape [..., num_classes].
    from_logits: Whether the y_pred are logits instead of probabilities.
      Defaults to False.
    num_thresholds: The number of thresholds to use to estimate the confusion
      matrix.
    **_: Unused keyword arguments.

  Returns:
    A AUC-PR metric accumulation.
  """
  return AUCPR.from_model_output(
      y_true,
      y_pred,
      weights,
      from_logits=from_logits,
      num_thresholds=num_thresholds,
  )


def aucroc(
    y_true: jax.Array,
    y_pred: jax.Array,
    weights: jax.Array | None = None,
    *,
    from_logits: bool = False,
    num_thresholds: int = 200,
    **_,
) -> AUCROC:
  """Computes the area under the receiver operating characteristic curve.

  Args:
    y_true: The true labels of shape [..., num_classes].
    y_pred: The predicted logits of shape [..., num_classes].
    weights: Optional weights of shape [..., num_classes].
    from_logits: Whether the y_pred are logits instead of probabilities.
      Defaults to False.
    num_thresholds: The number of thresholds to use to estimate the confusion
      matrix.
    **_: Unused keyword arguments.

  Returns:
    An AUC-ROC metric accumulation.
  """
  return AUCROC.from_model_output(
      y_true,
      y_pred,
      weights,
      from_logits=from_logits,
      num_thresholds=num_thresholds,
  )


def precision_at_recall(
    y_true: jax.Array,
    y_pred: jax.Array,
    weights: jax.Array | None = None,
    *,
    recall: float = 0.1,
    num_thresholds: int = 200,
    **_,
) -> PrecisionAtRecall:
  """Computes best precision where recall is >= specified value from observations.

  Args:
    y_true: The true labels of shape [..., num_classes].
    y_pred: The predicted logits of shape [..., num_classes].
    weights: Optional weights of shape [..., num_classes].
    recall: The recall value to find the corresponding precision.
    num_thresholds: The number of thresholds to use to estimate the confusion
      matrix.
    **_: Unused keyword arguments.

  Returns:
    A precision at recall metric accumulation.
  """
  return PrecisionAtRecall.from_model_output(
      y_true,
      y_pred,
      weights,
      recall=recall,
      num_thresholds=num_thresholds,
  )


def estimate_confusion_matrix_batched(
    predictions: jax.Array,
    labels: jax.Array,
    weights_batch: jax.Array,
    thresholds: Sequence[float] = (0.0,),
    *,
    sparse_positive_indices: jax.Array | None = None,
    threshold_batch_size: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Estimates the confusion matrix of binary predictions across weight batches.

  Args:
    predictions: The predictions of shape [*sample_dims, *class_dims].
    labels: The labels of shape [*sample_dims, *class_dims]. Ignored for
      positive detection when sparse_positive_indices is provided (only shape
      equality with predictions is checked).
    weights_batch: A batch of sample weights of shape [K, *sample_dims].
    thresholds: A 1D sequence of float threshold values.
    sparse_positive_indices: Optional integer indices of positive classes of
      shape [*sample_dims, L]. Negative or out-of-range padding IDs (< 0 or >=
      num_classes) and duplicate IDs along axis=-1 are ignored.
    threshold_batch_size: Optional batch size for threshold mapping. When None,
      thresholds are chunked in batches of 25 automatically once
      predictions.size * len(thresholds) >= 16_000_000, and evaluated in a
      single jax.vmap on smaller tensors.

  Returns:
    A tuple of (tp, tn, fp, fn) arrays each of shape [K, len(thresholds)] in
    float32.

  Raises:
    ValueError: If shapes of predictions, labels, weights_batch, or
    sparse_positive_indices are incompatible.
  """
  if predictions.shape != labels.shape:
    raise ValueError(
        "Predictions and labels must have the same shape. Got"
        f" {predictions.shape} and {labels.shape} respectively."
    )
  if weights_batch.ndim < 1:
    raise ValueError(
        "weights_batch must have at least 1 dimension. Got"
        f" {weights_batch.ndim}."
    )
  sample_ndims = weights_batch.ndim - 1
  if (
      sample_ndims >= predictions.ndim
      or weights_batch.shape[1:] != predictions.shape[:sample_ndims]
  ):
    raise ValueError(
        "weights_batch spatial dimensions do not match predictions. Got "
        f"{weights_batch.shape} and {predictions.shape}."
    )

  reduce_axes = tuple(range(sample_ndims, predictions.ndim))
  num_classes_per_sample = math.prod(predictions.shape[sample_ndims:])
  count_dtype = (
      jnp.int32 if num_classes_per_sample < 2_000_000_000 else jnp.float32
  )

  if sparse_positive_indices is not None:
    if (
        sample_ndims != predictions.ndim - 1
        or sparse_positive_indices.shape[:-1] != predictions.shape[:-1]
    ):
      raise ValueError(
          "sparse_positive_indices shape mismatch. Got "
          f"{sparse_positive_indices.shape} and {predictions.shape}."
      )
    num_classes = predictions.shape[-1]
    idx_int = jnp.asarray(sparse_positive_indices, dtype=jnp.int32)
    sorted_idx = jnp.sort(idx_int, axis=-1)
    is_new = jnp.concatenate(
        [
            jnp.ones_like(sorted_idx[..., :1], dtype=jnp.bool_),
            sorted_idx[..., 1:] != sorted_idx[..., :-1],
        ],
        axis=-1,
    )
    valid_unique = (sorted_idx >= 0) & (sorted_idx < num_classes) & is_new
    safe_idx = jnp.where(valid_unique, sorted_idx, 0)
    pos_preds = jnp.take_along_axis(predictions, safe_idx, axis=-1)
    pos_counts = jnp.sum(valid_unique, axis=-1, dtype=count_dtype)

    def _count_for_threshold(t):
      pred_pos = jnp.sum(predictions > t, axis=-1, dtype=count_dtype)
      tp_sample = jnp.sum(
          (pos_preds > t) & valid_unique, axis=-1, dtype=count_dtype
      )
      return tp_sample, pred_pos

  else:
    y_true_pos = jnp.asarray(labels, dtype=jnp.bool_)
    pos_counts = jnp.sum(y_true_pos, axis=reduce_axes, dtype=count_dtype)

    def _count_for_threshold(t):
      y_pred_pos = predictions > t
      pred_pos = jnp.sum(y_pred_pos, axis=reduce_axes, dtype=count_dtype)
      tp_sample = jnp.sum(
          y_pred_pos & y_true_pos, axis=reduce_axes, dtype=count_dtype
      )
      return tp_sample, pred_pos

  thresholds_arr = jnp.asarray(thresholds, dtype=jnp.float32)
  if threshold_batch_size is not None:
    effective_batch_size = threshold_batch_size
  elif predictions.size * thresholds_arr.shape[0] >= 16_000_000:
    effective_batch_size = 25
  else:
    effective_batch_size = None
  if (
      effective_batch_size is not None
      and thresholds_arr.shape[0] > effective_batch_size
  ):
    tp_t, pred_pos_t = jax.lax.map(
        _count_for_threshold, thresholds_arr, batch_size=effective_batch_size
    )
  else:
    tp_t, pred_pos_t = jax.vmap(_count_for_threshold)(thresholds_arr)

  fp_t = pred_pos_t - tp_t
  fn_t = pos_counts[None, ...] - tp_t
  tn_t = (num_classes_per_sample - pos_counts)[None, ...] - fp_t

  k_slices = weights_batch.shape[0]
  w_flat = jnp.asarray(weights_batch, dtype=jnp.float32).reshape(k_slices, -1)
  t_len = thresholds_arr.shape[0]

  all_counts = (
      jnp.stack([tp_t, tn_t, fp_t, fn_t], axis=0)
      .astype(jnp.float32)
      .reshape(4 * t_len, -1)
  )
  contracted = jnp.matmul(
      w_flat, all_counts.T, precision=jax.lax.Precision.HIGHEST
  ).reshape(k_slices, 4, t_len)
  return (
      contracted[:, 0, :],
      contracted[:, 1, :],
      contracted[:, 2, :],
      contracted[:, 3, :],
  )


def estimate_confusion_matrix(
    predictions: jt.Float[jt.Array, "* C"],
    labels: jt.Float[jt.Array, "* C"],
    weights: jt.Float[jt.Array, "* C"] | None = None,
    thresholds: Sequence[float] = (0.0,),
) -> tuple[
    jt.Float[jt.Array, "N"],
    jt.Float[jt.Array, "N"],
    jt.Float[jt.Array, "N"],
    jt.Float[jt.Array, "N"],
]:
  """Estimates the confusion matrix of binary predictions and labels.

  This uses jax.vmap to vectorize the computation over thresholds which avoids
  several inefficiencies in the keras.metrics._ConfusionMatrixConditionCount
  metrics.

  Args:
    predictions: The predictions of shape [..., num_classes].
    labels: The labels of shape [..., num_classes].
    weights: The weights of shape [..., num_classes].
    thresholds: A 1D sequence of float threshold values of len [num_thresholds].

  Returns:
    A tuple of true positives, true negatives, false positives, and false
    negatives, all of shape [num_thresholds].
  """

  if predictions.shape != labels.shape:
    raise ValueError(
        "Predictions and labels must have the same shape. Got"
        f" {predictions.shape} and {labels.shape} respectively."
    )

  if weights is not None and predictions.ndim != weights.ndim:
    if (
        weights.ndim > predictions.ndim
        or weights.shape != predictions.shape[: weights.ndim]
    ):
      raise ValueError(
          "Weights are not broadcastable to predictions shape. Got the"
          f" following shapes for predictions: {predictions.shape} and weights:"
          f" {weights.shape}."
      )
  elif weights is not None and predictions.shape != weights.shape:
    raise ValueError(
        "Predictions and weights must have the same shape. Got"
        f" {predictions.shape} and {weights.shape} respectively."
    )

  if weights is None:
    tp, tn, fp, fn = estimate_confusion_matrix_batched(
        predictions, labels, jnp.ones((1,), dtype=jnp.float32), thresholds
    )
    return tp[0], tn[0], fp[0], fn[0]
  if weights.ndim < predictions.ndim:
    tp, tn, fp, fn = estimate_confusion_matrix_batched(
        predictions, labels, jnp.expand_dims(weights, axis=0), thresholds
    )
    return tp[0], tn[0], fp[0], fn[0]

  y_true_pos = jnp.asarray(labels, dtype=jnp.bool_)
  y_true_neg = jnp.logical_not(y_true_pos)

  def _estimate_confusion_matrix(thresholds: jt.Scalar) -> tuple[
      jt.Float[jt.Array, "* C"],
      jt.Float[jt.Array, "* C"],
      jt.Float[jt.Array, "* C"],
      jt.Float[jt.Array, "* C"],
  ]:
    y_pred_pos = predictions > thresholds
    y_pred_neg = jnp.logical_not(y_pred_pos)

    tp = jnp.logical_and(y_pred_pos, y_true_pos)
    tn = jnp.logical_and(y_pred_neg, y_true_neg)
    fp = jnp.logical_and(y_pred_pos, y_true_neg)
    fn = jnp.logical_and(y_pred_neg, y_true_pos)

    tp = tp * weights
    tn = tn * weights
    fp = fp * weights
    fn = fn * weights

    return jnp.sum(tp), jnp.sum(tn), jnp.sum(fp), jnp.sum(fn)

  thresholds = jnp.asarray(thresholds, dtype=jnp.float32)  # pyrefly: ignore[bad-assignment]
  return jax.vmap(_estimate_confusion_matrix)(thresholds)  # pyrefly: ignore[bad-argument-type]


def default_thresholds(num_thresholds: int) -> np.ndarray:
  """Returns the default thresholds for AUC computation."""
  return np.array(
      [-EPSILON]
      + [
          (i + 1) * 1.0 / (num_thresholds - 1)
          for i in range(num_thresholds - 2)
      ]
      + [1.0 + EPSILON]
  )


def _np_divide_no_nan(x: Any, y: Any) -> np.ndarray:
  """Safe division of numpy arrays."""
  return np.where(y == 0, 0.0, x / y)


def _maybe_squeeze(x: Any) -> np.ndarray:
  """Squeezes the array if it has a single element."""
  return x[0] if len(x) == 1 else x

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
"""Tests for confusion metrics."""

from collections.abc import Sequence

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from recml.core.metrics import confusion_metrics


class ConfusionMetricsTest(parameterized.TestCase):

  def test_estimate_confusion_matrix_batched(self) -> None:
    predictions = np.array(
        [
            [0.9, 0.2, 0.8, 0.1],
            [0.2, 0.9, 0.7, 0.6],
            [0.1, 0.2, 0.4, 0.3],
            [0.0, 1.0, 0.7, 0.3],
        ],
        dtype=np.float32,
    )
    labels = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    weights_batch = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.5, 0.0, -1.0],
        ],
        dtype=np.float32,
    )

    thresholds = [0.15, 0.5, 0.85, 0.95, 1.0]

    tp_b, tn_b, fp_b, fn_b = (
        confusion_metrics.estimate_confusion_matrix_batched(
            predictions,
            labels,
            weights_batch,
            thresholds,
            threshold_batch_size=2,
        )
    )

    # Hand-computed literal arrays for weights_batch[0] (all ones)
    expected_tp_0 = np.array([6.0, 3.0, 1.0, 1.0, 0.0])
    expected_tn_0 = np.array([2.0, 5.0, 7.0, 9.0, 9.0])
    expected_fp_0 = np.array([7.0, 4.0, 2.0, 0.0, 0.0])
    expected_fn_0 = np.array([1.0, 4.0, 6.0, 6.0, 7.0])

    np.testing.assert_allclose(tp_b[0], expected_tp_0, rtol=1e-5)
    np.testing.assert_allclose(tn_b[0], expected_tn_0, rtol=1e-5)
    np.testing.assert_allclose(fp_b[0], expected_fp_0, rtol=1e-5)
    np.testing.assert_allclose(fn_b[0], expected_fn_0, rtol=1e-5)

    # Compare with estimate_confusion_matrix called with full-rank weights
    w_0_exp = np.broadcast_to(weights_batch[0][:, None], predictions.shape)
    tp_0_ind, tn_0_ind, fp_0_ind, fn_0_ind = (
        confusion_metrics.estimate_confusion_matrix(
            predictions, labels, w_0_exp, thresholds
        )
    )
    np.testing.assert_allclose(tp_b[0], tp_0_ind, rtol=1e-5)
    np.testing.assert_allclose(tn_b[0], tn_0_ind, rtol=1e-5)
    np.testing.assert_allclose(fp_b[0], fp_0_ind, rtol=1e-5)
    np.testing.assert_allclose(fn_b[0], fn_0_ind, rtol=1e-5)

    w_1_exp = np.broadcast_to(weights_batch[1][:, None], predictions.shape)
    tp_1_ind, tn_1_ind, fp_1_ind, fn_1_ind = (
        confusion_metrics.estimate_confusion_matrix(
            predictions, labels, w_1_exp, thresholds
        )
    )
    np.testing.assert_allclose(tp_b[1], tp_1_ind, rtol=1e-5)
    np.testing.assert_allclose(tn_b[1], tn_1_ind, rtol=1e-5)
    np.testing.assert_allclose(fp_b[1], fp_1_ind, rtol=1e-5)
    np.testing.assert_allclose(fn_b[1], fn_1_ind, rtol=1e-5)

  def test_estimate_confusion_matrix_batched_t150(self) -> None:
    predictions = np.zeros((4, 4), dtype=np.float32)
    labels = np.zeros((4, 4), dtype=np.float32)
    weights_batch = np.ones((1, 4), dtype=np.float32)
    thresholds = np.linspace(0.0, 1.0, 150).tolist()
    tp_b, tn_b, fp_b, fn_b = (
        confusion_metrics.estimate_confusion_matrix_batched(
            predictions,
            labels,
            weights_batch,
            thresholds,
            threshold_batch_size=2,
        )
    )
    w_0_exp = np.broadcast_to(weights_batch[0][:, None], predictions.shape)
    tp_0_ind, tn_0_ind, fp_0_ind, fn_0_ind = (
        confusion_metrics.estimate_confusion_matrix(
            predictions, labels, w_0_exp, thresholds
        )
    )
    np.testing.assert_allclose(tp_b[0], tp_0_ind, rtol=1e-5)
    np.testing.assert_allclose(tn_b[0], tn_0_ind, rtol=1e-5)
    np.testing.assert_allclose(fp_b[0], fp_0_ind, rtol=1e-5)
    np.testing.assert_allclose(fn_b[0], fn_0_ind, rtol=1e-5)

  def test_estimate_confusion_matrix_batched_sparse(self) -> None:
    predictions = np.array(
        [
            [0.9, 0.2, 0.8, 0.1],
            [0.2, 0.9, 0.7, 0.6],
            [0.1, 0.2, 0.4, 0.3],
            [0.0, 1.0, 0.7, 0.3],
        ],
        dtype=np.float32,
    )
    labels = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    weights_batch = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.5, 0.0, -1.0],
        ],
        dtype=np.float32,
    )
    thresholds = [0.15, 0.5, 0.85, 0.95, 1.0]
    sparse_positive_indices = np.array(
        [
            [1, 2, 2, 1],
            [0, -1, -1, -1],
            [-1, -1, -1, -1],
            [0, 1, 2, 3],
        ],
        dtype=np.int32,
    )
    predictions_bf16 = jax.numpy.array(predictions, dtype=jax.numpy.bfloat16)
    tp_b_sp, tn_b_sp, fp_b_sp, fn_b_sp = (
        confusion_metrics.estimate_confusion_matrix_batched(
            predictions_bf16,
            labels,
            weights_batch,
            thresholds,
            sparse_positive_indices=sparse_positive_indices,
        )
    )

    w_0_exp = np.broadcast_to(weights_batch[0][:, None], predictions_bf16.shape)
    tp_0_ind, tn_0_ind, fp_0_ind, fn_0_ind = (
        confusion_metrics.estimate_confusion_matrix(
            predictions_bf16, labels, w_0_exp, thresholds
        )
    )
    np.testing.assert_allclose(tp_b_sp[0], tp_0_ind, rtol=1e-5)
    np.testing.assert_allclose(tn_b_sp[0], tn_0_ind, rtol=1e-5)
    np.testing.assert_allclose(fp_b_sp[0], fp_0_ind, rtol=1e-5)
    np.testing.assert_allclose(fn_b_sp[0], fn_0_ind, rtol=1e-5)

    w_1_exp = np.broadcast_to(weights_batch[1][:, None], predictions_bf16.shape)
    tp_1_ind, tn_1_ind, fp_1_ind, fn_1_ind = (
        confusion_metrics.estimate_confusion_matrix(
            predictions_bf16, labels, w_1_exp, thresholds
        )
    )
    np.testing.assert_allclose(tp_b_sp[1], tp_1_ind, rtol=1e-5)
    np.testing.assert_allclose(tn_b_sp[1], tn_1_ind, rtol=1e-5)
    np.testing.assert_allclose(fp_b_sp[1], fp_1_ind, rtol=1e-5)
    np.testing.assert_allclose(fn_b_sp[1], fn_1_ind, rtol=1e-5)

  def test_estimate_confusion_matrix_batched_auto_chunking(self) -> None:
    predictions = np.zeros((80_000, 1), dtype=np.float32)
    predictions[0, 0] = 0.9
    predictions[1, 0] = 0.2
    labels = np.zeros((80_000, 1), dtype=np.float32)
    labels[0, 0] = 1.0
    labels[2, 0] = 1.0
    weights_batch = np.ones((2, 80_000), dtype=np.float32)
    weights_batch[1, 1] = 2.5
    thresholds = np.linspace(0.0, 1.0, 200).tolist()
    tp_b, tn_b, fp_b, fn_b = (
        confusion_metrics.estimate_confusion_matrix_batched(
            predictions,
            labels,
            weights_batch,
            thresholds,
            threshold_batch_size=None,
        )
    )
    w_0_exp = np.broadcast_to(weights_batch[0][:, None], predictions.shape)
    tp_0_ind, tn_0_ind, fp_0_ind, fn_0_ind = (
        confusion_metrics.estimate_confusion_matrix(
            predictions, labels, w_0_exp, thresholds
        )
    )
    np.testing.assert_allclose(tp_b[0], tp_0_ind, rtol=1e-5)
    np.testing.assert_allclose(tn_b[0], tn_0_ind, rtol=1e-5)
    np.testing.assert_allclose(fp_b[0], fp_0_ind, rtol=1e-5)
    np.testing.assert_allclose(fn_b[0], fn_0_ind, rtol=1e-5)

    w_1_exp = np.broadcast_to(weights_batch[1][:, None], predictions.shape)
    tp_1_ind, tn_1_ind, fp_1_ind, fn_1_ind = (
        confusion_metrics.estimate_confusion_matrix(
            predictions, labels, w_1_exp, thresholds
        )
    )
    np.testing.assert_allclose(tp_b[1], tp_1_ind, rtol=1e-5)
    np.testing.assert_allclose(tn_b[1], tn_1_ind, rtol=1e-5)
    np.testing.assert_allclose(fp_b[1], fp_1_ind, rtol=1e-5)
    np.testing.assert_allclose(fn_b[1], fn_1_ind, rtol=1e-5)

    np.testing.assert_allclose(tp_b[1, 0], 1.0, rtol=1e-5)
    np.testing.assert_allclose(tn_b[1, 0], 79997.0, rtol=1e-5)
    np.testing.assert_allclose(fp_b[1, 0], 2.5, rtol=1e-5)
    np.testing.assert_allclose(fn_b[1, 0], 1.0, rtol=1e-5)

    np.testing.assert_allclose(tp_b[1, 100], 1.0, rtol=1e-5)
    np.testing.assert_allclose(tn_b[1, 100], 79999.5, rtol=1e-5)
    np.testing.assert_allclose(fp_b[1, 100], 0.0, rtol=1e-5)
    np.testing.assert_allclose(fn_b[1, 100], 1.0, rtol=1e-5)

  def test_estimate_confusion_matrix_batched_jit(self) -> None:
    predictions = np.array([[0.9, 0.2], [0.2, 0.9]], dtype=np.float32)
    labels = np.array([[0, 1], [1, 0]], dtype=np.float32)
    weights_batch = np.array([[1.0, 1.0]], dtype=np.float32)
    thresholds = tuple([0.5])

    @jax.jit
    def _jit_dense(p, l, w):
      return confusion_metrics.estimate_confusion_matrix_batched(
          p, l, w, thresholds
      )

    tp_d, tn_d, fp_d, fn_d = _jit_dense(predictions, labels, weights_batch)

    sparse_ids = np.array([[1, -1], [0, -1]], dtype=np.int32)

    @jax.jit
    def _jit_sparse(p, l, w, sp):
      return confusion_metrics.estimate_confusion_matrix_batched(
          p, l, w, thresholds, sparse_positive_indices=sp
      )

    tp_s, tn_s, fp_s, fn_s = _jit_sparse(
        predictions, labels, weights_batch, sparse_ids
    )

    np.testing.assert_allclose(tp_d, tp_s, rtol=1e-5)
    np.testing.assert_allclose(tn_d, tn_s, rtol=1e-5)
    np.testing.assert_allclose(fp_d, fp_s, rtol=1e-5)
    np.testing.assert_allclose(fn_d, fn_s, rtol=1e-5)

  def test_estimate_confusion_matrix_batched_value_errors(self) -> None:
    predictions = np.zeros((4, 4), dtype=np.float32)
    labels = np.zeros((4, 4), dtype=np.float32)
    weights_batch = np.ones((1, 4), dtype=np.float32)
    thresholds = [0.0]
    sparse_positive_indices = np.zeros((4, 2), dtype=np.int32)

    with self.assertRaises(ValueError):
      confusion_metrics.estimate_confusion_matrix_batched(
          predictions, labels[1:], weights_batch, thresholds
      )
    with self.assertRaises(ValueError):
      confusion_metrics.estimate_confusion_matrix_batched(
          predictions, labels, weights_batch[..., :2], thresholds
      )
    with self.assertRaises(ValueError):
      confusion_metrics.estimate_confusion_matrix_batched(
          predictions,
          labels,
          weights_batch,
          thresholds,
          sparse_positive_indices=sparse_positive_indices[1:],
      )

  @parameterized.named_parameters(
      {
          'testcase_name': 'unweighted',
          'predictions': np.array([
              [0, 0, 1, 1, 0],
              [1, 1, 1, 1, 1],
              [0, 1, 0, 1, 0],
              [1, 1, 1, 1, 1],
          ]),
          'labels': np.array([
              [0, 1, 0, 1, 0],
              [0, 0, 1, 1, 1],
              [1, 1, 1, 1, 0],
              [0, 0, 0, 0, 1],
          ]),
          'weights': None,
          'thresholds': [0.5],
          'expected_tp': np.array([7.0]),
          'expected_tn': np.array([3.0]),
          'expected_fp': np.array([7.0]),
          'expected_fn': np.array([3.0]),
      },
      {
          'testcase_name': 'weighted',
          'predictions': np.array([
              [0, 0, 1, 1, 0],
              [1, 1, 1, 1, 1],
              [0, 1, 0, 1, 0],
              [1, 1, 1, 1, 1],
          ]),
          'labels': np.array([
              [0, 1, 0, 1, 0],
              [0, 0, 1, 1, 1],
              [1, 1, 1, 1, 0],
              [0, 0, 0, 0, 1],
          ]),
          'weights': np.array([1.0, 1.5, 2.0, 2.5]),
          'thresholds': [0.5],
          'expected_tp': np.array([12.0]),
          'expected_tn': np.array([4.0]),
          'expected_fp': np.array([14.0]),
          'expected_fn': np.array([5.0]),
      },
      {
          'testcase_name': 'unweighted_thresholds',
          'predictions': np.array([
              [0.9, 0.2, 0.8, 0.1],
              [0.2, 0.9, 0.7, 0.6],
              [0.1, 0.2, 0.4, 0.3],
              [0.0, 1.0, 0.7, 0.3],
          ]),
          'labels': np.array([
              [0, 1, 1, 0],
              [1, 0, 0, 0],
              [0, 0, 0, 0],
              [1, 1, 1, 1],
          ]),
          'weights': None,
          'thresholds': [0.15, 0.5, 0.85],
          'expected_tp': np.array([6.0, 3.0, 1.0]),
          'expected_tn': np.array([2.0, 5.0, 7.0]),
          'expected_fp': np.array([7.0, 4.0, 2.0]),
          'expected_fn': np.array([1.0, 4.0, 6.0]),
      },
      {
          'testcase_name': 'weighted_thresholds',
          'predictions': np.array([
              [0.9, 0.2, 0.8, 0.1],
              [0.2, 0.9, 0.7, 0.6],
              [0.1, 0.2, 0.4, 0.3],
              [0.0, 1.0, 0.7, 0.3],
          ]),
          'labels': np.array([
              [0, 1, 1, 0],
              [1, 0, 0, 0],
              [0, 0, 0, 0],
              [1, 1, 1, 1],
          ]),
          'weights': np.array([
              [1.0, 2.0, 3.0, 5.0],
              [7.0, 11.0, 13.0, 17.0],
              [19.0, 23.0, 29.0, 31.0],
              [5.0, 15.0, 10.0, 0],
          ]),
          'thresholds': [0.15, 0.5, 0.85],
          'expected_tp': np.array([37.0, 28.0, 15.0]),
          'expected_tn': np.array([24.0, 107.0, 137.0]),
          'expected_fp': np.array([125.0, 42.0, 12.0]),
          'expected_fn': np.array([5.0, 14.0, 27.0]),
      },
  )
  def test_estimate_confusion_matrix(
      self,
      predictions: jax.Array,
      labels: jax.Array,
      weights: jax.Array | None,
      thresholds: Sequence[float],
      expected_tp: jax.Array,
      expected_tn: jax.Array,
      expected_fp: jax.Array,
      expected_fn: jax.Array,
  ):
    tp, tn, fp, fn = confusion_metrics.estimate_confusion_matrix(
        predictions=predictions,
        labels=labels,
        weights=weights,
        thresholds=thresholds,
    )
    np.testing.assert_allclose(expected_tp, tp)
    np.testing.assert_allclose(expected_tn, tn)
    np.testing.assert_allclose(expected_fp, fp)
    np.testing.assert_allclose(expected_fn, fn)

  @parameterized.named_parameters(
      {
          'testcase_name': 'unweighted',
          'y_true': np.array([0, 1, 1, 0]),
          'y_pred': np.array([1, 0, 1, 0]),
          'weights': None,
          'thresholds': [0.5],
          'expected_outputs': 0.5,
      },
      {
          'testcase_name': 'weighted',
          'y_true': np.array([[0, 1, 1, 0], [1, 0, 0, 1]]),
          'y_pred': np.array([[1, 0, 1, 0], [1, 0, 1, 0]]),
          'weights': np.array([[1, 2, 3, 4], [4, 3, 2, 1]]),
          'thresholds': [0.5],
          'expected_outputs': (3.0 + 4.0) / ((1.0 + 3.0) + (4.0 + 2.0)),
      },
      {
          'testcase_name': 'div_by_zero',
          'y_true': np.array([0, 0, 0, 0]),
          'y_pred': np.array([0, 0, 0, 0]),
          'weights': None,
          'thresholds': [0.5],
          'expected_outputs': 0.0,
      },
      {
          'testcase_name': 'unweighted_thresholds',
          'y_true': np.array([0, 1, 1, 0]),
          'y_pred': np.array([1, 0, 0.6, 0]),
          'weights': None,
          'thresholds': [0.5, 0.7],
          'expected_outputs': [0.5, 0.0],
      },
      {
          'testcase_name': 'weighted_thresholds',
          'y_true': np.array([[0, 1], [1, 0]]),
          'y_pred': np.array([[1, 0], [0.6, 0]], dtype='float32'),
          'weights': np.array([[4, 0], [3, 1]], dtype='float32'),
          'thresholds': [0.5, 1.0],
          'expected_outputs': [(0 + 3.0) / ((0 + 3.0) + (4.0 + 0.0)), 0.0],
      },
  )
  def test_precision(
      self,
      y_true: jax.Array,
      y_pred: jax.Array,
      weights: jax.Array | None,
      thresholds: Sequence[float],
      expected_outputs: float | Sequence[float],
  ):
    precision = confusion_metrics.precision(
        y_true, y_pred, weights, threshold=thresholds
    )
    np.testing.assert_allclose(expected_outputs, precision.compute())
    np.testing.assert_allclose(expected_outputs, precision.localize().compute())

  @parameterized.named_parameters(
      {
          'testcase_name': 'unweighted',
          'y_true': np.array([0, 1, 1, 0]),
          'y_pred': np.array([1, 0, 1, 0]),
          'weights': None,
          'thresholds': [0.5],
          'expected_outputs': 0.5,
      },
      {
          'testcase_name': 'weighted',
          'y_true': np.array([[0, 1, 1, 0], [1, 0, 0, 1]]),
          'y_pred': np.array([[1, 0, 1, 0], [0, 1, 0, 1]]),
          'weights': np.array([[1, 2, 3, 4], [4, 3, 2, 1]]),
          'thresholds': [0.5],
          'expected_outputs': (3.0 + 1.0) / ((2.0 + 3.0) + (4.0 + 1.0)),
      },
      {
          'testcase_name': 'div_by_zero',
          'y_true': np.array([0, 0, 0, 0]),
          'y_pred': np.array([0, 0, 0, 0]),
          'weights': None,
          'thresholds': [0.5],
          'expected_outputs': 0.0,
      },
      {
          'testcase_name': 'unweighted_thresholds',
          'y_true': np.array([0, 1, 1, 0]),
          'y_pred': np.array([1, 0, 0.6, 0]),
          'weights': None,
          'thresholds': [0.5, 0.7],
          'expected_outputs': [0.5, 0.0],
      },
      {
          'testcase_name': 'weighted_thresholds',
          'y_true': np.array([[0, 1], [1, 0]]),
          'y_pred': np.array([[1, 0], [0.6, 0]], dtype='float32'),
          'weights': np.array([[1, 4], [3, 2]], dtype='float32'),
          'thresholds': [0.5, 1.0],
          'expected_outputs': [(0 + 3.0) / ((0 + 3.0) + (4.0 + 0.0)), 0.0],
      },
  )
  def test_recall(
      self,
      y_true: jax.Array,
      y_pred: jax.Array,
      weights: jax.Array | None,
      thresholds: Sequence[float],
      expected_outputs: float | Sequence[float],
  ):
    recall = confusion_metrics.recall(
        y_true, y_pred, weights, threshold=thresholds
    )
    np.testing.assert_allclose(expected_outputs, recall.compute())
    np.testing.assert_allclose(expected_outputs, recall.localize().compute())

  @parameterized.named_parameters({
      'testcase_name': 'unweighted',
      'y_true': np.array([
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
          [1, 0, 0],
          [1, 0, 0],
          [0, 0, 1],
      ]),
      'y_pred': np.array([
          [0.9, 0.1, 0],
          [0.2, 0.6, 0.2],
          [0, 0, 1],
          [0.4, 0.3, 0.3],
          [0, 0.9, 0.1],
          [0, 0, 1],
      ]),
      'weights': None,
      'thresholds': [0.5],
      'expected_outputs': 0.727,
  })
  def test_f1_score(
      self,
      y_true: jax.Array,
      y_pred: jax.Array,
      weights: jax.Array | None,
      thresholds: Sequence[float],
      expected_outputs: float,
  ):
    f1 = confusion_metrics.f1_score(
        y_true, y_pred, weights, threshold=thresholds
    )
    np.testing.assert_allclose(expected_outputs, f1.compute(), rtol=1e-3)
    np.testing.assert_allclose(
        expected_outputs, f1.localize().compute(), rtol=1e-3
    )

  @parameterized.named_parameters(
      {
          'testcase_name': 'weighted',
          'y_true': np.array([0, 0, 1, 1]),
          'y_pred': np.array([0, 0.5, 0.3, 0.9], dtype='float32'),
          'weights': np.array([1, 2, 3, 4]),
          'num_thresholds': 3,
          'from_logits': False,
          # auc = (slope / Total Pos) * [dTP - intercept * log(Pb/Pa)]
          # tp = [7, 4, 0], fp = [3, 0, 0], fn = [0, 3, 7], tn = [0, 3, 3]
          # P = tp + fp = [10, 4, 0]
          # dTP = [7-4, 4-0] = [3, 4]
          # dP = [10-4, 4-0] = [6, 4]
          # slope = dTP/dP = [0.5, 1]
          # intercept = (TPa+(slope*Pa) = [(4 - 0.5*4), (0 - 1*0)] = [2, 0]
          # (Pb/Pa) = (Pb/Pa) if Pb > 0 AND Pa > 0 else 1 = [10/4,4/0] = [2.5,1]
          # auc * TotalPos = [(0.5 * (3 + 2 * log(2.5))), (1 * (4 + 0))]
          #                = [2.416, 4]
          # auc = [2.416, 4]/(tp[1:]+fn[1:])
          'expected_outputs': 2.416 / 7 + 4 / 7,
      },
      {
          'testcase_name': 'weighted_negative',
          'y_true': np.array([0, 0, 1, 1]),
          'y_pred': np.array([0, 0.5, 0.3, 0.9], dtype='float32'),
          'weights': np.array([-1, -2, -3, -4]),
          'num_thresholds': 3,
          'from_logits': False,
          # Divisor in auc formula is max(tp[1:]+fn[1:], 0), which is all zeros
          # because the all values in tp and fn are negative, divide_no_nan will
          # produce all zeros.
          'expected_outputs': 0.0,
      },
  )
  def test_aucpr(
      self,
      y_true: jax.Array,
      y_pred: jax.Array,
      weights: jax.Array | None,
      num_thresholds: int,
      from_logits: bool,
      expected_outputs: float,
  ):
    aucpr = confusion_metrics.aucpr(
        y_true,
        y_pred,
        weights,
        from_logits=from_logits,
        num_thresholds=num_thresholds,
    )
    np.testing.assert_allclose(expected_outputs, aucpr.compute(), rtol=1e-3)
    np.testing.assert_allclose(
        expected_outputs, aucpr.localize().compute(), rtol=1e-3
    )

  @parameterized.named_parameters(
      {
          'testcase_name': 'unweighted_all_correct',
          'y_true': np.array([0, 0, 1, 1]),
          'y_pred': np.array([0, 0, 1, 1]),
          'weights': None,
          'num_thresholds': 3,
          'from_logits': False,
          'expected_outputs': 1.0,
      },
      {
          'testcase_name': 'unweighted',
          'y_true': np.array([0, 0, 1, 1]),
          'y_pred': np.array([0, 0.5, 0.3, 0.9]),
          'weights': None,
          'num_thresholds': 3,
          'from_logits': False,
          # tp = [2, 1, 0], fp = [2, 0, 0], fn = [0, 1, 2], tn = [0, 2, 2]
          # recall = [2/2, 1/(1+1), 0] = [1, 0.5, 0]
          # fp_rate = [2/2, 0, 0] = [1, 0, 0]
          # heights = [(1 + 0.5)/2, (0.5 + 0)/2] = [0.75, 0.25]
          # widths = [(1 - 0), (0 - 0)] = [1, 0]
          # aucroc = 0.75 * 1 + 0.25 * 0
          'expected_outputs': 0.75,
      },
      {
          'testcase_name': 'unweighted_from_logits',
          'y_true': np.array([0, 0, 1, 1]),
          'y_pred': -np.log(1.0 / (np.array([0, 0.5, 0.3, 0.9]) + 1e-12) - 1.0),
          'weights': None,
          'num_thresholds': 3,
          'from_logits': True,
          'expected_outputs': 0.75,  # Same as unweighted case.
      },
      {
          'testcase_name': 'weighted',
          'y_true': np.array([0, 0, 1, 1]),
          'y_pred': np.array([0, 0.5, 0.3, 0.9]),
          'weights': np.array([1, 2, 3, 4]),
          'num_thresholds': 3,
          'from_logits': False,
          # tp = [7, 4, 0], fp = [3, 0, 0], fn = [0, 3, 7], tn = [0, 3, 3]
          # recall = [7/7, 4/(4+3), 0] = [1, 0.571, 0]
          # fp_rate = [3/3, 0, 0] = [1, 0, 0]
          # heights = [(1 + 0.571)/2, (0.571 + 0)/2] = [0.7855, 0.2855]
          # widths = [(1 - 0), (0 - 0)] = [1, 0]
          # aucroc = 0.7855 * 1 + 0.2855 * 0
          'expected_outputs': 0.7855,
      },
  )
  def test_aucroc(
      self,
      y_true: jax.Array,
      y_pred: jax.Array,
      weights: jax.Array | None,
      num_thresholds: int,
      from_logits: bool,
      expected_outputs: float,
  ):
    aucroc = confusion_metrics.aucroc(
        y_true,
        y_pred,
        weights,
        from_logits=from_logits,
        num_thresholds=num_thresholds,
    )
    print(aucroc)
    np.testing.assert_allclose(expected_outputs, aucroc.compute(), rtol=1e-3)
    np.testing.assert_allclose(
        expected_outputs, aucroc.localize().compute(), rtol=1e-3
    )

  @parameterized.named_parameters(
      {
          'testcase_name': 'unweighted_high_recall',
          'y_true': np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
          'y_pred': np.array(
              [0.0, 0.1, 0.2, 0.5, 0.6, 0.2, 0.5, 0.6, 0.8, 0.9]
          ),
          'weights': None,
          'recall': 0.8,
          'expected_outputs': 2.0 / 3,
      },
      {
          'testcase_name': 'unweighted_low_recall',
          'y_true': np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
          'y_pred': np.array(
              [0.0, 0.1, 0.2, 0.5, 0.6, 0.2, 0.5, 0.6, 0.8, 0.9]
          ),
          'weights': None,
          'recall': 0.6,
          'expected_outputs': 0.75,
      },
      {
          'testcase_name': 'weighted',
          'y_true': np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
          'y_pred': np.array(
              [0.0, 0.1, 0.2, 0.5, 0.6, 0.2, 0.5, 0.6, 0.8, 0.9]
          ),
          'weights': np.array([2, 1, 2, 1, 2, 1, 2, 2, 1, 2]),
          'recall': 7.0 / 8,
          'expected_outputs': 0.7,
      },
  )
  def test_prediction_at_recall(
      self,
      y_true: jax.Array,
      y_pred: jax.Array,
      weights: jax.Array | None,
      recall: float,
      expected_outputs: float | Sequence[float],
  ):
    precision_at_recall = confusion_metrics.precision_at_recall(
        y_true, y_pred, weights=weights, recall=recall
    )
    np.testing.assert_allclose(
        expected_outputs, precision_at_recall.compute(), rtol=1e-3
    )
    np.testing.assert_allclose(
        expected_outputs, precision_at_recall.localize().compute(), rtol=1e-3
    )


if __name__ == '__main__':
  absltest.main()

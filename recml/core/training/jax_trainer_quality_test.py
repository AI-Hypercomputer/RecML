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
"""Tests for the quality of training loops."""

from collections.abc import Mapping
import functools

from absl import flags
from absl.testing import absltest
import clu.metrics as clu_metrics
import flax.linen as nn
from flax.training import train_state as ts
import jax
import jax.numpy as jnp
import jaxtyping as jt
import optax
from recml.core.training import jax_trainer
from recml.core.training import partitioning
import tensorflow as tf
import tensorflow_datasets as tfds


class _MNISTTask(jax_trainer.JaxTask):
  """Task for fitting a CNN on MNIST."""

  def create_datasets(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:

    def _preprocessor(batch: jt.PyTree) -> jt.PyTree:
      images = batch['image']
      labels = batch['label']
      images = tf.cast(images, tf.float32) / 255.0
      labels = tf.cast(labels, tf.int32)
      return images, labels

    def _create_dataset(training: bool) -> tf.data.Dataset:
      dataset = tfds.load(
          name='mnist',
          split='train' if training else 'test',
          batch_size=32,
          shuffle_files=training,
      )
      return dataset.map(_preprocessor).prefetch(buffer_size=tf.data.AUTOTUNE)

    return _create_dataset(training=True), _create_dataset(training=False)

  def create_state(self, batch: jt.PyTree, rng: jax.Array) -> ts.TrainState:
    images, _ = batch
    model = nn.Sequential([
        nn.Conv(32, kernel_size=(3, 3)),
        nn.relu,
        functools.partial(nn.max_pool, window_shape=(2, 2), strides=(2, 2)),
        nn.Conv(64, kernel_size=(3, 3)),
        nn.relu,
        functools.partial(nn.max_pool, window_shape=(2, 2), strides=(2, 2)),
        lambda x: x.reshape((x.shape[0], -1)),
        nn.Dense(256),
        nn.relu,
        nn.Dense(10),
    ])
    variables = model.init(rng, jnp.zeros_like(images))
    optimizer = optax.sgd(0.1)
    return ts.TrainState.create(
        apply_fn=model.apply, params=variables, tx=optimizer
    )

  def train_step(
      self, batch: jt.PyTree, state: ts.TrainState, rng: jax.Array
  ) -> tuple[ts.TrainState, Mapping[str, clu_metrics.Metric]]:
    images, labels = batch

    def _loss_fn(params):
      logits = state.apply_fn(params, images)
      loss = jnp.mean(
          optax.softmax_cross_entropy_with_integer_labels(logits, labels),
          axis=0,
      )
      return loss, (logits, labels)

    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
    (loss, (logits, labels)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = {
        'loss': clu_metrics.Average.from_model_output(loss),
        'accuracy': clu_metrics.Accuracy.from_model_output(
            logits=logits, labels=labels
        ),
    }
    return state, metrics

  def eval_step(
      self, batch: jt.PyTree, state: ts.TrainState
  ) -> Mapping[str, clu_metrics.Metric]:
    images, labels = batch
    logits = state.apply_fn(state.params, images)
    loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    )
    metrics = {
        'loss': clu_metrics.Average.from_model_output(loss),
        'accuracy': clu_metrics.Accuracy.from_model_output(
            logits=logits, labels=labels
        ),
    }
    return metrics


class JaxQualityTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Workaround to make `create_tempdir` work with pytest.
    if not flags.FLAGS.is_parsed():
      flags.FLAGS.mark_as_parsed()

  def test_mnist_e2e(self):
    model_dir = self.create_tempdir().full_path
    task = _MNISTTask()
    trainer = jax_trainer.JaxTrainer(
        partitioner=partitioning.DataParallelPartitioner(),
        train_steps=1000,
        steps_per_eval=50,
        steps_per_loop=100,
        continuous_eval_timeout=5,
        model_dir=model_dir,
        rng_seed=42,
    )
    logs = trainer.train_and_evaluate(task)
    self.assertGreater(logs['train']['accuracy'], 0.95)
    self.assertGreater(logs['val']['accuracy'], 0.95)

    self.assertTrue(tf.io.gfile.exists(model_dir))
    continuous_eval_logs = trainer.evaluate_continuously(task)
    self.assertGreater(continuous_eval_logs['val']['accuracy'], 0.95)


if __name__ == '__main__':
  absltest.main()

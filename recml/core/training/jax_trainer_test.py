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
"""Tests for Jax task and trainer."""

from collections.abc import Mapping, Sequence
import dataclasses
import os

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import clu.metrics as clu_metrics
import flax.linen as nn
from flax.training import train_state as ts
import jax
import jax.numpy as jnp
import jaxtyping as jt
import keras
import optax
import orbax.checkpoint as ocp
from recml.core.training import core
from recml.core.training import jax_trainer
from recml.core.training import partitioning
import tensorflow as tf


class _DummyFlaxModel(nn.Module):

  @nn.compact
  def __call__(self, inputs: jax.Array) -> jax.Array:
    return nn.Dense(1, kernel_init=nn.initializers.constant(-1.0))(inputs)


class _JaxTask(jax_trainer.JaxTask):

  def create_datasets(
      self,
  ) -> tuple[tf.data.Dataset, Mapping[str, tf.data.Dataset]]:
    def _map_fn(x: int):
      return (tf.cast(x, tf.float32), 0.1 * tf.cast(x, tf.float32) + 3)

    return tf.data.Dataset.range(1000).map(_map_fn).batch(2), {
        "eval_on_train": tf.data.Dataset.range(1000).map(_map_fn).batch(2),
        "eval_on_test": tf.data.Dataset.range(2000).map(_map_fn).batch(2),
    }

  def create_state(self, batch: jt.PyTree, rng: jax.Array) -> ts.TrainState:
    x, _ = batch
    model = _DummyFlaxModel()
    params = model.init(rng, x)
    optimizer = optax.adagrad(0.1)
    return ts.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

  def train_step(
      self, batch: jt.PyTree, state: ts.TrainState, rng: jax.Array
  ) -> tuple[ts.TrainState, Mapping[str, clu_metrics.Metric]]:
    x, y = batch

    def _loss_fn(params):
      y_pred = state.apply_fn(params, x)
      loss = keras.losses.mean_squared_error(y, y_pred)
      return loss

    grad_fn = jax.value_and_grad(_loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, {"loss": clu_metrics.Average.from_model_output(loss)}

  def eval_step(
      self, batch: jt.PyTree, state: ts.TrainState
  ) -> Mapping[str, clu_metrics.Metric]:
    x, y = batch
    y_pred = state.apply_fn(state.params, x)
    loss = keras.losses.mean_squared_error(y, y_pred)
    return {"loss": clu_metrics.Average.from_model_output(loss)}


class _KerasJaxTask(jax_trainer.JaxTask):

  def create_datasets(self) -> tf.data.Dataset:
    def _map_fn(x: int):
      return (
          tf.expand_dims(tf.cast(x, tf.float32), axis=-1),
          0.1 * tf.cast(x, tf.float32) + 3,
      )

    return (
        tf.data.Dataset.range(1000).map(_map_fn).batch(2),
        tf.data.Dataset.range(2000).map(_map_fn).batch(2),
    )

  def create_state(
      self, batch: jt.PyTree, rng: jax.Array
  ) -> jax_trainer.KerasState:
    x, _ = batch

    model = keras.Sequential(
        [
            keras.layers.Dense(
                1,
                kernel_initializer=keras.initializers.constant(-1.0),
                name="dense",
            ),
        ],
        name="model",
    )
    model.build(x.shape)

    optimizer = optax.adagrad(0.1)
    return jax_trainer.KerasState.create(model=model, tx=optimizer)

  def train_step(
      self, batch: jt.PyTree, state: jax_trainer.KerasState, rng: jax.Array
  ) -> tuple[jax_trainer.KerasState, Mapping[str, clu_metrics.Metric]]:
    x, y = batch

    def _loss_fn(tvars):
      y_pred, _ = state.model.stateless_call(tvars, state.ntvars, x)
      loss = keras.ops.mean(keras.losses.mean_squared_error(y, y_pred))
      return loss

    grad_fn = jax.value_and_grad(_loss_fn)
    loss, grads = grad_fn(state.tvars)
    state = state.update(grads=grads)
    return state, {"loss": clu_metrics.Average.from_model_output(loss)}

  def eval_step(
      self, batch: jt.PyTree, state: jax_trainer.KerasState
  ) -> Mapping[str, clu_metrics.Metric]:
    x, y = batch
    y_pred, _ = state.model.stateless_call(state.tvars, state.ntvars, x)
    loss = keras.losses.mean_squared_error(y, y_pred)
    return {"loss": clu_metrics.Average.from_model_output(loss)}


class JaxTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Workaround to make `create_tempdir` work with pytest.
    if not flags.FLAGS.is_parsed():
      flags.FLAGS.mark_as_parsed()

  @parameterized.named_parameters(
      {
          "testcase_name": "jax_task_train",
          "task_cls": _JaxTask,
          "mode": core.Experiment.Mode.TRAIN,
          "expected_keys": ["train"],
      },
      {
          "testcase_name": "keras_jax_task_train",
          "task_cls": _KerasJaxTask,
          "mode": core.Experiment.Mode.TRAIN,
          "expected_keys": ["train"],
      },
      {
          "testcase_name": "jax_task_eval",
          "task_cls": _JaxTask,
          "mode": core.Experiment.Mode.EVAL,
          "expected_keys": ["val_eval_on_train", "val_eval_on_test"],
      },
      {
          "testcase_name": "keras_jax_task_eval",
          "task_cls": _KerasJaxTask,
          "mode": core.Experiment.Mode.EVAL,
          "expected_keys": ["val"],
      },
      {
          "testcase_name": "jax_task_train_and_eval",
          "task_cls": _JaxTask,
          "mode": core.Experiment.Mode.TRAIN_AND_EVAL,
          "expected_keys": ["train", "val_eval_on_train", "val_eval_on_test"],
      },
      {
          "testcase_name": "keras_jax_task_train_and_eval",
          "task_cls": _KerasJaxTask,
          "mode": core.Experiment.Mode.TRAIN_AND_EVAL,
          "expected_keys": ["train", "val"],
      },
      {
          "testcase_name": "jax_task_continuous_eval",
          "task_cls": _JaxTask,
          "mode": core.Experiment.Mode.CONTINUOUS_EVAL,
          "expected_keys": ["val_eval_on_train", "val_eval_on_test"],
      },
      {
          "testcase_name": "keras_jax_task_continuous_eval",
          "task_cls": _KerasJaxTask,
          "mode": core.Experiment.Mode.CONTINUOUS_EVAL,
          "expected_keys": ["val"],
      },
  )
  def test_jax_trainer(
      self,
      task_cls: type[jax_trainer.JaxTask],
      mode: str,
      expected_keys: Sequence[str],
  ):
    model_dir = self.create_tempdir().full_path
    task = task_cls()
    trainer = jax_trainer.JaxTrainer(
        partitioner=partitioning.DataParallelPartitioner(data_axis="batch"),
        train_steps=12,
        steps_per_eval=3,
        steps_per_loop=4,
        model_dir=model_dir,
        continuous_eval_timeout=5,
    )
    experiment = core.Experiment(task, trainer)
    if mode == core.Experiment.Mode.CONTINUOUS_EVAL:
      # Produce one checkpoint so there is something to evaluate.
      core.run_experiment(experiment, core.Experiment.Mode.TRAIN)
    logs = core.run_experiment(experiment, mode)

    for key in expected_keys:
      self.assertIn(key, logs)
      self.assertIn("loss", logs[key])

    if mode in [
        core.Experiment.Mode.TRAIN,
        core.Experiment.Mode.TRAIN_AND_EVAL,
    ]:
      checkpointed_steps = ocp.utils.checkpoint_steps(
          os.path.join(model_dir, core.CHECKPOINT_DIR)
      )
      self.assertEqual([3, 7, 11], sorted(checkpointed_steps))

    # TODO(aahil): Check the logs for the correct summaries.
    # TODO(aahil): Test exporting here.

  def test_optimizer_metrics(self):
    @dataclasses.dataclass
    class State:
      step: int
      opt_state: optax.OptState

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.inject_stateful_hyperparams(optax.scale_by_learning_rate)(
            learning_rate=0.1
        ),
    )
    state = State(step=10, opt_state=tx.init({"a": jnp.ones((10, 10))}))
    metrics = jax_trainer._state_metrics(state)
    self.assertIn("optimizer/learning_rate", metrics)
    self.assertEqual(metrics["optimizer/learning_rate"].compute(), 0.1)


if __name__ == "__main__":
  absltest.main()

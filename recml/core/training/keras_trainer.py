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
"""Keras task and trainer."""

from __future__ import annotations

import abc
from collections.abc import Mapping
import dataclasses
import functools
import gc
import os
import time
from typing import Any

from absl import logging
from etils import epy
import fiddle as fdl
import jax
import keras
import orbax.checkpoint as ocp
import psutil
from recml.core.training import core
from recml.core.utils import py_utils
import tensorflow as tf

with epy.lazy_imports():
  from recml.core.utils import keras_utils  # pylint: disable=g-import-not-at-top, g-bad-import-order


class KerasTask(abc.ABC):
  """Keras 3 task for use with the Keras trainer."""

  @abc.abstractmethod
  def create_dataset(self, training: bool) -> tf.data.Dataset:
    """Creates a training or evaluation dataset.

    Args:
      training: A flag indicating whether the dataset is for training or
        evaluation.

    Returns:
      A `tf.data.Dataset` instance.
    """

  @abc.abstractmethod
  def create_model(self, **kwargs) -> keras.Model:
    """Creates a Keras model.

    You must call `model.compile` here and may also manually `build` the model.

    Args:
      **kwargs: Optional keyword arguments for model creation. Can be one of the
        following, `input_shapes`- a PyTree of input shapes of the same
        structure as the inputs to the model.

    Returns:
      A Keras model instance.
    """

  def create_model_for_eval(self, **kwargs) -> keras.Model:
    """Creates a Keras model for evaluation.

    Defaults to the implementation of `create_model`.

    Args:
      **kwargs: Optional keyword arguments for model creation. Can be one of the
        following, `input_shapes`- a PyTree of input shapes of the same
        structure as the inputs to the model.

    Returns:
      A Keras model instance.
    """
    return self.create_model(**kwargs)

  def export_model(self, model: keras.Model, model_dir: str):
    """Exports a Keras model.

    By default, the model is exported to '{model_dir}/model.keras' in Keras
    format, which is lossless. It can easily be loaded using
    `keras.saving.load_model` and re-exported.

    Args:
      model: The Keras model constructed by `create_model`.
      model_dir: The model directory passed to the trainer.
    """


class KerasTrainer(core.Trainer[KerasTask]):
  """Keras 3 trainer for use with the Keras task."""

  def __init__(
      self,
      *,
      distribution: (
          keras.distribution.DataParallel
          | keras.distribution.ModelParallel
          | None
      ) = None,
      model_dir: str | None = None,
      train_steps: int = 0,
      steps_per_eval: int | None = None,
      continuous_eval_timeout: int = 30,
      steps_per_loop: int = 1_000,
      max_checkpoints_to_keep: int = 5,
      checkpoint_save_interval_epochs: int = 1,
      rng_seed: int = core.DEFAULT_RNG_SEED,
      legacy_checkpoint_format: bool = True,
  ):
    """Initializes the instance."""

    keras.utils.set_random_seed(rng_seed)

    if model_dir is None:
      model_dir = "/tmp"

    # This should be set before any layers are constructed and this is a
    # fallback in case the trainer binary doesn't already do this.
    if (
        distribution is not None
        and keras.distribution.distribution() != distribution
    ):
      if hasattr(distribution, "_auto_shard_dataset"):
        setattr(distribution, "_auto_shard_dataset", False)
      keras.distribution.set_distribution(distribution)

    self._distribution = distribution
    self._model_dir = model_dir
    self._train_epochs = train_steps // steps_per_loop
    self._steps_per_eval = steps_per_eval
    self._continuous_eval_timeout = continuous_eval_timeout
    self._steps_per_loop = steps_per_loop
    self._marker_path = os.path.join(
        model_dir, core.TRAINING_COMPLETE_MARKER_FILE
    )
    self._checkpoint_dir = os.path.join(model_dir, core.CHECKPOINT_DIR)
    self._max_checkpoints_to_keep = max_checkpoints_to_keep
    self._checkpoint_save_interval_epochs = checkpoint_save_interval_epochs
    self._legacy_checkpoint_format = legacy_checkpoint_format

  @functools.cached_property
  def train_callbacks(self) -> list[keras.callbacks.Callback]:
    """Returns the training callbacks."""
    if keras.backend.backend() == "jax":
      if self._legacy_checkpoint_format:
        checkpoint_manager = keras_utils.KerasOrbaxCheckpointManager(
            checkpoint_dir=self._checkpoint_dir,
            max_to_keep=self._max_checkpoints_to_keep,
            save_interval_epochs=self._checkpoint_save_interval_epochs,
        )
      else:
        checkpoint_manager = keras_utils.KerasOrbaxCheckpointManagerV2(
            checkpoint_dir=self._checkpoint_dir,
            max_to_keep=self._max_checkpoints_to_keep,
            save_interval_epochs=self._checkpoint_save_interval_epochs,
        )
      callbacks = [
          keras_utils.EpochSummaryCallback(
              log_dir=os.path.join(self._model_dir, core.LOG_DIR),
              steps_per_epoch=self._steps_per_loop,
              write_steps_per_second=True,
          ),
          keras_utils.EpochOrbaxCheckpointAndRestoreCallback(
              checkpoint_manager=checkpoint_manager,
              marker_path=self._marker_path,
          ),
      ]
      return callbacks
    return [
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(self._model_dir, core.LOG_DIR),
            write_steps_per_second=True,
        ),
        keras.callbacks.BackupAndRestore(
            backup_dir=os.path.join(self._model_dir, core.BACKUP_DIR),
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                self._model_dir,
                core.CHECKPOINT_DIR,
                "ckpt-{epoch:d}.weights.h5",
            ),
            save_weights_only=True,
            verbose=1,
        ),
    ]

  @functools.cached_property
  def eval_callbacks(self) -> list[keras.callbacks.Callback]:
    """Returns the evaluation callbacks."""
    if keras.backend.backend() == "jax":
      return [
          keras_utils.EpochSummaryCallback(
              log_dir=os.path.join(self._model_dir, core.LOG_DIR),
              steps_per_epoch=self._steps_per_loop,
              write_steps_per_second=False,
          ),
      ]
    return [
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(self._model_dir, core.LOG_DIR),
            write_steps_per_second=True,
        ),
    ]

  def _maybe_get_model_kws(
      self, task: KerasTask, dataset: tf.data.Dataset
  ) -> Mapping[str, Any]:
    kws = {}
    if py_utils.has_argument(task.create_model, "input_shapes"):
      batch_spec = dataset.element_spec
      x, *_ = keras.utils.unpack_x_y_sample_weight(batch_spec)
      kws["input_shapes"] = keras.tree.map_structure(core.get_shape, x)

    return kws

  def train(self, task: KerasTask) -> core.Logs:
    """Trains a Keras model."""
    dataset = task.create_dataset(training=True)
    model = task.create_model(**self._maybe_get_model_kws(task, dataset))

    history = model.fit(
        dataset,
        epochs=self._train_epochs + 1,
        steps_per_epoch=self._steps_per_loop,
        callbacks=self.train_callbacks,
        initial_epoch=1,
    )
    model.summary(print_fn=logging.info)

    if keras.backend.backend() == "jax" and jax.process_index() != 0:
      return
    task.export_model(model, self._model_dir)
    return history

  def evaluate(self, task: KerasTask) -> core.Logs:
    """Evaluates a Keras model."""
    dataset = task.create_dataset(training=False)
    model = task.create_model_for_eval(
        **self._maybe_get_model_kws(task, dataset)
    )

    if keras.backend.backend() == "jax":
      [tb_cbk] = [
          cbk
          for cbk in self.eval_callbacks
          if isinstance(cbk, keras_utils.EpochSummaryCallback)
      ]
      epoch_start_time = time.time()
      history = model.evaluate(
          dataset,
          steps=self._steps_per_eval,
          callbacks=self.eval_callbacks,
          return_dict=True,
      )
      epoch_dt = time.time() - epoch_start_time
      steps_per_second = self._steps_per_eval / epoch_dt
      val_logs = {"val_" + k: v for k, v in history.items()}
      val_logs["val_steps_per_second"] = steps_per_second
      tb_cbk.on_epoch_end(0, val_logs)
      return history

    return model.evaluate(
        dataset,
        steps=self._steps_per_eval,
        callbacks=self.eval_callbacks,
    )

  def train_and_evaluate(self, task: KerasTask) -> core.Logs:
    """Trains and evaluates a Keras model."""
    train_dataset = task.create_dataset(training=True)
    eval_dataset = task.create_dataset(training=False)

    if self._steps_per_eval is not None:
      eval_dataset = eval_dataset.take(self._steps_per_eval)

    model = task.create_model(**self._maybe_get_model_kws(task, train_dataset))

    history = model.fit(
        train_dataset,
        validation_data=eval_dataset,
        epochs=self._train_epochs + 1,
        steps_per_epoch=self._steps_per_loop,
        # Explicitly set to None for deterministic evaluation.
        validation_steps=None,
        callbacks=self.train_callbacks,
        initial_epoch=1,
    )
    model.summary(print_fn=logging.info)

    if keras.backend.backend() == "jax" and jax.process_index() != 0:
      return
    task.export_model(model, self._model_dir)
    return history

  def evaluate_continuously(self, task: KerasTask) -> core.Logs | None:
    """Continuously evaluates a Keras model."""
    # This is typically used for a sidecar evaluation job.

    if keras.backend.backend() != "jax":
      raise NotImplementedError(
          "Continuous evaluation is only supported on the Jax backend."
      )

    eval_dataset = task.create_dataset(training=False)
    model = task.create_model_for_eval(
        **self._maybe_get_model_kws(task, eval_dataset)
    )

    def timeout_fn() -> bool:
      return tf.io.gfile.exists(self._marker_path)

    if self._steps_per_eval is not None:
      steps_msg = f"running {self._steps_per_eval} steps of evaluation..."
    else:
      steps_msg = "running complete evaluation..."

    use_legacy_checkpoint_format = self._legacy_checkpoint_format

    class _RestoreCallback(keras.callbacks.Callback):
      """Callback for restoring the model from the latest checkpoint."""

      def __init__(
          self,
          checkpoint_dir: str,
          epoch: int,
      ):
        self._checkpoint_dir = checkpoint_dir
        self._epoch = epoch

      def on_test_begin(self, logs: Mapping[str, Any] | None = None):
        if use_legacy_checkpoint_format:
          keras_utils.restore_keras_model(
              model, self._checkpoint_dir, step=self._epoch
          )
        else:
          keras_utils.restore_keras_checkpoint(
              self._checkpoint_dir, model=model, epoch=self._epoch
          )

    history = None
    for epoch in ocp.checkpoint_utils.checkpoints_iterator(
        self._checkpoint_dir,
        timeout=self._continuous_eval_timeout,
        timeout_fn=timeout_fn,
    ):
      restore_callback = _RestoreCallback(self._checkpoint_dir, epoch)
      [tb_cbk] = [
          cbk
          for cbk in self.eval_callbacks
          if isinstance(cbk, keras_utils.EpochSummaryCallback)
      ]
      try:
        logging.info(f"eval | epoch: {epoch: 6d} | {steps_msg}")
        epoch_start_time = time.time()

        logging.info(
            "[Before] Memory usage:"
            f" {psutil.Process().memory_info().rss / 1024 ** 2:.1f} MB"
        )
        history = model.evaluate(
            eval_dataset,
            steps=self._steps_per_eval,
            callbacks=[restore_callback] + self.eval_callbacks,
            return_dict=True,
        )

        logging.info(
            "[After] Memory usage:"
            f" {psutil.Process().memory_info().rss / 1024 ** 2:.1f} MB"
        )
        gc.collect()
        logging.info(
            "[After GC] Memory usage:"
            f" {psutil.Process().memory_info().rss / 1024 ** 2:.1f} MB"
        )
        epoch_dt = time.time() - epoch_start_time
        steps_per_second = self._steps_per_eval / epoch_dt

        val_logs = {"val_" + k: v for k, v in history.items()}
        val_logs["val_steps_per_second"] = steps_per_second
        tb_cbk.on_epoch_end(epoch, val_logs)
      except FileNotFoundError:
        logging.info("Checkpoint epoch: %s did not finish writing...", epoch)

    return history

  @classmethod
  def setup_experiment(cls, experiment_cfg: fdl.Config[core.Experiment]):
    # Disable Keras traceback filtering to get more informative stack traces.
    keras.config.disable_traceback_filtering()

    # Building the RNG seed and setting it globally before building the rest of
    # the experiment ensures that any random operations in `__init__` are also
    # deterministic. We do this before setting the distribution in case the
    # distribution uses random operations, for instance by overriding the seed
    # generator.
    if isinstance(experiment_cfg.trainer.rng_seed, fdl.Buildable):
      seed = fdl.build(experiment_cfg.trainer.rng_seed)
    else:
      seed = experiment_cfg.trainer.rng_seed
    keras.utils.set_random_seed(seed)

    # Building the distribution and setting it globally before building the
    # rest of the experiment ensures that any layers passed to the task are
    # constructed under the distribution.
    if (distribution_cfg := experiment_cfg.trainer.distribution) is not None:
      distribution = fdl.build(distribution_cfg)
      keras.distribution.set_distribution(distribution)

    # Set the threefry 2x32 key to be partitionable by default.
    jax.config.update("jax_threefry_partitionable", True)

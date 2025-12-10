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
"""Core training library for Jax."""

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
import dataclasses
import enum
from typing import Any, Generic, TypeVar

import fiddle as fdl
import jax
import jax.numpy as jnp
from recml.core.data import iterator
import tensorflow as tf


# pylint: disable=logging-fstring-interpolation

LOG_DIR = "logs"
BACKUP_DIR = "backup"
CHECKPOINT_DIR = "checkpoints"
TRAINING_COMPLETE_MARKER_FILE = "marker.txt"
TRAIN_LOG_DIRNAME = "train"
EVAL_LOG_DIRNAME = "val"
KERAS_MODEL_SAVEFILE = "model.keras"
ORBAX_CHECKPOINT_DEFAULT_KEY = "default"

DEFAULT_RNG_SEED = 0
STATE_CHECKPOINT_KEY = "state"

TaskT = TypeVar("TaskT")
DatasetT = TypeVar(
    "DatasetT",
    tf.data.Dataset,
    tuple[tf.data.Dataset, tf.data.Dataset],
    tuple[tf.data.Dataset, Mapping[str, tf.data.Dataset]],
    iterator.Iterator,
    tuple[iterator.Iterator, iterator.Iterator],
    tuple[iterator.Iterator, Mapping[str, iterator.Iterator]],
)
MetaT = TypeVar("MetaT")
Logs = Any  # Any metric logs returned by the training or evaluation task.


class Trainer(abc.ABC, Generic[TaskT]):
  """A base trainer interface for training and evaluation."""

  class Mode(enum.StrEnum):
    """Mode to run an experiment."""

    TRAIN = "train"
    EVAL = "eval"
    TRAIN_AND_EVAL = "train_and_eval"
    CONTINUOUS_EVAL = "continuous_eval"

  @abc.abstractmethod
  def __init__(self, model_dir: str, *args, **kwargs):
    """Initializes the instance."""

  @abc.abstractmethod
  def train(self, task: TaskT, *args, **kwargs) -> Logs | None:
    """Performs training for a fixed number of steps."""

  @abc.abstractmethod
  def evaluate(self, task: TaskT, *args, **kwargs) -> Logs | None:
    """Performs evaluation for a fixed number of steps."""

  @abc.abstractmethod
  def train_and_evaluate(self, task: TaskT, *args, **kwargs) -> Logs | None:
    """Performs training and evaluation for a fixed number of steps."""

  @abc.abstractmethod
  def evaluate_continuously(self, task: TaskT, *args, **kwargs) -> Logs | None:
    """Performs continuous evaluation until a condition is met."""

  def run(self, task: TaskT, mode: Any) -> Logs | None:
    """Runs the experiment in the given mode."""
    if mode == Trainer.Mode.TRAIN_AND_EVAL:
      return self.train_and_evaluate(task)
    elif mode == Trainer.Mode.TRAIN:
      return self.train(task)
    elif mode == Trainer.Mode.EVAL:
      return self.evaluate(task)
    elif mode == Trainer.Mode.CONTINUOUS_EVAL:
      return self.evaluate_continuously(task)
    else:
      raise ValueError(f"The job mode provided is not supported: {mode}.")

  @classmethod
  def setup_experiment(cls, experiment_cfg: fdl.Config[Experiment]):
    """Sets up the experiment before it is instantiated."""


@dataclasses.dataclass(frozen=True)
class Experiment(Generic[TaskT]):
  """Experiment definition.

  Properties:
    Mode: The mode to run the experiment in.

  Attributes:
    task: A user defined task that defines the training and evaluation logic.
    trainer: The trainer to use for the experiment.
  """

  task: TaskT
  trainer: Trainer[TaskT]


def run_experiment(experiment: Experiment, mode: Any) -> Logs | None:
  """Runs an experiment."""
  return experiment.trainer.run(experiment.task, mode)


def get_iterators(
    datasets: DatasetT,
) -> tuple[iterator.Iterator, Mapping[str, iterator.Iterator]]:
  """Creates and unpacks the datasets returned by the task."""
  if isinstance(datasets, (iterator.Iterator, tf.data.Dataset)):
    if isinstance(datasets, tf.data.Dataset):
      datasets = iterator.TFDatasetIterator(datasets)
    return datasets, {}
  elif not isinstance(datasets, tuple) and len(datasets) != 2:
    raise ValueError(
        "Expected `datasets` to be a single dataset or a tuple of training"
        f" and evaluation datasets, but got {type(datasets)}."
    )

  train_dataset, eval_datasets = datasets
  if isinstance(train_dataset, (iterator.Iterator, tf.data.Dataset)):
    if isinstance(train_dataset, tf.data.Dataset):
      train_dataset = iterator.TFDatasetIterator(train_dataset)
  else:
    raise ValueError(
        "Expected the training dataset in `datasets` to be a"
        " `tf.data.Dataset` or CLU `DatasetIterator` instance, but"
        f" {type(train_dataset)}."
    )

  if isinstance(eval_datasets, (iterator.Iterator, tf.data.Dataset)):
    if isinstance(eval_datasets, tf.data.Dataset):
      eval_datasets = iterator.TFDatasetIterator(eval_datasets)
    return train_dataset, {"": eval_datasets}

  if not isinstance(eval_datasets, Mapping):
    raise ValueError(
        "Expected the evaluation dataset in `datasets` to either be a"
        " `tf.data.Dataset` or CLU `DatasetIterator` instance or be a"
        " mapping of datasets keyed by name, but got"
        f" {type(eval_datasets)}."
    )

  if all(isinstance(v, tf.data.Dataset) for v in eval_datasets.values()):
    eval_datasets = {
        k: iterator.TFDatasetIterator(v) for k, v in eval_datasets.items()
    }

  if not all(isinstance(v, iterator.Iterator) for v in eval_datasets.values()):
    raise ValueError(
        "Expected all values in the evaluation datasets mapping to be either"
        " `tf.data.Dataset` instances or CLU `DatasetIterator` instances,"
        f" but got {eval_datasets}. You cannot mix both."
    )

  return train_dataset, eval_datasets  # pytype: disable=bad-return-type


def get_shape(
    x: tf.Tensor | tf.SparseTensor | tf.RaggedTensor | tf.TensorSpec,
) -> Sequence[int | None]:
  """Gets the shape of a dense / sparse / ragged tensor or tensor spec."""
  if isinstance(x, tf.SparseTensor):
    return [x.shape[0]] + [None for _ in x.shape[1:]]
  return x.shape.as_list()  # pylint: disable=attribute-error


def in_tracing_context() -> bool:
  """Returns whether the current context is a tracing context."""
  return isinstance(jnp.ones(()), jax.core.Tracer)

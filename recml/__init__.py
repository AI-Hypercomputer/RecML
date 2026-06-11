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
"""Public API for RecML."""

# pylint: disable=g-importing-member

from recml.core import data
from recml.core import metrics
from recml.core import utils
from recml.core.metrics.base_metrics import Metric
from recml.core.training.core import Experiment
from recml.core.training.core import run_experiment
from recml.core.training.core import Trainer
from recml.core.training.jax_trainer import JaxState
from recml.core.training.jax_trainer import JaxTask
from recml.core.training.jax_trainer import JaxTrainer
from recml.core.training.jax_trainer import KerasState
from recml.core.training.keras_trainer import KerasTask
from recml.core.training.keras_trainer import KerasTrainer
from recml.core.training.optax_factory import AdagradFactory
from recml.core.training.optax_factory import AdamFactory
from recml.core.training.optax_factory import OptimizerFactory
from recml.core.training.partitioning import DataParallelPartitioner
from recml.core.training.partitioning import ModelParallelPartitioner
from recml.core.training.partitioning import NullPartitioner
from recml.core.training.partitioning import Partitioner
from recml.core.utils.types import Factory
from recml.core.utils.types import FactoryProtocol
from recml.core.utils.types import ObjectFactory
from recml.layers.common import EmbeddingSpec

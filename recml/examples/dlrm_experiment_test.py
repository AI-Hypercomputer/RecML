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
"""Tests for the DLRM experiment."""

from absl.testing import absltest
import fiddle as fdl
from fiddle import selectors
import jax
import numpy as np
import recml
from recml.examples import dlrm_experiment


class DLRMExperimentTest(absltest.TestCase):

  def test_dlrm_experiment(self):
    if jax.devices()[0].platform != "tpu":
      self.skipTest("Test only supported on TPUs.")

    np.random.seed(1337)

    experiment = dlrm_experiment.experiment()

    experiment.task.train_data.global_batch_size = 4
    experiment.task.eval_data.global_batch_size = 4
    experiment.trainer.train_steps = 12
    experiment.trainer.steps_per_loop = 4
    experiment.trainer.steps_per_eval = 4

    for cfg in selectors.select(experiment, dlrm_experiment.SparseFeature):
      cfg.vocab_size = 200
      cfg.embedding_dim = 8

    experiment = fdl.build(experiment)
    recml.run_experiment(experiment, recml.Trainer.Mode.TRAIN_AND_EVAL)


if __name__ == "__main__":
  absltest.main()

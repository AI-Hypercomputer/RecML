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
"""Tests for Jax partitioners."""

from collections.abc import Mapping

from absl.testing import absltest
from absl.testing import parameterized
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from recml.core.training import partitioning


class PartitioningTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "data_parallel_partitioner",
          "partitioner_cls": partitioning.DataParallelPartitioner,
      },
      {
          "testcase_name": "model_parallel_partitioner",
          "partitioner_cls": partitioning.ModelParallelPartitioner,
      },
  )
  def test_data_parallelism(
      self, partitioner_cls: type[partitioning.Partitioner]
  ):
    if partitioner_cls is partitioning.ModelParallelPartitioner:
      kwargs = {"axes": [("data", jax.device_count()), ("model", 1)]}
    else:
      kwargs = {}
    partitioner = partitioner_cls(**kwargs)

    inputs = np.zeros((128, 16), dtype=np.float32)
    sharded_inputs = partitioner.shard_inputs(inputs)

    self.assertIsInstance(sharded_inputs, jax.Array)
    self.assertSequenceEqual(sharded_inputs.shape, (128, 16))
    self.assertEqual(sharded_inputs.sharding, partitioner.data_sharding)

    def _init(batch: jax.Array) -> jax.Array:
      return jnp.ones_like(batch)

    def _train_step(
        batch: jax.Array, state: jax.Array
    ) -> tuple[jax.Array, Mapping[str, jax.Array]]:
      return batch + state, {
          "batch_mean": jnp.mean(batch),
          "state_mean": jnp.mean(state),
      }

    def _eval_step(
        batch: jax.Array, state: jax.Array
    ) -> Mapping[str, jax.Array]:
      return {"batch_mean": jnp.mean(batch), "state_mean": jnp.mean(state)}

    state = partitioner.partition_init(_init, abstract_batch=sharded_inputs)(
        sharded_inputs
    )
    self.assertIsInstance(state, jax.Array)
    self.assertSequenceEqual(state.shape, (128, 16))
    self.assertEqual(state.sharding, partitioner.state_sharding)

    new_state, metrics = partitioner.partition_step(_train_step, training=True)(
        sharded_inputs, state
    )
    self.assertTrue(state.is_deleted())  # Buffer should be donated.
    self.assertIsInstance(new_state, jax.Array)
    self.assertSequenceEqual(new_state.shape, (128, 16))
    self.assertEqual(new_state.sharding, partitioner.state_sharding)
    for metric in jax.tree.flatten(metrics)[0]:
      self.assertIsInstance(metric, jax.Array)
      self.assertEqual(
          metric.sharding,
          jax.sharding.NamedSharding(
              partitioner.mesh, jax.sharding.PartitionSpec()
          ),
      )

    metrics = partitioner.partition_step(_eval_step, training=False)(
        sharded_inputs, new_state
    )
    self.assertFalse(new_state.is_deleted())  # Buffer should not be donated.
    for metric in jax.tree.flatten(metrics)[0]:
      self.assertIsInstance(metric, jax.Array)
      self.assertEqual(
          metric.sharding,
          jax.sharding.NamedSharding(
              partitioner.mesh, jax.sharding.PartitionSpec()
          ),
      )

    self.assertEqual(
        partitioner.state_sharding,
        jax.sharding.NamedSharding(
            partitioner.mesh, jax.sharding.PartitionSpec()
        ),
    )

  def test_model_parallelism(self):
    partitioner = partitioning.ModelParallelPartitioner(
        axes=[("data", 1), ("model", jax.device_count())]
    )

    inputs = np.zeros((128, 16), dtype=np.float32)
    sharded_inputs = partitioner.shard_inputs(inputs)

    self.assertIsInstance(sharded_inputs, jax.Array)
    self.assertSequenceEqual(sharded_inputs.shape, (128, 16))
    self.assertEqual(
        sharded_inputs.sharding,
        jax.sharding.NamedSharding(
            partitioner.mesh, jax.sharding.PartitionSpec("data")
        ),
    )

    def _init(batch: jax.Array) -> jax.Array:
      return nn.with_partitioning(
          jnp.ones_like, ("data", "model"), partitioner.mesh
      )(batch)

    state = partitioner.partition_init(_init, abstract_batch=sharded_inputs)(
        sharded_inputs
    )

    self.assertIsInstance(state, jax.Array)
    self.assertSequenceEqual(state.shape, (128, 16))
    self.assertEqual(state.sharding, partitioner.state_sharding)
    self.assertEqual(
        partitioner.state_sharding,
        jax.sharding.NamedSharding(
            partitioner.mesh,
            jax.sharding.PartitionSpec("data", "model"),
        ),
    )

    # TODO(aahil): Add tests for the steps.


if __name__ == "__main__":
  absltest.main()

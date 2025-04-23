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
"""Sparsecore tests."""

import functools

from absl.testing import absltest
import jax
from recml.layers.linen import sparsecore
from recml.core.training import partitioning


class SparsecoreTest(absltest.TestCase):

  def test_sparsecore_embedder_equivalence(self):
    if jax.devices()[0].platform != "tpu":
      self.skipTest("Test only supported on TPUs.")

    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)

    inputs = {
        "a": jax.random.randint(k1, (32, 16), minval=1, maxval=100),
        "b": jax.random.randint(k2, (32, 16), minval=1, maxval=100),
        "w": jax.random.normal(k3, (32, 16)),
    }

    dp_partitioner = partitioning.DataParallelPartitioner()
    embedder = sparsecore.SparsecoreEmbedder(
        specs={
            "a": sparsecore.EmbeddingSpec(
                input_dim=100,
                embedding_dim=16,
                combiner="mean",
                weight_name="w",
            ),
            "b": sparsecore.EmbeddingSpec(
                input_dim=100,
                embedding_dim=16,
                max_sequence_length=10,
            ),
        },
        optimizer=sparsecore.embedding_spec.AdagradOptimizerSpec(
            learning_rate=0.01
        ),
    )
    preprocessor = embedder.make_preprocessor(32)
    layer = embedder.make_sparsecore_module()

    sc_inputs = dp_partitioner.shard_inputs(preprocessor(inputs))
    sc_vars = dp_partitioner.partition_init(functools.partial(layer.init, k4))(
        sc_inputs
    )

    def step(inputs, params):
      return layer.apply(params, inputs)

    p_step = dp_partitioner.partition_step(step, training=False)
    sparsecore_activations = jax.device_get(p_step(sc_inputs, sc_vars))

    self.assertEqual(sparsecore_activations["a"].shape, (32, 16))
    self.assertEqual(sparsecore_activations["b"].shape, (32, 10, 16))


if __name__ == "__main__":
  absltest.main()

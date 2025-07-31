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
from absl import logging
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from recml.examples.DLRM_HSTU.stu import STULayer
from recml.examples.DLRM_HSTU.stu import STULayerConfig
from recml.examples.DLRM_HSTU.stu import STUStack


def get_test_configs():
  """Generates a list of test configurations."""
  test_params = []
  test_params.append((
      "basic_config",
      {
          "num_layers": 2,
          "num_heads": 2,
          "batch_size": 4,
          "max_len": 32,
          "embedding_dim": 16,
          "attention_dim": 8,
          "hidden_dim": 24,
          "use_group_norm": False,
          "target_aware": True,
      },
  ))
  test_params.append((
      "group_norm",
      {
          "num_layers": 1,
          "num_heads": 4,
          "batch_size": 2,
          "max_len": 16,
          "embedding_dim": 32,
          "attention_dim": 16,
          "hidden_dim": 20,
          "use_group_norm": True,
          "target_aware": True,
      },
  ))
  test_params.append((
      "not_target_aware",
      {
          "num_layers": 1,
          "num_heads": 1,
          "batch_size": 8,
          "max_len": 64,
          "embedding_dim": 8,
          "attention_dim": 4,
          "hidden_dim": 12,
          "use_group_norm": False,
          "target_aware": False,
      },
  ))
  test_params.append((
      "sliding_window_attention",
      {
          "num_layers": 1,
          "num_heads": 2,
          "batch_size": 2,
          "max_len": 20,
          "embedding_dim": 16,
          "attention_dim": 8,
          "hidden_dim": 16,
          "use_group_norm": False,
          "target_aware": True,
          "max_attn_len": 5,
      },
  ))
  return test_params


class StuJaxTest(parameterized.TestCase):
  """Unit tests for the JAX STU implementation."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    logging.info("Available devices: %s", jax.devices())
    # Assert that TPUs are available
    assert any(
        d.platform == "tpu" for d in jax.devices()
    ), "No TPU devices found."

  def setUp(self):
    """Set up a base key for all tests."""
    super().setUp()
    self.key = jax.random.PRNGKey(42)
    self.devices = jax.devices()
    self.num_devices = len(self.devices)
    self.mesh = Mesh(np.array(self.devices), ("data",))
    logging.info("Using device mesh: %s", self.mesh)

    self.batch_sharding = NamedSharding(self.mesh, PartitionSpec("data"))
    self.replicated_sharding = NamedSharding(self.mesh, PartitionSpec())

  @parameterized.named_parameters(get_test_configs())
  def test_output_shape_and_gradients(self, config_dict):
    """Tests STUStack for output shape and valid gradients.

    This test verifies that the STUStack runs, produces the correct output
    shape, and that gradients can be computed without errors (e.g., NaNs).

    Args:
      config_dict: A dictionary containing the configuration parameters for the
        STUStack.
    """
    self.assertEqual(jax.devices()[0].platform, "tpu")

    config = STULayerConfig(
        embedding_dim=config_dict["embedding_dim"],
        num_heads=config_dict["num_heads"],
        hidden_dim=config_dict["hidden_dim"],
        attention_dim=config_dict["attention_dim"],
        target_aware=config_dict["target_aware"],
        use_group_norm=config_dict["use_group_norm"],
        max_attn_len=config_dict.get("max_attn_len", 0),
    )

    stu_layers = [
        STULayer(config=config, name=f"stu_layer_{i}")
        for i in range(config_dict["num_layers"])
    ]
    model = STUStack(stu_layers=stu_layers)

    batch_size, max_len = config_dict["batch_size"], config_dict["max_len"]

    if batch_size % self.num_devices != 0:
      batch_size = (batch_size // self.num_devices + 1) * self.num_devices
      logging.warning("Adjusted batch size to %d for sharding", batch_size)

    init_key, data_key, dropout_key = jax.random.split(self.key, 3)

    dummy_x = jax.random.normal(
        data_key, (batch_size, max_len, config.embedding_dim)
    )
    dummy_num_targets = jax.random.randint(
        data_key, (batch_size,), minval=1, maxval=5
    )

    dummy_x = jax.device_put(dummy_x, self.batch_sharding)
    dummy_num_targets = jax.device_put(dummy_num_targets, self.batch_sharding)

    params = model.init(
        {"params": init_key, "dropout": dropout_key},
        x=dummy_x,
        num_targets=dummy_num_targets,
    )["params"]
    params = jax.device_put(params, self.replicated_sharding)

    @jax.jit
    def loss_fn(p, x, num_targets, rng_key):
      y = model.apply(
          {"params": p},
          x,
          num_targets=num_targets,
          rngs={"dropout": rng_key},
      )
      return jnp.sum(y**2)

    # Jitted apply function
    apply_fn = jax.jit(
        lambda p, x, num_targets: model.apply(
            {"params": p}, x, num_targets=num_targets
        ),
        out_shardings=self.batch_sharding,
    )

    output = apply_fn(params, dummy_x, dummy_num_targets)
    self.assertEqual(output.shape, dummy_x.shape)
    self.assertEqual(output.sharding, self.batch_sharding)

    grads = jax.grad(loss_fn)(params, dummy_x, dummy_num_targets, dropout_key)

    grad_leaves, _ = jax.tree_util.tree_flatten(grads)
    self.assertNotEmpty(grad_leaves)
    for g in grad_leaves:
      self.assertFalse(jnp.any(jnp.isnan(g)), "Found NaNs in gradients")
      self.assertFalse(jnp.all(g == 0), "Found all-zero gradients")
      self.assertEqual(g.sharding, self.replicated_sharding)

  def test_target_invariance(self):
    """Tests invariance of output with target section swaps.

    This test checks if swapping items within the target section of sequences
    results in an equivalently swapped output.
    """
    self.assertEqual(jax.devices()[0].platform, "tpu")

    batch_size, max_len, embedding_dim = 4, 32, 16
     # Adjust batch size to be divisible by the number of devices
    if batch_size % self.num_devices != 0:
      batch_size = (batch_size // self.num_devices + 1) * self.num_devices
      logging.warning("Adjusted batch size to %d for sharding", batch_size)

    config = STULayerConfig(
        embedding_dim=embedding_dim,
        num_heads=2,
        hidden_dim=24,
        attention_dim=8,
        target_aware=True,
        causal=True,
    )
    model = STUStack(stu_layers=[STULayer(config, name="stu_layer_0")])

    init_key, data_key = jax.random.split(self.key)
    x = jax.random.normal(data_key, (batch_size, max_len, embedding_dim))
    num_targets = jax.random.randint(
        data_key, (batch_size,), minval=2, maxval=10
    )

    # Shard inputs
    x = jax.device_put(x, self.batch_sharding)
    num_targets = jax.device_put(num_targets, self.batch_sharding)

    swap_from_offset = jnp.zeros((batch_size,), dtype=jnp.int32)
    swap_to_offset = jnp.ones((batch_size,), dtype=jnp.int32)
    swap_from_offset = jax.device_put(swap_from_offset, self.batch_sharding)
    swap_to_offset = jax.device_put(swap_to_offset, self.batch_sharding)

    swap_from_idx = max_len - 1 - swap_from_offset
    swap_to_idx = max_len - 1 - swap_to_offset

    params = model.init(
        {"params": init_key, "dropout": data_key},
        x=x,
        num_targets=num_targets,
    )["params"]
    params = jax.device_put(params, self.replicated_sharding)

    apply_fn = jax.jit(
        lambda p, x, num_targets: model.apply(
            {"params": p},
            x,
            num_targets=num_targets,
        ),
        out_shardings=self.batch_sharding,
    )

    output_original = apply_fn(params, x, num_targets)
    self.assertEqual(output_original.sharding, self.batch_sharding)

    def swap_rows(arr, idx1, idx2):
      val1 = arr[idx1]
      val2 = arr[idx2]
      return arr.at[idx1].set(val2).at[idx2].set(val1)

    swapped_x = jax.vmap(swap_rows)(x, swap_from_idx, swap_to_idx)
    self.assertEqual(swapped_x.sharding, self.batch_sharding)
    output_swapped_input = apply_fn(params, swapped_x, num_targets)
    self.assertEqual(output_swapped_input.sharding, self.batch_sharding)

    output_swapped_restored = jax.vmap(swap_rows)(
        output_swapped_input, swap_from_idx, swap_to_idx
    )
    self.assertEqual(output_swapped_restored.sharding, self.batch_sharding)

    np.testing.assert_allclose(
        output_original, output_swapped_restored, rtol=1e-5, atol=1e-5
    )


if __name__ == "__main__":
  absltest.main()


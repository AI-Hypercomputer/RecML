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
from absl.testing import absltest
import jax
import jax.numpy as jnp
from recml.examples.DLRM_HSTU.content_encoder import ContentEncoder


class ContentEncoderTest(absltest.TestCase):
  """Tests for JAX ContentEncoder."""

  def test_forward_and_backward_pass(self) -> None:
    """Verifies that the model's forward and backward passes execute without error."""
    batch_size = 2
    seq_len = 6
    num_targets = 2
    max_uih_len = seq_len - num_targets
    input_embedding_dim = 32
    additional_embedding_dim = 64
    enrich_embedding_dim = 16

    encoder = ContentEncoder(
        input_embedding_dim=input_embedding_dim,
        additional_content_features={
            "a0": additional_embedding_dim,
            "a1": additional_embedding_dim,
        },
        target_enrich_features={
            "t0": enrich_embedding_dim,
            "t1": enrich_embedding_dim,
        },
    )

    key = jax.random.PRNGKey(42)
    key, data_key, init_key = jax.random.split(key, 3)

    seq_embeddings = jax.random.normal(
        data_key, (batch_size, seq_len, input_embedding_dim)
    )
    seq_payloads = {
        "a0": jax.random.normal(
            data_key, (batch_size, seq_len, additional_embedding_dim)
        ),
        "a1": jax.random.normal(
            data_key, (batch_size, seq_len, additional_embedding_dim)
        ),
        "t0": jax.random.normal(
            data_key, (batch_size, num_targets, enrich_embedding_dim)
        ),
        "t1": jax.random.normal(
            data_key, (batch_size, num_targets, enrich_embedding_dim)
        ),
    }

    params = encoder.init(
        init_key,
        max_uih_len,
        seq_embeddings,
        seq_payloads,
    )["params"]

    content_embeddings = encoder.apply(
        {"params": params},
        max_uih_len,
        seq_embeddings,
        seq_payloads,
    )

    expected_dim = (
        input_embedding_dim
        + sum(encoder.additional_content_features.values())
        + sum(encoder.target_enrich_features.values())
    )
    self.assertEqual(
        content_embeddings.shape, (batch_size, seq_len, expected_dim)
    )

    def loss_fn(p):
      output = encoder.apply(
          {"params": p},
          max_uih_len,
          seq_embeddings,
          seq_payloads,
      )
      return jnp.sum(output)

    grads = jax.grad(loss_fn)(params)

    self.assertIsNotNone(grads)
    self.assertIn("target_enrich_dummy_param_t0", grads)
    self.assertIn("target_enrich_dummy_param_t1", grads)


if __name__ == "__main__":
  absltest.main()

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
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
# from third_party.py.pybase import googletest
from absl.testing import absltest
from recml.examples.DLRM_HSTU.action_encoder import ActionEncoder


class ActionEncoderJaxTest(absltest.TestCase):
  def test_forward_and_backward(self) -> None:
    """Tests the ActionEncoder's forward pass logic and differentiability."""

    batch_size = 2
    max_seq_len = 6
    action_embedding_dim = 32
    action_weights = [1, 2, 4, 8, 16]
    watchtime_to_action_thresholds_and_weights = [
        (30, 32), (60, 64), (100, 128),
    ]
    num_action_types = len(action_weights) + len(
        watchtime_to_action_thresholds_and_weights
    )
    output_dim = action_embedding_dim * num_action_types
    combined_action_weights = action_weights + [
        w for _, w in watchtime_to_action_thresholds_and_weights
    ]

    enabled_actions = [
        [0],          # Seq 1, Item 1
        [0, 1],       # Seq 1, Item 2
        [1, 3, 4],    # Seq 1, Item 3
        [1, 2, 3, 4], # Seq 1, Item 4
        [1, 2],       # Seq 2, Item 1
        [2],          # Seq 2, Item 2
    ]
    watchtimes_flat = [40, 20, 110, 31, 26, 55]

    # Add actions based on watchtime thresholds
    for i, wt in enumerate(watchtimes_flat):
      for j, (threshold, _) in enumerate(
          watchtime_to_action_thresholds_and_weights
      ):
        if wt > threshold:
          enabled_actions[i].append(j + len(action_weights))

    actions_flat = [
        sum([combined_action_weights[t] for t in x]) for x in enabled_actions
    ]

    padded_actions = np.zeros((batch_size, max_seq_len), dtype=np.int64)
    padded_watchtimes = np.zeros((batch_size, max_seq_len), dtype=np.int64)

    padded_actions[0, :4] = actions_flat[0:4]
    padded_actions[1, :2] = actions_flat[4:6]
    padded_watchtimes[0, :4] = watchtimes_flat[0:4]
    padded_watchtimes[1, :2] = watchtimes_flat[4:6]

    is_target_mask = np.zeros((batch_size, max_seq_len), dtype=bool)
    is_target_mask[0, 4:6] = True
    is_target_mask[1, 2] = True

    padding_mask = np.zeros((batch_size, max_seq_len), dtype=bool)
    padding_mask[0, :6] = True
    padding_mask[1, :3] = True

    seq_payloads = {
        "watchtimes": jnp.array(padded_watchtimes),
        "actions": jnp.array(padded_actions),
    }

    encoder = ActionEncoder(
        watchtime_feature_name="watchtimes",
        action_feature_name="actions",
        action_weights=action_weights,
        watchtime_to_action_thresholds_and_weights=(
            watchtime_to_action_thresholds_and_weights
        ),
        action_embedding_dim=action_embedding_dim,
    )

    key = jax.random.PRNGKey(0)
    variables = encoder.init(key, seq_payloads, is_target_mask)
    params = variables["params"]

    action_embeddings = encoder.apply(
        variables, seq_payloads, is_target_mask
    )

    self.assertEqual(
        action_embeddings.shape, (batch_size, max_seq_len, output_dim)
    )

    action_table = params["action_embedding_table"]
    target_table_flat = params["target_action_embedding_table"]
    target_table = target_table_flat.reshape(num_action_types, -1)

    history_item_idx = 0
    for b in range(batch_size):
      for s in range(max_seq_len):
        if not padding_mask[b, s]:
          npt.assert_allclose(action_embeddings[b, s], 0, atol=1e-6)
          continue

        embedding = action_embeddings[b, s].reshape(num_action_types, -1)

        if is_target_mask[b, s]:
          npt.assert_allclose(embedding, target_table, atol=1e-6)
        else:
          current_enabled = enabled_actions[history_item_idx]
          for atype in range(num_action_types):
            if atype in current_enabled:
              npt.assert_allclose(
                  embedding[atype], action_table[atype], atol=1e-6
              )
            else:
              npt.assert_allclose(embedding[atype],
                                  jnp.zeros_like(embedding[atype]),
                                  atol=1e-6)
          history_item_idx += 1

    def loss_fn(p):
      return encoder.apply({"params": p}, seq_payloads, is_target_mask).sum()

    grads = jax.grad(loss_fn)(params)
    self.assertIsNotNone(grads)
    self.assertFalse(np.all(np.isclose(grads["action_embedding_table"], 0)))
    self.assertFalse(np.all(
        np.isclose(grads["target_action_embedding_table"], 0)
    ))


if __name__ == "__main__":
    absltest.main()

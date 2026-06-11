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
"""Tests for the Mamba implementation."""

from absl.testing import absltest
from absl.testing import parameterized
import einops
import keras
from keras.src import testing
import numpy as np
from recml.layers.keras import mamba
import tensorflow as tf
import torch
import torch.nn.functional as F


# originally found here:
# https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/ssd_minimal.py
def segsum(x):
  """More stable segment sum calculation."""
  t = x.size(-1)
  x = einops.repeat(x, "... d -> ... d e", e=t)
  mask = torch.tril(torch.ones(t, t, device=x.device, dtype=bool), diagonal=-1)
  x = x.masked_fill(~mask, 0)
  x_segsum = torch.cumsum(x, dim=-2)
  mask = torch.tril(torch.ones(t, t, device=x.device, dtype=bool), diagonal=0)
  x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
  return x_segsum


# originally found here:
# https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/ssd_minimal.py
def ssd_minimal_discrete(x, a, b, c, block_len, initial_states=None):
  """Original Pytorch implementation of Mamba2.

  Args:
      x: (batch, length, n_heads, d_head)
      a: (batch, length, n_heads)
      b: (batch, length, n_groups, d_state)
      c: (batch, length, n_groups, d_state)
      block_len: int
      initial_states: tensor of initial state values.

  Returns:
      Y: (batch, length, n_heads, d_head)
  """
  assert x.dtype == a.dtype == b.dtype == c.dtype
  assert x.shape[1] % block_len == 0

  # Rearrange into blocks/chunks
  x, a, b, c = [
      einops.rearrange(x, "b (c l) ... -> b c l ...", l=block_len)
      for x in (x, a, b, c)
  ]

  a = einops.rearrange(a, "b c l h -> b h c l")
  a_cumsum = torch.cumsum(a, dim=-1)

  # 1. Compute the output for each intra-chunk (diagonal blocks)
  length = torch.exp(segsum(a))
  y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", c, b, length, x)

  # 2. Compute the state for each intra-chunk
  # (right term of low-rank factorization of off-diagonal blocks; B terms)
  decay_states = torch.exp((a_cumsum[:, :, :, -1:] - a_cumsum))
  states = torch.einsum("bclhn,bhcl,bclhp->bchpn", b, decay_states, x)

  # 3. Compute the inter-chunk SSM recurrence;
  # produces correct SSM states at chunk boundaries
  # (middle term of factorization of off-diag blocks; A terms)
  if initial_states is None:
    initial_states = torch.zeros_like(states[:, :1])
  states = torch.cat([initial_states, states], dim=1)
  decay_chunk = torch.exp(segsum(F.pad(a_cumsum[:, :, :, -1], (1, 0))))
  new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
  states, final_state = new_states[:, :-1], new_states[:, -1]

  # 4. Compute state -> output conversion per chunk
  # (left term of low-rank factorization of off-diagonal blocks; C terms)
  state_decay_out = torch.exp(a_cumsum)
  y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", c, states, state_decay_out)

  # Add output of intra-chunk any_offer-chunk terms
  # (diagonal and off-diagonal blocks)
  y = einops.rearrange(y_diag + y_off, "b c l h p -> b (c l) h p")
  return y, final_state


class MambaSSDTest(testing.TestCase):

  # Simple equivalence test
  @parameterized.parameters(
      # Multi-Query SSM (ngroups = 1, varying heads)
      dict(seed=40, ngroups=1, nheads=1),
      dict(seed=40, ngroups=1, nheads=8),
      dict(seed=40, ngroups=1, nheads=32),
      dict(seed=50, ngroups=1, nheads=1),
      dict(seed=50, ngroups=1, nheads=8),
      dict(seed=50, ngroups=1, nheads=32),
      # Multi-Head SSM (ngroups = nheads)
      dict(seed=70, ngroups=8, nheads=8),
      dict(seed=70, ngroups=32, nheads=32),
  )
  def test_ssd_correctness(self, seed: int, ngroups: int, nheads: int):
    keras.utils.set_random_seed(seed)

    ## Dimensions
    # Denoted (B, T, Q, D, P) in the paper
    batch, seqlen, chunk_size, dim = 1, 2048, 64, 2048
    dstate = 64  # (N) in the paper

    dtype = "float32"
    x = keras.random.normal((batch, seqlen, nheads, dim // nheads), dtype=dtype)
    dt = keras.ops.nn.softplus(
        keras.random.normal((batch, seqlen, nheads), dtype=dtype) - 4
    )
    a = keras.ops.multiply(
        -1, keras.ops.exp(keras.random.normal((nheads,), dtype=dtype))
    )
    b = keras.random.normal((batch, seqlen, ngroups, dstate), dtype=dtype)
    c = keras.random.normal((batch, seqlen, ngroups, dstate), dtype=dtype)

    torch_a = torch.tensor(np.array(a))
    torch_b = torch.tensor(np.array(b))
    torch_c = torch.tensor(np.array(c))
    torch_dt = torch.tensor(np.array(dt))
    torch_x = torch.tensor(np.array(x))
    ground_truth = ssd_minimal_discrete(
        torch_x * torch_dt.unsqueeze(-1),
        torch_a * torch_dt,
        torch_b,
        torch_c,
        chunk_size,
    )
    ours = mamba.ssd_minimal_discrete(
        keras.ops.multiply(x, keras.ops.expand_dims(dt, axis=-1)),
        keras.ops.multiply(a, dt),
        b,
        c,
        chunk_size,
    )

    self.assertAllClose(ground_truth[0], ours, atol=1e-5, rtol=1e-5)


class Mamba4RecTest(testing.TestCase):

  def test_mamba4rec(self):
    item_ids = keras.ops.array([[1, 2, 3, 4], [4, 5, 0, 0]], "int32")
    padding_mask = keras.ops.array([[1, 1, 1, 0], [1, 1, 0, 0]], "int32")
    init_kws = {
        "vocab_size": 500,
        "model_dim": 32,
        "mlp_expand": 4,
        "num_heads": 4,
        "num_layers": 3,
        "dropout": 0.1,
        "d_expand": 128,
        "d_state": 64,
        "d_conv": 4,
        "chunk_size": 2,
    }

    self.run_layer_test(
        mamba.Mamba4Rec,
        init_kwargs=init_kws,
        input_data=item_ids,
        call_kwargs={"padding_mask": padding_mask},
        expected_output_shape=(2, 4, 500),
        expected_output_dtype="float32",
        expected_num_seed_generators=1 + 3 * 3,
    )


if __name__ == "__main__":
  absltest.main()

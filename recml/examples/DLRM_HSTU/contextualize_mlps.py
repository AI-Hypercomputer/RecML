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
"""Contains Flax modules for contextualized MLPs used in DLRM-HSTU."""

from typing import Optional

from flax import linen as nn
import jax.numpy as jnp


class SwishLayerNorm(nn.Module):
  """Custom module for Swish(LayerNorm(x)) which is x * sigmoid(LayerNorm(x)).

  This mimics the SwishLayerNorm class in the PyTorch implementation.
  """

  epsilon: float = 1e-5

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Computes Swish(LayerNorm(x)).

    Args:
        x: Input tensor.

    Returns:
        The output tensor.
    """
    normed_x = nn.LayerNorm(epsilon=self.epsilon, name="layernorm")(x)
    return x * nn.sigmoid(normed_x)


class ContextualizedMLP(nn.Module):
  """Abstract base class for contextualized MLPs.

  JAX/Flax doesn't strictly require this, but it is included for structural
  parity with the PyTorch version.

  This module assumes dense inputs, where ragged tensors have been padded.
  """

  def __call__(
      self,
      seq_embeddings: jnp.ndarray,
      contextual_embeddings: Optional[jnp.ndarray],
  ) -> jnp.ndarray:
    """Forward pass for contextualized MLPs.

    Args:
        seq_embeddings: Dense tensor of shape (B, N, D_in).
        contextual_embeddings: Dense tensor of shape (B, D_ctx).

    Returns:
        Output tensor.
    """
    raise NotImplementedError()


class SimpleContextualizedMLP(ContextualizedMLP):
  """A simple MLP applied to sequential embeddings, ignoring contextual ones.

  This module is analogous to the PyTorch version and works on dense tensors.
  """

  sequential_input_dim: int
  sequential_output_dim: int
  hidden_dim: int

  @nn.compact
  def __call__(
      self,
      seq_embeddings: jnp.ndarray,
      contextual_embeddings: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    """Applies a simple MLP to the sequence embeddings.

    Args:
        seq_embeddings: Dense tensor of shape (B, N, sequential_input_dim).
        contextual_embeddings: Ignored.

    Returns:
        Output tensor of shape (B, N, sequential_output_dim).
    """
    x = nn.Dense(
        features=self.hidden_dim,
        kernel_init=nn.initializers.xavier_uniform(),
        name="mlp_0",
    )(seq_embeddings)
    x = SwishLayerNorm(name="mlp_1")(x)

    x = nn.Dense(
        features=self.sequential_output_dim,
        kernel_init=nn.initializers.xavier_uniform(),
        name="mlp_2",
    )(x)
    x = nn.LayerNorm(name="mlp_3")(x)
    return x


class ParameterizedContextualizedMLP(ContextualizedMLP):
  """An MLP whose weights are parameterized by contextual embeddings.

  This module is analogous to the PyTorch version and works on dense tensors.
  """

  contextual_embedding_dim: int
  sequential_input_dim: int
  sequential_output_dim: int
  hidden_dim: int

  @nn.compact
  def __call__(
      self,
      seq_embeddings: jnp.ndarray,
      contextual_embeddings: Optional[jnp.ndarray],
  ) -> jnp.ndarray:
    """Applies a parameterized MLP to the sequence embeddings.

    Args:
        seq_embeddings: Dense tensor of shape (B, N, sequential_input_dim).
        contextual_embeddings: Dense tensor of shape
          (B, contextual_embedding_dim).

    Returns:
        Output tensor of shape (B, N, sequential_output_dim).
    """
    if contextual_embeddings is None:
      raise ValueError(
          "contextual_embeddings cannot be None for "
          "ParameterizedContextualizedMLP"
      )

    shared_input = nn.Dense(
        features=self.hidden_dim,
        kernel_init=nn.initializers.xavier_uniform(),
        name="dense_features_compress"
    )(contextual_embeddings)

    attn_raw_weights_flat = nn.Dense(
        features=self.sequential_input_dim * self.sequential_output_dim,
        name="attn_raw_weights_0"
    )(shared_input)

    batch_size = contextual_embeddings.shape[0]
    attn_weights_unnorm = attn_raw_weights_flat.reshape(
        batch_size, self.sequential_input_dim, self.sequential_output_dim
    )

    attn_weights = nn.LayerNorm(
        feature_axes=(-2, -1),
        name="attn_weights_norm"
    )(attn_weights_unnorm)

    res_x = nn.Dense(
        features=self.hidden_dim,
        kernel_init=nn.initializers.xavier_uniform(),
        name="res_weights_0"
    )(shared_input)
    res_x = SwishLayerNorm(name="res_weights_1")(res_x)
    bias = nn.Dense(
        features=self.sequential_output_dim,
        kernel_init=nn.initializers.xavier_uniform(),
        name="res_weights_2"
    )(res_x)

    bmm_out = jnp.matmul(seq_embeddings, attn_weights)
    bias_broadcast = jnp.expand_dims(bias, axis=1)
    return bmm_out + bias_broadcast

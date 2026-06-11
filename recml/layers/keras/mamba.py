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
"""Implementation of Mamba2 [1] and Mamba4Rec [3].

[1] https://arxiv.org/abs/2405.21060
[2] https://arxiv.org/abs/2403.03900
"""

from collections.abc import Mapping, Sequence
from typing import Any, Self

import keras
import keras_hub
import numpy as np


Tensor = Any


def segsum(x: Tensor) -> Tensor:
  """More stable segment sum calculation."""
  time_steps = keras.ops.shape(x)[-1]
  x_dtype = keras.ops.dtype(x)

  x = keras.ops.expand_dims(x, axis=-1)
  x = keras.ops.repeat(x, repeats=time_steps, axis=-1)
  mask = keras.ops.tril(
      keras.ops.ones((time_steps, time_steps), dtype=x_dtype),
      k=-1,
  )
  x = keras.ops.multiply(x, mask)

  x_segsum = keras.ops.cumsum(x, axis=-2)
  mask = keras.ops.tril(
      keras.ops.ones((time_steps, time_steps), dtype=x_dtype),
      k=0,
  )
  x_segsum = keras.ops.where(mask, x_segsum, -np.inf)
  return x_segsum


def ssd_minimal_discrete(
    x: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    block_len: int,
    initial_states: Tensor | None = None,
) -> Tensor:
  """SSD minimal implementation in Keras.

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
  assert (
      keras.ops.dtype(x)
      == keras.ops.dtype(a)
      == keras.ops.dtype(b)
      == keras.ops.dtype(c)
  )
  assert keras.ops.shape(b)[-1] == keras.ops.shape(c)[-1]

  assert keras.ops.shape(x)[1] % block_len == 0

  c_value = keras.ops.shape(x)[1] // block_len
  n_heads = keras.ops.shape(x)[-2]
  n_groups = keras.ops.shape(b)[-2]
  d_state = keras.ops.shape(b)[-1]
  d_head = keras.ops.shape(x)[-1]
  batch_size = keras.ops.shape(x)[0]

  x = keras.ops.reshape(x, (batch_size, c_value, block_len, n_heads, d_head))
  a = keras.ops.reshape(a, (batch_size, c_value, block_len, n_heads))
  b = keras.ops.reshape(b, (batch_size, c_value, block_len, n_groups, d_state))
  c = keras.ops.reshape(c, (batch_size, c_value, block_len, n_groups, d_state))

  # Squeeze out n_groups dimension (axis=3) since n_groups is required to be 1.
  # This avoids the einsum string index clash on 'h' for n_groups vs n_heads.
  if n_groups == 1:
    b = keras.ops.squeeze(b, axis=3)
    c = keras.ops.squeeze(c, axis=3)

  # (batch, c, block_len, n_heads) -> (batch, n_heads, c, block_len)
  reshaped_a = keras.ops.transpose(a, (0, 3, 1, 2))

  cumsum_a = keras.ops.cumsum(reshaped_a, axis=-1)

  # 1. Compute the output for each intra-chunk (diagonal blocks)

  length = keras.ops.exp(segsum(reshaped_a))

  if n_groups == 1:
    y_diag = keras.ops.einsum("bcln,bcsn,bhcls,bcshp->bclhp", c, b, length, x)
  else:
    y_diag = keras.ops.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", c, b, length, x)

  # 2. Compute the state for each intra-chunk
  # (right term of low-rank factorization of off-diagonal blocks; B terms)

  decay_states = keras.ops.exp((cumsum_a[:, :, :, -1:] - cumsum_a))

  if n_groups == 1:
    states = keras.ops.einsum("bcln,bhcl,bclhp->bchpn", b, decay_states, x)
  else:
    states = keras.ops.einsum("bclhn,bhcl,bclhp->bchpn", b, decay_states, x)

  # 3. Compute the inter-chunk SSM recurrence;
  # produces correct SSM states at chunk boundaries
  # (middle term of factorization of off-diag blocks; A terms)

  if initial_states is None:
    initial_states = keras.ops.zeros_like(states[:, :1])

  states = keras.ops.concatenate([initial_states, states], axis=1)

  decay_chunk = keras.ops.exp(
      segsum(keras.ops.pad(cumsum_a[:, :, :, -1], ((0, 0), (0, 0), (1, 0))))
  )

  new_states = keras.ops.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)

  states = new_states[:, :-1]

  # 4. Compute state -> output conversion per chunk
  # (left term of low-rank factorization of off-diagonal blocks; C terms)

  state_decay_out = keras.ops.exp(cumsum_a)
  if n_groups == 1:
    y_off = keras.ops.einsum(
        "bcln,bchpn,bhcl->bclhp", c, states, state_decay_out
    )
  else:
    y_off = keras.ops.einsum(
        "bclhn,bchpn,bhcl->bclhp", c, states, state_decay_out
    )

  # Add output of intra-chunk and inter-chunk terms
  # (diagonal and off-diagonal blocks)

  y = keras.ops.reshape(
      keras.ops.add(y_diag, y_off),
      (batch_size, c_value * block_len, n_heads, d_head),
  )
  return y


@keras.saving.register_keras_serializable("recml")
class Mamba2(keras.layers.Layer):
  """Mamba2 implementation for sequential recommendations.

  This is a minimal implementation of the Mamba2 block that does not use any
  fused kernels.
  """

  def __init__(
      self,
      d_model: int,
      d_state: int = 64,
      d_conv: int = 4,
      conv_init: Tensor | None = None,
      expand: int = 4,
      nheads: int = 8,
      chunk_size: int = 256,
      init_range_a: tuple[int, int] = (1, 16),
      dt_min: float = 0.001,
      dt_max: float = 0.1,
      dt_init_floor: float = 1e-4,
      dt_limit: tuple[float, float] = (0.0, float("inf")),
      learn_init_states: bool = False,
      activation: str = "swish",
      use_dense_bias: bool = False,
      use_conv_bias: bool = False,
      **kwargs,
  ):
    """Mamba2 algorithm.

    Args:
      d_model: dimension of input and output embeddings
      d_state: dimension of the state of the B/C matrices
      d_conv: kernel size of convolution
      conv_init: maximal magnitude of the convolution kernel initialization. If
        None, use glorot_uniform.
      expand: expansion factor of the first dense layer
      nheads: how many heads to use
      chunk_size: chunk size for the Mamba block
      init_range_a: tuple[int, int] = range of initialization of a
      dt_min: float = minimum value of dt
      dt_max: float = maximum value of dt
      dt_init_floor: float = initial minimum value of dt
      dt_limit: tuple[float, float] = value range of dt
      learn_init_states: bool = whether to learn initial states
      activation: str = activation function to use
      use_dense_bias: bool = whether to use bias in dense layers
      use_conv_bias: bool = whether to use bias in convolution layer
      **kwargs: dict = additional arguments to pass to the layer
    """
    super().__init__(**kwargs)
    self.d_model: int = d_model
    self.d_state: int = d_state
    self.d_conv: int = d_conv
    self.conv_init: Tensor | None = conv_init
    self.expand = expand
    self.d_inner: int = expand * self.d_model
    self.nheads: int = nheads
    self.ngroups: int = 1  # ngroups is required to be 1
    self.learn_init_states: bool = learn_init_states
    self.activation: str = activation
    self.chunk_size: int = chunk_size
    self.use_dense_bias = use_dense_bias
    self.use_conv_bias = use_conv_bias
    self.dt_min: float = dt_min
    self.dt_max: float = dt_max
    self.dt_init_floor: float = dt_init_floor
    self.dt_limit: tuple[float, float] = dt_limit
    self.init_range_a: tuple[int, int] = init_range_a
    self.headdim = self.d_inner // self.nheads

    assert self.activation in ["silu", "swish"]

    # Dimension of last axis
    # z: self.d_inner,
    # x: self.d_inner,
    # B: self.ngroups * self.d_state,
    # C: self.ngroups * self.d_state,
    # dt: self.nheads

    self.z_proj = keras.layers.Dense(self.d_inner, use_bias=use_dense_bias)
    self.xbc_proj = keras.layers.Dense(
        self.d_inner + 2 * self.ngroups * self.d_state, use_bias=use_dense_bias
    )

    self.dt_proj = keras.layers.Dense(self.nheads, use_bias=use_dense_bias)

    conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
    if self.conv_init is None:
      uniform_init = "glorot_uniform"
    else:
      uniform_init = keras.initializers.RandomUniform(
          minval=-self.conv_init, maxval=self.conv_init
      )
    self.conv1d: keras.layers.Conv1D = keras.layers.Conv1D(
        filters=conv_dim,
        kernel_size=d_conv,
        use_bias=use_conv_bias,
        groups=conv_dim,
        padding="causal",
        kernel_initializer=uniform_init,
        activation="silu",
    )

    # A parameter
    assert init_range_a[0] > 0 and init_range_a[1] >= init_range_a[0]

    # Extra normalization layer right before output projection
    self.norm = keras.layers.LayerNormalization(
        name="rms_norm",
        rms_scaling=True,
        axis=-1,
        epsilon=1e-5,
        dtype="float32",
    )

    self.out_proj = keras.layers.Dense(self.d_model, use_bias=use_dense_bias)
    self.config = {
        "d_model": d_model,
        "d_state": d_state,
        "d_conv": d_conv,
        "conv_init": conv_init,
        "expand": expand,
        "nheads": nheads,
        "init_range_a": init_range_a,
        "dt_min": dt_min,
        "dt_max": dt_max,
        "dt_init_floor": dt_init_floor,
        "dt_limit": dt_limit,
        "learn_init_states": learn_init_states,
        "activation": activation,
        "use_dense_bias": use_dense_bias,
        "use_conv_bias": use_conv_bias,
    }

  def build(self, input_shape: Any):
    batch_size, seq_len, embedding_dim = input_shape

    if self.learn_init_states:
      self.init_states = self.add_weight(
          (self.nheads, self.headdim, self.d_state),
          keras.initializers.zeros(),
          name="no_weight_decay_init_states",
          dtype="float32",
      )

    self.z_proj.build((batch_size, seq_len, embedding_dim))
    self.xbc_proj.build((batch_size, seq_len, embedding_dim))
    self.dt_proj.build((batch_size, seq_len, embedding_dim))
    self.conv1d.build((
        batch_size,
        seq_len,
        self.d_inner + 2 * self.ngroups * self.d_state,
    ))
    self.out_proj.build((batch_size, seq_len, self.d_inner))

    # Initialize log dt bias
    def _inv_dt_initializer(shape, dtype):
      dt = keras.initializers.RandomUniform(minval=0, maxval=1)(shape, dtype)

      dt = keras.ops.exp(
          keras.ops.add(
              keras.ops.multiply(
                  dt,
                  keras.ops.subtract(
                      keras.ops.log(self.dt_max), keras.ops.log(self.dt_min)
                  ),
              ),
              keras.ops.log(self.dt_min),
          )
      )
      dt = keras.ops.clip(dt, x_min=self.dt_init_floor, x_max=1e10)
      # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
      inv_dt = keras.ops.add(
          dt, keras.ops.log(-keras.ops.expm1(keras.ops.negative(dt)))
      )
      return inv_dt

    self.dt_bias = self.add_weight(
        (self.nheads,),
        initializer=_inv_dt_initializer,
        trainable=True,
        name="no_weight_decay_dt_bias",
        dtype="float32",
    )

    def _a_log_initializer(shape, dtype):
      a = keras.initializers.RandomUniform(
          minval=self.init_range_a[0], maxval=self.init_range_a[1]
      )(shape, dtype)

      a = keras.ops.log(a)

      return a

    self.a_log_param = self.add_weight(
        (self.nheads,),
        initializer=_a_log_initializer,
        trainable=True,
        name="no_weight_decay_a",
    )

    self.norm.build((batch_size, seq_len, self.d_inner))

  def call(self, inputs: Tensor, seq_idx: Tensor | None = None):
    """Forward pass.

    Args:
      inputs: (B, L, D)
      seq_idx: (B, L)

    Returns:
      y: (B, L, D) same shape as inputs
    """
    batch, _, _ = keras.ops.shape(inputs)

    if self.learn_init_states:

      initial_states = keras.ops.expand_dims(self.init_states, axis=0)
      initial_states = keras.ops.repeat(initial_states, repeats=batch, axis=0)
    else:
      initial_states = None

    # (nheads) or (d_inner, d_state)
    a = keras.ops.negative(keras.ops.exp(self.a_log_param))
    z = self.z_proj(inputs)
    xbc = self.xbc_proj(inputs)
    dt = self.dt_proj(inputs)
    # (batch_size, seq_len, nheads)
    dt = keras.ops.nn.softplus(keras.ops.add(dt, self.dt_bias))
    xbc = self.conv1d(xbc)

    # Split into 3 main branches: X, B, C
    # These correspond to V, K, Q respectively in the SSM/attention duality
    x, b, c = keras.ops.split(
        xbc,
        [self.d_inner, self.d_inner + self.ngroups * self.d_state],
        axis=-1,
    )
    batch_size = keras.ops.shape(x)[0]
    sequence_length = keras.ops.shape(x)[1]
    x = keras.ops.reshape(
        x,
        (batch_size, sequence_length, self.nheads, self.headdim),
    )
    b = keras.ops.reshape(
        b,
        (batch_size, sequence_length, self.ngroups, self.d_state),
    )
    c = keras.ops.reshape(
        c,
        (batch_size, sequence_length, self.ngroups, self.d_state),
    )
    y = ssd_minimal_discrete(
        keras.ops.multiply(x, keras.ops.expand_dims(dt, axis=-1)),
        keras.ops.multiply(a, dt),
        b,
        c,
        block_len=self.chunk_size,
        initial_states=initial_states,
    )
    y = keras.ops.reshape(y, (keras.ops.shape(y)[0], keras.ops.shape(y)[1], -1))

    # Multiply "gate" branch and apply extra normalization layer
    y = self.norm(keras.ops.multiply(y, keras.ops.silu(z)))
    out = self.out_proj(y)
    return out

  def get_config(self) -> dict[str, Any]:
    return self.config

  @classmethod
  def from_config(cls, config: dict[str, Any]) -> Self:
    config["dt_limit"] = tuple(config["dt_limit"])
    config["init_range_a"] = tuple(config["init_range_a"])
    return cls(**config)


@keras.saving.register_keras_serializable("recml")
class Mamba2Block(keras.layers.Layer):
  """Mamba2 block implementation for sequential recommendations."""

  def __init__(
      self,
      d_model: int,
      d_state: int = 64,
      d_conv: int = 4,
      expand: int = 2,
      nheads: int = 8,
      chunk_size: int = 64,
      ffn_expand: int = 12,
      norm_eps: float = 1e-12,
      dropout: float = 0.0,
      **kwargs,
  ):
    """Mamba2 block.

    Args:
      d_model: Dimension of input and output embeddings.
      d_state: Dimension of the state of the B/C matrices.
      d_conv: Kernel size of the convolution.
      expand: Expansion factor of the first dense projection.
      nheads: How many heads to use.
      chunk_size: Chunk size of the Mamba block.
      ffn_expand: expansion factor of the feed forward network.
      norm_eps: The epsilon for the layer normalization. Defaults to 1e-12.
      dropout: The dropout rate. Defaults to 0.0.
      **kwargs: Arguments passed to base layer.
    """
    super().__init__(**kwargs)
    self.d_model: int = d_model
    self.d_state: int = d_state
    self.d_conv: int = d_conv
    self.expand: int = expand
    self.nheads: int = nheads
    self.chunk_size: int = chunk_size
    self.norm_eps: float = norm_eps
    self.dropout_rate: float = dropout
    self.ffn_expand: int = ffn_expand

  def build(self, input_shape: Any):
    self.mamba_ssm = Mamba2(
        d_model=self.d_model,
        d_state=self.d_state,
        d_conv=self.d_conv,
        expand=self.expand,
        nheads=self.nheads,
        chunk_size=self.chunk_size,
    )
    self.mamba_ssm.build(input_shape)

    self.layer_norm = keras.layers.LayerNormalization(
        name="layer_norm", axis=-1, epsilon=1e-12, dtype="float32"
    )
    self.layer_norm.build(input_shape)
    self.mamba_dropout = keras.layers.Dropout(
        self.dropout_rate, name="mamba_dropout"
    )

    self.ffn = keras.Sequential(
        [
            keras.layers.Dense(
                self.d_model * self.ffn_expand, activation="gelu"
            ),
            keras.layers.Dropout(self.dropout_rate),
            keras.layers.Dense(self.d_model),
            keras.layers.Dropout(self.dropout_rate),
        ],
        name="ffn",
    )
    self.ffn.build(input_shape)

    self.final_layer_norm = keras.layers.LayerNormalization(
        name="final_layer_norm", axis=-1, epsilon=self.norm_eps, dtype="float32"
    )
    self.final_layer_norm.build(input_shape)

  def call(
      self,
      inputs: Tensor,
      padding_mask: Tensor | None = None,
      training: bool | None = False,
      **_,
  ):
    """Forward pass of Mamba Block."""
    x = self.mamba_ssm(inputs=inputs, training=training)
    x = self.mamba_dropout(x, training=training)
    x = self.layer_norm(x + inputs)

    y = self.ffn(x, training=training)
    y = self.final_layer_norm(y + x)
    if padding_mask is not None:
      y = keras.ops.where(
          keras.ops.expand_dims(padding_mask, axis=-1),
          y,
          keras.ops.zeros_like(y),
      )
    return y

  def get_config(self) -> Mapping[str, Any]:
    return {
        **super().get_config(),
        "d_model": self.d_model,
        "d_state": self.d_state,
        "d_conv": self.d_conv,
        "expand": self.expand,
        "nheads": self.nheads,
        "chunk_size": self.chunk_size,
        "ffn_expand": self.ffn_expand,
        "norm_eps": self.norm_eps,
        "dropout": self.dropout_rate,
    }


@keras.saving.register_keras_serializable("recml")
class Mamba2Stack(keras.layers.Layer):
  """A stack of Mamba2 blocks."""

  EMBEDDINGS_KEY: str = "embeddings"
  MASK_KEY: str = "mask"

  def __init__(
      self,
      *,
      embedding_dim: int,
      num_layers: int,
      dropout: float,
      d_state: int = 64,
      d_conv: int = 4,
      expand: int = 4,
      chunk_size: int = 256,
      **kwargs,
  ):
    """Initializer.

    Args:
      embedding_dim: The dimension for embeddings.
      num_layers: Number of Mamba layers in the transformer.
      dropout: The dropout rate.
      d_state: The dimension of the state of the B/C matrices.
      d_conv: The kernel size of the convolution.
      expand: The expansion factor of the first dense layer.
      chunk_size: The chunk size of the Mamba block.
      **kwargs: Other arguments to pass to the keras.layers.Layer constructor.
    """
    super().__init__(**kwargs)

    mamba_args = dict(
        d_model=embedding_dim,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        chunk_size=chunk_size,
        dropout=dropout,
    )
    self._blocks = [
        Mamba2Block(**mamba_args, name=f"mamba_{i}") for i in range(num_layers)
    ]

    self._config = {
        "embedding_dim": embedding_dim,
        "dropout": dropout,
        "num_layers": num_layers,
        "d_state": d_state,
        "d_conv": d_conv,
        "expand": expand,
        "chunk_size": chunk_size,
    }

  def call(
      self, inputs: Mapping[str, Tensor], training: bool | None = False
  ) -> Tensor:
    """Applies Mamba2 blocks to input.

    Args:
        inputs: Dictionary mapping strings to tensors (embedding sequence, and
          mask).
        training: Whether the layer is in training mode.

    Returns:
        output tensor: (B, L, D) x float
    """
    input_embeddings = inputs[self.EMBEDDINGS_KEY]
    input_mask = keras.ops.cast(inputs[self.MASK_KEY], input_embeddings.dtype)
    expanded_mask = keras.ops.expand_dims(input_mask, axis=-1)

    x = keras.ops.multiply(input_embeddings, expanded_mask)
    for layer in self._blocks:
      x = layer(inputs=x, training=training)
      x = keras.ops.multiply(x, expanded_mask)

    return x

  def get_config(self) -> Mapping[str, Any]:
    return {**super().get_config(), **self._config}


@keras.saving.register_keras_serializable("recml")
class Mamba4Rec(keras.Model):
  """Mamba transformer for recommendations.

  An implementation of the Mamba4Rec paper [1].

  We use the Mamba2 block instead of the Mamba block they use in the paper.

  [1] https://arxiv.org/abs/2403.03900.
  """

  def __init__(
      self,
      vocab_size: int,
      model_dim: int,
      mlp_expand: int,
      num_heads: int,
      num_layers: int,
      d_expand: int,
      d_state: int = 64,
      d_conv: int = 4,
      chunk_size: int = 256,
      norm_eps: float = 1e-6,
      dropout: float = 0.0,
      add_head: bool = True,
      **kwargs,
  ):
    """Mamba Transformer.

    Args:
      vocab_size: The size of the item vocabulary.
      model_dim: The hidden dimension of the transformer.
      mlp_expand: The MLP expandsion factor in each transformer block.
      num_heads: The number of heads in each Mamba block.
      num_layers: Number of Mamba layers in the transformer.
      d_expand: The expansion factor of the dense projection.
      d_state: The dimension of the state of the B/C matrices.
      d_conv: The kernel size of the convolution.
      chunk_size: The chunk size of the Mamba block.
      norm_eps: The epsilon for the layer normalization. Defaults to 1e-12.
      dropout: The dropout rate. Defaults to 0.0.
      add_head: Whether to add a head to the transformer. Defaults to True.
      **kwargs: Other arguments to pass to the keras.layers.Layer constructor.
    """
    super().__init__(**kwargs)

    self.item_embedding = keras_hub.layers.ReversibleEmbedding(
        input_dim=vocab_size,
        output_dim=model_dim,
        embeddings_initializer=keras.initializers.RandomNormal(stddev=0.02),
        dtype=self.dtype_policy,
        reverse_dtype=self.compute_dtype,
        name="item_embedding",
    )
    self.dropout_layer = keras.layers.Dropout(dropout, name="embedding_dropout")
    self.embedding_norm = keras.layers.LayerNormalization(
        epsilon=norm_eps, name="embedding_norm"
    )

    self.decoder_blocks = [
        Mamba2Block(
            d_model=model_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=d_expand,
            nheads=num_heads,
            ffn_expand=mlp_expand,
            chunk_size=chunk_size,
            norm_eps=norm_eps,
            dropout=dropout,
            dtype=self.dtype_policy,
            name=f"mamba_block_{i}",
        )
        for i in range(num_layers)
    ]

    self._vocab_size = vocab_size
    self._model_dim = model_dim
    self._add_head = add_head
    self._config = {
        "vocab_size": vocab_size,
        "model_dim": model_dim,
        "mlp_expand": mlp_expand,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "d_expand": d_expand,
        "d_state": d_state,
        "d_conv": d_conv,
        "chunk_size": chunk_size,
        "norm_eps": norm_eps,
        "dropout": dropout,
        "add_head": add_head,
    }

  def build(self, inputs_shape: Sequence[int]):
    self.item_embedding.build(inputs_shape)
    self.embedding_norm.build((*inputs_shape, self._model_dim))

    for decoder_block in self.decoder_blocks:
      decoder_block.build((*inputs_shape, self._model_dim))

  def call(
      self,
      inputs: Tensor,
      padding_mask: Tensor | None = None,
      training: bool = False,
  ) -> Tensor:
    if padding_mask is not None:
      padding_mask = keras.ops.cast(padding_mask, dtype="bool")

    item_embeds = self.item_embedding(inputs)
    x = self.dropout_layer(item_embeds, training=training)
    x = self.embedding_norm(x)
    for decoder_block in self.decoder_blocks:
      x = decoder_block(x, padding_mask=padding_mask, training=training)

    if not self._add_head:
      return x

    return self.item_embedding(x, reverse=True)

  def compute_output_shape(self, inputs_shape: Sequence[int]) -> Sequence[int]:
    output_dim = self._vocab_size if self._add_head else self._model_dim
    return (*inputs_shape, output_dim)

  def get_config(self) -> Mapping[str, Any]:
    return {**super().get_config(), **self._config}

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
"""HSTU Encoder Block.

An implementation of the HSTU (Heirarchical Sequential Transducer Unit)
from "Actions Speak Louder than Words: Trillion-Parameter Sequential
Transducers for Generative Recommendations".
"""

from collections.abc import Callable, Mapping, MutableMapping, Sequence
import enum
from typing import Any, Self

import keras
import keras_hub
from recml.layers.keras import utils


Tensor = Any


@keras.utils.register_keras_serializable(package="recml")
def default_bucketization(tensor: Tensor, bucket_size: float = 0.301):
  """Default bucketization function for RelativeBucketedTimeAndPositionBasedBias.

  From the original paper's code base:
  https://github.com/facebookresearch/generative-recommenders/blob/main/modeling/sequential/hstu.py#L79

  Args:
    tensor: The tensor of timestamps to bucketize.
    bucket_size: value used to split the log timestamps into discrete integer
      buckets.

  Returns:
    Tensor of timestamps put into integer buckets.
  """
  clipped_values = keras.ops.clip(
      keras.ops.abs(tensor),
      x_min=1,
      x_max=keras.ops.cast(1e9, dtype="int32"),
  )
  log_values = keras.ops.log(clipped_values)
  bucket_values = keras.ops.divide(log_values, bucket_size)
  return keras.ops.cast(bucket_values, "int64")


@keras.saving.register_keras_serializable("recml")
class RelativePositionalBias(keras.layers.Layer):
  """Relative Positional Bias.

  This module creates positional biases that the model can learn position
  specific attention weights. Uses RandomNormal rather than
  TruncatedNormal to match the original paper.
  """

  def __init__(
      self,
      initializer: keras.initializers.Initializer = (
          keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None)
      ),
      **kwargs,
  ):
    """Creates the Relative Positional Bias.

    Args:
      initializer: The keras initializer to use for the weights. (can be a
        callable initializer)
      **kwargs: Other arguments to pass to the Keras layer initializer.
    """
    super().__init__(**kwargs)
    self._initializer: keras.initializers.Initializer = initializer

  def build(self, inputs_shape: Sequence[int]) -> None:
    self._pos_biases = self.add_weight(
        shape=(2 * inputs_shape[-1] - 1,),
        initializer=self._initializer,
        trainable=True,
    )

  def call(
      self,
      inputs: Tensor,
  ) -> Tensor:
    """Call the Relative Positional Bias.

    Args:
      inputs: A tensor of shape (batch_size, max_seq_len) containing the
        timestamps of each token in the sequence. (Not used in this module)

    Returns:
      A tensor of shape (max_seq_len, max_seq_len) containing the
      relative positional bias for each token in the sequence.
    """
    n = keras.ops.shape(inputs)[-1]
    tiled_positional_biases = keras.ops.tile(
        keras.ops.pad(self._pos_biases[: 2 * n - 1], [0, n]), (n,)
    )
    full_sequences = keras.ops.reshape(
        tiled_positional_biases[..., :-n], (1, n, 3 * n - 2)
    )
    excess_sequence = (2 * n - 1) // 2
    return full_sequences[:, :, excess_sequence:-excess_sequence]

  def get_config(self) -> dict[str, Any]:
    config = super().get_config()
    new_config = {
        "initializer": keras.utils.serialize_keras_object(self._initializer)
    }
    config.update(new_config)
    return config

  @classmethod
  def from_config(cls, config: MutableMapping[str, Any]) -> Self:
    config["initializer"] = keras.utils.deserialize_keras_object(
        config["initializer"]
    )
    return cls(**config)


@keras.saving.register_keras_serializable("recml")
class RelativeBucketedTimeAndPositionBasedBias(keras.layers.Layer):
  """Bucketizes timespans based on ts(next-item) - ts(current-item)."""

  def __init__(
      self,
      num_buckets: int = 128,
      bucketization_fn: Callable[[Tensor], Tensor] = default_bucketization,
      ts_initializer: keras.initializers.Initializer = (
          keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None)
      ),
      position_bias_layer: RelativePositionalBias | None = None,
      **kwargs,
  ):
    super().__init__(**kwargs)

    self._num_buckets: int = num_buckets
    self._bucketization_fn: Callable[[Tensor], Tensor] = bucketization_fn
    self._ts_initializer: keras.initializers.Initializer = ts_initializer
    self._positional_bias = position_bias_layer or RelativePositionalBias()

    self._config = {
        "ts_initializer": keras.utils.serialize_keras_object(
            self._ts_initializer
        ),
        "num_buckets": num_buckets,
        "bucketization_fn": keras.utils.serialize_keras_object(
            self._bucketization_fn
        ),
        "position_bias_layer": keras.utils.serialize_keras_object(
            self._positional_bias
        ),
    }

  def get_config(self) -> dict[str, Any]:
    config = super().get_config()
    config.update(self._config)
    return config

  @classmethod
  def from_config(cls, config: MutableMapping[str, Any]) -> Self:
    config_names = [
        "ts_initializer",
        "bucketization_fn",
        "position_bias_layer",
    ]

    for name in config_names:
      config[name] = keras.utils.deserialize_keras_object(config[name])

    return cls(**config)

  def build(self, inputs_shape: Sequence[int]):
    self._ts_w = self.add_weight(
        shape=(self._num_buckets + 1,),
        initializer=self._ts_initializer,
        trainable=True,
    )

  def time_bias(self, inputs: Tensor):
    b_size, n = keras.ops.shape(inputs)

    # [B, N + 1] to simplify tensor manipulations.
    ext_timestamps = keras.ops.concatenate(
        [inputs, inputs[:, n - 1 : n]], axis=1
    )
    # Expand dims to make the difference a 3D tensor with batch dimension.
    # Otherwise [:, :-1] - [:, 1:] works

    differenced_tensor = keras.ops.expand_dims(
        ext_timestamps[:, 1:], axis=2
    ) - keras.ops.expand_dims(ext_timestamps[:, :-1], axis=1)

    bucketed_timestamps = keras.ops.clip(
        self._bucketization_fn(differenced_tensor),
        x_min=0,
        x_max=self._num_buckets,
    )

    bucketed_timestamps = keras.ops.cast(
        keras.ops.floor(bucketed_timestamps), "int32"
    )

    flattened_timestamps = keras.ops.reshape(bucketed_timestamps, (-1,))
    one_hot_timestamps = keras.ops.one_hot(
        flattened_timestamps, self._num_buckets + 1
    )
    rel_ts_bias = keras.ops.matmul(one_hot_timestamps, self._ts_w)

    rel_ts_bias = keras.ops.reshape(rel_ts_bias, (b_size, n, n))
    return rel_ts_bias

  def call(self, inputs: Tensor) -> Tensor:
    """Relative Bucketed Time and Position Based Bias.

    Args:
        inputs: (B, N).

    Returns:
        (B, N, N).
    """
    if self._positional_bias is not None:
      return keras.ops.add(
          self._positional_bias(inputs), self.time_bias(inputs)
      )
    else:
      return self.time_bias(inputs)


@keras.saving.register_keras_serializable("recml")
class HSTUBlock(keras.layers.Layer):
  """HSTU - Hierarchical Sequential Transduction Unit block from [1].

  [1] https://arxiv.org/abs/2402.17152
  """

  def __init__(
      self,
      num_heads: int,
      use_bias: bool = False,
      activation: str | Callable[[Tensor], Tensor] = keras.activations.silu,
      layer_norm_epsilon: float = 1e-6,
      scaled_pointwise_attn: bool = False,
      softmax_attention: bool = False,
      dropout: float = 0.0,
      **kwargs,
  ):
    """Initializes the instance.

    Args:
      num_heads: The number of attention heads.
      use_bias: Whether to use a bias for dense layers. Defaults to False.
      activation: The activation to use for the linear layer. Defaults to silu.
      layer_norm_epsilon: Epsilon to use for layer norm.
      scaled_pointwise_attn: Whether to scale the attention logits 1 / sqrt(dim)
        instead of 1 / sequence length. Defaults to False.
      softmax_attention: Whether to use softmax attention. Defaults to False.
      dropout: The dropout rate.
      **kwargs: Other arguments to pass to the Keras layer initializer.
    """
    super().__init__(**kwargs)
    self._num_heads = num_heads
    self._use_bias = use_bias
    self._activation = activation
    self._layer_norm_eps = layer_norm_epsilon
    self._scaled_pointwise_attn = scaled_pointwise_attn
    self._use_softmax_attention = softmax_attention
    self._dropout = dropout

  def build(self, inputs_shape: Sequence[int]) -> None:
    self._input_layer_norm = keras.layers.LayerNormalization(
        epsilon=self._layer_norm_eps, name="input_layer_norm"
    )
    self._input_layer_norm.build(inputs_shape)

    hidden_dim = inputs_shape[-1]
    head_dim = hidden_dim // self._num_heads

    self._u_dense = keras.layers.EinsumDense(
        equation="btd,dnh->btnh",
        output_shape=(None, self._num_heads, head_dim),
        bias_axes="nh" if self._use_bias else None,
        activation=self._activation,
        kernel_initializer=keras.initializers.GlorotUniform(),
        bias_initializer=keras.initializers.Zeros(),
        dtype=self.dtype_policy,
        name="u_dense",
    )
    self._u_dense.build(inputs_shape)

    self._q_dense = keras.layers.EinsumDense(
        equation="btd,dnh->btnh",
        output_shape=(None, self._num_heads, head_dim),
        bias_axes="nh" if self._use_bias else None,
        activation=self._activation,
        kernel_initializer=keras.initializers.GlorotUniform(),
        bias_initializer=keras.initializers.Zeros(),
        dtype=self.dtype_policy,
        name="q_dense",
    )
    self._q_dense.build(inputs_shape)

    self._k_dense = keras.layers.EinsumDense(
        equation="btd,dnh->btnh",
        output_shape=(None, self._num_heads, head_dim),
        bias_axes="nh" if self._use_bias else None,
        activation=self._activation,
        kernel_initializer=keras.initializers.GlorotUniform(),
        bias_initializer=keras.initializers.Zeros(),
        dtype=self.dtype_policy,
        name="k_dense",
    )
    self._k_dense.build(inputs_shape)

    self._v_dense = keras.layers.EinsumDense(
        equation="btd,dnh->btnh",
        output_shape=(None, self._num_heads, head_dim),
        bias_axes="nh" if self._use_bias else None,
        activation=self._activation,
        kernel_initializer=keras.initializers.GlorotUniform(),
        bias_initializer=keras.initializers.Zeros(),
        dtype=self.dtype_policy,
        name="v_dense",
    )
    self._v_dense.build(inputs_shape)

    self._dropout_layer = keras.layers.Dropout(
        self._dropout, name="attn_dropout"
    )
    self._attn_layer_norm = keras.layers.LayerNormalization(
        axis=[-2, -1], epsilon=self._layer_norm_eps, name="attn_layer_norm"
    )
    self._attn_layer_norm.build((None, None, self._num_heads, head_dim))

    self._output_dense = keras.layers.EinsumDense(
        equation="btnh,nhd->btd",
        output_shape=(None, hidden_dim),
        bias_axes="d" if self._use_bias else None,
        kernel_initializer=keras.initializers.GlorotUniform(),
        bias_initializer=keras.initializers.Zeros(),
        dtype=self.dtype_policy,
        name="output_dense",
    )
    self._output_dense.build((None, None, self._num_heads, head_dim))

  def _pointwise_attention(
      self,
      q: Tensor,
      k: Tensor,
      v: Tensor,
      attention_mask: Tensor,
      attention_bias: Tensor | None = None,
      training: bool = False,
  ) -> Tensor:
    if self._scaled_pointwise_attn:
      q_head_dim = keras.ops.shape(q)[-1]
      q = keras.ops.divide(
          q, keras.ops.cast(q_head_dim**0.5, keras.ops.dtype(q))
      )

    qk_attn = keras.ops.einsum("bthd,bshd->bhts", q, k)

    if attention_bias is not None:
      qk_attn += keras.ops.expand_dims(attention_bias, axis=1)

    qk_attn = keras.ops.silu(qk_attn)
    if not self._scaled_pointwise_attn:
      qk_attn /= keras.ops.shape(q)[1]

    qk_attn = keras.ops.where(
        keras.ops.expand_dims(attention_mask, axis=1),
        qk_attn,
        keras.ops.zeros_like(qk_attn),
    )

    if training:
      qk_attn = self._dropout_layer(qk_attn, training=training)

    attn_output = keras.ops.einsum("bhts,bshd->bthd", qk_attn, v)
    return attn_output

  def _softmax_attention(
      self,
      q: Tensor,
      k: Tensor,
      v: Tensor,
      attention_mask: Tensor,
      attention_bias: Tensor | None = None,
      training: bool = False,
  ) -> Tensor:
    _, _, _, q_head_dim = keras.ops.shape(q)
    q = keras.ops.divide(q, keras.ops.cast(q_head_dim**0.5, keras.ops.dtype(q)))

    qk_attn = keras.ops.einsum("bthd,bshd->bhts", q, k)

    if attention_bias is not None:
      qk_attn += keras.ops.expand_dims(attention_bias, axis=1)

    qk_attn = keras.ops.where(
        keras.ops.expand_dims(attention_mask, axis=1),
        qk_attn,
        utils.large_negative_for_attention(keras.ops.dtype(qk_attn)),
    )
    qk_attn = keras.ops.softmax(qk_attn, axis=-1)

    if training:
      qk_attn = self._dropout_layer(qk_attn, training=training)

    attn_output = keras.ops.einsum("bhts,bshd->bthd", qk_attn, v)
    return attn_output

  def call(
      self,
      inputs: Tensor,
      attention_mask: Tensor,
      attention_bias: Tensor | None = None,
      training: bool = False,
      **_,
  ) -> Tensor:
    """Forward pass for Sequential Transduction Unit.

    Args:
      inputs: The input sequence embeddings of shape [B, L, D].
      attention_mask: An attention mask of shape [B, L, L].
      attention_bias: Optional attention bias of shape [B, L, L] to add to the
        attention scores.
      training: A boolean indicating whether to apply dropout or not. Defaults
        to False.

    Returns:
      Outputs of shape [B, L, D].
    """

    normed_inputs = self._input_layer_norm(inputs)

    u = self._u_dense(normed_inputs)
    q = self._q_dense(normed_inputs)
    k = self._k_dense(normed_inputs)
    v = self._v_dense(normed_inputs)

    if self._use_softmax_attention:
      attn_output = self._softmax_attention(
          q=q,
          k=k,
          v=v,
          attention_mask=attention_mask,
          attention_bias=attention_bias,
          training=training,
      )
    else:
      attn_output = self._pointwise_attention(
          q=q,
          k=k,
          v=v,
          attention_mask=attention_mask,
          attention_bias=attention_bias,
          training=training,
      )

    normed_attn_output = self._attn_layer_norm(attn_output)
    gated_attn_output = u * normed_attn_output

    out = self._output_dense(gated_attn_output)
    return out + inputs

  def get_config(self) -> dict[str, Any]:
    return {
        **super().get_config(),
        "num_heads": self._num_heads,
        "use_bias": self._use_bias,
        "activation": keras.activations.serialize(self._activation),
        "layer_norm_epsilon": self._layer_norm_eps,
        "scaled_pointwise_attn": self._scaled_pointwise_attn,
        "softmax_attention": self._use_softmax_attention,
        "dropout": self._dropout,
    }

  @classmethod
  def from_config(cls, config: MutableMapping[str, Any]) -> Self:
    config["activation"] = keras.activations.deserialize(config["activation"])
    return cls(**config)


@keras.saving.register_keras_serializable("recml")
class HSTU(keras.layers.Layer):
  """HSTU model.

  Implements the HSTU model architecture as described in 'Actions Speak Louder
  than Words: Trillion-Parameter Sequential Transducers for Generative
  Recommendations' [1].

  [1] https://arxiv.org/abs/2402.17152
  """

  def __init__(
      self,
      *,
      vocab_size: int,
      max_positions: int | None = None,
      model_dim: int,
      num_heads: int,
      num_layers: int,
      dropout: float = 0.0,
      norm_eps: float = 1e-6,
      scale_by_sqrt_dim: bool = False,
      add_head: bool = True,
      **kwargs,
  ):
    super().__init__(**kwargs)

    self.item_embedding = keras_hub.layers.ReversibleEmbedding(
        input_dim=vocab_size,
        output_dim=model_dim,
        embeddings_initializer=keras.initializers.RandomNormal(stddev=0.02),
        dtype=self.dtype_policy,
        reverse_dtype=self.compute_dtype,
        tie_weights=False,
        name="item_embedding",
    )

    if max_positions is not None:
      self.position_embedding = keras_hub.layers.PositionEmbedding(
          sequence_length=max_positions,
          initializer=keras.initializers.RandomNormal(stddev=0.02),
          dtype=self.dtype_policy,
          name="position_embedding",
      )
    else:
      self.position_embedding = None

    self.embeddings_dropout = keras.layers.Dropout(
        dropout, name="embedding_dropout"
    )

    self.decoder_blocks = [
        HSTUBlock(
            num_heads=num_heads,
            dropout=dropout,
            layer_norm_epsilon=norm_eps,
            dtype=self.dtype_policy,
            name=f"hstu_block_{i}",
        )
        for i in range(num_layers)
    ]
    self.final_norm = keras.layers.LayerNormalization(
        epsilon=norm_eps, name="final_norm"
    )

    self._vocab_size = vocab_size
    self._model_dim = model_dim
    self._scale_by_sqrt_dim = scale_by_sqrt_dim
    self._add_head = add_head
    self._config = {
        "vocab_size": vocab_size,
        "max_positions": max_positions,
        "model_dim": model_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "dropout": dropout,
        "norm_eps": norm_eps,
        "scale_by_sqrt_dim": scale_by_sqrt_dim,
        "add_head": add_head,
    }

  def build(self, inputs_shape: Sequence[int]):
    self.item_embedding.build(inputs_shape)
    if self.position_embedding is not None:
      self.position_embedding.build((*inputs_shape, self._model_dim))

    for encoder_block in self.decoder_blocks:
      encoder_block.build((*inputs_shape, self._model_dim))

    self.final_norm.build((*inputs_shape, self._model_dim))

  def call(
      self,
      inputs: Tensor,
      padding_mask: Tensor | None = None,
      attention_mask: Tensor | None = None,
      mask_positions: Tensor | None = None,
      training: bool = False,
  ) -> Tensor:
    embeddings = self.item_embedding(inputs)
    if self._scale_by_sqrt_dim:
      embeddings *= keras.ops.cast(
          self._model_dim**0.5, keras.ops.dtype(embeddings)
      )
    if self.position_embedding is not None:
      embeddings += self.position_embedding(embeddings)

    embeddings = self.final_norm(embeddings)
    embeddings = self.embeddings_dropout(embeddings, training=training)

    if attention_mask is None:
      if padding_mask is None:
        raise ValueError(
            "Either `attention_mask` or `padding_mask` must be set."
        )
      attention_mask = utils.make_causal_mask(padding_mask)

    for decoder_block in self.decoder_blocks:
      embeddings = decoder_block(
          embeddings, attention_mask=attention_mask, training=training
      )

    embeddings = self.final_norm(embeddings)

    if not self._add_head:
      return embeddings

    return self.item_embedding(embeddings, reverse=True)

  def compute_output_shape(self, inputs_shape: Sequence[int]) -> Sequence[int]:
    output_dim = self._vocab_size if self._add_head else self._model_dim
    return (*inputs_shape, output_dim)

  def get_config(self) -> Mapping[str, Any]:
    return {**super().get_config(), **self._config}

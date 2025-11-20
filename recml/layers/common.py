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
"""Common utilities for RecML layers."""

from collections.abc import Callable
import dataclasses
from typing import Any, Literal


@dataclasses.dataclass
class EmbeddingSpec:
  """Generic embedding spec for accelerated embedding lookups.

  Attributes:
    input_dim: The cardinality of the input feature or size of its vocabulary.
    embedding_dim: The length of each embedding vector.
    max_sequence_length: An optional maximum sequence length. If set, the looked
      up embeddings will not be aggregated over the sequence dimension.
      Otherwise the embeddings will be aggregated over the sequence dimension
      using the `combiner`. Defaults to None.
    combiner: The combiner to use to aggregate the embeddings over the sequence
      dimension. This is ignored when `max_sequence_length` is set. Allowed
      values are 'sum', 'mean', and 'sqrtn'. Defaults to 'mean'.
    initializer: The initializer to use for the embedding table. Defaults to
      truncated_normal(stddev=1 / sqrt(embedding_dim)) if not set.
    optimizer: An optional custom optimizer to use for the embedding table. The
      type of this is use-case dependent. Defaults to None.
    weight_name: An optional weight feature name to use for performing a
      weighted aggregation on the output of the embedding lookup. Defaults to
      None.
    param_dtype: The dtype to use for the embedding variables. Defaults to None.
  """

  input_dim: int
  embedding_dim: int
  max_sequence_length: int | None = None
  combiner: Literal['sum', 'mean', 'sqrtn'] = 'mean'
  initializer: Callable[..., Any] | None = None
  optimizer: Any | None = None
  weight_name: str | None = None
  param_dtype: str | None = None

  def __post_init__(self):
    if self.max_sequence_length is not None and self.weight_name is not None:
      raise ValueError(
          '`max_sequence_length` and `weight_name` cannot both be set. Weighted'
          ' aggregation can only be performed when the embeddings are'
          ' aggregated over the sequence dimension.'
      )

  def __hash__(self):
    return id(self)

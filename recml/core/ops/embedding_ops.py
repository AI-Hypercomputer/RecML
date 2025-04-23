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
"""Embedding lookup ops."""

from collections.abc import Mapping, Sequence
import dataclasses
import functools

from etils import epy
import jax
from jax.experimental import shard_map
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec


with epy.lazy_imports():
  # pylint: disable=g-import-not-at-top
  from jax_tpu_embedding.sparsecore.lib.nn import embedding
  # pylint: enable=g-import-not-at-top


@dataclasses.dataclass
class SparsecoreParams:
  """Embedding parameters."""

  feature_specs: embedding.Nested[embedding_spec.FeatureSpec]
  abstract_mesh: jax.sharding.AbstractMesh
  data_axes: Sequence[str | None]
  embedding_axes: Sequence[str | None]
  sharding_strategy: str


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def sparsecore_lookup(
    sparsecore_params: SparsecoreParams,
    tables: Mapping[str, tuple[jax.Array, ...]],
    csr_inputs: tuple[jax.Array, ...],
):
  return shard_map.shard_map(
      functools.partial(
          embedding.tpu_sparse_dense_matmul,
          global_device_count=sparsecore_params.abstract_mesh.size,
          feature_specs=sparsecore_params.feature_specs,
          sharding_strategy=sparsecore_params.sharding_strategy,
      ),
      mesh=sparsecore_params.abstract_mesh,
      in_specs=(
          jax.sharding.PartitionSpec(*sparsecore_params.data_axes),
          jax.sharding.PartitionSpec(*sparsecore_params.data_axes),
          jax.sharding.PartitionSpec(*sparsecore_params.data_axes),
          jax.sharding.PartitionSpec(*sparsecore_params.data_axes),
          jax.sharding.PartitionSpec(*sparsecore_params.embedding_axes),
      ),
      out_specs=jax.sharding.PartitionSpec(*sparsecore_params.data_axes),
      check_rep=False,
  )(*csr_inputs, tables)


def _emb_lookup_fwd(
    sparsecore_params: SparsecoreParams,
    tables: Mapping[str, tuple[jax.Array, ...]],
    csr_inputs: tuple[jax.Array, ...],
):
  out = sparsecore_lookup(sparsecore_params, tables, csr_inputs)
  return out, (tables, csr_inputs)


def _emb_lookup_bwd(
    sparsecore_params: SparsecoreParams,
    res: tuple[Mapping[str, tuple[jax.Array, ...]], tuple[jax.Array, ...]],
    gradients: embedding.Nested[jax.Array],
) -> tuple[embedding.Nested[jax.Array], None]:
  """Backward pass for embedding lookup."""
  (tables, csr_inputs) = res

  emb_table_grads = shard_map.shard_map(
      functools.partial(
          embedding.tpu_sparse_dense_matmul_grad,
          feature_specs=sparsecore_params.feature_specs,
          sharding_strategy=sparsecore_params.sharding_strategy,
      ),
      mesh=sparsecore_params.abstract_mesh,
      in_specs=(
          jax.sharding.PartitionSpec(*sparsecore_params.data_axes),
          jax.sharding.PartitionSpec(*sparsecore_params.data_axes),
          jax.sharding.PartitionSpec(*sparsecore_params.data_axes),
          jax.sharding.PartitionSpec(*sparsecore_params.data_axes),
          jax.sharding.PartitionSpec(*sparsecore_params.data_axes),
          jax.sharding.PartitionSpec(*sparsecore_params.embedding_axes),
      ),
      out_specs=jax.sharding.PartitionSpec(*sparsecore_params.data_axes),
      check_rep=False,
  )(gradients, *csr_inputs, tables)

  # `tpu_sparse_dense_matmul_grad` returns a general mapping (usually a dict).
  # It may not be the same type as the embedding table (e.g. FrozenDict).
  # Here we use flatten / unflatten to ensure the types are the same.
  emb_table_grads = jax.tree.unflatten(
      jax.tree.structure(tables), jax.tree.leaves(emb_table_grads)
  )

  return emb_table_grads, None


sparsecore_lookup.defvjp(_emb_lookup_fwd, _emb_lookup_bwd)

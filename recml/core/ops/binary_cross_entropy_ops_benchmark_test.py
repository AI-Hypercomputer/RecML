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
"""Mini benchmarks for binary_cross_entropy_ops.

These benchmarks run hundreds of steps on a large vocabulary, so they are
considerably slower than the correctness tests in
`binary_cross_entropy_ops_test.py`. The corresponding BUILD target is tagged
`manual` so that it does not slow down presubmit TAP and only runs in
post-submit.
"""

from collections.abc import Callable
import dataclasses
import json
import os
import pathlib
import time
from typing import Any

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import keras
from recml.core.ops import binary_cross_entropy_ops

_WARMUP_STEPS = 3


@dataclasses.dataclass(frozen=True)
class BenchmarkShape:
  """Input shape and tiling configuration for a CutBCE benchmark case.

  Attributes:
    name: Short identifier for logging and XProf artifact naming.
    batch: Batch dimension B.
    seq_len: Sequence / mask dimension N.
    vocab_size: Vocabulary size V.
    hidden_dim: Hidden / embedding dimension D.
    num_labels: Multi-hot positive label slots L per position.
    num_steps: Number of timed steady-state iterations.
    block_v: Vocabulary tile size passed to CutBCE.
    block_n: Token tile size passed to CutBCE.
    run_keras: Whether to attempt the dense Keras baseline.
    run_optl_false: Whether to also benchmark `optimize_large_l=False`.
    capture_xprof: Whether to capture `.response_pb` XProf traces.
  """

  name: str
  batch: int
  seq_len: int
  vocab_size: int
  hidden_dim: int
  num_labels: int
  num_steps: int = 200
  block_v: int = 4096
  block_n: int = 512
  run_keras: bool = True
  run_optl_false: bool = True
  capture_xprof: bool = False


def _make_inputs(
    shape: BenchmarkShape,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Returns random (activations, embeddings, targets) benchmark inputs."""
  key_act, key_emb, key_tgt = jax.random.split(jax.random.PRNGKey(42), 3)
  activations = jax.random.normal(
      key_act, (shape.batch, shape.seq_len, shape.hidden_dim), dtype=jnp.float32
  )
  embeddings = jax.random.normal(
      key_emb, (shape.vocab_size, shape.hidden_dim), dtype=jnp.float32
  )
  targets = jax.random.randint(
      key_tgt,
      (shape.batch, shape.seq_len, shape.num_labels),
      0,
      shape.vocab_size,
      dtype=jnp.int32,
  )
  jax.block_until_ready((activations, embeddings, targets))
  return activations, embeddings, targets


def _extract_memory_mb(
    compiled: jax.stages.Compiled,
) -> tuple[float, float, float]:
  """Returns (temp_mb, argument_mb, output_mb) from compiled memory analysis."""
  mem = compiled.memory_analysis()
  if mem is None:
    return -1.0, -1.0, -1.0
  return (
      mem.temp_size_in_bytes / 1e6,
      mem.argument_size_in_bytes / 1e6,
      mem.output_size_in_bytes / 1e6,
  )


def _benchmark_callable(
    shape: BenchmarkShape,
    variant: str,
    mode: str,
    fn: Callable[[jax.Array, jax.Array], Any],
    activations: jax.Array,
    embeddings: jax.Array,
    matmul_flops: float,
    xprof_tag: str = '',
) -> dict[str, Any]:
  """Compiles, times, and optionally profiles a JITTed benchmark callable."""
  device_kind = jax.devices()[0].device_kind
  label = f'{shape.name}:{variant}:{mode}'
  logging.info('Compiling %s on %s...', label, device_kind)
  t0 = time.perf_counter()
  jitted = jax.jit(fn)
  compiled = jitted.lower(activations, embeddings).compile()
  temp_mb, arg_mb, out_mb = _extract_memory_mb(compiled)

  for _ in range(_WARMUP_STEPS):
    out = compiled(activations, embeddings)
    jax.block_until_ready(out)
  compile_s = time.perf_counter() - t0

  t0 = time.perf_counter()
  for _ in range(shape.num_steps):
    out = compiled(activations, embeddings)
    jax.block_until_ready(out)
  step_ms = (time.perf_counter() - t0) * 1000.0 / shape.num_steps
  achieved_tflops = (matmul_flops / (step_ms * 1e-3)) / 1e12

  xprof_file = ''
  del xprof_tag

  record = {
      'device_kind': device_kind,
      'case': shape.name,
      'batch': shape.batch,
      'seq_len': shape.seq_len,
      'tokens': shape.batch * shape.seq_len,
      'vocab_size': shape.vocab_size,
      'hidden_dim': shape.hidden_dim,
      'num_labels': shape.num_labels,
      'block_v': shape.block_v,
      'block_n': shape.block_n,
      'variant': variant,
      'mode': mode,
      'status': 'ok',
      'step_ms': round(step_ms, 4),
      'achieved_tflops': round(achieved_tflops, 2),
      'temp_mb': round(temp_mb, 2),
      'arg_mb': round(arg_mb, 2),
      'out_mb': round(out_mb, 2),
      'compile_s': round(compile_s, 2),
      'xprof_file': xprof_file,
  }
  logging.info(
      'BENCH_RESULT | dev=%s | case=%s | var=%s | mode=%s | ms=%.4f | '
      'tflops=%.2f | temp_mb=%.2f | xprof=%s',
      device_kind,
      shape.name,
      variant,
      mode,
      step_ms,
      achieved_tflops,
      temp_mb,
      xprof_file,
  )
  return record


def _run_suite_for_shape(shape: BenchmarkShape) -> list[dict[str, Any]]:
  """Runs Keras and CutBCE forward-only + fwd+bwd benchmarks for `shape`."""
  activations, embeddings, targets = _make_inputs(shape)
  tokens = shape.batch * shape.seq_len
  fwd_flops = 2.0 * tokens * shape.vocab_size * shape.hidden_dim
  fwd_bwd_flops = 6.0 * tokens * shape.vocab_size * shape.hidden_dim
  records: list[dict[str, Any]] = []

  if shape.run_keras:
    try:
      one_hot = jax.nn.one_hot(targets, shape.vocab_size, axis=-1)
      multi_hot = jnp.max(one_hot, axis=-2)
      jax.block_until_ready(multi_hot)

      def run_keras(act: jax.Array, emb: jax.Array) -> jax.Array:
        logits = jnp.matmul(act, emb.T)
        loss_per_token = keras.losses.binary_crossentropy(
            multi_hot, logits, from_logits=True
        )
        return jnp.mean(loss_per_token)

      records.append(
          _benchmark_callable(
              shape,
              'keras',
              'fwd',
              run_keras,
              activations,
              embeddings,
              fwd_flops,
          )
      )
      records.append(
          _benchmark_callable(
              shape,
              'keras',
              'fwd_bwd',
              jax.grad(run_keras, argnums=(0, 1)),
              activations,
              embeddings,
              fwd_bwd_flops,
          )
      )
    except Exception as err:  # pylint: disable=broad-exception-caught
      logging.warning('Keras OOM/failed on %s: %s', shape.name, err)
      records.append({
          'device_kind': jax.devices()[0].device_kind,
          'case': shape.name,
          'variant': 'keras',
          'mode': 'fwd_bwd',
          'status': f'oom_or_err: {type(err).__name__}: {err}',
      })

  opt_l_modes = (False, True) if shape.run_optl_false else (True,)
  for opt_l in opt_l_modes:
    variant = f'cut_optl_{int(opt_l)}'

    def run_cut(
        act: jax.Array,
        emb: jax.Array,
        optimize_large_l: bool = opt_l,
    ) -> jax.Array:
      return binary_cross_entropy_ops.cut_binary_cross_entropy(
          act,
          emb,
          targets,
          block_v=shape.block_v,
          block_n=shape.block_n,
          optimize_large_l=optimize_large_l,
      )

    want_xprof = shape.capture_xprof and opt_l
    records.append(
        _benchmark_callable(
            shape,
            variant,
            'fwd',
            run_cut,
            activations,
            embeddings,
            fwd_flops,
            xprof_tag=f'{shape.name}_{variant}_fwd' if want_xprof else '',
        )
    )
    records.append(
        _benchmark_callable(
            shape,
            variant,
            'fwd_bwd',
            jax.grad(run_cut, argnums=(0, 1)),
            activations,
            embeddings,
            fwd_bwd_flops,
            xprof_tag=f'{shape.name}_{variant}_fwd_bwd' if want_xprof else '',
        )
    )
    records.append(
        _benchmark_callable(
            shape,
            variant,
            'val_grad',
            jax.value_and_grad(run_cut, argnums=(0, 1)),
            activations,
            embeddings,
            fwd_bwd_flops,
        )
    )

  out_dir = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', '')
  if out_dir:
    json_path = pathlib.Path(out_dir) / f'bench_{shape.name}.json'
    json_path.write_text(json.dumps(records, indent=2))
  return records


class BinaryCrossEntropyOpsBenchmarkTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Skipping TPU benchmark.')

  @parameterized.named_parameters(
      dict(
          testcase_name='paper_32x32_v100k',
          shape=BenchmarkShape(
              name='paper_32x32_v100k',
              batch=32,
              seq_len=32,
              vocab_size=100_000,
              hidden_dim=128,
              num_labels=8,
              num_steps=200,
              block_v=4096,
              block_n=512,
              run_keras=True,
              capture_xprof=True,
          ),
      ),
      dict(
          testcase_name='paper_32x32_v200k',
          shape=BenchmarkShape(
              name='paper_32x32_v200k',
              batch=32,
              seq_len=32,
              vocab_size=200_000,
              hidden_dim=128,
              num_labels=8,
              num_steps=200,
              block_v=4096,
              block_n=512,
              run_keras=True,
              capture_xprof=False,
          ),
      ),
      dict(
          testcase_name='paper_128x128_v100k',
          shape=BenchmarkShape(
              name='paper_128x128_v100k',
              batch=128,
              seq_len=128,
              vocab_size=100_000,
              hidden_dim=128,
              num_labels=8,
              num_steps=100,
              block_v=4096,
              block_n=512,
              run_keras=True,
              run_optl_false=False,
              capture_xprof=True,
          ),
      ),
      dict(
          testcase_name='gum_16x80_v121k_bv11008',
          shape=BenchmarkShape(
              name='gum_16x80_v121k_bv11008',
              batch=16,
              seq_len=80,
              vocab_size=121_088,
              hidden_dim=256,
              num_labels=64,
              num_steps=150,
              block_v=11008,
              block_n=256,
              run_keras=True,
              run_optl_false=False,
              capture_xprof=True,
          ),
      ),
      dict(
          testcase_name='gum_16x80_v121k_bv5504',
          shape=BenchmarkShape(
              name='gum_16x80_v121k_bv5504',
              batch=16,
              seq_len=80,
              vocab_size=121_088,
              hidden_dim=256,
              num_labels=64,
              num_steps=150,
              block_v=5504,
              block_n=256,
              run_keras=False,
              run_optl_false=False,
              capture_xprof=True,
          ),
      ),
  )
  def test_benchmark_shape(self, shape: BenchmarkShape):
    records = _run_suite_for_shape(shape)
    self.assertNotEmpty(records)


if __name__ == '__main__':
  absltest.main()

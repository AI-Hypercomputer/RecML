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
import collections
import functools
import os
import threading
import time
from typing import Any, Callable, List, Mapping

import orbax.checkpoint as ocp
from absl import app, flags, logging
from clu import metrics as clu_metrics
import flax
import jax
import jax.numpy as jnp
from jax.experimental.layout import DeviceLocalLayout as DLL
from jax.experimental.layout import Layout
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import metrax
import numpy as np
import optax

from dataloader import CriteoDataLoader, DataConfig
from dlrm_model import DLRMDCNV2, uniform_init
from jax_tpu_embedding.sparsecore.lib.flax import embed, embed_optimizer
from jax_tpu_embedding.sparsecore.lib.nn import (
    embedding,
    embedding_spec,
)

jax.distributed.initialize()
import jax.profiler

jax.profiler.start_server(9999)
partial = functools.partial
info = logging.info
shard_map = jax.experimental.shard_map.shard_map
Nested = embedding.Nested

FLAGS = flags.FLAGS

# --- Data and Model Flags ---
VOCAB_SIZES = [
    40000000,
    39060,
    17295,
    7424,
    20265,
    3,
    7122,
    1543,
    63,
    40000000,
    3067956,
    405282,
    10,
    2209,
    11938,
    155,
    4,
    976,
    14,
    40000000,
    40000000,
    40000000,
    590152,
    12973,
    108,
    36,
]
MULTI_HOT_SIZES = [
    3,
    2,
    1,
    2,
    6,
    1,
    1,
    1,
    1,
    7,
    3,
    8,
    1,
    6,
    9,
    5,
    1,
    1,
    1,
    12,
    100,
    27,
    10,
    3,
    1,
    1,
]
_NUM_DENSE_FEATURES = flags.DEFINE_integer(
    "num_dense_features", 13, "Number of dense features."
)
_EMBEDDING_SIZE = flags.DEFINE_integer("embedding_size", 16, "Embedding size.")
_EMBEDDING_THRESHOLD = flags.DEFINE_integer(
    "embedding_threshold",
    21000,
    "Threshold for placing features on TensorCore or SparseCore.",
)

# --- Mode Flag ---
_MODE = flags.DEFINE_enum(
    "mode",
    "train",
    ["train", "eval"],
    "Mode to run: 'train' for training, 'eval' for evaluation-only.",
)

# --- Training Flags ---
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 8192, "Batch size.")
_FILE_PATTERN = flags.DEFINE_string(
    "file_pattern", None, "File pattern for the training data."
)
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 0.0034, "Learning rate.")
_NUM_STEPS = flags.DEFINE_integer("num_steps", 28000, "Number of steps to train for.")
_LOGGING_INTERVAL = flags.DEFINE_integer(
    "logging_interval", 6000, "Frequency of logging training metrics."
)
_ALLOW_ID_DROPPING = flags.DEFINE_bool(
    "allow_id_dropping", True, "Allow dropping ids during embedding lookup."
)

# --- Evaluation Flags ---
_EVAL_FILE_PATTERN = flags.DEFINE_string(
    "eval_file_pattern", None, "File pattern for the evaluation data."
)
_EVAL_INTERVAL = flags.DEFINE_integer(
    "eval_interval", 5000, "Run evaluation every N steps."
)
_EVAL_STEPS = flags.DEFINE_integer(
    "eval_steps", 0, "Number of steps for each eval. 0 for all."
)

# --- Checkpointing and Misc Flags ---
_MODEL_DIR = flags.DEFINE_string(
    "model_dir", "/tmp/dlrm_jax", "Model working directory."
)
_SAVE_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    "save_checkpoint_interval", 5000, "Frequency of saving checkpoints."
)
_RESTORE_CHECKPOINT = flags.DEFINE_bool(
    "restore_checkpoint", False, "Restore from the latest checkpoint."
)


def create_feature_specs(
    vocab_sizes: List[int],
) -> tuple[
    Mapping[str, embedding_spec.TableSpec],
    Mapping[str, embedding_spec.FeatureSpec],
]:
    """Creates the feature specs for the DLRM model."""
    table_specs = {}
    feature_specs = {}
    for i, vocab_size in enumerate(vocab_sizes):
        if vocab_size <= _EMBEDDING_THRESHOLD.value:
            continue

        table_name = f"{i}"
        feature_name = f"{i}"
        bound = jnp.sqrt(1.0 / vocab_size)
        table_spec = embedding_spec.TableSpec(
            vocabulary_size=vocab_size,
            embedding_dim=_EMBEDDING_SIZE.value,
            initializer=uniform_init(bound),
            optimizer=embedding_spec.AdagradOptimizerSpec(
                learning_rate=_LEARNING_RATE.value
            ),
            combiner="sum",
            name=table_name,
            max_ids_per_partition=2048,
            max_unique_ids_per_partition=512,
        )
        feature_spec = embedding_spec.FeatureSpec(
            table_spec=table_spec,
            input_shape=(_BATCH_SIZE.value, 1),
            output_shape=(_BATCH_SIZE.value, _EMBEDDING_SIZE.value),
            name=feature_name,
        )
        feature_specs[feature_name] = feature_spec
        table_specs[table_name] = table_spec
    return table_specs, feature_specs


@flax.struct.dataclass
class TrainMetrics(clu_metrics.Collection):
    """Metrics for the training loop."""

    loss: metrax.Average
    accuracy: metrax.Accuracy
    auc: metrax.AUCROC


@flax.struct.dataclass
class EvalMetrics(clu_metrics.Collection):
    """Metrics for the evaluation loop."""

    loss: metrax.Average
    accuracy: metrax.Accuracy
    auc: metrax.AUCROC


class DLRMDataLoader:
    """Parallel data producer for the DLRM model."""

    def __init__(
        self,
        file_pattern: str,
        batch_size,
        is_training: bool,
        num_workers=4,
        buffer_size=128,
        feature_specs=None,
        mesh=None,
        global_sharding=None,
    ):
        """Initialize the producer."""
        self.data_config = DataConfig(
            global_batch_size=batch_size,
            is_training=is_training,
            use_cached_data=file_pattern is None,
        )
        self._dataloader = CriteoDataLoader(
            file_pattern=file_pattern,
            params=self.data_config,
            num_dense_features=_NUM_DENSE_FEATURES.value,
            vocab_sizes=VOCAB_SIZES,
            multi_hot_sizes=MULTI_HOT_SIZES,
            embedding_threshold=_EMBEDDING_THRESHOLD.value,
        )
        self._iterator = self._dataloader.get_iterator()
        self.feature_specs = feature_specs
        self.mesh = mesh
        self.global_sharding = global_sharding

        self.buffer = collections.deque(maxlen=buffer_size)
        self._sync = threading.Condition()
        self._workers = []

        for _ in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self._workers.append(worker)

    def process_inputs(self, feature_batch):
        """Process input features into the required format."""
        dense_features = feature_batch["dense_features"]
        sparse_features = feature_batch["sparse_features"]
        dense_lookups = {}
        for i in range(len(VOCAB_SIZES)):
            if VOCAB_SIZES[i] <= _EMBEDDING_THRESHOLD.value:
                dense_lookups[str(i)] = sparse_features[str(i)]
                sparse_features.pop(str(i))

        labels = feature_batch["clicked"]

        feature_weights = jax.tree_util.tree_map(
            lambda x: np.array(np.ones_like(x, shape=x.shape, dtype=np.float32)),
            sparse_features,
        )

        processed_sparse = embedding.preprocess_sparse_dense_matmul_input(
            sparse_features,
            feature_weights,
            self.feature_specs,
            self.mesh.local_mesh.size,
            self.mesh.size,
            num_sc_per_device=4,
            sharding_strategy="MOD",
            allow_id_dropping=_ALLOW_ID_DROPPING.value,
        )[0]
        make_global_view = lambda x: jax.tree.map(
            lambda y: jax.make_array_from_process_local_data(self.global_sharding, y),
            x,
        )
        labels = make_global_view(labels)
        dense_features = make_global_view(dense_features)
        dense_lookups = make_global_view(dense_lookups)
        processed_sparse = map(make_global_view, processed_sparse)
        processed_sparse = embedding.SparseDenseMatmulInput(*processed_sparse)
        return [labels, dense_features, dense_lookups, processed_sparse]

    def _worker_loop(self):
        """Worker thread that continuously generates and processes batches."""
        while True:
            try:
                # This will fail if the main thread deletes _iterator.
                batch = next(self._iterator)
                processed_batch = self.process_inputs(batch)
                with self._sync:
                    self._sync.wait_for(lambda: len(self.buffer) < self.buffer.maxlen)
                    self.buffer.append(processed_batch)
                    self._sync.notify_all()
            except (StopIteration, AttributeError):  # Catch error if iterator is gone
                with self._sync:
                    self.buffer.append(None)
                    self._sync.notify_all()
                return

    def __iter__(self):
        return self

    def __next__(self):
        """Get next batch from the buffer."""
        with self._sync:
            self._sync.wait_for(lambda: self.buffer)
            item = self.buffer.popleft()
            self._sync.notify_all()
            if item is None:
                raise StopIteration
            return item

    def stop(self):
        """Stop all worker threads and clear the buffer."""
        if hasattr(self, "_iterator"):
            del self._iterator


def eval_loop(
    eval_producer: DLRMDataLoader,
    eval_step_fn: Callable,
    params: Any,
    apply_fn: Callable,
    max_steps: int = 0,
):
    """Runs the evaluation loop."""
    logging.info("Starting evaluation...")
    eval_metrics_collection = EvalMetrics.empty()
    step_count = 0
    for batch in eval_producer:
        labels, dense_features, dense_lookups, embedding_lookups = batch
        eval_metrics_collection = eval_step_fn(
            apply_fn,
            params,
            labels,
            dense_features,
            dense_lookups,
            embedding_lookups,
            eval_metrics_collection,
        )
        step_count += 1
        if max_steps > 0 and step_count >= max_steps:
            logging.info("Reached max evaluation steps (%d).", max_steps)
            break

    logging.info("Finished evaluation after %d steps.", step_count)
    metrics_on_host = jax.device_get(eval_metrics_collection)
    loss_val = metrics_on_host.loss.compute()
    accuracy_val = metrics_on_host.accuracy.compute()
    try:
        auc_val = metrics_on_host.auc.compute()
    except (ValueError, ZeroDivisionError):
        auc_val = 0.5
    logging.info(
        "Evaluation results: loss=%.5f, accuracy=%.5f, auc=%.5f",
        loss_val,
        accuracy_val,
        auc_val,
    )


@partial(jax.jit, static_argnums=0)
def eval_step(
    apply_fn: Callable,
    params: Any,
    labels: jax.Array,
    dense_features: jax.Array,
    dense_lookups: Any,
    embedding_lookups: embed.EmbeddingLookupInput,
    metrics_collection,
):
    logits = apply_fn(params, dense_features, dense_lookups, embedding_lookups)
    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))
    preds = jax.nn.sigmoid(logits)
    binarized_preds = (preds > 0.5).astype(jnp.int32)
    metric_updates = EvalMetrics.empty().replace(
        loss=metrax.Average.from_model_output(values=loss),
        accuracy=metrax.Accuracy.from_model_output(binarized_preds, labels),
        auc=metrax.AUCROC.from_model_output(preds, labels),
    )
    return metrics_collection.merge(metric_updates)


def train_loop(
    model: DLRMDCNV2,
    feature_specs: Mapping[str, embedding_spec.FeatureSpec],
    mesh: jax.sharding.Mesh,
    global_sharding=None,
):
    """Main training and evaluation loop."""
    producer = DLRMDataLoader(
        file_pattern=_FILE_PATTERN.value,
        batch_size=_BATCH_SIZE.value,
        is_training=True,
        num_workers=16,
        buffer_size=256,
        feature_specs=feature_specs,
        mesh=mesh,
        global_sharding=global_sharding,
    )

    _, dense_features, dense_lookups, embedding_lookups = next(producer)
    params = model.init(
        jax.random.key(42), dense_features, dense_lookups, embedding_lookups
    )
    tx = embed_optimizer.create_optimizer_for_sc_model(
        params, optax.adagrad(learning_rate=_LEARNING_RATE.value)
    )
    opt_state = tx.init(params)

    checkpoint_dir = os.path.join(_MODEL_DIR.value, "checkpoints")
    checkpointer = ocp.CheckpointManager(checkpoint_dir)

    initial_step = 0
    if _RESTORE_CHECKPOINT.value:
        latest_step = checkpointer.latest_step()
        if latest_step is not None:
            logging.info("Found checkpoint at step %d. Restoring...", latest_step)
            ckpt_target = {"params": params, "opt_state": opt_state, "step": 0}
            restored = checkpointer.restore(
                latest_step, args=ocp.args.PyTreeRestore(ckpt_target)
            )
            params = restored["params"]
            opt_state = restored["opt_state"]
            initial_step = restored["step"]
            logging.info("Restored state from step %d.", initial_step)
        else:
            logging.info("No checkpoint found to restore from. Starting from scratch.")

    @partial(jax.jit, donate_argnums=(0, 5))
    def train_step_fn(
        params: Any,
        labels: jax.Array,
        dense_features: jax.Array,
        dense_lookups: Any,
        embedding_lookups: embed.EmbeddingLookupInput,
        opt_state,
        metrics_collection,
    ):
        def forward_pass(p, lbl, dense, dense_lkp, embed_lkp):
            logits = model.apply(p, dense, dense_lkp, embed_lkp)
            xentropy = optax.sigmoid_binary_cross_entropy(logits, lbl)
            return jnp.mean(xentropy), logits

        train_fn = jax.value_and_grad(forward_pass, has_aux=True, allow_int=True)
        (loss_val, logits), grads = train_fn(
            params, labels, dense_features, dense_lookups, embedding_lookups
        )
        preds = jax.nn.sigmoid(logits)
        binarized_preds = (preds > 0.5).astype(jnp.int32)
        metric_updates = TrainMetrics.empty().replace(
            loss=metrax.Average.from_model_output(values=loss_val),
            accuracy=metrax.Accuracy.from_model_output(binarized_preds, labels),
            auc=metrax.AUCROC.from_model_output(preds, labels),
        )
        metrics_collection = metrics_collection.merge(metric_updates)
        updates, new_opt_state = tx.update(grads, opt_state)
        new_params = embed_optimizer.apply_updates_for_sc_model(params, updates)
        return new_params, new_opt_state, metrics_collection

    start_time = time.time()
    train_metrics_collection = TrainMetrics.empty()
    for step in range(initial_step, _NUM_STEPS.value):
        with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
            labels, dense_features, dense_lookups, embedding_lookups = next(producer)
            params, opt_state, train_metrics_collection = train_step_fn(
                params,
                labels,
                dense_features,
                dense_lookups,
                embedding_lookups,
                opt_state,
                train_metrics_collection,
            )

        current_step = step + 1
        if current_step % _LOGGING_INTERVAL.value == 0:
            end_time = time.time()
            metrics_on_host = jax.device_get(train_metrics_collection)
            try:
                auc_val = metrics_on_host.auc.compute()
            except (ValueError, ZeroDivisionError):
                auc_val = 0.5
            logging.info(
                "Step %d: loss=%.5f, accuracy=%.5f, auc=%.5f, step_time=%.2fms",
                current_step,
                metrics_on_host.loss.compute(),
                metrics_on_host.accuracy.compute(),
                auc_val,
                (end_time - start_time) * 1000 / _LOGGING_INTERVAL.value,
            )
            train_metrics_collection = TrainMetrics.empty()
            start_time = time.time()

        if current_step % _EVAL_INTERVAL.value == 0 and _EVAL_FILE_PATTERN.value:
            eval_producer = DLRMDataLoader(
                file_pattern=_EVAL_FILE_PATTERN.value,
                batch_size=_BATCH_SIZE.value,
                is_training=False,
                num_workers=4,
                buffer_size=128,
                feature_specs=feature_specs,
                mesh=mesh,
                global_sharding=global_sharding,
            )
            eval_loop(
                eval_producer,
                eval_step,
                params,
                model.apply,
                max_steps=_EVAL_STEPS.value,
            )
            eval_producer.stop()

        if current_step % _SAVE_CHECKPOINT_INTERVAL.value == 0:
            ckpt_to_save = {
                "params": params,
                "opt_state": opt_state,
                "step": current_step,
            }
            checkpointer.save(
                current_step, args=ocp.args.PyTreeSave(ckpt_to_save), force=True
            )

    producer.stop()
    checkpointer.wait_until_finished()
    checkpointer.close()


def run_evaluation_only(
    model: DLRMDCNV2,
    feature_specs: Mapping[str, embedding_spec.FeatureSpec],
    mesh: jax.sharding.Mesh,
    global_sharding: NamedSharding,
):
    """Runs evaluation on a saved checkpoint."""
    if not _EVAL_FILE_PATTERN.value:
        raise ValueError("--eval_file_pattern must be set in 'eval' mode.")

    checkpoint_dir = os.path.join(_MODEL_DIR.value, "checkpoints")
    checkpointer = ocp.CheckpointManager(checkpoint_dir)
    latest_step = checkpointer.latest_step()

    if latest_step is None:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir} to evaluate.")

    logging.info(
        "Found checkpoint at step %d. Restoring for evaluation...", latest_step
    )

    dummy_producer = DLRMDataLoader(
        file_pattern=_EVAL_FILE_PATTERN.value,
        batch_size=_BATCH_SIZE.value,
        is_training=False,
        num_workers=1,
        feature_specs=feature_specs,
        mesh=mesh,
        global_sharding=global_sharding,
    )
    _, dense_features, dense_lookups, embedding_lookups = next(dummy_producer)
    # Do not call stop() on the dummy_producer. Let its daemon threads exit with the program.

    params_structure = jax.eval_shape(
        lambda: model.init(
            jax.random.key(0), dense_features, dense_lookups, embedding_lookups
        )
    )

    dummy_tx = embed_optimizer.create_optimizer_for_sc_model(
        params_structure, optax.adagrad(learning_rate=0.0)
    )
    opt_state_structure = jax.eval_shape(lambda: dummy_tx.init(params_structure))

    restore_target = {
        "params": params_structure,
        "opt_state": opt_state_structure,
        "step": 0,
    }

    restored_full_dict = checkpointer.restore(
        latest_step, args=ocp.args.PyTreeRestore(restore_target)
    )
    params = restored_full_dict["params"]

    logging.info("Parameters restored. Starting evaluation...")

    eval_producer = DLRMDataLoader(
        file_pattern=_EVAL_FILE_PATTERN.value,
        batch_size=_BATCH_SIZE.value,
        is_training=False,
        num_workers=16,
        buffer_size=128,
        feature_specs=feature_specs,
        mesh=mesh,
        global_sharding=global_sharding,
    )

    eval_loop(
        eval_producer, eval_step, params, model.apply, max_steps=_EVAL_STEPS.value
    )

    eval_producer.stop()
    checkpointer.close()


def main(argv):
    del argv

    pd = P("x")
    global_devices = jax.devices()
    mesh = jax.sharding.Mesh(global_devices, "x")
    global_sharding = jax.sharding.NamedSharding(mesh, pd)

    _, feature_specs = create_feature_specs(VOCAB_SIZES)

    def _get_max_ids_per_partition(name: str, batch_size: int) -> int:
        return 16384

    def _get_max_unique_ids_per_partition(name: str, batch_size: int) -> int:
        return 8192

    embedding.auto_stack_tables(
        feature_specs,
        global_device_count=jax.device_count(),
        stack_to_max_ids_per_partition=_get_max_ids_per_partition,
        stack_to_max_unique_ids_per_partition=_get_max_unique_ids_per_partition,
        num_sc_per_device=4,
    )
    embedding.prepare_feature_specs_for_training(
        feature_specs,
        global_device_count=jax.device_count(),
        num_sc_per_device=4,
    )

    model = DLRMDCNV2(
        feature_specs=feature_specs,
        mesh=mesh,
        sharding_axis="x",
        global_batch_size=_BATCH_SIZE.value,
        embedding_size=_EMBEDDING_SIZE.value,
        bottom_mlp_dims=[512, 256, _EMBEDDING_SIZE.value],
        vocab_sizes=VOCAB_SIZES,
    )

    if _MODE.value == "train":
        train_loop(model, feature_specs, mesh, global_sharding)
    elif _MODE.value == "eval":
        run_evaluation_only(model, feature_specs, mesh, global_sharding)


if __name__ == "__main__":
    app.run(main)

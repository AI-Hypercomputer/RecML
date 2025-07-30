"""DLRM DCN v2 model."""

import collections
import functools
import os
import threading
import time
from typing import Any, List, Mapping

from absl import app
from absl import flags
from absl import logging
import flax
import jax
from jax.experimental.layout import Layout as DLL
from jax.experimental.layout import Format
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

# Checkpointing and embedding spec proto imports
import orbax.checkpoint as ocp
from jax_tpu_embedding.sparsecore.lib.proto import embedding_spec_pb2

from dataloader import CriteoDataLoader
from dataloader import DataConfig
from dlrm_model import DLRMDCNV2
from dlrm_model import uniform_init
from jax_tpu_embedding.sparsecore.lib.flax import embed
from jax_tpu_embedding.sparsecore.lib.flax import embed_optimizer
from jax_tpu_embedding.sparsecore.lib.fdo import file_fdo_client
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
import numpy as np
import optax

jax.distributed.initialize()
import jax.profiler

jax.profiler.start_server(9999)
partial = functools.partial
info = logging.info
shard_map = jax.experimental.shard_map.shard_map
Nested = embedding.Nested

FLAGS = flags.FLAGS

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

np.set_printoptions(threshold=np.inf)

_BATCH_SIZE = flags.DEFINE_integer("batch_size", 8192, "Batch size.")

# MLPerf official dataset:
# "/cns/ig-d/home/tpu-perf-team/chandrasekhard/criteo/terabyte_processed_golden_shuffled/final/train-*tfrecord"
_FILE_PATTERN = flags.DEFINE_string(
    "file_pattern",
    None,
    "File pattern for the training data.",
)

_FDO_DIR = flags.DEFINE_string(
    "fdo_dir",
    "/tmp",
    "If set, FDO dumps will be written to the directory.",
)


_LOG_FREQUENCY = flags.DEFINE_integer(
    "log_frequency", 10, "Frequency to log metrics."
)

_LEARNING_RATE = flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")

_NUM_DENSE_FEATURES = flags.DEFINE_integer(
    "num_dense_features", 13, "Number of dense features."
)

_NUM_STEPS = flags.DEFINE_integer(
    "num_steps", 28000, "Number of steps to train for."
)

_NUM_EPOCHS = flags.DEFINE_integer(
    "num_epochs",
    1,
    "Number of separate epochs.",
)

_EMBEDDING_SIZE = flags.DEFINE_integer("embedding_size", 16, "Embedding size.")

_ALLOW_ID_DROPPING = flags.DEFINE_bool(
    "allow_id_dropping",
    True,
    "If set, allow dropping ids during embedding lookup.",
)

_EMBEDDING_THRESHOLD = flags.DEFINE_integer(
    "embedding_threshold",
    21000,
    "Embedding threshold for placing features on TensorCore or SparseCore.",
)

_EXPERIMENT_NAME = flags.DEFINE_string(
    "experiment_name",
    "jax_dlrm",
    "The name of the experiment. If not set the name of the fixture that"
    " returns the fiddle configuration will be used instead.",
)
_MODEL_DIR = flags.DEFINE_string(
    "model_dir",
    "/cns/{cell}-d/home/tpu-perf-team/{username}/{exp_name}/{xid}",
    "Model working directory.",
)

# --- Checkpointing Flags Start ---
_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir',
    None,
    'If set, checkpoints will be written to the directory.',
)

_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    'checkpoint_interval', 1000, 'Number of steps per checkpoint'
)

_CHECKPOINT_RESUME = flags.DEFINE_bool(
    'checkpoint_resume',
    True,
    'If set True and checkpoint_dir is specified, try to resume the training'
    ' from the latest checkpoint available in the checkpoint_dir.',
)

_CHECKPOINT_MAX_TO_KEEP = flags.DEFINE_integer(
    'checkpoint_max_to_keep',
    5,
    'Number of checkpoints to keep.',
)
# --- Checkpointing Flags End ---

_MAX_IDS = flags.DEFINE_integer(
    "max_ids",
    2048,
    "max_ids",
)

_MAX_UNIQUE_IDS = flags.DEFINE_integer(
    "max_unique_ids",
    512,
    "max_unique_ids",
)




@flax.struct.dataclass
class TrainState:
  """State of the model and the training."""
  params: Any
  opt_state: Any


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
        optimizer=embedding_spec.AdagradOptimizerSpec(learning_rate=0.01),
        combiner="sum",
        name=table_name,
        max_ids_per_partition=_MAX_IDS.value,
        max_unique_ids_per_partition=_MAX_UNIQUE_IDS.value,
    )
    feature_spec = embedding_spec.FeatureSpec(
        table_spec=table_spec,
        input_shape=(_BATCH_SIZE.value, 1),
        output_shape=(
            _BATCH_SIZE.value,
            _EMBEDDING_SIZE.value,
        ),
        name=feature_name,
    )
    feature_specs[feature_name] = feature_spec
    table_specs[table_name] = table_spec
  print(table_specs)
  print(feature_specs)
  return table_specs, feature_specs


class DLRMDataLoader:
  """Parallel data producer for the DLRM model."""

  def __init__(
      self,
      file_pattern: str,
      batch_size,
      num_workers=4,
      buffer_size=128,
      feature_specs=None,
      mesh=None,
      global_sharding=None,
      fdo_client = None,
  ):
    """Initialize the producer.

    Args:
        file_pattern: File pattern for the training data.
        batch_size: The size of each batch.
        num_workers: Number of threads generating batches.
        buffer_size: Maximum number of batches to store in the buffer.
        feature_specs: Specifications for features.
        mesh: Mesh configuration for distributed processing.
        global_sharding: Global sharding configuration.
    """
    # Here we pass global batch size. TF.data will shard it.
    self.data_config = DataConfig(
        global_batch_size=batch_size,
        pre_batch_size=4224,
        is_training=True,
        use_cached_data=False if file_pattern else True,
    )
    self._dataloader = CriteoDataLoader(
        file_pattern=file_pattern,
        params=self.data_config,
        num_dense_features=_NUM_DENSE_FEATURES.value,
        vocab_sizes=VOCAB_SIZES,
        multi_hot_sizes=MULTI_HOT_SIZES,
        # embedding_threshold=_EMBEDDING_THRESHOLD.value,
    )
    self._iterator = self._dataloader.get_iterator()
    self.batch_size = batch_size
    self.feature_specs = feature_specs
    self.mesh = mesh
    self.global_sharding = global_sharding

    self.buffer = collections.deque(maxlen=buffer_size)
    self._sync = threading.Condition()
    self._workers = []
    self.fdo_client = fdo_client

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

    # Process sparse features
    processed_sparse, stats = embedding.preprocess_sparse_dense_matmul_input(
        sparse_features,
        feature_weights,
        self.feature_specs,
        self.mesh.local_mesh.size,
        self.mesh.size,
        num_sc_per_device=2,
        sharding_strategy="MOD",
        allow_id_dropping=_ALLOW_ID_DROPPING.value,
        # static_buffer_size_multiplier=256,
    )
    self.fdo_client.record(stats)
    # Preprocess the inputs and build Jax global views of the data.
    # Global view is required for embedding lookup.
    make_global_view = lambda x: jax.tree.map(
        lambda y: jax.make_array_from_process_local_data(
            self.global_sharding, y
        ),
        x,
    )
    labels = make_global_view(labels)
    dense_features = make_global_view(dense_features)
    dense_lookups = make_global_view(dense_lookups)
    processed_sparse = make_global_view(processed_sparse)
    return [labels, dense_features, dense_lookups, processed_sparse]

  def _worker_loop(self):
    """Worker thread that continuously generates and processes batches."""
    while True:
      batch = next(self._iterator)
      processed_batch = self.process_inputs(batch)
      # processed_batch = jax.device_put(processed_batch, self.global_sharding)

      with self._sync:
        self._sync.wait_for(lambda: len(self.buffer) < self.buffer.maxlen)
        self.buffer.append(processed_batch)
        self._sync.notify_all()

  def __iter__(self):
    return self

  def __next__(self):
    """Get next batch from the buffer."""
    with self._sync:
      self._sync.wait_for(lambda: self.buffer)
      item = self.buffer.popleft()
      self._sync.notify_all()
      return item

  def stop(self):
    """Stop all worker threads and clear the buffer."""
    for worker in self._workers:
      worker.join(timeout=5)

    with self._sync:
      self.buffer.clear()
      self._sync.notify_all()

    # del self._iterator


def test_dlrm_dcnv2_model():
  """Test DLRM DCN v2 model."""

  # define fdo client
  out_path = os.path.join(_MODEL_DIR.value, "fdo_dump")
  os.makedirs(out_path, exist_ok=True)
  logging.info("FDO storage path: %s", out_path)
  fdo_client = file_fdo_client.NPZFileFDOClient(out_path)

  # Define sharding.
  pd = P("x")
  global_devices = jax.devices()
  mesh = jax.sharding.Mesh(global_devices, "x")
  global_sharding = jax.sharding.NamedSharding(mesh, pd)

  _, feature_specs = create_feature_specs(VOCAB_SIZES)

  def _get_max_ids_per_partition(name: str, batch_size: int) -> int:
    """Reference implementation for calculating max ids per partition on the fly."""
    logging.info("max_ids_per_partition: %s : # of ids: %s", name, batch_size)
    return _MAX_IDS.value

  def _get_max_unique_ids_per_partition(name: str, batch_size: int) -> int:
    """Reference implementation for calculating max unique ids per partition."""
    logging.info(
        "max_unique_ids_per_partition: %s : # of ids: %s", name, batch_size
    )
    return _MAX_UNIQUE_IDS.value

  # Table stacking
  embedding.auto_stack_tables(
      feature_specs,
      global_device_count=jax.device_count(),
      stack_to_max_ids_per_partition=_get_max_ids_per_partition,
      stack_to_max_unique_ids_per_partition=_get_max_unique_ids_per_partition,
      num_sc_per_device=2,
  )

  embedding.prepare_feature_specs_for_training(
      feature_specs,
      global_device_count=jax.device_count(),
      num_sc_per_device=2,
  )

  # Construct the model.
  model = DLRMDCNV2(
      feature_specs=feature_specs,
      mesh=mesh,
      sharding_axis="x",
      global_batch_size=_BATCH_SIZE.value,
      embedding_size=_EMBEDDING_SIZE.value,
      bottom_mlp_dims=[512, 256, _EMBEDDING_SIZE.value],
      vocab_sizes=VOCAB_SIZES,
  )

  # Initialize model, optimizer, and state
  producer = DLRMDataLoader(
      file_pattern=_FILE_PATTERN.value,
      batch_size=_BATCH_SIZE.value,
      num_workers=2,
      buffer_size=2,
      feature_specs=feature_specs,
      mesh=mesh,
      global_sharding=global_sharding,
      fdo_client=fdo_client,
  )

  _, dense_features, dense_lookups, embedding_lookups = next(producer)
  params = model.init(
      jax.random.key(42), dense_features, dense_lookups, embedding_lookups
  )
  tx = embed_optimizer.create_optimizer_for_sc_model(
      params,
      optax.adagrad(learning_rate=_LEARNING_RATE.value),
  )
  # Create a single TrainState
  train_state = TrainState(params=params, opt_state=tx.init(params))
  del params  # No longer needed

  # --- Checkpointing Setup Start ---
  chkpt_mgr = None
  if _CHECKPOINT_DIR.value:
    chkpt_mgr = ocp.CheckpointManager(
        directory=os.path.join(_CHECKPOINT_DIR.value, "checkpoints"),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=_CHECKPOINT_MAX_TO_KEEP.value,
            save_interval_steps=_CHECKPOINT_INTERVAL.value,
            enable_background_delete=True,
        ),
    )

  latest_step = -1
  if chkpt_mgr and _CHECKPOINT_RESUME.value:
    latest_step = chkpt_mgr.latest_step()
    if latest_step is not None:
      logging.info("Resuming from checkpoint at step %d", latest_step)
      restore_target = {
          'train_state': ocp.args.PyTreeRestore(train_state),
          'stacking_proto': ocp.args.ProtoRestore(
              embedding_spec_pb2.EmbeddingSpecProto
          ),
      }
      restored = chkpt_mgr.restore(
          latest_step, args=ocp.args.Composite(**restore_target)
      )
      train_state = restored['train_state']
    else:
      logging.info("No checkpoint found, starting from scratch.")
      latest_step = -1
  # --- Checkpointing Setup End ---

  # Define output shardings for JIT compilation
  def get_shape(x):
    return x.shape

  logging.info(
      "[chandra-debug] params: %s",
      jax.tree_util.tree_map(get_shape, train_state.params),
  )

  dense_out_shardings = {
      f"Dense_{i}": {
          "bias": Format(
              DLL(major_to_minor=(0,)),
              NamedSharding(mesh, P()),
          ),
          "kernel": Format(
              DLL(major_to_minor=(0, 1)),
              NamedSharding(mesh, P()),
          ),
      }
      for i in range(8)
  }

  stacked_table_names = set(
      [f.table_spec.setting_in_stack.stack_name for _, f in feature_specs.items()]
  )

  embedding_out_shardings = {
      "SparseCoreEmbed_0": {
          "sc_embedding_variables": embed.WithSparseCoreLayout(
              value={
                  f"{key}": Format(
                      DLL(
                          major_to_minor=(0, 1),
                          tiling=((8,),),
                      ),
                      NamedSharding(mesh, P("x", None)),
                  )
                  for key in stacked_table_names
              },
              names=("x",),
              mesh=mesh,
          )
      }
  }

  dense_embed_out_shardings = {
      f"Embed_{i}": Format(DLL(major_to_minor=(0, 1)), NamedSharding(mesh, P()))
      for i in range(26 - len(feature_specs.items()))
  }

  dcn_out_shardings = {
      f"bias_{i}": Format(DLL(major_to_minor=(0,)), NamedSharding(mesh, P()))
      for i in range(3)
  }
  dcn_out_shardings.update(
      {f"u_kernel_{i}": Format(DLL(major_to_minor=(0, 1)), NamedSharding(mesh, P())) for i in range(3)}
  )
  dcn_out_shardings.update(
      {f"v_kernel_{i}": Format(DLL(major_to_minor=(0, 1)), NamedSharding(mesh, P())) for i in range(3)}
  )

  # Sharding for the entire TrainState object
  # opt_state sharding is inferred by JAX to match params, so we pass None
  train_state_shardings = TrainState(
      params={
          "params": {
              **dense_out_shardings,
              **embedding_out_shardings,
              **dcn_out_shardings,
              **dense_embed_out_shardings,
          }
      },
      opt_state=None,
  )

  @partial(
      jax.jit,
      out_shardings=(train_state_shardings, None),
      donate_argnums=(0,),
  )
  def train_step(
      train_state: TrainState,
      labels: jax.Array,
      dense_features: jax.Array,
      dense_lookups: Any,
      embedding_lookups: embed.EmbeddingLookupInput,
  ):
    def forward_pass(
        params, labels, dense_features, dense_lookups, embedding_lookups
    ):
      logits = model.apply(
          params, dense_features, dense_lookups, embedding_lookups
      )
      xentropy = optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels)
      return jnp.mean(xentropy), logits

    # Run model forward/backward pass.
    train_step_fn = jax.value_and_grad(
        forward_pass, has_aux=True, allow_int=True
    )

    (loss_val, unused_logits), grads = train_step_fn(
        train_state.params,
        labels,
        dense_features,
        dense_lookups,
        embedding_lookups,
    )

    updates, new_opt_state = tx.update(grads, train_state.opt_state)
    new_params = embed_optimizer.apply_updates_for_sc_model(
        train_state.params, updates
    )

    return train_state.replace(
        params=new_params, opt_state=new_opt_state
    ), loss_val

  start_time = time.time()
  start_step = latest_step + 1

  # with XprofHelper(start_step=100 + 100, num_steps=50) as xprof_helper:
  for step in range(start_step, _NUM_STEPS.value):
    # xprof_helper.step()
    if step == 100:
      jax.profiler.start_trace("/tmp/tensorboard")
    with jax.profiler.StepTraceAnnotation("step", step_num=step):
      # --------------------------------------------------------------------------
      # Step 1: SC input processing.
      # --------------------------------------------------------------------------
      labels, dense_features, dense_lookups, embedding_lookups = next(producer)
      # ------------------------------------------------------------------------
      # Step 2: run model.
      # ------------------------------------------------------------------------
      train_state, loss_val = train_step(
          train_state,
          labels,
          dense_features,
          dense_lookups,
          embedding_lookups,
      )

    if step == 200:
      jax.profiler.stop_trace()

    # --- Checkpointing Save Logic Start ---
    if chkpt_mgr and (step + 1) % _CHECKPOINT_INTERVAL.value == 0:
      logging.info("Saving checkpoint at step %d", step)
      stacking_proto = embedding.create_proto_from_feature_specs(
          feature_specs,
          global_device_count=jax.device_count(),
          num_sparsecore_per_device=2,
      )
      chkpt_mgr.save(
          step,
          args=ocp.args.Composite(
              train_state=ocp.args.PyTreeSave(train_state),
              stacking_proto=ocp.args.ProtoSave(stacking_proto),
          ),
      )
    # --- Checkpointing Save Logic End ---

    if step % 1000 == 0:
      fdo_client.publish()
      jax.experimental.multihost_utils.sync_global_devices(
          "FDO CLIENT BARRIER"
      )

    if step == 1500:
      loaded_stats = fdo_client.load()
      embedding.update_preprocessing_parameters(
          feature_specs,
          loaded_stats,
          num_sc_per_device=2,
      )

    if (step % 1500 == 0) and (step > start_step):
      end_time = time.time()
      logging.info(
          "Step %s: %lf, mean_step_time = %lf",
          step,
          loss_val,
          (end_time - start_time) / 1500,
      )
      start_time = time.time()

  # --- Checkpointing Finalize Start ---
  if chkpt_mgr:
    # Save final step
    stacking_proto = embedding.create_proto_from_feature_specs(
        feature_specs,
        global_device_count=jax.device_count(),
        num_sparsecore_per_device=2,
    )
    chkpt_mgr.save(
        _NUM_STEPS.value - 1,
        args=ocp.args.Composite(
            train_state=ocp.args.PyTreeSave(train_state),
            stacking_proto=ocp.args.ProtoSave(stacking_proto),
        ),
        force=True, # force save even if not on interval
    )
    logging.info("Waiting for final checkpoint to save...")
    chkpt_mgr.wait_until_finished()
    chkpt_mgr.close()
    logging.info("Checkpoint manager closed.")
  # --- Checkpointing Finalize End ---

  producer.stop()


def main(argv):
  del argv
  # webserver_start.StartXprofServerBasedOnFlags()
  
  test_dlrm_dcnv2_model()


if __name__ == "__main__":
  app.run(main)

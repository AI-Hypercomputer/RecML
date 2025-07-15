import dataclasses
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf


# Define dataclass and constants at the top level
dataclass = dataclasses.dataclass
PARALELLISM = 32


VOCAB_SIZES = [
    40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000,
    3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14, 40000000, 40000000,
    40000000, 590152, 12973, 108, 36,
]
MULTI_HOT_SIZES = [
    3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10,
    3, 1, 1,
]


@dataclass
class DataConfig:
  """Configuration for data loading parameters."""
  global_batch_size: int  
  pre_batch_size: int     
  is_training: bool
  use_cached_data: bool = False


def get_dummy_batch(batch_size, multi_hot_sizes, vocab_sizes):
  """Returns a dummy batch of data in the final desired structure."""
  data = {
      'clicked': np.random.randint(0, 2, size=(batch_size,), dtype=np.int64)
  }
  dense_features_list = [
      np.random.uniform(0.0, 0.9, size=(batch_size, 1)).astype(np.float32)
      for _ in range(13)
  ]
  data['dense_features'] = np.concatenate(dense_features_list, axis=-1)
  sparse_features = {}
  for i in range(len(multi_hot_sizes)):
    vocab_size = vocab_sizes[i] if i < len(vocab_sizes) else 1000
    multi_hot_size = multi_hot_sizes[i]
    output_key = str(i)
    sparse_features[output_key] = np.random.randint(
        low=0, high=vocab_size, size=(batch_size, multi_hot_size), dtype=np.int64)
  data['sparse_features'] = sparse_features
  return data


class CriteoDataLoader:
  """
  Data loader for Criteo dataset optimized for JAX training.

  This loader is designed for a distributed environment and handles sharding
  of input files across multiple JAX processes. It efficiently reads
  pre-batched TFRecords, unbatches them into individual examples, and then
  re-batches them into a single, large global batch of a precise target size.
  """

  def __init__(
      self,
      file_pattern: str,
      params: DataConfig,
      num_dense_features: int,
      vocab_sizes: List[int],
      multi_hot_sizes: List[int],
      shuffle_buffer: int = 256,
  ):
    if params.global_batch_size % jax.process_count() != 0:
        raise ValueError(
            f"global_batch_size ({params.global_batch_size}) must be divisible "
            f"by the number of JAX processes ({jax.process_count()}).")

    self._file_pattern = file_pattern
    print(f'file_pattern: {file_pattern}')
    self._params = params
    self._num_dense_features = num_dense_features
    self._vocab_sizes = vocab_sizes
    self._multi_hot_sizes = multi_hot_sizes
    self._shuffle_buffer = shuffle_buffer

    self.label_features = 'clicked'
    self.dense_features = [f'int-feature-{i}' for i in range(1, 14)]
    self.sparse_features = [f'categorical-feature-{i}' for i in range(14, 14 + len(vocab_sizes))]

  def _get_feature_spec(self, batch_size: int) -> Dict[str, tf.io.FixedLenFeature]:
    """Creates the feature specification for parsing TFRecords."""
    feature_spec = {
        self.label_features: tf.io.FixedLenFeature([batch_size], dtype=tf.int64)
    }
    for dense_feat in self.dense_features:
      feature_spec[dense_feat] = tf.io.FixedLenFeature([batch_size], dtype=tf.float32)
    for sparse_feat in self.sparse_features:
      feature_spec[sparse_feat] = tf.io.FixedLenFeature([batch_size], dtype=tf.string)
    return feature_spec

  def _parse_example(self, serialized_example: tf.Tensor) -> Dict[str, tf.Tensor]:
    """Parses a serialized TFRecord example into features."""
    feature_spec = self._get_feature_spec(self._params.pre_batch_size)
    parsed_features = tf.io.parse_single_example(serialized_example, feature_spec)
    labels = parsed_features[self.label_features]
    dense_features = tf.stack(
        [parsed_features[feat] for feat in self.dense_features], axis=-1)
    sparse_features = {}
    for i, sparse_ft in enumerate(self.sparse_features):
      output_key = str(i)
      cat_ft_int64 = tf.io.decode_raw(parsed_features[sparse_ft], tf.int64)
      cat_ft_int64 = tf.reshape(cat_ft_int64, [self._params.pre_batch_size, self._multi_hot_sizes[i]])
      sparse_features[output_key] = cat_ft_int64
    return {'clicked': labels, 'dense_features': dense_features, 'sparse_features': sparse_features}

  def _get_direct_dummy_dataset(self, batch_size: int) -> tf.data.Dataset:
    """Creates a TF dataset from cached dummy data of the final batch size."""
    print(f'Returning a dummy dataset of final batch size {batch_size}')
    dummy_data = get_dummy_batch(batch_size, self._multi_hot_sizes, self._vocab_sizes)
    dataset = tf.data.Dataset.from_tensors(dummy_data)
    return dataset.repeat()

  def _create_dataset(self) -> tf.data.Dataset:
    """Creates and configures the TensorFlow dataset."""
    final_per_host_batch_size = self._params.global_batch_size // jax.process_count()

    if self._params.use_cached_data:
      return self._get_direct_dummy_dataset(final_per_host_batch_size)

    print(
        f"[Process {jax.process_index()}] This host will produce batches of size {final_per_host_batch_size}. "
        f"It will read TFRecords with pre_batch_size={self._params.pre_batch_size}, "
        f"unbatch them, and then re-batch to the target size."
    )

    dataset = tf.data.Dataset.list_files(self._file_pattern, shuffle=self._params.is_training)
    dataset = dataset.shard(jax.process_count(), jax.process_index())
    if self._params.is_training:
      dataset = dataset.repeat()

    dataset = tf.data.TFRecordDataset(
        dataset, buffer_size=32 * 1024 * 1024, num_parallel_reads=PARALELLISM)

    dataset = dataset.map(
        self._parse_example, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.unbatch()

    if self._params.is_training and self._shuffle_buffer > 0:
      dataset = dataset.shuffle(self._shuffle_buffer)

    dataset = dataset.batch(
        final_per_host_batch_size,
        drop_remainder=True,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.take(1).cache().repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_deterministic = False
    options.threading.private_threadpool_size = 96
    dataset = dataset.with_options(options)
    return dataset

  def get_iterator(self):
    """Returns an iterator over the dataset that provides NumPy arrays."""
    dataset = self._create_dataset()
    return dataset.as_numpy_iterator()

  def get_jax_arrays(self, batch: Dict[str, np.ndarray]) -> Dict[str, jnp.ndarray]:
    """Converts a batch of NumPy arrays to JAX arrays."""
    return jax.tree_util.tree_map(jnp.array, batch)

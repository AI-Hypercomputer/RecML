"""Data loader for Criteo dataset optimized for JAX training."""

import dataclasses
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf


dataclass = dataclasses.dataclass
PARALELLISM = 32


@dataclass
class DataConfig:
  """Configuration for data loading parameters."""

  global_batch_size: int
  is_training: bool
  use_cached_data: bool = False


def get_dummy_batch(batch_size, multi_hot_sizes=None, vocab_sizes=None):
  """Returns a dummy batch of data.

  Args:
    batch_size: The size of the batch to generate.
    multi_hot_sizes: A list of sizes for the multi-hot features.
    vocab_sizes: A list of sizes for the vocabularies of the sparse features.

  Our features dict is an inputs of dictionary - {
      'label': (0,1)
      'dense_features': {
          '1': 0.123,
          '2': 0.456,
          ...
          '13': 0.987,
      },
      'sparse_features': {
          '1': sparse_tensor,
          '2': sparse_tensor,
          ...
          '26': sparse_tensor,
      },
  }
  """
  data = {}

  data['clicked'] = np.random.randint(0, 2, size=(batch_size,), dtype=np.int64)
  data['dense_features'] = np.random.uniform(
      0.0, 0.9, size=(batch_size, 13)
  ).astype(np.float32)

  sparse_features = {}

  for i in range(len(multi_hot_sizes)):
    sparse_features[str(i)] = np.random.randint(
        low=0,
        high=vocab_sizes[i],
        size=(batch_size, multi_hot_sizes[i]),
    )

  data['sparse_features'] = sparse_features
  return data


class CriteoDataLoader:
  """Data loader for Criteo dataset optimized for JAX training."""

  def __init__(
      self,
      file_pattern: str,
      params: DataConfig,
      num_dense_features: int,
      vocab_sizes: List[int],
      multi_hot_sizes: List[int],
      embedding_threshold: int = 21000,
      shuffle_buffer: int = 256,
      prefetch_size: int = 256,
  ):
    self._file_pattern = file_pattern
    print(f' file_pattern: {file_pattern}')
    self._params = params
    self._num_dense_features = num_dense_features
    self._vocab_sizes = vocab_sizes
    self._multi_hot_sizes = multi_hot_sizes
    # Embedding threshold is used to determine whether a feature should be
    # placed on TensorCore or SparseCore.
    self._embedding_threshold = embedding_threshold
    self._shuffle_buffer = shuffle_buffer
    self._prefetch_size = prefetch_size
    self._cached_dummy_data = False

    self.label_features = 'clicked'
    self.dense_features = [f'int-feature-{x}' for x in range(1, 14)]
    self.sparse_features = [f'categorical-feature-{x}' for x in range(14, 40)]

  def _get_cached_dummy_dataset(
      self, batch_size: int, vocab_sizes: List[int]
  ) -> tf.data.Dataset:
    """Creates a TensorFlow dataset from cached dummy data."""
    print(f'returning dummy dataset')
    if self._cached_dummy_data is None:
      # Generate dummy data once
      self._cached_dummy_data = get_dummy_batch(
          batch_size,
          self._multi_hot_sizes,
          vocab_sizes=vocab_sizes,
      )

    # Convert numpy arrays to tf.data.Dataset
    dataset = tf.data.Dataset.from_tensors({
        'clicked': tf.convert_to_tensor(self._cached_dummy_data['clicked']),
        'dense_features': tf.convert_to_tensor(
            self._cached_dummy_data['dense_features']
        ),
        'sparse_features': {
            k: tf.convert_to_tensor(v)
            for k, v in self._cached_dummy_data['sparse_features'].items()
        },
    })
    dataset = dataset.take(1).repeat()
    dataset = dataset.prefetch(buffer_size=2048)
    options = tf.data.Options()
    options.deterministic = False
    options.threading.private_threadpool_size = 96
    dataset = dataset.with_options(options)
    return dataset

  def _get_feature_spec(
      self, batch_size: int
  ) -> Dict[str, tf.io.FixedLenFeature]:
    """Creates the feature specification for parsing TFRecords."""
    feature_spec = {
        self.label_features: tf.io.FixedLenFeature(
            [
                batch_size,
            ],
            dtype=tf.int64,
        )
    }

    for dense_feat in self.dense_features:
      feature_spec[dense_feat] = tf.io.FixedLenFeature(
          [
              batch_size,
          ],
          dtype=tf.float32,
      )

    for sparse_feat in self.sparse_features:
      feature_spec[sparse_feat] = tf.io.FixedLenFeature(
          [
              batch_size,
          ],
          dtype=tf.string,
      )

    return feature_spec

  def _parse_example(
      self, serialized_example: tf.Tensor, batch_size: int
  ) -> Dict[str, tf.Tensor]:
    """Parses a serialized TFRecord example into features."""
    feature_spec = self._get_feature_spec(batch_size)
    parsed_features = tf.io.parse_single_example(
        serialized_example, feature_spec
    )

    # Process labels
    labels = tf.reshape(
        parsed_features[self.label_features],
        [
            batch_size,
        ],
    )

    # Process dense features
    dense_features = []
    for dense_ft in self.dense_features:
      cur_feature = tf.reshape(
          parsed_features[dense_ft],
          [
              batch_size,
              1,
          ],
      )
      dense_features.append(cur_feature)
    dense_features = tf.concat(dense_features, axis=-1)

    # Process sparse features
    sparse_features = {}
    for i, sparse_ft in enumerate(self.sparse_features):
      cat_ft_int64 = tf.io.decode_raw(parsed_features[sparse_ft], tf.int64)
      cat_ft_int64 = tf.reshape(
          cat_ft_int64,
          [
              batch_size,
              self._multi_hot_sizes[i],
          ],
      )

      # TODO(b/396189671): Logic needed for PartialTPUEmbedding.
      # if self._vocab_sizes[i] > self._embedding_threshold:
      #   sparse_features[str(i)] = tf.sparse.from_dense(cat_ft_int64)
      # else:
      #   sparse_features[str(i)] = cat_ft_int64

      sparse_features[str(i)] = cat_ft_int64

    return {
        'clicked': labels,
        'dense_features': dense_features,
        'sparse_features': sparse_features,
    }

  def _create_dataset(self) -> tf.data.Dataset:
    """Creates and configures the TensorFlow dataset."""
    batch_size = self._params.global_batch_size // jax.process_count()
    if self._params.use_cached_data:
      return self._get_cached_dummy_dataset(
          batch_size, vocab_sizes=self._vocab_sizes
      )

    dataset = tf.data.Dataset.list_files(self._file_pattern, shuffle=False)

    dataset = dataset.shard(jax.process_count(), jax.process_index())

    if self._params.is_training:
      # dataset = dataset.shuffle(4)
      dataset = dataset.repeat()

    # We make 4 * num_processes TFRecord shards.
    parallelism = PARALELLISM

    dataset = tf.data.TFRecordDataset(
        dataset, buffer_size=32 * 1024 * 1024, num_parallel_reads=parallelism
    )

    # Parse examples
    dataset = dataset.map(
        lambda x: self._parse_example(x, batch_size),
        num_parallel_calls=parallelism,
    )

    if self._params.is_training and self._shuffle_buffer > 0:
      dataset = dataset.shuffle(self._shuffle_buffer)

    if not self._params.is_training:
      def _mark_as_padding(features):
        return {
            'clicked': -1 * tf.ones(
                [
                    batch_size,
                ],
                dtype=tf.int64,
            ),
            'dense_features': features['dense_features'],
            'sparse_features': features['sparse_features'],
        }

      padding_ds = dataset.take(1)
      padding_ds = padding_ds.map(_mark_as_padding).repeat(200)
      dataset = dataset.concatenate(padding_ds).take(660).cache().repeat()

    # dataset = dataset.prefetch(self._prefetch_size)

    dataset = dataset.prefetch(buffer_size=2048)
    options = tf.data.Options()
    options.deterministic = False
    options.threading.private_threadpool_size = 96
    dataset = dataset.with_options(options)
    return dataset

  def get_iterator(self):
    """Returns an iterator over the dataset that provides NumPy arrays."""
    dataset = self._create_dataset()

    def _convert_to_numpy(batch):
      return {
          'clicked': batch['clicked'].numpy(),
          'dense_features': batch['dense_features'].numpy(),
          'sparse_features': {
              k: v.numpy() for k, v in batch['sparse_features'].items()
          },
      }

    return map(_convert_to_numpy, iter(dataset))

  def get_jax_arrays(
      self, batch: Dict[str, np.ndarray]
  ) -> Dict[str, jnp.ndarray]:
    """Converts a batch of NumPy arrays to JAX arrays."""
    features = {
        'clicked': jnp.array(batch['clicked']),
        'dense_features': jnp.array(batch['dense_features']),
        'sparse_features': {
            k: jnp.array(v) for k, v in batch['sparse_features'].items()
        },
    }
    return features


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
"""TF dataset factory."""

from __future__ import annotations

import collections
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import enum
import functools
import os
import re
from typing import Any, Protocol

from absl import flags
from absl import logging
import jax
from recml.core.utils import types
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

TensorType = tf.Tensor | tf.SparseTensor | tf.RaggedTensor
FeaturesDictType = dict[str, TensorType]
ParserFn = Callable[[Sequence[bytes]], TensorType]
FeatureTransformationFn = Callable[[FeaturesDictType], FeaturesDictType]
TransformFn = Callable[
    [FeaturesDictType],
    tuple[FeaturesDictType, tf.Tensor]
    | tuple[FeaturesDictType, FeaturesDictType]
    | FeaturesDictType,
]
FilterFn = Callable[[FeaturesDictType], tf.Tensor]
IO_Feature = (
    tf.io.FixedLenFeature
    | tf.io.VarLenFeature
    | tf.io.RaggedFeature
    | tf.io.SparseFeature
)
TFDS_REGEX = r"(.*):(.*)"

_DEFAULT_FILE_SHUFFLE_SEED = 42


class TFTransformOutput(Protocol):
  """Interface for `tft.TFTransformOutput` for typing."""

  def transform_features_layer(self) -> tf.keras.layers.Layer:
    """Returns a layer that applies the transform function."""

  def raw_feature_spec(self) -> Mapping[str, IO_Feature]:
    """Returns the raw feature spec."""


class FileFormat(enum.StrEnum):
  """Supported file formats for the dataset creator."""

  TFRECORD = "tfrecord"
  RECORDIO = "recordio"
  SSTABLE = "sstable"
  ARRAY_RECORD = "array_record"


READER_MAP = {
    FileFormat.TFRECORD: tf.data.TFRecordDataset,
}


class DatasetShardingInfo(types.Dataclass):
  """Sharding information for the dataset."""

  num_processes: int = dataclasses.field(default_factory=jax.process_count)
  process_index: int = dataclasses.field(default_factory=jax.process_index)

  def per_process_batch_size(self, global_batch_size: int) -> int:
    """Returns the per-process batch size."""
    if global_batch_size % self.num_processes != 0:
      raise ValueError(
          f"The global batch size: {global_batch_size} must be divisible by the"
          f" number of processes: {self.num_processes}."
      )
    return global_batch_size // self.num_processes


class TFDSMetadata(types.Dataclass):
  """TFDS metadata for the dataset."""

  source: Sequence[str]
  input_paths: Sequence[str]
  file_format: FileFormat
  feature_spec: Mapping[str, IO_Feature] | None = None


class TFDatasetFactory(types.Factory[tf.data.Dataset]):
  """A class used to create a TF data dataset for training.

  Note that the returned dataset is prefetched and does not require the
  application of additional dataset ops.

  Attributes:
    input_path: A string or sequence of string paths / patterns pointing to the
      training or validation data. This or `tfds_source` must be set.
    tfds_source: A colon separated string of form "dataset_name:split_name",
      which will be used to get the input paths for the dataset from TFDS.
      Optionally, a sequence of such strings can be provided to create an evenly
      distributed mixture of datasets. This or `input_path` must be set.
    file_format: The file format of the input files. Must be one of 'tfrecord',
      'recordio', 'sstable', 'array_record'. Defaults to recordio.
    global_batch_size: The global batch size across all replicas.
    drop_remainder: Whether the last batch should be dropped in the case it has
      fewer than `global_batch_size` elements.
    shuffle: Whether to shuffle the dataset. Note that shuffling happens before
      and after `interleave`, i.e. on a shard / file group level and on an
      example level. Defaults to False.
    shuffle_buffer_size: The shuffle buffer size when shuffling batches of
      examples. Defaults to 1000.
    repeat: Whether to repeat the dataset infinitely. This happens on the file
      dataset level.
    repeat_files: Whether to repeat the files infinitely before sharding. This
      is valid on when `repeat` is True.
    sharding: Whether to enable sharding in the input pipeline. This will shard
      files across different workers when an input context is present.
    cycle_length: The number of files that will be processed concurrently when
      interleaving files. If None, the tf.data runtime decides what it should be
      based on the available CPU.
    block_length: The number of consecutive elements to produce from each input
      element before cycling to another input element when interleaving files.
      Defaults to 1.
    deterministic: An optional boolean controlling whether determinism should be
      enforced during file interleaving. If None, the
      `tf.data.Options.deterministic` option, True by default, determines the
      behaviour.
    num_parser_threads: The number of parallel threads to use when mapping
      `tf.io.parse_example` over the batched dataset.
    num_parallel_threads: The number of parallel threads to use in generic map
      operations over the dataset. Defaults to `tf.data.AUTOTUNE`.
    prefetch_buffer_size: The maximum number of batches to buffer while
      prefetching. Defaults to `tf.data.AUTOTUNE`.
    readahead: An optional readahead to add to input paths. If passed, prefixes
      all input paths with a readahead with the passed prefix. i.e. '64M'.
    group_uris_by_dir: A boolean indicating whether to group the file uris by
      their directory and sort the groups in descending order. If True,
      `interleave` will cycle through file groups instead of individual shards,
      with the shards for the file groups with the lexically largest folder name
      being read first. Any operations on the dataset before `interleave`, such
      as shuffling or sharding, will happen on a file group level. This
      behaviour is useful when the files are stored in the form
      'Span-0/Version-0/Split-Training/Shard-0-of-2', where the lexically
      largest folder for the training data will consist of the most recent data,
      resulting in the input function cycling through the most recent files
      first. See the docstring of `get_file_groups` for more information.
      Defaults to False.
    seed: An optional seed to use for deterministic shuffling / preprocessing.
      Defaults to None.
    enable_tf_data_service: Whether to apply tf.data service for this dataset.
      If True, flag `tf_data_service_address` must be set.
    tf_data_service_policy: Sharding policy to use for tf.data service when it
      is enabled.
    tf_data_service_job_name: Job name to use for tf.data service. If None, the
      default job name will be used.
    feature_spec: A mapping of feature keys to `FixedLenFeature`,
      `VarLenFeature`, `SparseFeature`, or `RaggedFeature` values. This will be
      used to parse the TF examples, or as context_features spec to parse TF
      sequence examples if sequence_feature_spec is not None.
    sequence_feature_spec: sequence feature spec for parsing TF sequence
      examples. Leaving as None and the raw data will be considered as regular
      TF examples and parsed with feature_spec; iff not None, the data will be
      treated as TF sequence examples and parsed with
      context_features=feature_spec and sequence_features=sequence_feature_spec.
      The parsed context and sequence features will be merged into a single
      feature dictionary.
    tf_transform_output: An optional `tft.TFTransformOutput` instance to parse
      the features and transform them. This supersedes `feature_spec` and any
      feature spec inferred from the `tfds_source`. tf_transform_output is not
      supported for TF sequence examples.
    filter_fn: An optional vectorized filter function to apply to the dataset.
      This will be applied after batching and the dataset will be re-batched
      after it is applied to discard filtered out examples. This must be a
      callable that accepts a dictionary of features, where each value can be a
      dense, sparse, or ragged tensor of shape [B, ...], and returns a dense
      bool tensor of shape [B], which should be a mask indicating whether or not
      to keep the feature.
    preprocessor: An optional preprocessing function to map over the dataset. If
      `tf_transform_output` is also supplied this will be composed with
      `tf_transform_output.transform_features_layer()` before being mapped over
      the dataset.
    postprocessors: A sequence of postprocessing functions to apply to the
      dataset. These will be applied in the order they are provided after the
      preprocessor if specified.
    label_name: The name of the label feature. If passed, this will be popped
      from the features dictionary after performing any transformations and the
      dataset returned will consist of tuples of the features dictionary and the
      corresponding label.
    data_options: Optional data options to apply to the dataset.
    sharding_info: A `ShardingInfo` instance that specifies how to shard the
      dataset. Defaults to `ShardingInfo(num_processes=jax.process_count(),
      process_index=jax.process_index())`. This is similar to `InputContext` in
      tensorflow.
    debug: An optional boolean indicating whether to debug input boundedness. If
      `True`, the dataset will consist of a single batch that's cached and
      infinitely repeated.
  """

  cache_reading: bool = False
  input_path: str | Sequence[str] = ""
  tfds_source: str | Sequence[str] = ""
  file_format: FileFormat = FileFormat.RECORDIO
  global_batch_size: int = 1
  drop_remainder: bool = True
  shuffle: bool = False
  shuffle_buffer_size: int = 1000
  repeat: bool = False
  repeat_files: bool = False
  sharding: bool = True
  cycle_length: int | None = None
  block_length: int = 1
  deterministic: bool | None = None
  num_parser_threads: int = tf.data.AUTOTUNE
  num_parallel_threads: int = tf.data.AUTOTUNE
  prefetch_buffer_size: int = tf.data.AUTOTUNE
  readahead: str | None = None
  group_uris_by_dir: bool = False
  seed: int | None = None
  enable_tf_data_service: bool = False
  tf_data_service_job_name: str | None = None
  tf_data_service_policy: tf.data.experimental.service.ShardingPolicy = (
      tf.data.experimental.service.ShardingPolicy.OFF
  )
  feature_spec: Mapping[str, IO_Feature] | None = None
  sequence_feature_spec: Mapping[str, IO_Feature] | None = None
  tf_transform_output: TFTransformOutput | None = None
  filter_fn: FilterFn | None = None
  preprocessor: FeatureTransformationFn | None = None
  postprocessors: Sequence[FeatureTransformationFn] = ()
  label_name: str | None = None
  data_options: tf.data.Options | None = None
  sharding_info: DatasetShardingInfo = dataclasses.field(
      default_factory=DatasetShardingInfo
  )
  debug: bool = False

  def __post_init__(self):
    if self.enable_tf_data_service:
      if flags.FLAGS.tf_data_service_address is None:
        raise ValueError(
            "Flag `tf_data_service_address` must be set when"
            " `enable_tf_data_service` is True."
        )
      if self.seed is not None:
        raise ValueError("`seed` must be None for data service.")
      if self.sharding:
        raise ValueError("`sharding` must be set to False for data service.")

  @functools.cached_property
  def tfds_metadata(self) -> TFDSMetadata | None:
    """Returns the TFDS metadata for the dataset."""
    if not self.tfds_source:
      return None
    if isinstance(self.tfds_source, str):
      tfds_sources = [self.tfds_source]
    else:
      tfds_sources = self.tfds_source

    uris = []
    feature_specs = []
    file_formats = []
    for source in tfds_sources:
      match = re.fullmatch(TFDS_REGEX, source)
      if not match:
        raise ValueError(
            f"Invalid `tfds_source`: {self.tfds_source}. Expected format:"
            " 'dataset_name:split_name'."
        )
      name, split = match.groups()
      info = tfds.builder(name).info

      input_paths = list(map(str, info.splits[split].filepaths))
      uris.extend(input_paths)

      if info.file_format == tfds.core.FileFormat.TFRECORD:
        file_format = FileFormat.TFRECORD
      elif info.file_format == tfds.core.FileFormat.SSTABLE:
        file_format = FileFormat.SSTABLE
      elif info.file_format == tfds.core.FileFormat.ARRAY_RECORD:
        file_format = FileFormat.ARRAY_RECORD
      else:
        raise ValueError(f"Unsupported file format: {info.file_format}.")
      file_formats.append(file_format)

      if info.features is not None and hasattr(
          info.features, "tf_example_spec"
      ):
        feature_spec = info.features.tf_example_spec
      else:
        feature_spec = None
      feature_specs.append(feature_spec)

      logging.info("Using TFDS dataset: '%s' split: '%s'", name, split)
      logging.info("Found %d uris for TFDS dataset: %s", len(uris), source)

    if not all(file_format == file_formats[0] for file_format in file_formats):
      raise ValueError(
          "All TFDS sources must have the same file format. Got file formats:"
          f" {list(zip(tfds_sources, file_formats))}."
      )

    if not all(
        feature_spec == feature_specs[0] for feature_spec in feature_specs
    ):
      raise ValueError(
          "All TFDS sources must have the same feature spec. Got feature specs:"
          f" {list(zip(tfds_sources, feature_specs))}."
      )

    return TFDSMetadata(
        source=self.tfds_source,
        input_paths=uris,
        feature_spec=feature_specs[0],
        file_format=file_formats[0],
    )

  @functools.cached_property
  def input_filepaths(self) -> Sequence[str]:
    """Returns the input file paths for the dataset."""
    tfds_metadata = self.tfds_metadata
    if self.input_path and tfds_metadata is not None:
      raise ValueError("`input_path` and `tfds_source` cannot both be set.")
    elif self.input_path:
      input_patterns = self.input_path
      if isinstance(input_patterns, str):
        input_patterns = [input_patterns]

      uris = []
      for input_pattern in input_patterns:
        uris.extend(tf.io.gfile.glob(input_pattern))

      if not uris:
        raise ValueError(
            f"No input files found for patterns: {input_patterns}."
        )

    elif tfds_metadata:
      uris = tfds_metadata.input_paths
      if not uris:
        raise ValueError(
            f"No input files found for TFDS sources: {tfds_metadata.source}."
        )

    else:
      raise ValueError("One of `input_path` or `tfds_source` must be set.")

    return uris

  @functools.cached_property
  def reader(self) -> type[tf.data.Dataset]:
    """Gets the file format for the dataset."""
    tfds_metadata = self.tfds_metadata
    if tfds_metadata is not None:
      file_format = tfds_metadata.file_format
    else:
      file_format = self.file_format

    if file_format not in READER_MAP:
      raise ValueError(
          f"File format: {file_format} is not supported."
          f" Expected one of: {list(READER_MAP)}."
      )

    return READER_MAP[file_format]

  @functools.cached_property
  def parsing_fn(self) -> Callable[..., TensorType]:
    """Returns a function that parses the dataset."""
    if self.tf_transform_output is not None:
      feature_spec = self.tf_transform_output.raw_feature_spec()
    else:
      feature_spec = self.feature_spec

    tfds_metadata = self.tfds_metadata
    if not feature_spec and tfds_metadata is not None:
      if tfds_metadata.feature_spec is None:
        raise ValueError(
            "TFDS dataset must have a `FeaturesDict` for parsing to work."
        )
      feature_spec = tfds_metadata.feature_spec

    parser = build_parser_fn(
        feature_spec=feature_spec,
        sequence_feature_spec=self.sequence_feature_spec,
    )
    tfds_metadata = self.tfds_metadata
    if (
        tfds_metadata is not None
        and tfds_metadata.file_format == FileFormat.SSTABLE
    ) or self.file_format == FileFormat.SSTABLE:
      return lambda _, v: parser(v)
    return parser

  @functools.cached_property
  def file_shuffle_seed(self) -> int | None:
    """Returns the file shuffle seed."""
    if self.seed is not None:
      return self.seed
    if self.sharding:
      return _DEFAULT_FILE_SHUFFLE_SEED
    return None

  @functools.cached_property
  def map_fns(self) -> Sequence[TransformFn]:
    """Returns the map functions for the dataset."""
    if (
        self.tf_transform_output is None
        and self.preprocessor is None
        and not self.postprocessors
        and self.label_name is None
    ):
      return []
    return [
        build_transform_fn(
            tf_transform_output=self.tf_transform_output,
            feature_transformations=(
                [self.preprocessor] if self.preprocessor is not None else []
            )
            + list(self.postprocessors),
            label_name=self.label_name,
            batch_size=self.sharding_info.per_process_batch_size(
                self.global_batch_size
            ),
        )
    ]

  def _create_dataset(self) -> tf.data.Dataset:
    """Creates an examples dataset from the input files."""

    uris = self.input_filepaths
    reader = self.reader

    # Prefix all input paths with a readahead.
    if self.readahead:
      uris = [
          os.path.join(f"/readahead/{self.readahead}", filename)
          for filename in uris
      ]

    # Group the uris by directory.
    if self.group_uris_by_dir:

      def _file_group_reader(file_group: str) -> tf.data.Dataset:
        return self.reader(tf.strings.split(file_group, sep=","))

      uris = get_file_groups(uris)
      reader = _file_group_reader

    # Shuffle the uris before creating the dataset. This ensures that all uris
    # aren't prefetched to one worker during a shuffle when using tf.data
    # service with dynamic sharding is enabled.
    if self.shuffle:
      uris = tf.random.shuffle(uris, seed=self.file_shuffle_seed)

    # Create a dataset of file / file group uris.
    dataset = tf.data.Dataset.from_tensor_slices(uris)

    # Repeat the dataset. We might need to repeat the dataset here in case the
    # issue is encountered: internal screenshot link:6jAKKoEMT3afXRe
    # even we do have enough shards for the input data.
    if self.repeat and self.repeat_files:
      dataset = dataset.repeat()

    # Shard the uri dataset into a separate dataset for each worker during
    # distributed training.
    if self.sharding and self.sharding_info.num_processes > 1:
      dataset = dataset.shard(
          self.sharding_info.num_processes, self.sharding_info.process_index
      )

    # Generate a tf.Example dataset by cycling through all uris in parallel.
    return dataset.interleave(
        map_func=reader,
        cycle_length=self.cycle_length,
        block_length=self.block_length,
        num_parallel_calls=(
            self.cycle_length
            if self.cycle_length is not None
            else tf.data.experimental.AUTOTUNE
        ),
        deterministic=self.deterministic,
    )

  def _parse_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Batches and parses an examples dataset."""
    # Batch the dataset to the global or per replica batch size.
    per_process_batch_size = self.sharding_info.per_process_batch_size(
        self.global_batch_size
    )
    dataset = dataset.batch(
        per_process_batch_size,
        drop_remainder=self.drop_remainder,
    )
    logging.info("Per process batch size: %s", per_process_batch_size)
    logging.info("Number of processes: %s", self.sharding_info.num_processes)

    # Parse the batches of serialized examples using the feature spec.
    return dataset.map(
        self.parsing_fn, num_parallel_calls=self.num_parser_threads
    )

  def _maybe_filter_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Filters a batched examples dataset."""
    if self.filter_fn is not None:
      dataset = vectorized_filter(
          dataset,
          filter_fn=self.filter_fn,
          batch_size=self.sharding_info.per_process_batch_size(
              self.global_batch_size
          ),
          drop_remainder=self.drop_remainder,
      )
    return dataset

  def _maybe_shuffle_and_repeat(self, dataset: tf.data.Dataset):
    """Shuffles and / or repeats an examples dataset."""
    if self.shuffle:
      dataset = dataset.shuffle(self.shuffle_buffer_size, seed=self.seed)
    if self.repeat and not self.repeat_files:
      dataset = dataset.repeat()
    return dataset

  def _maybe_apply_tf_data_service(
      self, dataset: tf.data.Dataset
  ) -> tf.data.Dataset:
    """Applies the tf.data service to the dataset."""
    if not self.enable_tf_data_service:
      return dataset

    tf_data_service_address = flags.FLAGS.tf_data_service_address

    per_proc_batch_size = self.sharding_info.per_process_batch_size(
        self.global_batch_size
    )
    logging.info(
        "Applying tf.data service with address %s and per replica batch"
        " size %s",
        tf_data_service_address,
        per_proc_batch_size,
    )
    return dataset.apply(
        tf.data.experimental.service.distribute(
            processing_mode=self.tf_data_service_policy,
            service=tf_data_service_address,
            job_name=self.tf_data_service_job_name
            or "tf_data_service_shared_job_name",
        )
    )

  def make(self) -> tf.data.Dataset:
    """Creates a `tf.data.Dataset` instance with all dataset ops applied."""
    # Create an examples dataset.
    if self.cache_reading:
      dataset = self._create_dataset().cache()
    else:
      dataset = self._create_dataset()
    # Shuffle and repeat the dataset.
    dataset = self._maybe_shuffle_and_repeat(dataset)
    # Batch and parse the examples dataset.
    dataset = self._parse_dataset(dataset)
    # Apply filters to the batched dataset.
    dataset = self._maybe_filter_dataset(dataset)
    # Apply data service.
    dataset = self._maybe_apply_tf_data_service(dataset)
    # Apply transformations on the dataset.
    for fn in self.map_fns:
      dataset = dataset.map(fn, num_parallel_calls=self.num_parallel_threads)

    if self.debug:
      dataset = dataset.take(1).cache().repeat()

    dataset = dataset.prefetch(buffer_size=self.prefetch_buffer_size)

    if self.data_options is not None:
      dataset = dataset.with_options(self.data_options)

    return dataset


def build_parser_fn(
    feature_spec: Mapping[str, IO_Feature] | None = None,
    sequence_feature_spec: Mapping[str, IO_Feature] | None = None,
    label_name_to_pop_for_serving: str | None = None,
) -> ParserFn:
  """Build a function to parse the inputs."""
  feature_spec = {**feature_spec} if feature_spec else {}
  sequence_feature_spec = (
      {**sequence_feature_spec} if sequence_feature_spec else {}
  )

  if label_name_to_pop_for_serving:
    feature_spec.pop(label_name_to_pop_for_serving, None)
    sequence_feature_spec.pop(label_name_to_pop_for_serving, None)

  if sequence_feature_spec:
    logging.info(
        "Data will be parsed as sequence examples using "
        "`tf.io.parse_sequence_example` with `context_features=%s` and "
        "`sequence_features=%s",
        feature_spec,
        sequence_feature_spec,
    )
  else:
    logging.info(
        "Data will be parsed as regular tf examples using "
        "`tf.io.parse_example` with `features=%s`",
        feature_spec,
    )

  def _parse_sequence_features(e, context_features, sequence_features):
    c, f, _ = tf.io.parse_sequence_example(
        e,
        context_features=context_features,
        sequence_features=sequence_features,
    )
    return {**c, **f}

  if sequence_feature_spec:  # replace to sequence example parser
    return functools.partial(
        _parse_sequence_features,
        context_features=feature_spec,
        sequence_features=sequence_feature_spec,
    )
  else:
    return functools.partial(tf.io.parse_example, features=feature_spec)


def build_transform_fn(
    tf_transform_output: TFTransformOutput | None = None,
    tft_layer: tf.keras.Model | None = None,
    feature_transformations: Sequence[FeatureTransformationFn] | None = None,
    label_name: str | Mapping[str, str] | None = None,
    batch_size: int | None = None,
) -> TransformFn:
  """Build a function to transform the inputs.

  This function will be used during training, evaluation, and serving, to avoid
  training / serving skew.

  Args:
    tf_transform_output: An optional `tft.TFTransformOutput` instance that will
      will be used to transform the features dictionary. Use this for training
      and evaluation.
    tft_layer: An optional `tf.keras.Model` instance that will be used to
      transform the features dictionary. Use this for serving. This is expected
      to be the output of `tft.TFTransformOutput.transform_features_layer()` and
      must be attached to the Keras model that is exported.
    feature_transformations: An optional list of functions to apply to the
      features dictionary.
    label_name: The name of the label feature. A list of label names are passed
      for multi-task model training. If passed, this will be popped from the
      features dictionary after performing any transformations and the dataset
      returned will consist of tuples of the features dictionary and the
      corresponding label.
    batch_size: The batch size of the data. If passed the all dense tensors will
      be passed through `tf.ensure_shape` so that the batch size can be inferred
      by TF. Defaults to None.

  Returns:
    A callable that can be applied to a dictionary of features or mapped over
    a batched features `tf.data.Dataset` instance.

  Raises:
    ValueError: If the parameters `tf_transform_output`, `tft_layer`,
      `feature_transformations`, and `label_name`, are all None.
    ValueError: If both `tf_transform_output` and `tft_layer` are not None.
  """
  if (
      tf_transform_output is None
      and tft_layer is None
      and feature_transformations is None
      and label_name is None
      and batch_size is None
  ):
    raise ValueError(
        "At least one of `tf_transform_output`, `tft_layer`,"
        " `feature_transformations`, and `label_name` must be passed."
    )

  if tf_transform_output is not None and tft_layer is not None:
    raise ValueError(
        "At most one of `tf_transform_output` and `tft_layer` can be passed."
    )

  def _ensure_shape(x: Any) -> Any:
    if isinstance(x, tf.Tensor):
      return tf.ensure_shape(x, [batch_size] + x.shape[1:])
    return x

  def _transform_example(
      features: FeaturesDictType,
  ) -> (
      tuple[FeaturesDictType, tf.Tensor]
      | tuple[FeaturesDictType, FeaturesDictType]
      | FeaturesDictType
  ):
    if tf_transform_output is not None:
      features = tf_transform_output.transform_features_layer()(features)

    if tft_layer is not None:
      features = tft_layer(features)

    if feature_transformations is not None:
      for transformation in feature_transformations:
        features = transformation(features)

    if batch_size is not None:
      features = tf.nest.map_structure(_ensure_shape, features)

    if label_name is not None:
      if isinstance(label_name, Mapping):
        label = {}
        for label_key in label_name:
          label[label_key] = features[label_name[label_key]]

        # Pop labels in separate loops because multiple label_key can map to
        # the same label_name.
        for label_key in label_name:
          if label_name[label_key] in features:
            features.pop(label_name[label_key])

      else:
        label = features.pop(label_name)
      return features, label

    return features

  return _transform_example


def vectorized_filter(
    dataset: tf.data.Dataset,
    filter_fn: FilterFn,
    batch_size: int,
    drop_remainder: bool,
    tighten_sparse_shapes: bool = True,
) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
  """Performs a vectorized filter on a dataset.

  This function does the following dataset transformations in order:
    - Apply the boolean mask returned by the filter function on each batch.
    - Un-batch the dataset into individual examples.
    - Re-batch the dataset to the batch size.
    - Optionally, tighten the shape of sparse features to ensure that the shape
      of the variable length dimension is consistent with the longest example
      in the new batch. This assumes that any sparse features are 2D, which
      should be the case if this is applied after `tf.io.parse_example`.

  Args:
    dataset: A batched dataset to perform the vectorized filter on.
    filter_fn: A callable that accepts a dictionary of features, where each
      value can be a dense, sparse, or ragged tensor of shape [B, ...], and
      returns a dense bool tensor of shape [B], which should be a mask
      indicating whether or not to keep the feature.
    batch_size: The per replica batch size.
    drop_remainder: Whether the last batch should be dropped in the case it has
      fewer than `batch_size` elements.
    tighten_sparse_shapes: If True, applies an additional transformation to
      tighten the shape of sparse features to to ensure that the shape of the
      variable length dimension is consistent with the longest example in the
      new batch. This is useful when downstream feature processing /
      computations depend on the shape of the sparse tensor. Defaults to True.

  Returns:
    A dataset with the filtering and other transformations applied.
  """

  def _vectorized_filter(features: FeaturesDictType) -> FeaturesDictType:
    mask = tf.reshape(filter_fn(features), [-1])
    outputs = {}
    for name in sorted(features):
      if isinstance(features[name], tf.SparseTensor):
        outputs[name] = tf.sparse_boolean_mask(features[name], mask)
      elif isinstance(features[name], tf.RaggedTensor):
        # TODO(b/307323524): Support this when we start using Ragged tensors.
        raise ValueError("Filtering ragged tensors is not supported.")
      else:
        outputs[name] = tf.boolean_mask(features[name], mask)
    return outputs

  def _tighten_2d_sparse(features: FeaturesDictType) -> FeaturesDictType:
    outputs = {}
    for key in features:
      if (
          isinstance(features[key], tf.SparseTensor)
          and len(features[key].shape.as_list()) == 2
      ):
        outputs[key] = tighten_2d_sparse_tensor_shape(features[key])
      else:
        outputs[key] = features[key]
    return outputs

  dataset = dataset.map(
      _vectorized_filter, num_parallel_calls=tf.data.AUTOTUNE
  ).rebatch(batch_size, drop_remainder=drop_remainder)
  if tighten_sparse_shapes:
    dataset = dataset.map(
        _tighten_2d_sparse, num_parallel_calls=tf.data.AUTOTUNE
    )
  return dataset


def get_file_groups(files: Sequence[str]) -> Sequence[str]:
  """Parse and return file groups from file pattern.

  Groups files by their folders. Each group of file names is joined to string
  with comma as separator.

  Args:
    files: A sequence of string file names to be grouped.

  Returns:
    A sequence of strings containing a list of file groups in reverse order,
    each consisting of the file paths in the group separated by commas.

  Example usage:
  >>> files = [
  ...   'Span-0/Version-0/Split-Training/Shard-0-of-2',
  ...   'Span-0/Version-0/Split-Training/Shard-1-of-2',
  ...   'Span-1/Version-0/Split-Training/Shard-0-of-1'
  ... ]
  >>> get_file_groups(files)
  [
    'Span-1/Version-0/Split-Training/Shard-0-of-1',
    'Span-0/Version-0/Split-Training/Shard-0-of-2,Span-0/Version-0/Split-Training/Shard-1-of-2'
  ]

  Raises:
    ValueError: If `files` is empty.
  """
  if not files:
    raise ValueError("`files` is empty.")

  def _prefix(file_name):
    return file_name[: file_name.rfind("/")]

  file_name_groups = collections.defaultdict(list)
  for file_name in files:
    file_name_groups[_prefix(file_name)].append(file_name)

  # The file groups are sorted by folder in reverse order.
  sorted_prefix_file_list_tuple = sorted(
      file_name_groups.items(),
      key=lambda prefix_files: prefix_files[0],
      reverse=True,
  )

  logging.info(
      "First 10 file groups and number of files: %s",
      {
          prefix: len(group)
          for prefix, group in sorted_prefix_file_list_tuple[:10]
      },
  )
  return [",".join(sorted(files)) for _, files in sorted_prefix_file_list_tuple]


def tighten_2d_sparse_tensor_shape(
    sparse_tensor: tf.SparseTensor,
) -> tf.SparseTensor:
  """Reset the 2nd dimension of a SparseTensor to the tightest bounding shape.

  For example, given a SparseTensor:
    tf.SparseTensor(
        indices=tf.constant(
            [
                [0, 0], [0, 1], [0, 2],
                [1, 0], [1, 1],
            ],
            dtype=tf.int64,
        ),
        values=tf.constant([0, 0, 1, 2, 3], dtype=tf.int64),
        dense_shape=(2, 6),
    )

  The function returns:
    tf.SparseTensor(
        indices=tf.constant(
            [
                [0, 0], [0, 1], [0, 2],
                [1, 0], [1, 1],
            ],
            dtype=tf.int64,
        ),
        values=tf.constant([0, 0, 1, 2, 3], dtype=tf.int64),
        dense_shape=(2, 3),
    )
  The new SparseTensor has the same indices and values as the original input,
  but the dense_shape is the tightest bounding shape for the 2nd dimension.

  Args:
    sparse_tensor: a tf.SparseTensor with a potentially loose dense_shape.

  Returns:
    A SparseTensor with tight dense_shape.

  Raises:
    tf.RuntimeError: If the rank of the input is not 2.
  """
  with tf.control_dependencies([tf.assert_rank(sparse_tensor, 2)]):
    # This is required since reduce max returns the smallest possible int value
    # when the list is empty.
    max_index = tf.reduce_max(
        tf.concat(
            [
                tf.constant([-1], dtype=sparse_tensor.indices.dtype),
                sparse_tensor.indices[:, 1],
            ],
            axis=0,
        ),
        axis=0,
    )
    max_length = max_index + tf.constant(1, dtype=max_index.dtype)
    batch_size = sparse_tensor.dense_shape[0]
    return tf.SparseTensor(
        indices=sparse_tensor.indices,
        values=sparse_tensor.values,
        dense_shape=tf.stack([batch_size, max_length]),
    )

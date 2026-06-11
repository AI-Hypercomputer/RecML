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
"""HSTU Experiment Configuration using Fiddle and RecML with KerasTrainer"""

import dataclasses
from typing import Optional
import sys
import os

import fiddle as fdl
import keras
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

# Add the RecML folder to the system path
sys.path.append(os.path.join(os.getcwd(), "../../../RecML"))

from recml.core.training import core
from recml.core.training import keras_trainer
from recml.layers.keras import hstu
import recml
import jax
print(jax.devices())

# Define command-line flags
FLAGS = flags.FLAGS

flags.DEFINE_string("train_path", None, "Path (or pattern) to training data")
flags.DEFINE_string("eval_path", None, "Path (or glob pattern) to evaluation data")

flags.DEFINE_string("model_dir", "/tmp/hstu_model", "Where to save the model")
flags.DEFINE_integer("vocab_size", 5_000_000, "Vocabulary size")
flags.DEFINE_integer("train_steps", 2000, "Total training steps")

# Mark flags as required
flags.mark_flag_as_required("train_path")
flags.mark_flag_as_required("eval_path")

@dataclasses.dataclass
class HSTUModelConfig:
    """Configuration for the HSTU model architecture"""
    vocab_size: int = 5_000_000
    max_sequence_length: int = 50
    model_dim: int = 64
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.5
    learning_rate: float = 1e-3

class TFRecordDataFactory(recml.Factory[tf.data.Dataset]):
    """Reusable Data Factory for TFRecord datasets"""

    path: str
    batch_size: int
    max_sequence_length: int
    feature_key: str = "sequence"
    target_key: str = "target"
    is_training: bool = True

    def make(self) -> tf.data.Dataset:
        """Builds the tf.data.Dataset"""
        if not self.path:
            logging.warning("No path provided for dataset factory")
            return tf.data.Dataset.empty()

        dataset = tf.data.Dataset.list_files(self.path)
        dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=tf.data.AUTOTUNE)

        def _parse_fn(serialized_example):
            features = {
                self.feature_key: tf.io.VarLenFeature(tf.int64),
                self.target_key: tf.io.FixedLenFeature([1], tf.int64),
            }
            parsed = tf.io.parse_single_example(serialized_example, features)
            
            seq = tf.sparse.to_dense(parsed[self.feature_key])
            padding_needed = self.max_sequence_length - tf.shape(seq)[0]
            seq = tf.pad(seq, [[0, padding_needed]])
            seq = tf.ensure_shape(seq, [self.max_sequence_length])
            seq = tf.cast(seq, tf.int32)
            
            target = tf.cast(parsed[self.target_key], tf.int32)
            return seq, target

        dataset = dataset.map(_parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
        
        if self.is_training:
            dataset = dataset.repeat()
            
        return dataset.batch(self.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

class HSTUTask(keras_trainer.KerasTask):
    """KerasTask that receives its dependencies via injection"""

    def __init__(
        self,
        model_config: HSTUModelConfig,
        train_data_factory: recml.Factory[tf.data.Dataset],
        eval_data_factory: recml.Factory[tf.data.Dataset],
    ):
        self.config = model_config
        self.train_data_factory = train_data_factory
        self.eval_data_factory = eval_data_factory

    def create_dataset(self, training: bool) -> tf.data.Dataset:
        if training:
            return self.train_data_factory.make()
        return self.eval_data_factory.make()

    def create_model(self) -> keras.Model:
        inputs = keras.Input(
            shape=(self.config.max_sequence_length,), dtype="int32", name="input_ids"
        )
        padding_mask = keras.ops.cast(keras.ops.not_equal(inputs, 0), "int32")

        hstu_layer = hstu.HSTU(
            vocab_size=self.config.vocab_size,
            max_positions=self.config.max_sequence_length,
            model_dim=self.config.model_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )

        logits = hstu_layer(inputs, padding_mask=padding_mask)

        def get_last_token_logits(args):
            seq_logits, mask = args
            lengths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
            last_indices = lengths - 1
            return tf.gather(seq_logits, last_indices, batch_dims=1)

        output_logits = keras.layers.Lambda(get_last_token_logits)([logits, padding_mask])
        output_logits = keras.layers.Activation("linear", dtype="float32")(output_logits)

        model = keras.Model(inputs=inputs, outputs=output_logits)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseTopKCategoricalAccuracy(k=10, name="HR_10"),
                keras.metrics.SparseTopKCategoricalAccuracy(k=50, name="HR_50"),
                keras.metrics.SparseTopKCategoricalAccuracy(k=200, name="HR_200"),
            ],
        )
        return model

def experiment() -> fdl.Config[recml.Experiment]:
    """Defines the experiment structure using Fiddle configs"""
    
    max_seq_len = 50
    batch_size = 128
    
    model_cfg = fdl.Config(
        HSTUModelConfig,
        vocab_size=5_000_000, 
        max_sequence_length=max_seq_len,
        model_dim=64,
        num_layers=4,
        dropout=0.5
    )

    train_data = fdl.Config(
        TFRecordDataFactory,
        path="", # Placeholder
        batch_size=batch_size,
        max_sequence_length=max_seq_len,
        is_training=True
    )

    eval_data = fdl.Config(
        TFRecordDataFactory,
        path="", # Placeholder
        batch_size=batch_size,
        max_sequence_length=max_seq_len,
        is_training=False
    )

    task = fdl.Config(
        HSTUTask,
        model_config=model_cfg,
        train_data_factory=train_data,
        eval_data_factory=eval_data
    )

    trainer = fdl.Config(
        keras_trainer.KerasTrainer,
        model_dir="/tmp/default_dir", # Placeholder
        train_steps=2000,
        steps_per_eval=10,
        steps_per_loop=10,
    )

    return fdl.Config(recml.Experiment, task=task, trainer=trainer)

def main(_):
    keras.mixed_precision.set_global_policy("mixed_bfloat16")
    logging.info("Mixed precision policy set to mixed_bfloat16")

    config = experiment()

    logging.info(f"Setting Train Path to: {FLAGS.train_path}")
    config.task.train_data_factory.path = FLAGS.train_path
    
    logging.info(f"Setting Eval Path to: {FLAGS.eval_path}")
    config.task.eval_data_factory.path = FLAGS.eval_path
    
    config.task.model_config.vocab_size = FLAGS.vocab_size
    
    logging.info(f"Setting Model Dir to: {FLAGS.model_dir}")
    config.trainer.model_dir = FLAGS.model_dir
    config.trainer.train_steps = FLAGS.train_steps

    expt = fdl.build(config)
    
    logging.info("Starting experiment execution...")
    core.run_experiment(expt, core.Experiment.Mode.TRAIN_AND_EVAL)


if __name__ == "__main__":
    app.run(main)
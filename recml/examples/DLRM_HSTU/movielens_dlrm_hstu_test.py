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
"""Tests for DLRM with MovieLens dataset."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
import optax
import pandas as pd
from recml.examples.DLRM_HSTU.dlrm_hstu import DlrmHSTU
from recml.examples.DLRM_HSTU.dlrm_hstu import DlrmHSTUConfig
from recml.examples.DLRM_HSTU.dlrm_hstu import EmbeddingConfig
from recml.examples.DLRM_HSTU.movielens_dataloader import MovieLensDataLoader
from recml.examples.DLRM_HSTU.multitask_module import MultitaskTaskType
from recml.examples.DLRM_HSTU.multitask_module import TaskConfig


USER_ID = 'user_id'
ITEM_ID = 'item_id'
TIMESTAMP = 'timestamp'
USER_RATING = 'user_rating'


def create_dummy_movielens_df(num_users, num_items, num_events):
  user_ids = np.random.randint(0, num_users, num_events)
  item_ids = np.random.randint(0, num_items, num_events)
  ratings = np.random.uniform(0.5, 5.0, num_events).round(1)
  timestamps = np.arange(num_events) * 1000  # Increasing timestamps
  df = pd.DataFrame({
      USER_ID: [f'user_{u}' for u in user_ids],
      ITEM_ID: [f'item_{i}' for i in item_ids],
      USER_RATING: ratings,
      TIMESTAMP: timestamps,
  })
  return df


class DlrmMovielensTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.batch_size = 16
    self.max_uih_len = 16
    self.max_candidates = 4
    self.embed_dim = 32
    self.hstu_dim = 64

    dummy_df = create_dummy_movielens_df(
        num_users=50, num_items=100, num_events=500
    )

    self.dataloader = MovieLensDataLoader(
        self.batch_size,
        self.max_uih_len,
        self.max_candidates,
        raw_df=dummy_df,
    )

    self.user_vocab = self.dataloader.user_vocab_size
    self.movie_vocab = self.dataloader.movie_vocab_size
    self.rating_vocab = self.dataloader.rating_vocab_size
    self.genre_vocab = self.dataloader.genre_vocab_size

    self.config = DlrmHSTUConfig(
        max_seq_len=self.max_uih_len + self.max_candidates,
        hstu_embedding_table_dim=self.embed_dim,
        hstu_transducer_embedding_dim=self.hstu_dim,
        hstu_preprocessor_hidden_dim=16,
        hstu_attn_num_layers=2,
        hstu_attn_linear_dim=16,
        hstu_attn_qk_dim=16,
        item_embedding_feature_names=['movie_id'],
        user_embedding_feature_names=['user_id'],
        uih_post_id_feature_name='movie_id',
        uih_action_time_feature_name='action_time',
        candidates_querytime_feature_name='query_time',
        candidates_weight_feature_name='candidates_weight',
        candidates_watchtime_feature_name='candidates_watch_time',
        uih_weight_feature_name='uih_weight',
        action_weights=[1],
        merge_uih_candidate_feature_mapping=[
            ('movie_id', 'movie_id'),
            ('rating', 'rating'),
            ('action_time', 'query_time'),
            ('user_id', 'user_id'),
            ('uih_weight', 'candidates_weight'),
            ('uih_watch_time', 'candidates_watch_time'),
        ],
        multitask_configs=[
            TaskConfig('RatingPrediction', 1, MultitaskTaskType.REGRESSION),
        ],
        contextual_feature_to_max_length={},
        additional_content_features={},
        target_enrich_features={},
    )
    self.tables = {
        'user_id': EmbeddingConfig('user_id', self.user_vocab, self.embed_dim),
        'movie_id': EmbeddingConfig(
            'movie_id', self.movie_vocab, self.embed_dim
        ),
        'rating': EmbeddingConfig('rating', self.rating_vocab, self.embed_dim),
    }

  @parameterized.named_parameters(
      ('train', False),
      ('eval', True),
  )
  def test_forward_pass(self, deterministic):
    key = jax.random.PRNGKey(0)
    prng_keys = jax.random.split(key, 3)
    model = DlrmHSTU(hstu_configs=self.config, embedding_tables=self.tables)

    if not self.dataloader:
      self.skipTest(
          'No batches were created, potentially too few users or max_candidates'
          ' too high for debug data.'
      )

    batch = self.dataloader.get_batch(0)
    uih_features = batch['uih_features']
    candidate_features = batch['candidate_features']
    uih_lengths = batch['uih_lengths']
    num_candidates = batch['num_candidates']

    variables = model.init(
        {'params': prng_keys[0], 'dropout': prng_keys[1]},
        uih_features,
        candidate_features,
        uih_lengths,
        num_candidates,
        deterministic=deterministic,
    )

    user_emb, item_emb, aux_losses, preds, labels, weights = model.apply(
        variables,
        uih_features,
        candidate_features,
        uih_lengths,
        num_candidates,
        deterministic=deterministic,
        rngs={'dropout': prng_keys[2]} if not deterministic else None,
    )

    num_tasks = len(self.config.multitask_configs)
    self.assertEqual(
        user_emb.shape, (self.batch_size, self.max_candidates, self.hstu_dim)
    )
    self.assertEqual(
        item_emb.shape, (self.batch_size, self.max_candidates, self.hstu_dim)
    )
    self.assertEqual(
        preds.shape, (num_tasks, self.batch_size, self.max_candidates)
    )

    if not deterministic:
      self.assertNotEmpty(aux_losses)
      self.assertEqual(
          labels.shape, (num_tasks, self.batch_size, self.max_candidates)
      )
      self.assertEqual(
          weights.shape, (num_tasks, self.batch_size, self.max_candidates)
      )

  def test_backward_pass(self):
    key = jax.random.PRNGKey(1)
    init_key, data_key, train_key = jax.random.split(key, 3)
    model = DlrmHSTU(hstu_configs=self.config, embedding_tables=self.tables)

    if not self.dataloader:
      self.skipTest(
          'No batches were created, potentially too few users or max_candidates'
          ' too high for debug data.'
      )

    batch = self.dataloader.get_batch(0)
    uih_features = batch['uih_features']
    candidate_features = batch['candidate_features']
    uih_lengths = batch['uih_lengths']
    num_candidates = batch['num_candidates']

    variables = model.init(
        {'params': init_key, 'dropout': train_key},
        uih_features,
        candidate_features,
        uih_lengths,
        num_candidates,
        deterministic=False,
    )
    params = variables['params']

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    def loss_fn(params, dropout_key):
      _, _, aux_losses, _, _, _ = model.apply(
          {'params': params},
          uih_features,
          candidate_features,
          uih_lengths,
          num_candidates,
          deterministic=False,
          rngs={'dropout': dropout_key},
      )
      return sum(val.sum() for val in aux_losses.values())

    @jax.jit
    def train_step(params, opt_state, dropout_key):
      loss, grads = jax.value_and_grad(loss_fn)(params, dropout_key)
      updates, opt_state = optimizer.update(grads, opt_state, params)
      params = optax.apply_updates(params, updates)
      return params, opt_state, loss

    step_key, train_key = jax.random.split(train_key)
    params, opt_state, loss = train_step(params, opt_state, step_key)
    logging.info('MovieLens Test Loss: %f', loss)
    self.assertIsNotNone(params)


if __name__ == '__main__':
  absltest.main()

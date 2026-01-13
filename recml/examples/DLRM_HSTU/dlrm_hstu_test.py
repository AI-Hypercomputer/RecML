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
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import tree_util
import jax.numpy as jnp
import numpy as np
import optax
from recml.core.training import partitioning
from recml.examples.DLRM_HSTU.dlrm_hstu import DlrmHSTU
from recml.examples.DLRM_HSTU.dlrm_hstu import DlrmHSTUConfig
from recml.examples.DLRM_HSTU.dlrm_hstu import EmbeddingConfig
from recml.examples.DLRM_HSTU.multitask_module import MultitaskTaskType
from recml.examples.DLRM_HSTU.multitask_module import TaskConfig
from recml.layers.linen import sparsecore
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec


def get_sparsecore_config(embed_dim, post_id_vocab, cat_feat_vocab, seq_len):
  """Returns a SparsecoreConfig for DLRM-HSTU."""
  post_id_spec = sparsecore.EmbeddingSpec(
      input_dim=post_id_vocab,
      embedding_dim=embed_dim,
      max_sequence_length=seq_len,
  )
  cat_feat_spec = sparsecore.EmbeddingSpec(
      input_dim=cat_feat_vocab,
      embedding_dim=embed_dim,
      max_sequence_length=seq_len,
  )
  return sparsecore.SparsecoreConfig(
      specs={
          'post_id': post_id_spec,
          'cat_feat': cat_feat_spec,
      },
      optimizer=embedding_spec.AdagradOptimizerSpec(learning_rate=0.01),
  )


class DlrmHstuTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.batch_size = 16
    self.max_uih_len = 4
    self.max_candidates = 2
    self.embed_dim = 128
    self.hstu_dim = 128
    self.post_id_vocab = 20000
    self.cat_feat_vocab = 20000

    sc_config = get_sparsecore_config(
        self.embed_dim,
        self.post_id_vocab,
        self.cat_feat_vocab,
        seq_len=self.max_uih_len + self.max_candidates,
    )

    self.config = DlrmHSTUConfig(
        sparsecore_config=sc_config,
        max_seq_len=self.max_uih_len + self.max_candidates,
        hstu_embedding_table_dim=self.embed_dim,
        hstu_transducer_embedding_dim=self.hstu_dim,
        hstu_preprocessor_hidden_dim=8,
        hstu_attn_num_layers=4,
        hstu_attn_linear_dim=8,
        hstu_attn_qk_dim=8,
        item_embedding_feature_names=['post_id', 'cat_feat'],
        user_embedding_feature_names=['post_id'],
        uih_post_id_feature_name='post_id',
        uih_action_time_feature_name='action_time',
        candidates_querytime_feature_name='query_time',
        candidates_weight_feature_name='weight',
        candidates_watchtime_feature_name='watch_time',
        uih_weight_feature_name='action_type',
        action_weights=[1, 2, 4],
        merge_uih_candidate_feature_mapping=[
            ('post_id', 'post_id'),
            ('cat_feat', 'cat_feat'),
            ('action_type', 'action_type'),
        ],
        multitask_configs=[
            TaskConfig('CTR', 1, MultitaskTaskType.BINARY_CLASSIFICATION),
            TaskConfig('WT', 2, MultitaskTaskType.REGRESSION),
        ],
        contextual_feature_to_max_length={},
        additional_content_features={'cat_feat': self.embed_dim},
        target_enrich_features={'cat_feat': self.embed_dim},
    )
    if jax.devices()[0].platform == 'tpu':
      self.mesh = jax.sharding.Mesh(np.array(jax.devices()), ('data',))
    else:
      self.mesh = None

  def _get_mock_data(self, key):
    k1, k2, k3, k4, k5, k6, k7, k8, k9 = jax.random.split(key, 9)
    uih_features = {
        'post_id': jax.random.randint(
            k1, (self.batch_size, self.max_uih_len), 0, self.post_id_vocab
        ),
        'cat_feat': jax.random.randint(
            k2, (self.batch_size, self.max_uih_len), 0, self.cat_feat_vocab
        ),
        'action_time': jax.random.randint(
            k3, (self.batch_size, self.max_uih_len), 0, 1000
        ),
        'action_type': jax.random.randint(
            k7, (self.batch_size, self.max_uih_len), 0, 8
        ),
    }
    candidate_features = {
        'post_id': jax.random.randint(
            k1, (self.batch_size, self.max_candidates), 0, self.post_id_vocab
        ),
        'cat_feat': jax.random.randint(
            k2, (self.batch_size, self.max_candidates), 0, self.cat_feat_vocab
        ),
        'query_time': jax.random.randint(
            k4, (self.batch_size, self.max_candidates), 1000, 2000
        ),
        'weight': jax.random.randint(
            k5, (self.batch_size, self.max_candidates), 0, 2
        ),  # for CTR bitmask
        'watch_time': jax.random.randint(
            k6, (self.batch_size, self.max_candidates), 0, 100
        ),  # for WT regression
        'action_type': jax.random.randint(
            k7, (self.batch_size, self.max_candidates), 0, 8
        ),
    }
    uih_lengths = jax.random.randint(
        k8, (self.batch_size,), 1, self.max_uih_len + 1
    ).astype(jnp.int32)
    num_candidates = jax.random.randint(
        k9, (self.batch_size,), 1, self.max_candidates + 1
    ).astype(jnp.int32)
    return uih_features, candidate_features, uih_lengths, num_candidates

  @parameterized.named_parameters(
      ('train', False),
      ('eval', True),
  )
  def test_forward_pass(self, deterministic):
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Test only supported on TPUs.')
    key = jax.random.PRNGKey(0)
    prng_keys = jax.random.split(key, 3)
    model = DlrmHSTU(hstu_configs=self.config, mesh=self.mesh)
    uih_features, candidate_features, uih_lengths, num_candidates = (
        self._get_mock_data(key)
    )

    features = uih_features | candidate_features
    features['post_id'] = np.concatenate(
        [uih_features['post_id'], candidate_features['post_id']], axis=1
    )
    features['cat_feat'] = np.concatenate(
        [uih_features['cat_feat'], candidate_features['cat_feat']], axis=1
    )
    features['action_type'] = np.concatenate(
        [uih_features['action_type'], candidate_features['action_type']], axis=1
    )

    preprocessor = sparsecore.SparsecorePreprocessor(
        self.config.sparsecore_config, self.batch_size
    )
    sc_features = preprocessor(features)

    variables = model.init(
        {'params': prng_keys[0], 'dropout': prng_keys[1]},
        sc_features,
        uih_lengths,
        num_candidates,
        deterministic=deterministic,
    )

    user_emb, item_emb, aux_losses, preds, labels, weights = model.apply(
        variables,
        sc_features,
        uih_lengths,
        num_candidates,
        deterministic=deterministic,
        rngs={'dropout': prng_keys[2]} if not deterministic else None,
    )

    num_tasks = len(self.config.multitask_configs)
    expected_user_emb_shape = (
        self.batch_size,
        self.max_candidates,
        self.hstu_dim,
    )
    self.assertEqual(user_emb.shape, expected_user_emb_shape)
    expected_item_emb_shape = (
        self.batch_size,
        self.max_candidates,
        self.hstu_dim,
    )
    self.assertEqual(item_emb.shape, expected_item_emb_shape)
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
    else:
      self.assertEmpty(aux_losses)
      self.assertIsNone(labels)
      self.assertIsNone(weights)

  def test_backward_pass_and_training(self):
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Test only supported on TPUs.')
    key = jax.random.PRNGKey(1)
    init_key, data_key, train_key = jax.random.split(key, 3)
    model = DlrmHSTU(hstu_configs=self.config, mesh=self.mesh)
    uih_features, candidate_features, uih_lengths, num_candidates = (
        self._get_mock_data(data_key)
    )

    features = uih_features | candidate_features
    features['post_id'] = np.concatenate(
        [uih_features['post_id'], candidate_features['post_id']], axis=1
    )
    features['cat_feat'] = np.concatenate(
        [uih_features['cat_feat'], candidate_features['cat_feat']], axis=1
    )
    features['action_type'] = np.concatenate(
        [uih_features['action_type'], candidate_features['action_type']], axis=1
    )

    preprocessor = sparsecore.SparsecorePreprocessor(
        self.config.sparsecore_config, self.batch_size
    )
    sc_features = preprocessor(features)

    variables = model.init(
        {'params': init_key, 'dropout': train_key},
        sc_features,
        uih_lengths,
        num_candidates,
        deterministic=False,
    )
    params = variables['params']
    cache = variables['cache']

    logging.info(
        'Model parameter shapes: %s',
        tree_util.tree_map(lambda x: x.shape, params),
    )
    logging.info('Model parameters: %s', params)

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    def loss_fn(params, dropout_key):
      user_emb, item_emb, aux_losses, preds, labels, weights = model.apply(
          {'params': params, 'cache': cache},
          sc_features,
          uih_lengths,
          num_candidates,
          deterministic=False,
          rngs={'dropout': dropout_key},
      )
      return (
          user_emb.sum()
          + item_emb.sum()
          + preds.sum()
          + sum(val.sum() for val in aux_losses.values())
      )

    @jax.jit
    def train_step(params, opt_state, dropout_key):
      loss, grads = jax.value_and_grad(loss_fn)(params, dropout_key)
      updates, opt_state = optimizer.update(grads, opt_state, params)
      params = optax.apply_updates(params, updates)
      return params, opt_state, loss

    logging.info('Starting training loop...')
    for i in range(10):
      step_key, train_key = jax.random.split(train_key)
      params, opt_state, loss = train_step(params, opt_state, step_key)
      logging.info('Step %d, Loss: %f', i, loss)

    self.assertIsNotNone(params)

  def test_dlrm_hstu_with_sparsecore(self):
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Test only supported on TPUs.')
    key = jax.random.PRNGKey(0)
    prng_keys = jax.random.split(key, 3)
    model = DlrmHSTU(hstu_configs=self.config, mesh=self.mesh)
    uih_features, candidate_features, uih_lengths, num_candidates = (
        self._get_mock_data(key)
    )

    features = uih_features | candidate_features
    features['post_id'] = np.concatenate(
        [uih_features['post_id'], candidate_features['post_id']], axis=1
    )
    features['cat_feat'] = np.concatenate(
        [uih_features['cat_feat'], candidate_features['cat_feat']], axis=1
    )
    features['action_type'] = np.concatenate(
        [uih_features['action_type'], candidate_features['action_type']], axis=1
    )

    preprocessor = sparsecore.SparsecorePreprocessor(
        self.config.sparsecore_config, self.batch_size
    )
    sc_features = preprocessor(features)

    variables = model.init(
        {'params': prng_keys[0], 'dropout': prng_keys[1]},
        sc_features,
        uih_lengths,
        num_candidates,
        deterministic=True,
    )

    user_emb, item_emb, _, preds, _, _ = model.apply(
        variables,
        sc_features,
        uih_lengths,
        num_candidates,
        deterministic=True,
        rngs={'dropout': prng_keys[2]},
    )
    num_tasks = len(self.config.multitask_configs)
    expected_user_emb_shape = (
        self.batch_size,
        self.max_candidates,
        self.hstu_dim,
    )
    self.assertEqual(user_emb.shape, expected_user_emb_shape)
    expected_item_emb_shape = (
        self.batch_size,
        self.max_candidates,
        self.hstu_dim,
    )
    self.assertEqual(item_emb.shape, expected_item_emb_shape)
    self.assertEqual(
        preds.shape, (num_tasks, self.batch_size, self.max_candidates)
    )


if __name__ == '__main__':
  absltest.main()

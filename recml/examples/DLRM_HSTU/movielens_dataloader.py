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
"""Dataloader for MovieLens dataset using jax_recommenders."""

import jax.numpy as jnp
import pandas as pd

USER_ID = 'user_id'
ITEM_ID = 'item_id'
TIMESTAMP = 'timestamp'
USER_RATING = 'user_rating'


class MovieLensDataLoader:
  """Dataloader for MovieLens dataset."""

  def __init__(
      self,
      batch_size,
      max_uih_len,
      max_candidates,
      raw_df: pd.DataFrame,
  ):
    self.batch_size = batch_size
    self.max_uih_len = max_uih_len
    self.max_candidates = max_candidates

    if raw_df is None:
      raise ValueError('raw_df must be provided')
    self.raw_df = raw_df
    self._create_vocabs()
    self.processed_data = self._preprocess_data()

  def _create_vocabs(self):
    """Creates vocabularies from the raw dataframe."""
    self.user_vocab = sorted(self.raw_df[USER_ID].unique())
    self.movie_vocab = sorted(self.raw_df[ITEM_ID].unique())
    # Movielens ratings are 0.5 to 5.0. We can map them to 0-9
    self.rating_vocab = sorted(self.raw_df[USER_RATING].unique())

    self.user_map = {name: i for i, name in enumerate(self.user_vocab)}
    self.movie_map = {name: i for i, name in enumerate(self.movie_vocab)}
    self.rating_map = {name: i for i, name in enumerate(self.rating_vocab)}

    self.user_vocab_size = len(self.user_vocab)
    self.movie_vocab_size = len(self.movie_vocab)
    self.rating_vocab_size = len(self.rating_vocab)
    self.genre_vocab_size = 1  # Genre not directly used in this simple version

  def _pad_seq(self, seq, max_len, pad_value=0):
    """Pads a sequence to max_len."""
    if len(seq) > max_len:
      return seq[:max_len]
    return seq + [pad_value] * (max_len - len(seq))

  def _preprocess_data(self):
    """Preprocesses the raw data into batches of UIH and candidates."""
    df = self.raw_df.copy()
    df[USER_ID] = df[USER_ID].map(self.user_map)
    df[ITEM_ID] = df[ITEM_ID].map(self.movie_map)
    df[USER_RATING] = df[USER_RATING].map(self.rating_map)

    df = df.sort_values(by=[USER_ID, TIMESTAMP])
    grouped = df.groupby(USER_ID)

    batched_data = []
    current_batch = []

    for user_id, user_df in grouped:
      history = user_df[:-self.max_candidates]
      candidates = user_df[-self.max_candidates:]

      if len(history) < 1 or len(candidates) < 1:
        continue

      uih_len = min(len(history), self.max_uih_len)
      num_cands = len(candidates)

      uih_features = {
          'user_id': self._pad_seq(
              [user_id] * uih_len, self.max_uih_len, pad_value=-1
          ),
          'movie_id': self._pad_seq(
              history[ITEM_ID].tolist(), self.max_uih_len
          ),
          'rating': self._pad_seq(
              history[USER_RATING].tolist(), self.max_uih_len
          ),
          'action_time': self._pad_seq(
              history[TIMESTAMP].tolist(), self.max_uih_len
          ),
          'uih_weight': self._pad_seq([1] * uih_len, self.max_uih_len, 0),
          'uih_watch_time': self._pad_seq(
              history[USER_RATING].tolist(), self.max_uih_len, 0
          ),
      }

      candidate_features = {
          'movie_id': self._pad_seq(
              candidates[ITEM_ID].tolist(), self.max_candidates
          ),
          'query_time': self._pad_seq(
              candidates[TIMESTAMP].tolist(), self.max_candidates
          ),
          # candidates_weight is used as a mask for valid candidates in the loss
          # calculation.
          'candidates_weight': self._pad_seq(
              [1] * num_cands, self.max_candidates, 0
          ),
          # candidates_watch_time carries the true rating values for the
          # candidate items, which are used as labels for the regression task
          # in the MultitaskModule.
          'candidates_watch_time': self._pad_seq(
              candidates[USER_RATING].tolist(), self.max_candidates, 0
          ),
      }

      current_batch.append({
          'uih_features': uih_features,
          'candidate_features': candidate_features,
          'uih_lengths': uih_len,
          'num_candidates': num_cands,
      })

      if len(current_batch) == self.batch_size:
        batched_data.append(self._collate_batch(current_batch))
        current_batch = []

    # Add the last partial batch if any
    if current_batch:
      # To keep things simple for the test, we'll drop the last partial batch
      # pass # batched_data.append(self._collate_batch(current_batch))
      pass
    return batched_data

  def _collate_batch(self, batch):
    """Collates a list of samples into a single batch of numpy arrays."""
    collated = {}
    if not batch:
      return collated

    keys = batch[0].keys()

    for key in keys:
      example_value = batch[0][key]
      if isinstance(example_value, dict):
        collated[key] = {}
        sub_keys = example_value.keys()
        for sub_key in sub_keys:
          collated[key][sub_key] = jnp.array(
              [sample[key][sub_key] for sample in batch]
          )
      elif isinstance(example_value, int):
        collated[key] = jnp.array([sample[key] for sample in batch])
      else:
        # Handle other potential types if necessary
        pass
    return collated

  def get_batch(self, idx):
    """Returns a single batch by index."""
    if idx >= len(self.processed_data):
      raise IndexError("Batch index out of range")
    return self.processed_data[idx]

  def __len__(self):
    return len(self.processed_data)

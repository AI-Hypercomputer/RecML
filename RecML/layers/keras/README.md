## Models

### SASRec

Uses self-attention to predict a user's next action based on their past
activities. It aims to understand long-term user interests while also making
good predictions based on just the most recent actions. It smartly adapts which
past actions to focus on depending on how much history a user has. Built
entirely with efficient attention blocks, SASRec avoids the complex structures
of older RNN or CNN models, leading to faster training and better performance on
diverse datasets.

#### Architecture Overview

-   **Embedding Layer** - Converts item IDs into dense vectors. Adds a learnable
    absolute positional embedding to the item embedding to incorporate sequence
    order information. Dropout is applied to the combined embedding.
-   **Multi-Head Self-Attention Layer** Computes attention scores between all
    pairs of items within the allowed sequence window. Employs causality by
    masking out attention to future positions to prevent information leakage
    when training with a causal prediction objective.
-   **Feed-Forward Network** Applied independently to each embedding vector
    output by the attention layer. Uses two linear layers with a GeLU activation
    in between to add non-linearity.
-   **Residual Connections and Pre-Layernorm** Applied around both the
    self-attention and feed-forward network sub-layers for stable and faster
    training of deeper models. Dropout is also used within the block.
-   **Prediction Head** Decodes the sequence embeddings into logits using the
    input item embedding table and computes a causal categorical cross entropy
    loss between the inputs and the inputs shifted right.

### BERT4Rec

Models how user preferences change based on their past actions for
recommendations. Unlike older methods that only look at history in chronological
order, BERT4Rec uses a transformer based approach to look at the user's sequence
of actions in both directions. This helps capture context better, as user
behavior isn't always strictly ordered. To learn effectively, it is trained
using a mask prediction objective: some items are randomly masked and the model
learns to predict them based on the context.. BERT4Rec consistently performs
better than many standard sequential models.

#### Architecture Overview

-   **Embedding Layer** - Converts item IDs into dense vectors. Adds a learnable
    absolute positional embedding to the item embedding to incorporate sequence
    order information. An optional type embedding can be added to the item
    embedding. Embedding dropout is applied to the combined embedding. Uses a
    separate embedding for masked features to prevent other item tokens from
    attending to them.

-   **Multi-Head Self-Attention Layer** Computes attention scores between all
    pairs of items within the allowed sequence window. Uses a separate embedding
    for masked features to prevent other item tokens from attending to them.

-   **Feed-Forward Network** Applied independently to each embedding vector
    output by the attention layer. Uses two linear layers with a GeLU activation
    in between to add non-linearity.

-   **Residual Connections and Post-Layernorm** Applied around both the
    self-attention and feed-forward network sub-layers for stable and faster
    training of deeper models. Dropout is also used within the block.

-   **Masked Prediction Head** Gathers and projects the masked sequence
    embeddings, and decodes them using the item embedding layer. Computes a
    categorical cross entropy loss between the masked item ids and the predicted
    logits for the corresponding masked item embeddings.

### HSTU

HSTU is a novel architecture designed for sequential recommendation,
particularly suited for high cardinality, non-stationary streaming data. It
reformulates recommendation as a sequential transduction task within a
generative modeling framework -"Generative Recommenders". HSTU aims to provide
state-of-the-art results while being highly scalable and efficient, capable of
handling models with up to trillions of parameters. It has demonstrated
significant improvements over baselines in offline benchmarks and online A/B
tests, leading to deployment on large-scale internet platforms.

#### Architecture Overview

-   **Embedding Layer** Converts various action tokens into dense vectors in the
    same space. Optionally, adds a learnable absolute positional embedding to
    incorporate sequence order information. Embedding dropout is applied to the
    combined embedding.
-   **Gated Pointwise Aggregated Attention** - Uses a multi-head gated pointwise
    attention mechanism with a Layernorm on the attention outputs before
    projecting them. This captures the intensity of interactions between
    actions, which is lost in softmax attention.
-   **Relative Attention Bias** - Uses a T5 style relative attention bias
    computed using the positions and timestamps of the actions to improve the
    position encoding.
-   **Residual Connections and Pre-Layernorm** Applied around both the pointwise
    attention blocks for stable and faster training of deeper models.
-   **No Feedforward Network** - The feedforward network is removed.
-   **Prediction Head** - Decodes the sequence embeddings into logits using
    separately learnt weights and computes a causal categorical cross entropy
    loss between the inputs and the inputs shifted right.

### Mamba4Rec

A linear recurrent Mamba 2 architecture to model sequences of items for
recommendations. This scales better on longer sequences than attention based
methods due to its linear complexity compared to the former's quadratic
complexity. Mamba4Rec performs better than RNNs and matches the quality of
standard attention models while being more efficient at both training and
inference time.

#### Architecture Overview

-   **Embedding Layer** Converts item IDs into dense vectors. No position
    embedding is used since the recurrent nature of Mamba inherently encodes
    positional information as an inductive bias.
-   **Mamba SSD** Computes a causal interaction between different item
    embeddings in the sequence using the Mamba state space duality algorithm.
-   **Feedforward Network** Applied independently to each embedding vector
    output by the Mamba layer. Uses two linear layers with a GeLU activation in
    between to add non-linearity.
-   **Residual Connections and Post-Layernorm** Applied around both the Mamba
    and feed-forward network sub-layers for stable and faster training of deeper
    models. Dropout is also used within the block.
-   **Prediction Head** Decodes the sequence embeddings into logits using the
    input item embedding table and computes a causal categorical cross entropy
    loss between the inputs and the inputs shifted right.

## References

-   SASRec Paper: Kang, W. C., & McAuley, J. (2018). Self-Attentive Sequential
    Recommendation. arXiv preprint arXiv:1808.09781v1.
    https://arxiv.org/abs/1808.09781
-   Transformer Paper: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J.,
    Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you
    need. Advances in neural information processing systems, 30.
-   Mamba4Rec Paper: Liu, C., Lin, J., Liu, H., Wang, J., & Caverlee, J. (2024).
    Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State
    Space Models. arXiv preprint arXiv:2403.03900v2.
    https://arxiv.org/abs/2403.03900
-   Mamba Paper: Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling
    with Selective State Spaces. arXiv preprint arXiv:2312.00752.
-   BERT4Rec Paper: Sun, F., Liu, J., Wu, J., Pei, C., Lin, X., Ou, W., & Jiang,
    P. (2019). BERT4Rec: Sequential Recommendation with Bidirectional Encoder
    Representations from Transformer. arXiv preprint arXiv:1904.06690v2.
    https://arxiv.org/abs/1904.06690
-   BERT Paper: Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT:
    Pre-training of Deep Bidirectional Transformers for Language Understanding.
    arXiv preprint arXiv:1810.04805.
-   HSTU Paper: Actions Speak Louder than Words: Trillion-Parameter Sequential
    Transducers for Generative Recommendations (arXiv:2402.17152)

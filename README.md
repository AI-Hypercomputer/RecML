# RecML: High-Performance Recommender Library

## Vision

RecML is envisioned as a high-performance, large-scale deep learning recommender
system library optimized for Cloud TPUs. It aims to provide researchers and
practitioners state-of-the-art reference implementations, tools, and best
practice guidelines for building and deploying recommender systems.

The key goals of RecML are:

*   **Performance & Scalability:** Leverage Cloud TPUs (including SparseCore
    acceleration) to deliver exceptional performance for training and serving
    massive models with large embeddings on datasets with millions or billions
    of items/users. RecML can additionally target Cloud GPUs.
*   **State-of-the-Art Models:** Provide production-ready, easy-to-understand
    reference implementations of popular and cutting-edge models, with a strong
    focus on LLM-based recommenders.
*   **Ease of Use:** Offer a user-friendly API, intuitive abstractions, and
    comprehensive documentation/examples for rapid prototyping and deployment.
*   **Flexibility:** Primarily built with Keras and JAX, but designed with
    potential future expansion to other frameworks like PyTorch/XLA.
*   **Open Source:** Foster community collaboration and provide components to
    help users get started with advanced recommender workloads on Google Cloud.

## Features

*   **High Performance:** Optimized for Cloud TPU (SparseCore) training and
    inference.
*   **Scalable Architecture:** Designed for massive datasets and models with
    large embedding tables. Includes support for efficient data loading
    (tf.data, potentially Grain) and sharding/SPMD.
*   **State-of-the-Art Model Implementations:** Reference implementations for
    various recommendation tasks (ranking, retrieval, sequential).
*   **Reusable Building Blocks:**
    *   Common recommendation layers (e.g., DCN, BERT4Rec).
    *   Specialized Embedding APIs (e.g. JAX Embedding API for SparseCore).
    *   Standardized metrics (e.g., AUC, Accuracy, NDCG@K, MRR, Recall@K).
    *   Common loss functions.
*   **Unified Trainer:** A high-level trainer abstraction capable of targeting
    different hardware (TPU/GPU) and frameworks. Includes customizable training
    and evaluation loops.
*   **End-to-End Support:** Covers aspects from data pipelines to training,
    evaluation, checkpointing, metrics logging (e.g., to BigQuery), and model
    export/serving considerations.

## Models Included

This library aims to house implementations for a variety of recommender models,
including:

*   **SASRec:** Self-Attention based Sequential Recommendation
*   **BERT4Rec:** Bidirectional Encoder Representations from Transformer for
    Sequential Recommendation.
*   **Mamba4Rec:** Efficient Sequential Recommendation with Selective State
    Space Models.
*   **HSTU:** Hierarchical Sequential Transduction Units for Generative
    Recommendations.
*   **DLRM v2:** Deep Learning Recommendation Model

## Roadmap / Future Work

*   Expand reference model implementations (Ranking,
    Retrieval, Uplift, foundation user model).
*   Add support for optimized configurations and lower precision training
    (bfloat16, fp16).
*   Improve support for Cloud GPU training and inference
*   Enhance sharding and quantization support.
*   Improve integration with Keras (and Keras Recommenders) and potentially
    PyTorch/XLA.
*   Develop comprehensive model serving examples and integrations.
*   Refine data loading pipelines (e.g., Grain support).
*   Add more common layers, losses, and metrics.

## Responsible Use

As with any machine learning model, potential risks exist. The performance and
behavior depend heavily on the training data, which may contain biases reflected
in the recommendations. Developers should carefully evaluate the model's
fairness and potential limitations in their specific application context.

## License

RecML is released under the Apache 2.0. Please see the `LICENSE` file for full
details.

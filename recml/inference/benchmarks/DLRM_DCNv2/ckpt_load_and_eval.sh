#!/bin/bash


export LIBTPU_INIT_ARGS=
export XLA_FLAGS=

export TPU_NAME=<TPU_NAME>
export LEARNING_RATE=0.0034
export BATCH_SIZE=4224
export EMBEDDING_SIZE=128
export MODEL_DIR=/tmp/
export FILE_PATTERN=gs://qinyiyan-vm/mlperf-dataset/criteo_merge_balanced_4224/train-*
export NUM_STEPS=28000
export CHECKPOINT_INTERVAL=1500
export EVAL_INTERVAL=1500
export EVAL_FILE_PATTERN=gs://qinyiyan-vm/mlperf-dataset/criteo_merge_balanced_4224/eval-*
export EVAL_STEPS=660
export MODE=eval
export EMBEDDING_THRESHOLD=21000
export LOGGING_INTERVAL=1500
export RESTORE_CHECKPOINT=true


python RecML/recml/inference/models/jax/DLRM_DCNv2/dlrm_main.py \
--learning_rate=${LEARNING_RATE} \
--batch_size=${BATCH_SIZE} \
--embedding_size=${EMBEDDING_SIZE} \
--embedding_threshold=${EMBEDDING_THRESHOLD} \
--model_dir=${MODEL_DIR} \
--file_pattern=${FILE_PATTERN} \
--num_steps=${NUM_STEPS} \
--save_checkpoint_interval=${CHECKPOINT_INTERVAL} \
--restore_checkpoint=${RESTORE_CHECKPOINT} \
--eval_interval=${EVAL_INTERVAL} \
--eval_file_pattern=${EVAL_FILE_PATTERN} \
--eval_steps=${EVAL_STEPS}  \
--mode=${MODE} \
--logging_interval=${LOGGING_INTERVAL}
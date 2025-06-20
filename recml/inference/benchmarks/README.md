

# Running RecML Inference benchmarks

## Setup environment

### Export Env 

```
export TPU_NAME=
export QR_NODE_NAME=
export PROJECT=
export ZONE=
export ACCELERATOR_TYPE=
export RUNTIME_VERSION=
```

### Launch a TPU VM

```
gcloud alpha compute tpus queued-resources create ${TPU_NAME} --node-id ${QR_NODE_NAME}$ --project ${PROJECT} --zone ${ZONE} --accelerator-type ${ACCELERATOR_TYPE} --runtime-version ${RUNTIME_VERSION}
```

### Install dependencies


#### Clone the RecML repository

```
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all  --command="git clone https://github.com/AI-Hypercomputer/RecML.git"
```

#### Install requirements

```
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all  --command="cd RecML && pip install -r requirements.txt" 
```

#### Install jax and jaxlib nightly

```
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all  --command="pip install -U --pre jax jaxlib libtpu requests -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/libtpu_releases.html --force"
```

#### Install JAX Sparsecore  (jax-tpu-embedding)

```
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all  --command="pip install -U https://storage.googleapis.com/jax-tpu-embedding-whls/20250604/jax_tpu_embedding-0.1.0.dev20250604-cp310-cp310-manylinux_2_35_x86_64.whl --force"
```

#### Install other dependencies

```
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all  --command="pip install -U tensorflow  dm-tree flax google-metrax"
```

#### Make script executable and Run workload

Note: Please update the MODEL_NAME & TASK_NAME before running the below command

```
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT} --zone ${ZONE} --worker=all  --command="cd RecML && chmod +x ./recml/inference/benchmarks/<MODEL_NAME>/<TASK_NAME> && TPU_NAME=${TPU_NAME} ./recml/inference/benchmarks/<MODEL_NAME>/<TASK_NAME>"
```

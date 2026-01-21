# Model Training Guide

This guide explains how to set up the environment and train the HSTU/DLRM models on Cloud TPU v6.

## Option 1: Virtual Environment (Recommended for Dev)

If you are developing on a TPU VM directly, use a virtual environment to avoid conflicts with the system-level Python packages.

#### 1. Prerequisites
Ensure you have **Python 3.12+** installed.
```bash
python3 --version
```

### 2. Create and Activate Virtual Environment
Run the following from the root of the repository:
```bash
# Create the venv
python3 -m venv venv

# Activate it
source venv/bin/activate
```

### 3. Install Dependencies

Install the latest version of the jax-tpu-embedding library:
```bash
pip install ./jax_tpu_embedding-0.1.0.dev20260121-cp312-cp312-manylinux_2_31_x86_64.whl
```
```bash
pip install -r requirements.txt
```
We need to force a specific version of Protobuf to ensure compatibility with our TPU stack. Run this exactly as shown:
```bash
pip install "protobuf>=6.31.1" --no-deps
```
The `--no-deps` flag is required to prevent pip from downgrading it due to strict dependency pinning in other libraries.

### 4. Run the Training for DLRM
```bash
python dlrm_experiment_test.py
```

## Option 2: Docker (Recommended for Production)

If you prefer not to manage a virtual environment or want to deploy this as a container, you can build a Docker image.

### 1. Build the Image

Run this command from the root of the repository. It reads the `Dockerfile`, installs all dependencies, and creates a ready-to-run image.

```bash
docker build -t recml-training .
```

### 2. Run the Image

This will run the docker image and execute the command specified, which is currently set to run DLRM. 

```bash
docker run --rm --privileged \
  --net=host \
  --ipc=host \
  --name recml-experiment \
  recml-training
```

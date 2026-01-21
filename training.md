# Model Training Guide

This guide explains how to set up the environment and train the HSTU/DLRM models on Cloud TPU v6.

## Option 1: Virtual Environment (Recommended for Dev)

If you are developing on a TPU VM directly, use a virtual environment to avoid conflicts with the system-level Python packages.

#### 1. Prerequisites
Ensure you have **Python 3.11+** installed.
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

## 1. Build the Image
Create a file named `Dockerfile` in the root of the repository:

```dockerfile
# Use an official Python 3.12 runtime as a parent image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# This tells Python to look in /app for the 'recml' package
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Install system tools if needed (e.g., git)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install the latest jax-tpu-embedding wheel
COPY jax_tpu_embedding-0.1.0.dev20260121-cp312-cp312-manylinux_2_31_x86_64.whl ./
RUN pip install ./jax_tpu_embedding-0.1.0.dev20260121-cp312-cp312-manylinux_2_31_x86_64.whl

# Copy requirements.txt to current directory
COPY  requirements.txt ./

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt

# Force install the specific protobuf version
RUN pip install "protobuf>=6.31.1" --no-deps

# Copy the current directory contents into the container
COPY . /app

# Default command to run the training script
CMD ["python", "recml/examples/dlrm_experiment_test.py"]
```

You can use this dockerfile to run the DLRM model experiment from this repo in your own environment. 

### 2. Build the Image

Run this command from the root of the repository. It reads the `Dockerfile`, installs all dependencies, and creates a ready-to-run image.

```bash
docker build -t recml-training .
```

### 3. Run the Image

```bash
docker run --rm --privileged \
  --net=host \
  --ipc=host \
  --name recml-experiment \
  recml-training
```

### What is happening here?
* **`--rm`**: Automatically deletes the container after the script finishes to keep your disk clean.
* **`--privileged`**: Grants the container direct access to the host's hardware devices, which is required to see the physical TPU chips.
* **`--net=host`**: Removes the container's network isolation, allowing the script to connect to the TPU runtime listening on local ports (e.g., 8353).
* **`--ipc=host`**: Allows the container to use the host's Shared Memory (IPC), which is critical for high-speed data transfer between the CPU and TPU.

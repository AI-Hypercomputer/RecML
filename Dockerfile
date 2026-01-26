# Use an official Python 3.12 runtime as a parent image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# This tells Python to look in /app for the 'recml' package
ENV PYTHONPATH="${PYTHONPATH}:/app"

# This prevents the "MessageFactory" crash when using Protobuf
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# This prevents the "Unable to register cuFFT/cuBLAS" log spam and initialization errors
ENV CUDA_VISIBLE_DEVICES=-1

# Install system tools
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install standard requirements
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt

# Force install the specific protobuf version
RUN pip install "protobuf>=6.31.1"

# Install the latest jax-tpu-embedding wheel
COPY jax_tpu_embedding-0.1.0.dev20260121-cp312-cp312-manylinux_2_31_x86_64.whl ./
RUN pip install ./jax_tpu_embedding-0.1.0.dev20260121-cp312-cp312-manylinux_2_31_x86_64.whl

# Copy the current directory contents into the container
COPY . /app

# Default command to run the training script
CMD ["python", "recml/examples/dlrm_experiment_test.py"]
# Use a base image with Python 3.11
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repository into the container
COPY . .

# Add the current directory to PYTHONPATH so imports work correctly
ENV PYTHONPATH="${PYTHONPATH}:/app"

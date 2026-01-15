# 1. Base image
FROM python:3.10-slim

# 2. Set the working directory (keeps things clean)
WORKDIR /app

# 3. Install requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the ENTIRE repo structure
# This will copy the 'recml' folder into /app/recml
COPY . .

# 5. Add current directory to PYTHONPATH
# This ensures python can find the 'recml' package imports
ENV PYTHONPATH="${PYTHONPATH}:/app"

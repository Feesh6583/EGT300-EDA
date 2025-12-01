# Use slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY . /app

# Set environment variable so Python can find src as a package
ENV PYTHONPATH=/app

# Default command: run pipeline
CMD ["python", "-m", "src.pipeline"]

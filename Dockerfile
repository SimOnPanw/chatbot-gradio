# Use the NVIDIA CUDA base image
FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create a working directory
WORKDIR /app

# Install required system packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    software-properties-common wget && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip && \
    apt-get clean

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Copy the entrypoint script
COPY entrypoint.sh /app/entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Expose the port for Gradio
EXPOSE 7860

# Run the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]

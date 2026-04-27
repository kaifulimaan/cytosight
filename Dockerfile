FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Set working directory inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from backend folder
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend source code
COPY backend/ .

# Create a non-root user for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

# Set environment paths
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose port
EXPOSE 7860

# Start application (app.main:app works because we copied contents of backend/)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

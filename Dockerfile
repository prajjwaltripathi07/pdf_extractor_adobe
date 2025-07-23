# Use Python 3.9 slim image for AMD64 architecture
FROM --platform=linux/amd64 python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Install required system packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Copy the entire project into the container
# This includes: model/, input/, output/, .py files, etc.
COPY . .

# Set environment variables for cleaner Python behavior
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command to run the PDF heading extractor
CMD ["python", "pdf_outline_extractor.py"]

# Use a standard Python 3.9 base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# --- NEW: Set environment variables for cache directories ---
# This tells Hugging Face libraries to use a local, writable directory for caching
ENV HF_HOME=/app/huggingface
ENV TRANSFORMERS_CACHE=/app/huggingface/transformers

# Create the directory and grant permissions (best practice)
RUN mkdir -p $HF_HOME/transformers && \
    chmod -R 777 $HF_HOME

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port that the app will run on
EXPOSE 7860

# The command to run your FastAPI application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

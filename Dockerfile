# Stage 1: Build - install dependencies and copy application code
FROM continuumio/miniconda3 AS builder

WORKDIR /app

# Copy the environment file first so that changes to your code won’t invalidate this cache
COPY environment.yml .

# Install system dependencies and create the conda environment
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    sox \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/* && \
    conda env create -f environment.yml

# Copy your application code and model file
COPY main.py .
COPY models/ models/

# Stage 2: Final runtime image
FROM continuumio/miniconda3

WORKDIR /app

# Install only the system libraries needed at runtime.
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    sox \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Copy the pre-built conda environment and app code from the builder stage
COPY --from=builder /opt/conda/envs/audio_classification_api /opt/conda/envs/audio_classification_api
COPY --from=builder /app/ /app/

# Make sure the conda environment’s binaries are in the PATH
ENV PATH=/opt/conda/envs/audio_classification_api/bin:$PATH

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

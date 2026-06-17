FROM mambaorg/micromamba:1.5.10 AS base

USER root

# 1. Install system dependencies INCLUDING procps for Nextflow
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget unzip ca-certificates openjdk-17-jdk-headless \
        libgl1 libglib2.0-0 libxrender1 libxtst6 libxi6 libxext6 \
        procps && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Setup working directory
WORKDIR /app

# 3. Setup conda environment
COPY --chown=mambauser:mambauser microscopy_env.yml .
# Cache pip downloads separately — aydin and its tree (numpy-indexed,
# dask, scikit-learn, …) is heavy and a fresh pip cache makes rebuilds slow.
RUN --mount=type=cache,target=/opt/conda/pkgs \
    --mount=type=cache,target=/home/mambauser/.cache/pip \
    micromamba create -f microscopy_env.yml -y

# 5. Switch to non-root user
USER mambauser

# 6. Set up the micromamba environment activation
ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV ENV_NAME=microscopy_env

# 7. Copy application code
COPY --chown=mambauser:mambauser . /app

# 8. Default command
CMD ["/bin/bash"]
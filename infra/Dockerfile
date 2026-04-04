# Parameter Golf — Training Container
# Pre-bakes CUDA, Python 3.13, PyTorch 2.11 (cu130), Triton 3.6, and all deps.
# Training data is NOT included — downloaded from GCS at job start via entrypoint.sh.
#
# Build from project root:
#   docker build -t parameter-golf-training -f infra/Dockerfile .

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps + Python 3.13
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl gnupg2 ca-certificates && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.13 python3.13-venv python3.13-dev && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13 && \
    ln -sf /usr/bin/python3.13 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.13 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# gcloud CLI (for GCS data download inside the container)
RUN curl -fsSL https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir=/opt
ENV PATH="/opt/google-cloud-sdk/bin:${PATH}"

# PyTorch (CUDA 13.0 build) — must be installed separately due to --index-url
RUN pip install --no-cache-dir \
    torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130

# Remaining Python deps (pinned from migration.md)
RUN pip install --no-cache-dir \
    triton==3.6.0 \
    numpy==2.4.4 \
    sentencepiece==0.2.1 \
    tqdm==4.67.3 \
    typing_extensions==4.15.0 \
    setuptools==81.0.0 \
    psutil==7.2.2

WORKDIR /app

# Training scripts — rebuild this layer when scripts change
COPY train_gpt.py experiment1.py experiment2.py kernels.py /app/

# Entrypoint
COPY infra/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

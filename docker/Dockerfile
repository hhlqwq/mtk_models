FROM python:3.11.11-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
        openssh-client \
        procps \
        unzip \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade \
        pip==25.2 \
        setuptools==80.9.0 \
        wheel==0.45.1

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

CMD ["sleep", "infinity"]

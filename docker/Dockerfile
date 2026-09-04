FROM python:3.11.11-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ARG DEBIAN_MIRROR=https://mirrors.aliyun.com/debian
ARG PYPI_MIRROR=https://mirrors.aliyun.com/pypi/simple/

RUN sed -i "s#http://deb.debian.org/debian#${DEBIAN_MIRROR}#g" \
        /etc/apt/sources.list.d/debian.sources

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

RUN python -m pip config --global set global.index-url "${PYPI_MIRROR}" \
    && python -m pip install --no-cache-dir --upgrade \
        pip==25.2 \
        setuptools==80.9.0 \
        wheel==0.45.1

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

CMD ["sleep", "infinity"]

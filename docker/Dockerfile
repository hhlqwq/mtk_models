# 使用 Docker 内置前端, 避免额外拉取 docker/dockerfile 镜像。
FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# 在 Ubuntu 内编译 Python, 保持 SDK 的 cp311 ABI。
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl git patch unzip openssh-client procps time \
    libgl1 libglib2.0-0 libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev libffi-dev liblzma-dev uuid-dev \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /tmp/hailongcodex/python-build \
    && cd /tmp/hailongcodex/python-build \
    && curl --fail --location --retry 3 https://www.python.org/ftp/python/3.11.11/Python-3.11.11.tgz -o python.tgz \
    && tar -xzf python.tgz && cd Python-3.11.11 \
    && ./configure --prefix=/usr/local --with-ensurepip=install \
    && make -j8 && make install && ln -s python3.11 /usr/local/bin/python
RUN python -m pip config --global set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && python -m pip install pip==25.2 setuptools==80.9.0 wheel==0.45.1
COPY requirements.txt yolov5-requirements.txt /opt/mtk-build/
RUN python -m pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118 \
    && python -m pip install -r /opt/mtk-build/requirements.txt -r /opt/mtk-build/yolov5-requirements.txt
# sdk 是构建脚本传入的服务器本地 SDK, 无远程下载。
COPY --from=sdk / /opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/
RUN python -m pip install \
    /opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/offline_tool/mtk_converter-8.16.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
    /opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/offline_tool/mtk_quantization-8.2.1-py3-none-any.whl \
    && ln -s /opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host/lib/libc++.so.1 /usr/local/lib/libc++.so.1 \
    && ln -s libnvrtc-672ee683.so.11.2 /usr/local/lib/python3.11/site-packages/torch/lib/libnvrtc.so && ldconfig
COPY setup_container.sh verify_environment.py /opt/mtk-build/
RUN bash /opt/mtk-build/setup_container.sh --build \
    && python -m pip freeze > /opt/mtk-build/installed-requirements.txt
LABEL mtk.environment="ubuntu22.04-np8.0.11-py311-cu118"
WORKDIR /workspace
CMD ["sleep", "infinity"]

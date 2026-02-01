FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/src/tensorrt/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    git \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgeos-dev \
    libmagic-dev \
    libexiv2-dev \
    libboost-all-dev \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# TensorRT 10.9.0.34 (CUDA 12.8)
RUN TRT_VERSION="10.9.0.34-1+cuda12.8" && \
    apt-get update && apt-get install -y \
    libnvinfer-dev=${TRT_VERSION} \
    libnvinfer-dispatch-dev=${TRT_VERSION} \
    libnvinfer-dispatch10=${TRT_VERSION} \
    libnvinfer-headers-dev=${TRT_VERSION} \
    libnvinfer-headers-plugin-dev=${TRT_VERSION} \
    libnvinfer-lean-dev=${TRT_VERSION} \
    libnvinfer-lean10=${TRT_VERSION} \
    libnvinfer-plugin-dev=${TRT_VERSION} \
    libnvinfer-plugin10=${TRT_VERSION} \
    libnvinfer-vc-plugin-dev=${TRT_VERSION} \
    libnvinfer-vc-plugin10=${TRT_VERSION} \
    libnvinfer10=${TRT_VERSION} \
    libnvonnxparsers-dev=${TRT_VERSION} \
    libnvonnxparsers10=${TRT_VERSION} \
    tensorrt-dev=${TRT_VERSION} \
    libnvinfer-bin=${TRT_VERSION}

# PyTorch 2.7.1 + torchvision for CUDA 12.8
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir onnx==1.16.1 onnxruntime-gpu==1.19.2 onnxslim==0.1.82

RUN pip install --no-cache-dir tensorrt-cu12==10.9.0.34

RUN pip install --no-cache-dir ultralytics>=8.3.0

# ARG NSYS_URL=https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_4/
# ARG NSYS_PKG=NsightSystems-linux-cli-public-2024.4.1.61-3431596.deb
# RUN apt-get update && apt install -y wget libglib2.0-0 && \
#     wget ${NSYS_URL}${NSYS_PKG} && dpkg -i $NSYS_PKG && rm $NSYS_PKG

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/src/tensorrt/lib:$LD_LIBRARY_PATH
WORKDIR /workspace

CMD ["/bin/bash"]
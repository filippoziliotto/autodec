# Base image: Ubuntu 22.04 with CUDA 12.4.1 developer tooling.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

LABEL org.opencontainers.image.title="superdec-autodec"
LABEL org.opencontainers.image.description="CUDA 12.4 training image for SuperDec and AutoDec"

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV FORCE_CUDA=1
ENV TORCH_EXTENSIONS_DIR=/workspace/superdec/.torch_extensions
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/workspace/superdec

# -------------------------
# 1. System packages
# -------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libegl1-mesa-dev \
    libgl1 \
    libglib2.0-0 \
    libglfw3-dev \
    libglm-dev \
    libgomp1 \
    libjpeg-dev \
    libomp-dev \
    libpng-dev \
    libsm6 \
    libx11-dev \
    libxext6 \
    libxrender1 \
    ninja-build \
    pkg-config \
    vim \
    wget \
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# 2. Install Miniforge
# -------------------------
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh && \
    conda clean -afy && \
    conda config --set channel_priority strict

SHELL ["/bin/bash", "-lc"]

# -------------------------
# 3. Create conda env
# -------------------------
RUN conda create -y -n superdec python=3.10 cmake && \
    conda clean -afy

# -------------------------
# 4. Install Python/CUDA dependencies
# -------------------------
ARG TORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1
ARG TORCH_CUDA=cu124
ARG PYG_TORCH_VERSION=2.5.0
ARG TORCH_GEOMETRIC_VERSION=2.6.1

RUN conda run -n superdec python -m pip install --upgrade pip setuptools wheel

RUN conda run -n superdec python -m pip install \
    numpy==1.26.4 \
    Cython==0.29.37 \
    ninja==1.11.1.3 \
    packaging==24.2

RUN conda run -n superdec python -m pip install \
    --extra-index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
    "torch==${TORCH_VERSION}+${TORCH_CUDA}" \
    "torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA}"

RUN conda run -n superdec python -m pip install \
    -f "https://data.pyg.org/whl/torch-${PYG_TORCH_VERSION}+${TORCH_CUDA}.html" \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv && \
    conda run -n superdec python -m pip install "torch-geometric==${TORCH_GEOMETRIC_VERSION}"

RUN conda run -n superdec python -m pip install \
    easydict==1.13 \
    gdown==5.2.1 \
    h5py==3.12.1 \
    hydra-core==1.3.2 \
    matplotlib==3.10.0 \
    networkx==3.4.2 \
    omegaconf==2.3.0 \
    open3d==0.19.0 \
    opencv-python-headless==4.13.0.90 \
    pandas==2.2.3 \
    pillow==11.1.0 \
    plyfile==1.1.3 \
    psutil==5.9.8 \
    pydantic==2.10.6 \
    PyYAML==6.0.2 \
    requests==2.32.3 \
    rich==14.3.1 \
    rtree==1.4.1 \
    scikit-image==0.25.1 \
    scikit-learn==1.6.1 \
    scipy==1.15.1 \
    shapely==2.1.2 \
    tqdm==4.67.1 \
    trimesh==3.23.5 \
    typing_extensions==4.12.2 \
    viser==1.0.21 \
    wandb==0.24.0 \
    websockets==15.0.1

# -------------------------
# 5. Copy repo and install local modules
# -------------------------
WORKDIR /workspace/superdec
COPY . /workspace/superdec

RUN mkdir -p "${TORCH_EXTENSIONS_DIR}" data checkpoints outputs wandb logs && \
    conda run -n superdec python -m pip install --no-deps -e . && \
    conda run -n superdec python setup_sampler.py build_ext --inplace && \
    chmod -R a+rwX "${TORCH_EXTENSIONS_DIR}" data checkpoints outputs wandb logs

# To use "conda activate" in interactive bash.
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda activate superdec" >> /root/.bashrc && \
    echo "cd /workspace/superdec" >> /root/.bashrc

# Standard command for interactive runs.
CMD ["bash", "-lc", "source /opt/conda/etc/profile.d/conda.sh && conda activate superdec && cd /workspace/superdec && bash"]

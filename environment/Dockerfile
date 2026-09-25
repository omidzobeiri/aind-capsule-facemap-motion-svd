# facemap motion-SVD capsule environment.
#
# The base image is the one the existing AIND facemap capsules build from
# (facemap-capsule-template / facemap-capsule-fixed-face-rois), so the install is
# known to work in this deployment. Torch is a hard dependency of facemap's import
# chain but nothing here runs on the GPU -- the motion SVD is numpy/scipy -- so the
# CPU build is installed explicitly to keep the image small and the run deterministic.
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch-cpu-only:2.4.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-ask-pass /

RUN apt-get update -qq > /dev/null 2>&1 \
    && apt-get install -y -qq --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
    > /dev/null 2>&1 \
    && rm -rf /var/lib/apt/lists/*

# --extra-index-url keeps PyPI as the primary index (so build deps like flit_core
# are resolvable) while adding the PyTorch CPU wheel index for the torch wheel itself.
# The +cpu local version tag forces selection of the CPU-only wheel over PyPI's CUDA build.
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch==2.5.1+cpu"

# -U (upgrade) removed from this step: it could cause pip to pull a CUDA-enabled
# torch from the default PyPI index (overriding the CPU torch above).
# torch is intentionally not listed here -- it is already pinned in the step above.
RUN pip install --no-cache-dir \
    facemap==1.0.8 \
    av==13.1.0 \
    opencv-python-headless==4.9.0.80 \
    numpy==1.26.4 \
    scipy==1.14.1 \
    pandas==2.2.3 \
    matplotlib==3.9.2 \
    numba==0.60.0 \
    natsort==8.4.0 \
    scikit-learn==1.5.2 \
    h5py==3.12.1 \
    tables==3.10.1 \
    tqdm==4.66.5 \
    aind-data-schema==1.2.0 \
    aind-qcportal-schema==0.2.3 \
    npc-session==0.1.32

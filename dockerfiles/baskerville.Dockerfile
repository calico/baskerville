# GPU image for gcprunner (--backend gcp). Build and push it to your own
# registry: see docs/gcprunner.md §8.2. Nothing here needs replacing; you may
# change the base image (keep torch within pyproject's range, and CUDA within
# your GPU driver's support) or pass --build-arg UV_INDEX_URL=<mirror>.
# Baskerville's Triton kernels need no nvcc.
FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime

RUN apt-get -y update && \
    # gcprunner needs gcsfuse/gcloud; sorted-nearest needs build-essential.
    apt-get install -y --no-install-recommends \
        python3-pip python3-dev build-essential git bedtools \
        curl gnupg fuse ca-certificates patch && \
    # Shared keyring for the Google repositories.
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    GCSFUSE_REPO=gcsfuse-$(. /etc/os-release && echo $VERSION_CODENAME) && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" \
        > /etc/apt/sources.list.d/gcsfuse.list && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
        > /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends gcsfuse google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

# Optional package mirror; unset means PyPI.
ARG UV_INDEX_URL
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Keep downloaded wheels out of the image layers.
ENV UV_NO_CACHE=1

# Allow pip and uv to install alongside torch in the system Python (PEP 668).
RUN rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED && \
    python3 -m pip install --upgrade pip uv

COPY . /workspace/baskerville

RUN uv pip install --system Cython
# False markers exclude dependencies: preserve the base torch/triton and omit
# quack-kernels (unused CUTE decoding). Build Mamba against the installed torch,
# skipping its Mamba-1 CUDA extension; the patch below makes its import optional.
RUN printf '%s\n' \
        'torch; sys_platform == "never"' \
        'triton; sys_platform == "never"' \
        'quack-kernels; sys_platform == "never"' \
        > /tmp/overrides.txt && \
    MAMBA_SKIP_CUDA_BUILD=TRUE \
    uv pip install --system -e /workspace/baskerville[cuda] \
        --no-build-isolation --override /tmp/overrides.txt && \
    # Check imports and log installed versions.
    python3 -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__)"

# Locate Mamba without importing it until patched. Retire patches as the pin
# advances to releases containing the fixes.
RUN SP=$(python3 -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec('mamba_ssm').submodule_search_locations[0]))") && \
    for p in /workspace/baskerville/dockerfiles/patches/*.patch; do \
        patch -p1 --forward --fuzz=0 -d "$SP" < "$p"; \
    done && \
    python3 -c "import mamba_ssm; print(mamba_ssm.__version__)"

RUN install -m 0755 /workspace/baskerville/src/gcprunner/entry.sh \
    /usr/local/bin/gcprunner-entry

WORKDIR /workspace

CMD ["/bin/bash"]

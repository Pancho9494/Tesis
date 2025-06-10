#syntax=docker/dockerfile:1.7-labs
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder
RUN useradd -m appuser

WORKDIR /home/appuser/LIM
COPY src/ /home/appuser/LIM/src/
COPY src/submodules/IAE /home/appuser/LIM/src/submodules/IAE
COPY pyproject.toml /home/appuser/LIM/pyproject.toml
USER root
RUN chown -R appuser:appuser /home/appuser && \
  apt-get update -o Acquire::Check-Valid-Until=false -o Acquire::AllowInsecureRepositories=true && \
  apt-get install --no-install-recommends -y \
  ca-certificates=20240203~22.04.1\
  libegl1=1.4.0-1\
  libgl1=1.4.0-1\
  libgomp1=12.3.0-1ubuntu1~22.04\
  build-essential=12.9ubuntu3\
  libx11-6=2:1.7.5-1ubuntu0.3\
  mesa-utils=8.4.0-1ubuntu1\
  libgl1-mesa-glx=23.0.4-0ubuntu1~22.04.1\
  libglib2.0-0=2.72.4-0ubuntu2.5\
  wget=1.21.2-2ubuntu1.1 && \
  apt-get clean && rm -rf /var/lib/apt/lists/* && \
  update-ca-certificates

USER appuser
RUN bash -c "set -euo pipefail; wget -q https://astral.sh/uv/install.sh; bash install.sh; rm install.sh"
ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN uv sync && uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

# Compile cpp libraries
WORKDIR /home/appuser/LIM/src/LIM/cpp/
RUN bash -c "set -euo pipefail; bash compile.sh -j$(nproc)"
WORKDIR /home/appuser/LIM/src/submodules/IAE/src/encoder/pointnet2
ARG TORCH_CUDA_ARCH_LIST
RUN TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CXXFLAGS="-DAT_CHECK=TORCH_CHECK" \
    bash -c "set -euo pipefail; uv run python setup.py install"
WORKDIR /home/appuser/LIM/

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS production
RUN useradd -m appuser
WORKDIR /home/appuser/LIM
RUN apt-get update && apt-get install --no-install-recommends -y \
  libegl1\
  libgl1\
  libgomp1\
  libx11-6\
  mesa-utils\
  libgl1-mesa-glx\
  libglib2.0-0 && \
  rm -rf /var/lib/apt/lists/**

ENV PATH="/home/appuser/.local/bin:${PATH}"
USER appuser
COPY --from=builder --chown=appuser /home/appuser/LIM/src/ /home/appuser/LIM/src
COPY --from=builder --chown=appuser /home/appuser/.local /home/appuser/.local
COPY --from=builder --chown=appuser /home/appuser/LIM/.venv .venv
RUN uv aim up

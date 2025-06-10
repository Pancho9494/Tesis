#syntax=docker/dockerfile:1.7-labs
FROM bitnami/minideb:bookworm AS builder
RUN useradd -m appuser

WORKDIR /home/appuser/LIM
COPY src/ /home/appuser/LIM/src/
COPY src/submodules/IAE /home/appuser/LIM/src/submodules/IAE
COPY pyproject.toml /home/appuser/LIM/pyproject.toml
USER root
RUN chown -R appuser:appuser /home/appuser && \
  apt-get update -o Acquire::Check-Valid-Until=false -o Acquire::AllowInsecureRepositories=true && \
  apt-get install --no-install-recommends -y \
  ca-certificates=20230311 \
  libegl1=1.6.0-1 \
  libgl1=1.6.0-1 \
  libgomp1=12.2.0-14+deb12u1 \
  build-essential=12.9 \
  libx11-6=2:1.8.4-2+deb12u2 \
  mesa-utils=8.5.0-1 \
  libgl1-mesa-glx=22.3.6-1+deb12u1 \
  libglib2.0-0=2.74.6-2+deb12u6 \
  wget=1.21.3-1+deb12u1 && \
  apt-get clean && rm -rf /var/lib/apt/lists/* && \
  update-ca-certificates

USER appuser
RUN bash -c "set -euo pipefail; wget -q https://astral.sh/uv/install.sh; bash install.sh; rm install.sh"
ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN uv sync && uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
WORKDIR /home/appuser/LIM/src/LIM/cpp/
RUN bash -c "set -euo pipefail; bash compile.sh"
WORKDIR /home/appuser/LIM/

FROM bitnami/minideb:bookworm AS production 
RUN useradd -m appuser
WORKDIR /home/appuser/LIM
RUN apt-get update && apt-get install --no-install-recommends -y \
  libegl1=1.6.0-1 \
  libgl1=1.6.0-1 \
  libgomp1=12.2.0-14+deb12u1 \
  build-essential=12.9 \
  libx11-6=2:1.8.4-2+deb12u2 \
  mesa-utils=8.5.0-1 \
  libgl1-mesa-glx=22.3.6-1+deb12u1 \
  libglib2.0-0=2.74.6-2+deb12u6 && \
  rm -rf /var/lib/apt/lists/**

ENV PATH="/home/appuser/.local/bin:${PATH}"
USER appuser
COPY --from=builder --chown=appuser /home/appuser/LIM/src/ /home/appuser/LIM/src
COPY --from=builder --chown=appuser /home/appuser/.local /home/appuser/.local
COPY --from=builder --chown=appuser /home/appuser/LIM/.venv .venv

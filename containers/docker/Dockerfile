FROM bitnami/minideb:bookworm AS builder
RUN useradd -m appuser

WORKDIR /home/appuser/LIM
COPY src/ /home/appuser/LIM/src/
COPY pyproject.toml /home/appuser/LIM/pyproject.toml
RUN chown -R appuser:appuser /home/appuser
USER root
RUN apt-get update -o Acquire::Check-Valid-Until=false -o Acquire::AllowInsecureRepositories=true && \
    apt-get install --no-install-recommends -y \
      ca-certificates libegl1 libgl1 libgomp1 build-essential \
      libx11-6 mesa-utils libgl1-mesa-glx libglib2.0-0 wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    update-ca-certificates

USER appuser
RUN wget -qO- https://astral.sh/uv/install.sh | sh
ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN uv sync
RUN uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
RUN cd ./src/LIM/cpp && bash compile.sh && echo "compile.sh exit code = $?" && cd ../../../

FROM bitnami/minideb:bookworm AS production 
RUN useradd -m appuser
WORKDIR /home/appuser/LIM
# RUN rm -rf /home/appuser/LIM/src/LIM/cpp/neighbors/build /home/appuser/LIM/src/LIM/cpp/subsampling/build
RUN apt-get update && apt-get install --no-install-recommends -y \
	libegl1 libgl1 libgomp1 build-essential libx11-6 mesa-utils libgl1-mesa-glx libglib2.0-0 
ENV PATH="/home/appuser/.local/bin:${PATH}"
COPY .aim/ /home/appuser/LIM/.aim/
COPY --from=builder /home/appuser/LIM/src/ /home/appuser/LIM/src
COPY --from=builder /home/appuser/.local /home/appuser/.local
COPY --from=builder /home/appuser/LIM/.venv .venv
USER appuser
CMD ["uv", "run", "./src/main.py"]

#!/bin/bash

FORCE_DOCKER_REBUILD=false
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --force-docker-rebuild)
      FORCE_DOCKER_REBUILD=true
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--force-docker-rebuild]"
      exit 1
      ;;
  esac
  shift
done

# Does the docker image already exist?
if $FORCE_DOCKER_REBUILD || [ -z "$(docker images -q lim/cuda12.4-cudnn9:latest 2>/dev/null)" ]; then
  echo "Docker image lim/cuda12.4-cudnn9:latest not found, building from scratch..."
  docker build \
    --build-arg TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr '\n' ';' | sed 's/;$//')+PTX" \
    -f containers/docker/Dockerfile \
    -t lim/cuda12.4-cudnn9:latest .
fi

apptainer build --fakeroot containers/apptainer/iae.sif containers/apptainer/iae.def

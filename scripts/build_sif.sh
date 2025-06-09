#!/usr/bin/bash

# Does the docker image already exist?
if [ -z "$(docker images -q lim/cuda12.4-cudnn9:latest 2>/dev/null)" ]; then
  echo "Docker image lim/cuda12.4-cudnn9:latest not found, building from scratch..."
  docker build -f containers/docker/Dockerfile -t lim/cuda12.4-cudnn9:latest . --no-cache
fi

apptainer build --fakeroot containers/apptainer/iae.sif containers/apptainer/iae.def

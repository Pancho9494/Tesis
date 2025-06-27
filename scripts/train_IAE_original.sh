#!/bin/bash
echo "Launching train_PREDATOR_original"
export CURRENT_USER=$(whoami)
docker run \
  --gpus all \
  -v ./src/LIM/data/raw/scannet/:/home/appuser/IAE/data/scannet \
  -it iae/cuda10.1-cudnn7:latest
echo "Finished train_PREDATOR_recreation"

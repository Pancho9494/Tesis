#!/bin/bash
docker run -it \
  --gpus all \
  --ipc=host \
  --network=host \
  --cap-add=SYS_PTRACE \
  --security-opt=seccomp=unconfined \
  --shm-size=8g \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $HOME/.Xauthority:/home/appuser/.Xauthority:ro \
  -e DISPLAY=$DISPLAY \
  -e XAUTHORITY=/home/appuser/.Xauthority \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v ./src/LIM/data/:/home/appuser/LIM/src/LIM/data \
  -v ./.aim/:/home/appuser/LIM/.aim \
  -v ./src/clean_scannet.py:/home/appuser/LIM/src/clean_scannet.py \
  -v ./src/LIM/models/layers/kpconv.py:/home/appuser/LIM/src/LIM/models/layers/kpconv.py \
  -v ./src/LIM/models/IAE/trainer.py:/home/appuser/LIM/src/LIM/models/IAE/trainer.py \
  -v ./src/config/:/home/appuser/LIM/src/config/ \
  -v ./src/LIM/training/trainer.py:/home/appuser/LIM/src/LIM/training/trainer.py \
  -v ./src/LIM/training/backups/:/home/appuser/LIM/src/LIM/training/backups \
  -v ./src/LIM/models/PREDATOR/:/home/appuser/LIM/src/LIM/models/PREDATOR/ \
  -v ./src/main.py:/home/appuser/LIM/src/main.py \
  -v ./src/LIM/training/run_state.py:/home/appuser/LIM/src/LIM/training/run_state.py \
  -v ./src/LIM/metrics/metrics.py:/home/appuser/LIM/src/LIM/metrics/metrics.py \
  -v ./run_freeze_sequence.sh:/home/appuser/LIM/run_freeze_sequence.sh \
  lim/cuda12.4-cudnn9:leftraru

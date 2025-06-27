#!/usr/bin/bash
#SBATCH -J train_PREDATOR_original
#SBATCH -p v100
#SBATCH -n 4
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu 12000
#SBATCH -o train_IAE_%j.err.out
#SBATCH -e train_IAE_%j.err.out
#SBATCH --mail-user=fmolinaleiv@gmail.com
#SBATCH --mail-type=ALL

# ml apptainer/1.3.6-zen4-i
#
#docker run -it --rm --ipc=host --gpus all -v ./src/LIM/data/raw/:/home/appuser/OverlapPredator/data -v ./src/submodules/OverlapPredator/snapshot:/home/appuser/OverlapPredator/snapshot predator/cuda11.0-cudnn8:latest

echo "Launching train_PREDATOR_original"
export CURRENT_USER=$(whoami)
cd /home/$CURRENT_USER/code/LIM &&
  apptainer run --fakeroot --nv \
    --pwd /home/appuser/LIM \
    --bind /home/$CURRENT_USER/code/LIM/src/LIM/data:/home/appuser/LIM/src/LIM/data \
    --bind /home/$CURRENT_USER/code/LIM/src/LIM/training/backups/:/home/appuser/LIM/src/LIM/training/backups \
    --bind /home/$CURRENT_USER/code/LIM/.aim/:/home/appuser/LIM/.aim/ \
    containers/apptainer/trainer.sif \
    ./src/config/PREDATOR_recreation.yaml
echo "Finished train_PREDATOR_recreation"

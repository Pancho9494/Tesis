#!/bin/bash
#SBATCH -J train_IAE_recreation
#SBATCH -p v100
#SBATCH -n 2
#SBATCH -c 2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu 2768
#SBATCH -o train_IAE_%j.err.out
#SBATCH -e train_IAE_%j.err.out
#SBATCH --mail-user=fmolinaleiv@gmail.com
#SBATCH --mail-type=ALL

ml apptainer/1.3.6-zen4-i

echo "Launching train_IAE_recreation"
cd /home/$(whoami)/code/LIM &&
  apptainer exec --fakeroot --nv \
    --pwd /home/appuser/LIM \
    --bind /home/$(whoami)/code/LIM/src/config:/home/appuser/LIM/src/config \
    --bind /home/$(whoami)/code/LIM/src/LIM/data:/home/appuser/LIM/src/LIM/data \
    --bind /home/$(whoami)/code/LIM/src/LIM/training/backups/:/home/appuser/LIM/src/LIM/training/backups \
    --bind /home/$(whoami)/code/LIM/.aim/:/home/appuser/LIM/.aim/ \
    containers/apptainer/trainer.sif \
    torchrun \
    --nproc_per_node=22 --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    src/main.py ./src/config/IAE_recreation.yaml
echo "Finished train_IAE_recreation"

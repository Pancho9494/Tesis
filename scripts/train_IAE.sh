docker build -f containers/docker/Dockerfile -t lim/cuda12.4-cudnn9:latest . --no-cache
docker run -it --rm -v ./src/LIM/data/:/home/appuser/LIM/src/LIM/data lim/cuda12.cudnn9

#!/usr/bin/bash
# SBATCH -J train_IAE
# SBATCH -p v100
# SBATCH -n 4
# SBATCH -c 4
# SBATCH --gres=gpu:1
# SBATCH --mem-per-cpu 12000
# SBATCH -o train_IAE_%j.err.out
# SBATCH -e train_IAE_%j.err.out
# SBATCH --mail-user=fmolinaleiv@gmail.com
# SBATCH --mail-type=ALL

ml apptainer/1.3.6-zen4-i

echo "Launching train_IAE"
cd /home/fmolinal/code/Tesis && apptainer run --fakeroot --pwd /home/appuser/LIM --bind /home/fmolinal/code/Tesis/src/LIM/data:/home/appuser/LIM/src/LIM/data containers/apptainer/iae.sif
echo "Finished train_IAE"

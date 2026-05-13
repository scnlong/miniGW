#!/bin/bash -x
#SBATCH --job-name=minigw-cosma
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:2

set -euo pipefail

module --force purge
module use "$OTHERSTAGES"
module load Stages/2025 GCC/13.3.0 OpenMPI/5.0.5 ScaLAPACK/2.2.0-fb CMake/3.29.3 CUDA/12

cd /e/home/jusers/liu21/jupiter/software/miniGW/regression_tests/h2o_cosma

export COSMA_ROOT="/e/home/jusers/liu21/jupiter/software/COSMA/install"

srun ../../build/gw --linalg-backend cosma --frequency-parallel mpi > slurm.log

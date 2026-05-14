#!/bin/bash -x
#SBATCH --job-name=minigw-regression
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:2

set -euo pipefail

echo "[$(date)] job started"
echo "[$(date)] host: $(hostname)"
echo "[$(date)] SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "[$(date)] SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unset}"
echo "[$(date)] SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}"

module --force purge
module use "$OTHERSTAGES"
module load Stages/2025 GCC/13.3.0 OpenMPI/5.0.5 ScaLAPACK/2.2.0-fb CMake/3.29.3 CUDA/12

cd /e/home/jusers/liu21/jupiter/software/miniGW

export COSMA_ROOT="/e/home/jusers/liu21/jupiter/software/COSMA/install"

rm -rf build

cmake -B build -C cmake_install.cmake \
  -DGW_TEST_MPI_LAUNCHER=srun \
  -DGW_TEST_MPI_RANKS=4 \
  -DGW_TEST_MPI_NUMPROC_FLAG=-n

cmake --build build -j 4

cd build

export OMP_PROC_BIND=close
export OMP_PLACES=cores

: > ../slurm.log

echo "[$(date)] build finished" | tee -a ../slurm.log

echo " " | tee -a ../slurm.log
echo "== serial CPU ==" | tee -a ../slurm.log
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
ctest -L "serial" -LE "cuda|lapack|mpi|scalapack|cosma" -j1 --output-on-failure | tee -a ../slurm.log

echo " " | tee -a ../slurm.log
echo "== BLAS/LAPACK serial ==" | tee -a ../slurm.log
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
ctest -L "lapack"  -LE "scalapack" -j1 --output-on-failure | tee -a ../slurm.log

echo " " | tee -a ../slurm.log
echo "== MPI CPU ==" | tee -a ../slurm.log
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
ctest -L "mpi" -LE "cuda|scalapack|cosma" -j1 --output-on-failure | tee -a ../slurm.log

echo " " | tee -a ../slurm.log
echo "== SCALAPACK CPU ==" | tee -a ../slurm.log
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
ctest -L "mpi" -L "scalapack" -LE "cuda|cosma" -j1 --output-on-failure | tee -a ../slurm.log

echo " " | tee -a ../slurm.log
echo "== CUDA serial ==" | tee -a ../slurm.log
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
ctest -L "cuda" -LE "mpi" -j1 --output-on-failure | tee -a ../slurm.log

echo " " | tee -a ../slurm.log
echo "== CUDA MPI ==" | tee -a ../slurm.log
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
ctest -L "cuda" -L "mpi" -LE "cosma" -j1 --output-on-failure | tee -a ../slurm.log

echo " " | tee -a ../slurm.log
echo "== CUDA COSMA ==" | tee -a ../slurm.log
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
ctest -L "cuda" -L "mpi" -L "cosma" -j1 --output-on-failure | tee -a ../slurm.log

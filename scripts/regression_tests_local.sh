#!/usr/bin/env bash
set -euo pipefail

rm -rf build

cmake -B build -C cmake_install.cmake \
  -DGW_TEST_MPI_LAUNCHER=mpirun \
  -DGW_TEST_MPI_RANKS="${GW_TEST_MPI_RANKS:-4}" \
  -DGW_TEST_MPI_NUMPROC_FLAG=-np

cmake --build build -j 4

cd build 

: > ../regression_tests.log

echo "[$(date)] build finished" | tee -a ../regression_tests.log

echo " " | tee -a ../regression_tests.log
echo "== serial CPU ==" | tee -a ../regression_tests.log
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
ctest -L "serial" -LE "cuda|lapack|mpi|scalapack|cosma" -j1 --output-on-failure | tee -a ../regression_tests.log

echo " " | tee -a ../regression_tests.log
echo "== BLAS/LAPACK serial ==" | tee -a ../regression_tests.log
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
ctest -L "lapack"  -LE "scalapack" -j1 --output-on-failure | tee -a ../regression_tests.log

echo " " | tee -a ../regression_tests.log
echo "== MPI CPU ==" | tee -a ../regression_tests.log
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
ctest -L "mpi" -LE "cuda|scalapack|cosma" -j1 --output-on-failure | tee -a ../regression_tests.log

echo " " | tee -a ../regression_tests.log
echo "== SCALAPACK CPU ==" | tee -a ../regression_tests.log
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
ctest -L "mpi" -L "scalapack" -LE "cuda|cosma" -j1 --output-on-failure | tee -a ../regression_tests.log

#!/bin/bash
# Run FDS GPU simulation

set -e

echo "=============================================="
echo " Running FDS GPU Simulation with AmgX"
echo "=============================================="

# Set up environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/mnt/c/Users/ji/Documents/amgx/AMGX/build:$LD_LIBRARY_PATH

# Print GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Go to test directory
cd /mnt/c/Users/ji/Documents/amgx/test

# Clean previous results
rm -f simple_test.out simple_test.smv simple_test*.s3d* simple_test*.sf* simple_test*.csv simple_test_git.txt 2>/dev/null || true

# Run simulation
echo "Starting GPU-accelerated FDS simulation..."
echo "Input file: simple_test.fds with SOLVER='GPU'"
echo ""

# Run with MPI (single process for now)
mpirun --allow-run-as-root -np 1 /mnt/c/Users/ji/Documents/amgx/Build_WSL/fds_ompi_gnu_linux_amgx simple_test.fds

echo ""
echo "=============================================="
echo " Simulation Complete!"
echo "=============================================="

#!/bin/bash
# Build FDS with AmgX in WSL2

set -e

echo "=============================================="
echo " Building FDS with AmgX GPU Support in WSL2"
echo "=============================================="

# Set up environment
export AMGX_HOME=/mnt/c/Users/ji/Documents/amgx/AMGX/build
export CUDA_HOME=/usr/local/cuda
export COMP_FC=mpifort
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Print environment
echo ""
echo "Environment:"
echo "  AMGX_HOME = $AMGX_HOME"
echo "  CUDA_HOME = $CUDA_HOME"
echo "  COMP_FC   = $COMP_FC"
echo ""

# Check dependencies
echo "Checking dependencies..."
which mpifort
which nvcc
ls $AMGX_HOME/libamgx.a
echo "Dependencies OK"
echo ""

# Create and enter build directory
cd /mnt/c/Users/ji/Documents/amgx
mkdir -p Build_WSL
cd Build_WSL

# Clean previous build
rm -f *.o *.mod fds_* 2>/dev/null || true

# Run make
echo "Starting build..."
make -f ../Source/makefile_amgx ompi_gnu_linux_amgx

echo ""
echo "=============================================="
echo " Build Complete!"
echo "=============================================="
ls -la fds_*

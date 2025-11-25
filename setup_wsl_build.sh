#!/bin/bash
# ============================================================================
# WSL2 Environment Setup for FDS + AmgX GPU Build
# ============================================================================
#
# This script installs necessary packages for building GPU-accelerated FDS
# Run with: sudo bash setup_wsl_build.sh
#
# ============================================================================

set -e

echo "=============================================="
echo " WSL2 FDS + AmgX Build Environment Setup"
echo "=============================================="
echo

# Update package list
echo "Updating package list..."
apt-get update

# Install build essentials
echo "Installing build essentials..."
apt-get install -y build-essential gfortran

# Install OpenMPI
echo "Installing OpenMPI..."
apt-get install -y openmpi-bin libopenmpi-dev

# Install CUDA Toolkit (WSL2 version)
echo "Installing CUDA Toolkit for WSL2..."

# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
apt-get update

# Install CUDA toolkit (without driver - WSL uses Windows driver)
apt-get install -y cuda-toolkit-12-6

# Set up environment variables
echo "Setting up environment variables..."
cat >> /etc/profile.d/cuda.sh << 'EOF'
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF

source /etc/profile.d/cuda.sh

echo
echo "=============================================="
echo " Installation Complete!"
echo "=============================================="
echo
echo "Please restart your WSL2 session, then run:"
echo "  source /etc/profile.d/cuda.sh"
echo
echo "To build FDS with AmgX:"
echo "  cd /mnt/c/Users/ji/Documents/amgx"
echo "  export AMGX_HOME=/mnt/c/Users/ji/Documents/amgx/AMGX/build"
echo "  export CUDA_HOME=/usr/local/cuda"
echo "  export COMP_FC=mpifort"
echo "  make -f Source/makefile_amgx ompi_gnu_linux_amgx"
echo

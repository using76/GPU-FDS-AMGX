# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FDS (Fire Dynamics Simulator) with NVIDIA AmgX GPU acceleration for pressure Poisson equation solving. The project integrates Fortran-based FDS with NVIDIA's AmgX GPU solver library through a C wrapper.

## Architecture

```
FDS Simulation (Fortran)
    ↓
Pressure Solver (pres.f90)
    ↓
    ├─→ CPU: MKL PARDISO / HYPRE
    └─→ GPU: AmgX (via amgx_fortran.f90 → amgx_c_wrapper.c → CUDA)
```

### Key Directories
- `FDS_CPU_Source/` - FDS Fortran source with AmgX integration
- `AMGX/` - NVIDIA AmgX GPU solver library
- `Build_WSL/` - WSL2 build output directory
- `test/` - Test cases (simple_test.fds, test_les3.fds)

### Core Files for AmgX Integration
| File | Purpose |
|------|---------|
| `amgx_c_wrapper.c` | C wrapper for NVIDIA AmgX API |
| `amgx_fortran.f90` | Fortran ISO_C_BINDING interface |
| `pres.f90` | Pressure solver (AMGX_FLAG case) |
| `cons.f90` | Constants including `AMGX_FLAG=3`, `FORCE_GPU_SOLVER` |
| `read.f90` | Parses `&PRES SOLVER='GPU'/` option |
| `main.f90` | AmgX init/finalize, GPU monitoring |

## Build Commands

### Environment Setup (WSL2)
```bash
export AMGX_HOME=/mnt/c/Users/ji/Documents/amgx/AMGX/build
export CUDA_HOME=/usr/local/cuda
export COMP_FC=mpifort
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$AMGX_HOME:$LD_LIBRARY_PATH
```

### Build FDS with AmgX
```bash
cd Build_WSL
make -f ../FDS_CPU_Source/makefile_amgx clean
make -f ../FDS_CPU_Source/makefile_amgx ompi_gnu_linux_amgx
```

### Build Targets
- `ompi_gnu_linux_amgx` - OpenMPI + GNU Fortran (recommended for WSL2)
- `ompi_gnu_linux_amgx_db` - Debug build
- `impi_intel_linux_amgx` - Intel MPI + Intel Fortran
- `test_amgx` - Unit tests for AmgX interface

### Rebuild Only C Wrapper (fast iteration)
```bash
gcc -c -O2 -fPIC -I$AMGX_HOME/../include -I$CUDA_HOME/include ../FDS_CPU_Source/amgx_c_wrapper.c
mpifort -O3 -fopenmp -o fds_ompi_gnu_linux_amgx *.o $AMGX_HOME/libamgx.a -L$CUDA_HOME/lib64 -lcudart -lcublas -lcusparse -lcusolver -lstdc++
```

## Running Simulations

### FDS Input File Configuration
```fortran
&PRES SOLVER='GPU'/   ! Enable GPU solver
```

### Run Test
```bash
cd test
mpirun --allow-run-as-root -np 1 ../Build_WSL/fds_ompi_gnu_linux_amgx simple_test.fds
```

### Using run_gpu_sim.sh
```bash
bash run_gpu_sim.sh
```

## Key Implementation Details

### Zone ID Mapping
FDS uses zone IDs like `NM*1000 + IPZ` (e.g., 1001). The C wrapper uses `find_zone_slot()` to map these to internal array indices (0-255).

### Matrix Format
- Input: Upper triangular CSR (1-based indexing, Fortran)
- Converted to: Full symmetric matrix (0-based indexing, C/AmgX)

### Solver Configuration (in amgx_c_wrapper.c)
- Algorithm: FGMRES with AMG preconditioner
- Tolerance: 1e-8, Max iterations: 100
- Smoother: MULTICOLOR_DILU, V-cycle

### Compilation Flags
- `-DWITH_AMGX` - Enable AmgX GPU solver
- `-DWITH_MKL` - Enable Intel MKL PARDISO (CPU)
- `-DWITH_NVML` - Enable GPU monitoring

## Performance Notes

| Mesh Size | GPU Benefit |
|-----------|-------------|
| < 50K cells | Slower (overhead dominates) |
| 50K-500K | 2-5x speedup |
| > 500K | 5-20x speedup |

For small meshes (32x32x32), CPU FFT solver is faster. Use GPU for large problems.

## Documentation

See `FDS_AMGX_MODIFICATION_PLAN.md` for detailed implementation guide including:
- Line-by-line modification instructions
- Data structure mapping
- Troubleshooting guide

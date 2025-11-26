# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FDS (Fire Dynamics Simulator) with NVIDIA GPU acceleration. The project integrates Fortran-based FDS with two GPU acceleration layers:
1. **AmgX** - GPU-accelerated pressure Poisson solver
2. **CUDA Kernels** - GPU-accelerated advection, diffusion, and velocity flux computations

## Architecture

```
FDS Simulation (Fortran)
    ↓
┌───────────────────────────────────────────────────────────┐
│  Pressure Solver (pres.f90)                               │
│    ├─→ CPU: MKL PARDISO / HYPRE                           │
│    └─→ GPU: AmgX (amgx_fortran.f90 → amgx_c_wrapper.c)    │
├───────────────────────────────────────────────────────────┤
│  Physics Kernels (velo.f90, divg.f90, mass.f90)           │
│    ├─→ CPU: Fortran loops                                 │
│    └─→ GPU: CUDA (gpu_fortran.f90 → gpu_c_wrapper.c       │
│              → gpu_kernels.cu)                            │
└───────────────────────────────────────────────────────────┘
```

### Key Directories
- `FDS_CPU_Source/` - FDS Fortran source with GPU integration
- `Build_WSL/` - WSL2 build output directory
- `test/` - Test cases (simple_test.fds)
- `docs/` - Technical documentation (Korean)

### Core Integration Files

| File | Purpose |
|------|---------|
| **AmgX Pressure Solver** | |
| `amgx_c_wrapper.c` | C wrapper for NVIDIA AmgX API, zone management |
| `amgx_fortran.f90` | Fortran ISO_C_BINDING interface for AmgX |
| **GPU Compute Kernels** | |
| `gpu_kernels.cu` | CUDA kernels for advection, diffusion, velocity flux |
| `gpu_c_wrapper.c` | C wrapper with persistent GPU memory management |
| `gpu_fortran.f90` | Fortran interface for GPU kernels |
| `gpu_data_manager.c` | GPU memory pool and mesh data management |
| `gpu_data_fortran.f90` | Fortran interface for GPU data manager |
| **Modified FDS Files** | |
| `pres.f90` | Pressure solver (AMGX_FLAG case) |
| `velo.f90` | Velocity computation (GPU kernel calls) |
| `divg.f90` | Divergence computation (GPU kernel calls) |
| `mass.f90` | Mass/density update (GPU kernel calls) |
| `cons.f90` | Constants: `AMGX_FLAG=3`, `FORCE_GPU_SOLVER`, GPU kernel flags |
| `read.f90` | Parses `&PRES SOLVER='GPU'/` option |
| `main.f90` | GPU init/finalize, cache invalidation |

## Build Commands

### Environment Setup (WSL2)
```bash
export AMGX_HOME=/path/to/AMGX/build
export CUDA_HOME=/usr/local/cuda
export COMP_FC=mpifort
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$AMGX_HOME:$LD_LIBRARY_PATH
```

### Build Targets
```bash
cd Build_WSL
make -f ../FDS_CPU_Source/makefile_amgx clean

# AmgX pressure solver only
make -f ../FDS_CPU_Source/makefile_amgx ompi_gnu_linux_amgx

# Full GPU (AmgX + CUDA kernels for advection/diffusion)
make -f ../FDS_CPU_Source/makefile_amgx ompi_gnu_linux_full_gpu

# Debug builds (add _db suffix)
make -f ../FDS_CPU_Source/makefile_amgx ompi_gnu_linux_full_gpu_db
```

### Quick Rebuild (C wrapper only)
```bash
gcc -c -O2 -fPIC -I$AMGX_HOME/../include -I$CUDA_HOME/include ../FDS_CPU_Source/amgx_c_wrapper.c
mpifort -O3 -fopenmp -o fds_ompi_gnu_linux_amgx *.o $AMGX_HOME/libamgx.a -L$CUDA_HOME/lib64 -lcudart -lcublas -lcusparse -lcusolver -lstdc++
```

## Running Simulations

### FDS Input File
```fortran
&PRES SOLVER='GPU'/   ! Enable GPU solver
```

### Run Test
```bash
cd test
mpirun --allow-run-as-root -np 1 ../Build_WSL/fds_ompi_gnu_linux_amgx simple_test.fds
```

## Key Implementation Details

### Zone ID Mapping
FDS uses zone IDs like `NM*1000 + IPZ` (e.g., 1001). The C wrapper uses `find_zone_slot()` to map these to internal array indices (0-255).

### Matrix Format Conversion
- Input: Upper triangular CSR (1-based indexing, Fortran)
- Output: Full symmetric matrix (0-based indexing, C/AmgX)

### GPU Kernel Persistent Memory
The `GPUKernelContext` structure in `gpu_c_wrapper.c` maintains persistent GPU arrays:
- Velocity arrays: `d_UU`, `d_VV`, `d_WW`
- Thermodynamic: `d_RHOP`, `d_MU`, `d_DP`, `d_TMP`, `d_KP`
- Pinned memory buffers for fast DMA transfers
- Cache flags to avoid redundant uploads within a timestep

### Compilation Flags
- `-DWITH_AMGX` - Enable AmgX GPU pressure solver
- `-DWITH_GPU_KERNELS` - Enable CUDA compute kernels
- `-DWITH_MKL` - Enable Intel MKL PARDISO (CPU fallback)
- `-DWITH_NVML` - Enable GPU monitoring

### Solver Configuration (amgx_c_wrapper.c)
- Algorithm: FGMRES with AMG preconditioner
- Tolerance: 1e-8, Max iterations: 100
- Smoother: MULTICOLOR_DILU, V-cycle

## Performance Notes

| Mesh Size | GPU Benefit |
|-----------|-------------|
| < 50K cells | Slower (overhead dominates) |
| 50K-500K | 2-5x speedup |
| > 500K | 5-20x speedup |

For small meshes (32x32x32), CPU FFT solver is faster. Use GPU for large problems.

## Documentation

- `docs/GPU_FDS_Technical_Documentation.md` - Detailed mathematical background and implementation (Korean)
- `FDS_AMGX_MODIFICATION_PLAN.md` - Line-by-line modification guide

# FDS-AmgX: GPU-Accelerated Fire Dynamics Simulator

GPU-accelerated pressure Poisson solver for [Fire Dynamics Simulator (FDS)](https://github.com/firemodels/fds) using NVIDIA AmgX library.

## Overview

This project integrates NVIDIA's [AmgX](https://github.com/NVIDIA/AMGX) GPU solver library into FDS to accelerate the pressure Poisson equation solving, which is one of the most computationally intensive parts of CFD fire simulations.

### Key Features

- **GPU-Accelerated Pressure Solver**: Replaces CPU-based iterative solvers with NVIDIA AmgX
- **FGMRES + AMG**: Flexible GMRES solver with Algebraic Multigrid preconditioner
- **Seamless Integration**: Simple `&PRES SOLVER='GPU'/` configuration in FDS input files
- **Multi-Mesh Support**: Dynamic zone ID mapping for complex FDS configurations
- **Drop-in Replacement**: No changes required to existing FDS input files (except solver option)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FDS Simulation (Fortran)                 │
│                         main.f90                            │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 Pressure Solver (pres.f90)                  │
│                                                             │
│    ┌─────────────┐                    ┌─────────────┐       │
│    │  CPU Path   │                    │  GPU Path   │       │
│    │ (FFT/ULMAT) │                    │   (AmgX)    │       │
│    └─────────────┘                    └──────┬──────┘       │
└──────────────────────────────────────────────┼──────────────┘
                                               │
                          ┌────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Fortran Interface (amgx_fortran.f90)           │
│                      ISO_C_BINDING                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                C Wrapper (amgx_c_wrapper.c)                 │
│           Zone Management, Matrix Conversion                │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    NVIDIA AmgX Library                      │
│              FGMRES Solver + AMG Preconditioner             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      NVIDIA GPU (CUDA)                      │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

### Hardware
- NVIDIA GPU with Compute Capability 6.0+ (Pascal or newer)
- Recommended: 8GB+ VRAM for large simulations

### Software
- CUDA Toolkit 11.0+
- NVIDIA AmgX library
- OpenMPI or Intel MPI
- GCC/GFortran 9+ or Intel Fortran Compiler
- CMake 3.18+

### Tested Environment
- WSL2 Ubuntu 22.04
- CUDA 12.x
- NVIDIA RTX 3080/4090
- GCC 11, GFortran 11
- OpenMPI 4.1

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/fds-amgx.git
cd fds-amgx
```

### 2. Build NVIDIA AmgX

```bash
cd AMGX
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89"
make -j$(nproc)
cd ../..
```

### 3. Set Environment Variables

```bash
export AMGX_HOME=$(pwd)/AMGX/build
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$AMGX_HOME:$LD_LIBRARY_PATH
```

### 4. Build FDS with AmgX

```bash
mkdir -p Build_WSL && cd Build_WSL
make -f ../FDS_CPU_Source/makefile_amgx ompi_gnu_linux_amgx
```

## Usage

### Enable GPU Solver in FDS Input File

Add the following line to your `.fds` input file:

```fortran
&PRES SOLVER='GPU'/
```

### Run Simulation

```bash
mpirun -np 1 ./fds_ompi_gnu_linux_amgx your_simulation.fds
```

Or use the provided script:

```bash
bash run_gpu_sim.sh
```

### Example Input File

```fortran
&HEAD CHID='fire_test', TITLE='GPU Accelerated Fire Simulation'/
&TIME T_END=60.0/
&PRES SOLVER='GPU'/

&MESH IJK=64,64,64, XB=0.0,2.0,0.0,2.0,0.0,2.0/

&REAC FUEL='PROPANE', SOOT_YIELD=0.01/
&SURF ID='BURNER', HRRPUA=500.0, COLOR='RED'/
&OBST XB=0.8,1.2,0.8,1.2,0.0,0.1, SURF_IDS='BURNER','INERT','INERT'/

&VENT XB=0.0,2.0,0.0,2.0,2.0,2.0, SURF_ID='OPEN'/

&TAIL/
```

## Performance

### Speedup vs Mesh Size

| Mesh Size | Cells | CPU (FFT) | GPU (AmgX) | Speedup |
|-----------|-------|-----------|------------|---------|
| 32³ | 32K | Faster | Slower | <1x |
| 64³ | 262K | Baseline | 2-3x | 2-3x |
| 128³ | 2M | Baseline | 5-10x | 5-10x |
| 256³ | 16M | Baseline | 10-20x | 10-20x |

### When to Use GPU Solver

**Recommended:**
- Large meshes (> 100K cells)
- Multi-mesh simulations
- Complex boundary conditions
- Long-duration simulations

**Not Recommended:**
- Small meshes (< 50K cells)
- Simple geometries where FFT applies
- Memory-constrained GPUs

## Project Structure

```
fds-amgx/
├── FDS_CPU_Source/          # Modified FDS source files
│   ├── amgx_c_wrapper.c     # C wrapper for AmgX API
│   ├── amgx_fortran.f90     # Fortran interface (ISO_C_BINDING)
│   ├── pres.f90             # Pressure solver (modified)
│   ├── cons.f90             # Constants (AMGX_FLAG, FORCE_GPU_SOLVER)
│   ├── read.f90             # Input parser (SOLVER='GPU' option)
│   ├── main.f90             # AmgX initialization/finalization
│   └── makefile_amgx        # Build configuration
├── AMGX/                    # NVIDIA AmgX library
├── Build_WSL/               # Build output directory
├── test/                    # Test cases
│   └── simple_test.fds      # Example input file
├── docs/                    # Documentation
│   └── GPU_FDS_Technical_Documentation.md
├── run_gpu_sim.sh           # Run script
└── README.md
```

## Technical Details

### Pressure Poisson Equation

The pressure correction in FDS solves:

$$\nabla^2 p = \frac{\partial}{\partial t}(\nabla \cdot \mathbf{u}) + \nabla \cdot \mathbf{F}$$

This is discretized using second-order finite differences on a staggered grid, resulting in a sparse linear system **Ax = b**.

### AmgX Solver Configuration

```
Solver: FGMRES (Flexible Generalized Minimal Residual)
Preconditioner: AMG (Algebraic Multigrid)
  - Algorithm: Aggregation
  - Smoother: Multicolor DILU
  - Cycle: V-cycle
Tolerance: 1e-8
Max Iterations: 100
```

### Matrix Format Conversion

- **Input (FDS)**: Upper triangular CSR, 1-based indexing (Fortran)
- **Output (AmgX)**: Full symmetric CSR, 0-based indexing (C)

## Documentation

- [Technical Documentation](docs/GPU_FDS_Technical_Documentation.md) - Detailed mathematical background and implementation
- [CLAUDE.md](CLAUDE.md) - Development guide for AI assistants

## Troubleshooting

### Common Issues

**1. "AmgX: Invalid zone_id"**
- Zone IDs use format `NM*1000 + IPZ`
- Wrapper handles mapping automatically

**2. "CUDA out of memory"**
- Reduce mesh size or use multiple GPUs
- Check `nvidia-smi` for memory usage

**3. "Library not found"**
- Verify `LD_LIBRARY_PATH` includes AmgX and CUDA paths
- Run `ldd fds_ompi_gnu_linux_amgx` to check dependencies

**4. Slow performance on small meshes**
- This is expected; use CPU solver for meshes < 50K cells
- GPU overhead dominates for small problems

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is based on [FDS](https://github.com/firemodels/fds) which is public domain software.
NVIDIA AmgX is licensed under the [BSD 3-Clause License](https://github.com/NVIDIA/AMGX/blob/main/LICENSE).

## Acknowledgments

- [NIST Fire Research Division](https://www.nist.gov/el/fire-research-division-73300) - Original FDS development
- [NVIDIA AmgX Team](https://github.com/NVIDIA/AMGX) - GPU solver library

## Citation

If you use this work in your research, please cite:

```bibtex
@software{fds_amgx,
  title = {FDS-AmgX: GPU-Accelerated Fire Dynamics Simulator},
  year = {2024},
  url = {https://github.com/yourusername/fds-amgx}
}
```

---

**Note**: This is a research project. For production fire simulations, please validate results against standard FDS.

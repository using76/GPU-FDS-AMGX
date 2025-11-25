# FDS AmgX Integration Plan
## GPU-Accelerated Pressure Solver Implementation

---

## 1. Overview

This document describes the modifications required to integrate NVIDIA AmgX into FDS for GPU-accelerated pressure Poisson equation solving.

### Current Architecture (CPU)
```
FDS (Fortran) --> MKL PARDISO (Intel CPU) --> Solution
```

### Target Architecture (GPU)
```
FDS (Fortran) --> AmgX Fortran Module --> C Wrapper --> AmgX (NVIDIA GPU) --> Solution
```

### Expected Performance Improvement
- **5-20x speedup** for pressure solver on large meshes
- Most significant impact on meshes > 100,000 cells

---

## 2. Files to Modify

### 2.1 New Files (Already Created)

| File | Description |
|------|-------------|
| `amgx_c_wrapper.c` | C wrapper for AmgX API |
| `amgx_fortran.f90` | Fortran interface module |

### 2.2 Existing Files to Modify

| File | Lines | Modification Type |
|------|-------|-------------------|
| `pres.f90` | ~5700 | Add AmgX solver calls |
| `cons.f90` | ~550 | Add AmgX flag constant |
| `type.f90` | ~1800 | Add AmgX handles to ZONE_MESH_TYPE |
| `main.f90` | ~4500 | Initialize/finalize AmgX |
| `read.f90` | ~17000 | Parse AmgX configuration |
| `Makefile` | - | Add AmgX compilation |

---

## 3. Detailed Modification Instructions

### 3.1 cons.f90 - Add AmgX Solver Flag

**Location**: After line 535 (ULMAT_SOLVER_LIBRARY definition)

```fortran
! Add after MKL_PARDISO_FLAG and HYPRE_FLAG definitions (around line 532-536):
INTEGER, PARAMETER :: AMGX_FLAG=3                              !< Integer matrix solver library flag for NVIDIA AmgX
```

**Location**: Around line 535

```fortran
! Modify the default solver library selection:
! Original:
! INTEGER :: ULMAT_SOLVER_LIBRARY=MKL_PARDISO_FLAG

! New (with compile flag):
#ifdef WITH_AMGX
INTEGER :: ULMAT_SOLVER_LIBRARY=AMGX_FLAG                      !< Use AmgX by default when available
#else
INTEGER :: ULMAT_SOLVER_LIBRARY=MKL_PARDISO_FLAG               !< Use MKL PARDISO by default
#endif
```

---

### 3.2 type.f90 - Add AmgX Handle to ZONE_MESH_TYPE

**Location**: Lines 1722-1746 (TYPE ZONE_MESH_TYPE definition)

```fortran
TYPE ZONE_MESH_TYPE
#ifdef WITH_MKL
   TYPE(MKL_PARDISO_HANDLE), ALLOCATABLE  :: PT_H(:)  !< Internal solver memory pointer
#else
   INTEGER, ALLOCATABLE :: PT_H(:)
#endif /* WITH_MKL */
#ifdef WITH_HYPRE
   TYPE(HYPRE_ZM_TYPE) HYPRE_ZM
#else
   INTEGER :: HYPRE_ZM
#endif /* WITH_HYPRE */

   ! ===== ADD THIS SECTION FOR AMGX =====
#ifdef WITH_AMGX
   INTEGER :: AMGX_ZONE_ID = 0                        !< AmgX zone identifier for this ZONE_MESH
#endif /* WITH_AMGX */
   ! ===== END AMGX SECTION =====

   INTEGER :: NUNKH=0                                 !< Number of unknowns in pressure solution
   ! ... rest of existing fields ...
END TYPE ZONE_MESH_TYPE
```

---

### 3.3 pres.f90 - Main Pressure Solver Modifications

#### 3.3.1 Add USE statement at module top

**Location**: Around line 1070 (after existing USE statements)

```fortran
#ifdef WITH_MKL
USE MKL_PARDISO
#endif
! ===== ADD THIS =====
#ifdef WITH_AMGX
USE AMGX_FORTRAN
#endif
! ===== END ADD =====
```

#### 3.3.2 Modify ULMAT_SOLVER_SETUP (Setup Phase)

**Location**: Around line 1127-1136 (Library check section)

```fortran
! Original error check for missing libraries:
#ifndef WITH_MKL
#ifndef WITH_HYPRE
IF (MY_RANK==0) WRITE(LU_ERR,'(A)') &
'Error: MKL or HYPRE Library compile flag not defined for ULMAT pressure solver.'
STOP_STATUS = SETUP_STOP
RETURN
#endif
#endif

! ===== MODIFY TO: =====
#ifndef WITH_MKL
#ifndef WITH_HYPRE
#ifndef WITH_AMGX
IF (MY_RANK==0) WRITE(LU_ERR,'(A)') &
'Error: MKL, HYPRE, or AmgX Library compile flag not defined for ULMAT pressure solver.'
STOP_STATUS = SETUP_STOP
RETURN
#endif
#endif
#endif
```

#### 3.3.3 Modify ULMAT_H_MATRIX_SOLVER_SETUP (Matrix Factorization)

**Location**: Around line 2758-2938 (LIBRARY_SELECT section)

Add new case after HYPRE_FLAG case:

```fortran
LIBRARY_SELECT: SELECT CASE(ULMAT_SOLVER_LIBRARY)

CASE(MKL_PARDISO_FLAG) LIBRARY_SELECT
#ifdef WITH_MKL
   ! ... existing MKL PARDISO code ...
#endif /* WITH_MKL */

CASE(HYPRE_FLAG) LIBRARY_SELECT
#ifdef WITH_HYPRE
   ! ... existing HYPRE code ...
#endif /* WITH_HYPRE */

! ===== ADD THIS NEW CASE =====
CASE(AMGX_FLAG) LIBRARY_SELECT
#ifdef WITH_AMGX
   ! Compute unique zone identifier: MESH_NUMBER * 1000 + ZONE_NUMBER
   ZM%AMGX_ZONE_ID = NM * 1000 + IPZ + 1

   ! Setup AmgX zone
   CALL AMGX_SETUP_ZONE(ZM%AMGX_ZONE_ID, ZM%NUNKH, TOT_NNZ_H, ERROR)
   IF (ERROR /= 0) THEN
      IF (MY_RANK==0) WRITE(LU_ERR,'(A,I5)') 'AmgX zone setup error: ', ERROR
      STOP_STATUS = SETUP_STOP
      RETURN
   ENDIF

   ! Build full CSR matrix (AmgX needs full matrix, not just upper triangular)
   ! Allocate temporary arrays for full matrix assembly
   IF (ALLOCATED(ZM%A_H))  DEALLOCATE(ZM%A_H)
   IF (ALLOCATED(ZM%IA_H)) DEALLOCATE(ZM%IA_H)
   IF (ALLOCATED(ZM%JA_H)) DEALLOCATE(ZM%JA_H)
   ALLOCATE ( ZM%A_H(TOT_NNZ_H) , ZM%IA_H(ZM%NUNKH+1) , ZM%JA_H(TOT_NNZ_H) )

   ! Store upper triangular part in CSR format
   INNZ = 0
   DO IROW=1,ZM%NUNKH
      ZM%IA_H(IROW) = INNZ + 1
      DO JCOL=1,NNZ_H_MAT(IROW)
         IF ( JD_H_MAT(JCOL,IROW) < IROW ) CYCLE ! Only upper triangular part
         INNZ = INNZ + 1
         ZM%A_H(INNZ)  =  D_H_MAT(JCOL,IROW)
         ZM%JA_H(INNZ) = JD_H_MAT(JCOL,IROW)
      ENDDO
   ENDDO
   ZM%IA_H(ZM%NUNKH+1) = INNZ + 1

   ! Upload matrix to GPU (C wrapper handles conversion to full matrix)
   CALL AMGX_UPLOAD_MATRIX(ZM%AMGX_ZONE_ID, ZM%NUNKH, INNZ, ZM%IA_H, ZM%JA_H, ZM%A_H, ERROR)
   IF (ERROR /= 0) THEN
      IF (MY_RANK==0) WRITE(LU_ERR,'(A,I5)') 'AmgX matrix upload error: ', ERROR
      STOP_STATUS = SETUP_STOP
      RETURN
   ENDIF

   ! Deallocate temporary arrays
   IF (ALLOCATED(NNZ_H_MAT)) DEALLOCATE(NNZ_H_MAT)
   IF (ALLOCATED(D_H_MAT))   DEALLOCATE(D_H_MAT)
   IF (ALLOCATED(JD_H_MAT))  DEALLOCATE(JD_H_MAT)

   IF (CHECK_POISSON .AND. MY_RANK==0) &
      WRITE(LU_ERR,*) 'AmgX: Matrix uploaded for MESH,ZONE=', NM, IPZ, ZM%NUNKH
#endif /* WITH_AMGX */
! ===== END AMGX CASE =====

END SELECT LIBRARY_SELECT
```

#### 3.3.4 Modify ULMAT_SOLVE_ZONE (Solution Phase)

**Location**: Around line 1690-1710 (Solve the system section)

```fortran
! Solve the system...

LIBRARY_SELECT: SELECT CASE(ULMAT_SOLVER_LIBRARY)

CASE(MKL_PARDISO_FLAG) LIBRARY_SELECT
#ifdef WITH_MKL
   ! ... existing PARDISO solve code (Phase 33) ...
   PHASE    = 33 ! only solving
   CALL PARDISO(ZM%PT_H, MAXFCT, MNUM, ZM%MTYPE, PHASE, ZM%NUNKH, &
                ZM%A_H, ZM%IA_H, ZM%JA_H, PERM, NRHS, IPARM, MSGLVL, ZM%F_H, ZM%X_H, ERROR)
   IF (ERROR /= 0) WRITE(0,*) 'ULMAT_SOLVER: The following ERROR was detected: ', ERROR
#endif

CASE(HYPRE_FLAG) LIBRARY_SELECT
#ifdef WITH_HYPRE
   ! ... existing HYPRE solve code ...
#endif

! ===== ADD THIS NEW CASE =====
CASE(AMGX_FLAG) LIBRARY_SELECT
#ifdef WITH_AMGX
   ! Solve using AmgX GPU solver
   CALL AMGX_SOLVE(ZM%AMGX_ZONE_ID, ZM%NUNKH, ZM%F_H, ZM%X_H, ERROR)
   IF (ERROR /= 0) THEN
      IF (MY_RANK==0) WRITE(LU_ERR,'(A,I5)') 'AmgX solve error: ', ERROR
      ! Optionally get iteration info for debugging
      CALL AMGX_GET_ITERATIONS(ZM%AMGX_ZONE_ID, ITERS, ERROR)
      CALL AMGX_GET_RESIDUAL(ZM%AMGX_ZONE_ID, RESIDUAL, ERROR)
      IF (MY_RANK==0) WRITE(LU_ERR,'(A,I5,A,ES12.5)') '  Iterations: ', ITERS, '  Residual: ', RESIDUAL
   ENDIF
#endif /* WITH_AMGX */
! ===== END AMGX CASE =====

END SELECT LIBRARY_SELECT
```

#### 3.3.5 Add FINISH_ULMAT_SOLVER AmgX Cleanup

**Location**: Around line 2947 (FINISH_ULMAT_SOLVER subroutine)

```fortran
SUBROUTINE FINISH_ULMAT_SOLVER(NM)
#ifdef WITH_HYPRE
USE HYPRE_INTERFACE, ONLY: HYPRE_FINALIZE, HYPRE_IERR
#endif
! ===== ADD THIS =====
#ifdef WITH_AMGX
USE AMGX_FORTRAN
#endif
! ===== END ADD =====

! ... existing code ...

! ===== ADD AmgX cleanup in the cleanup loop =====
#ifdef WITH_AMGX
   IF (ULMAT_SOLVER_LIBRARY == AMGX_FLAG .AND. ZM%AMGX_ZONE_ID > 0) THEN
      CALL AMGX_DESTROY_ZONE(ZM%AMGX_ZONE_ID, ERROR)
   ENDIF
#endif
! ===== END ADD =====
```

---

### 3.4 main.f90 - Initialization and Finalization

#### 3.4.1 Add AmgX Initialization

**Location**: Early in program execution (around CALL INITIALIZE_MESH section)

```fortran
! ===== ADD AMGX INITIALIZATION =====
#ifdef WITH_AMGX
IF (PRES_FLAG == ULMAT_FLAG .AND. ULMAT_SOLVER_LIBRARY == AMGX_FLAG) THEN
   CALL AMGX_INIT(IERR)
   IF (IERR /= 0) THEN
      IF (MY_RANK==0) WRITE(LU_ERR,'(A,I5)') 'Error: AmgX initialization failed with code ', IERR
      STOP_STATUS = SETUP_STOP
   ENDIF
ENDIF
#endif
! ===== END AMGX INITIALIZATION =====
```

#### 3.4.2 Add AmgX Finalization

**Location**: At program termination (near END PROGRAM)

```fortran
! ===== ADD AMGX FINALIZATION =====
#ifdef WITH_AMGX
IF (PRES_FLAG == ULMAT_FLAG .AND. ULMAT_SOLVER_LIBRARY == AMGX_FLAG) THEN
   CALL AMGX_FINALIZE(IERR)
ENDIF
#endif
! ===== END AMGX FINALIZATION =====
```

---

### 3.5 Makefile Modifications

Add the following to the Makefile:

```makefile
# AmgX Configuration
AMGX_DIR = /path/to/amgx
CUDA_DIR = /usr/local/cuda

# AmgX flags
ifdef WITH_AMGX
  FFLAGS += -DWITH_AMGX
  CFLAGS += -DWITH_AMGX
  INCLUDES += -I$(AMGX_DIR)/include -I$(CUDA_DIR)/include
  LIBS += -L$(AMGX_DIR)/lib -lamgx -L$(CUDA_DIR)/lib64 -lcudart -lcublas -lcusparse
endif

# Object files
AMGX_OBJS = amgx_c_wrapper.o amgx_fortran.o

# Compilation rules
amgx_c_wrapper.o: amgx_c_wrapper.c
	$(NVCC) -c $(CFLAGS) $(INCLUDES) $< -o $@

amgx_fortran.o: amgx_fortran.f90
	$(FC) -c $(FFLAGS) $< -o $@
```

---

## 4. Compilation Instructions

### 4.1 Prerequisites

1. **CUDA Toolkit** (11.0 or newer)
2. **AmgX Library** (v2.4.0 or newer)
3. **NVIDIA GPU** (Compute Capability 7.0+, e.g., V100, A100, H100)

### 4.2 Build Steps

```bash
# 1. Set environment variables
export AMGX_DIR=/path/to/amgx
export CUDA_DIR=/usr/local/cuda
export LD_LIBRARY_PATH=$AMGX_DIR/lib:$CUDA_DIR/lib64:$LD_LIBRARY_PATH

# 2. Build FDS with AmgX
make clean
make WITH_AMGX=1 all

# 3. Verify AmgX linkage
ldd fds | grep amgx
```

---

## 5. Runtime Configuration

### 5.1 FDS Input File

To use AmgX solver, add to your FDS input file:

```
&PRES SOLVER='ULMAT', VELOCITY_TOLERANCE=0.5, PRESSURE_TOLERANCE=50.0 /
```

The solver library (MKL vs HYPRE vs AmgX) is determined at compile time by the `WITH_AMGX` flag.

### 5.2 AmgX Solver Configuration

The default configuration in `amgx_c_wrapper.c` is optimized for pressure Poisson:

```json
{
  "solver": "FGMRES",
  "gmres_n_restart": 10,
  "max_iters": 100,
  "tolerance": 1e-8,
  "preconditioner": {
    "solver": "AMG",
    "algorithm": "AGGREGATION",
    "smoother": "MULTICOLOR_DILU",
    "cycle": "V"
  }
}
```

To customize, modify the `default_config` string in `amgx_c_wrapper.c`.

---

## 6. Verification Steps

### 6.1 Correctness Verification

1. Run same case with MKL PARDISO and AmgX
2. Compare pressure field (H array) - should match within tolerance
3. Compare velocity divergence - should be similar

### 6.2 Performance Benchmarking

```bash
# Run benchmark case with timing
mpirun -np 1 fds benchmark.fds

# Compare T_USED(5) (pressure solver time) between MKL and AmgX builds
```

Expected results:
- **Small mesh (<50k cells)**: AmgX may be slower (GPU overhead)
- **Medium mesh (50k-500k cells)**: 2-5x speedup
- **Large mesh (>500k cells)**: 5-20x speedup

---

## 7. Troubleshooting

### 7.1 Common Issues

| Error | Solution |
|-------|----------|
| `AmgX initialization failed` | Check CUDA driver, GPU availability |
| `Matrix upload failed` | Check matrix dimensions match |
| `Solver did not converge` | Increase max_iters, check matrix conditioning |
| `CUDA out of memory` | Reduce mesh size or use multi-GPU |

### 7.2 Debug Output

Set `CHECK_POISSON=.TRUE.` in input file and `MSGLVL=1` in code to enable verbose output.

---

## 8. Future Enhancements

1. **Multi-GPU support**: Use MPI + NCCL for distributed GPU solving
2. **Mixed precision**: Use FP32 for preconditioner, FP64 for solve
3. **Persistent GPU data**: Keep matrix on GPU across timesteps
4. **Automatic solver selection**: Switch between FFT/AmgX based on mesh characteristics

---

## Appendix A: Code Location Reference

| Function | File | Line | Purpose |
|----------|------|------|---------|
| `ULMAT_SOLVER_SETUP` | pres.f90 | 1102 | Solver initialization |
| `ULMAT_SOLVER` | pres.f90 | 1380 | Main solver entry point |
| `ULMAT_SOLVE_ZONE` | pres.f90 | 1420 | Per-zone solve |
| `ULMAT_H_MATRIX_SOLVER_SETUP` | pres.f90 | 2733 | Matrix factorization |
| `ZONE_MESH_TYPE` | type.f90 | 1722 | Zone data structure |
| `ULMAT_SOLVER_LIBRARY` | cons.f90 | 535 | Solver selection flag |

---

## Appendix B: Data Structure Mapping

| FDS Variable | AmgX Equivalent | Description |
|--------------|-----------------|-------------|
| `ZM%NUNKH` | `n` | Number of unknowns |
| `ZM%IA_H` | `row_ptrs` | CSR row pointers |
| `ZM%JA_H` | `col_indices` | CSR column indices |
| `ZM%A_H` | `values` | Matrix values |
| `ZM%F_H` | `rhs` | RHS vector |
| `ZM%X_H` | `sol` | Solution vector |

**Note**: FDS stores upper triangular part with 1-based indexing. The C wrapper converts to full symmetric matrix with 0-based indexing for AmgX.

!> @file amgx_fortran.f90
!> @brief Fortran interface module for NVIDIA AmgX GPU solver
!>
!> This module provides Fortran bindings to the AmgX C wrapper functions.
!> It replaces MKL PARDISO for pressure Poisson equation solving on GPU.
!>
!> Usage in FDS:
!>   1. Call AMGX_INIT() at program start (in main.f90)
!>   2. Call AMGX_SETUP_ZONE() during solver setup
!>   3. Call AMGX_UPLOAD_MATRIX() to transfer matrix to GPU
!>   4. Call AMGX_SOLVE() each timestep
!>   5. Call AMGX_FINALIZE() at program end

MODULE AMGX_FORTRAN

USE ISO_C_BINDING
USE PRECISION_PARAMETERS, ONLY: EB

IMPLICIT NONE

PRIVATE

! Public procedures
PUBLIC :: AMGX_INIT
PUBLIC :: AMGX_FINALIZE
PUBLIC :: AMGX_SETUP_ZONE
PUBLIC :: AMGX_UPLOAD_MATRIX
PUBLIC :: AMGX_UPDATE_MATRIX
PUBLIC :: AMGX_SOLVE
PUBLIC :: AMGX_SOLVE_ZERO_INIT
PUBLIC :: AMGX_GET_ITERATIONS
PUBLIC :: AMGX_GET_RESIDUAL
PUBLIC :: AMGX_DESTROY_ZONE

! GPU Monitoring functions
PUBLIC :: AMGX_GPU_MONITOR_INIT
PUBLIC :: AMGX_GPU_MONITOR_SHUTDOWN
PUBLIC :: AMGX_GET_GPU_UTILIZATION
PUBLIC :: AMGX_GET_GPU_MEMORY
PUBLIC :: AMGX_GET_GPU_TEMPERATURE
PUBLIC :: AMGX_GET_GPU_POWER
PUBLIC :: AMGX_GET_GPU_NAME
PUBLIC :: AMGX_LOG_GPU_STATUS
PUBLIC :: AMGX_GET_GPU_STATS

! GPU-Resident Data functions (zero-copy optimization)
PUBLIC :: AMGX_UPLOAD_RHS_GPU
PUBLIC :: AMGX_SOLVE_GPU_RESIDENT
PUBLIC :: AMGX_DOWNLOAD_SOLUTION_GPU
PUBLIC :: AMGX_IS_GPU_RESIDENT
PUBLIC :: AMGX_SOLVE_GPU_OPTIMIZED

! C interface declarations
INTERFACE

   !> Initialize AmgX library
   SUBROUTINE amgx_initialize_c(ierr) BIND(C, NAME='amgx_initialize_')
      IMPORT :: C_INT
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE amgx_initialize_c

   !> Finalize AmgX library
   SUBROUTINE amgx_finalize_c(ierr) BIND(C, NAME='amgx_finalize_')
      IMPORT :: C_INT
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE amgx_finalize_c

   !> Setup zone for AmgX solver
   SUBROUTINE amgx_setup_zone_c(zone_id, n, nnz, config, ierr) BIND(C, NAME='amgx_setup_zone_')
      IMPORT :: C_INT, C_CHAR
      INTEGER(C_INT), INTENT(IN) :: zone_id
      INTEGER(C_INT), INTENT(IN) :: n
      INTEGER(C_INT), INTENT(IN) :: nnz
      CHARACTER(KIND=C_CHAR), INTENT(IN) :: config(*)
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE amgx_setup_zone_c

   !> Upload sparse matrix to GPU
   SUBROUTINE amgx_upload_matrix_c(zone_id, n, nnz, row_ptrs, col_indices, values, ierr) &
              BIND(C, NAME='amgx_upload_matrix_')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), INTENT(IN) :: zone_id
      INTEGER(C_INT), INTENT(IN) :: n
      INTEGER(C_INT), INTENT(IN) :: nnz
      INTEGER(C_INT), INTENT(IN) :: row_ptrs(*)
      INTEGER(C_INT), INTENT(IN) :: col_indices(*)
      REAL(C_DOUBLE), INTENT(IN) :: values(*)
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE amgx_upload_matrix_c

   !> Update matrix coefficients
   SUBROUTINE amgx_update_matrix_c(zone_id, n, nnz, row_ptrs, col_indices, values, ierr) &
              BIND(C, NAME='amgx_update_matrix_')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), INTENT(IN) :: zone_id
      INTEGER(C_INT), INTENT(IN) :: n
      INTEGER(C_INT), INTENT(IN) :: nnz
      INTEGER(C_INT), INTENT(IN) :: row_ptrs(*)
      INTEGER(C_INT), INTENT(IN) :: col_indices(*)
      REAL(C_DOUBLE), INTENT(IN) :: values(*)
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE amgx_update_matrix_c

   !> Solve linear system
   SUBROUTINE amgx_solve_c(zone_id, n, rhs, sol, ierr) BIND(C, NAME='amgx_solve_')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), INTENT(IN) :: zone_id
      INTEGER(C_INT), INTENT(IN) :: n
      REAL(C_DOUBLE), INTENT(IN) :: rhs(*)
      REAL(C_DOUBLE), INTENT(INOUT) :: sol(*)
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE amgx_solve_c

   !> Solve with zero initial guess
   SUBROUTINE amgx_solve_zero_init_c(zone_id, n, rhs, sol, ierr) BIND(C, NAME='amgx_solve_zero_init_')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), INTENT(IN) :: zone_id
      INTEGER(C_INT), INTENT(IN) :: n
      REAL(C_DOUBLE), INTENT(IN) :: rhs(*)
      REAL(C_DOUBLE), INTENT(OUT) :: sol(*)
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE amgx_solve_zero_init_c

   !> Get iteration count
   SUBROUTINE amgx_get_iterations_c(zone_id, iters, ierr) BIND(C, NAME='amgx_get_iterations_')
      IMPORT :: C_INT
      INTEGER(C_INT), INTENT(IN) :: zone_id
      INTEGER(C_INT), INTENT(OUT) :: iters
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE amgx_get_iterations_c

   !> Get final residual
   SUBROUTINE amgx_get_residual_c(zone_id, residual, ierr) BIND(C, NAME='amgx_get_residual_')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), INTENT(IN) :: zone_id
      REAL(C_DOUBLE), INTENT(OUT) :: residual
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE amgx_get_residual_c

   !> Destroy zone resources
   SUBROUTINE amgx_destroy_zone_c(zone_id, ierr) BIND(C, NAME='amgx_destroy_zone_')
      IMPORT :: C_INT
      INTEGER(C_INT), INTENT(IN) :: zone_id
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE amgx_destroy_zone_c

   !> Initialize GPU monitoring (NVML)
   SUBROUTINE amgx_gpu_monitor_init_c(ierr) BIND(C, NAME='amgx_gpu_monitor_init_')
      IMPORT :: C_INT
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE amgx_gpu_monitor_init_c

   !> Shutdown GPU monitoring
   SUBROUTINE amgx_gpu_monitor_shutdown_c(ierr) BIND(C, NAME='amgx_gpu_monitor_shutdown_')
      IMPORT :: C_INT
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE amgx_gpu_monitor_shutdown_c

   !> Get GPU utilization percentage
   SUBROUTINE amgx_get_gpu_utilization_c(util, ierr) BIND(C, NAME='amgx_get_gpu_utilization_')
      IMPORT :: C_INT
      INTEGER(C_INT), INTENT(OUT) :: util
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE amgx_get_gpu_utilization_c

   !> Get GPU memory usage
   SUBROUTINE amgx_get_gpu_memory_c(used_mb, total_mb, ierr) BIND(C, NAME='amgx_get_gpu_memory_')
      IMPORT :: C_INT, C_DOUBLE
      REAL(C_DOUBLE), INTENT(OUT) :: used_mb
      REAL(C_DOUBLE), INTENT(OUT) :: total_mb
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE amgx_get_gpu_memory_c

   !> Get GPU temperature
   SUBROUTINE amgx_get_gpu_temperature_c(temp, ierr) BIND(C, NAME='amgx_get_gpu_temperature_')
      IMPORT :: C_INT
      INTEGER(C_INT), INTENT(OUT) :: temp
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE amgx_get_gpu_temperature_c

   !> Get GPU power usage
   SUBROUTINE amgx_get_gpu_power_c(power_w, ierr) BIND(C, NAME='amgx_get_gpu_power_')
      IMPORT :: C_INT, C_DOUBLE
      REAL(C_DOUBLE), INTENT(OUT) :: power_w
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE amgx_get_gpu_power_c

   !> Get GPU name
   SUBROUTINE amgx_get_gpu_name_c(name, name_len, ierr) BIND(C, NAME='amgx_get_gpu_name_')
      IMPORT :: C_INT, C_CHAR
      CHARACTER(KIND=C_CHAR), INTENT(OUT) :: name(*)
      INTEGER(C_INT), INTENT(IN) :: name_len
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE amgx_get_gpu_name_c

   !> Log GPU status to console
   SUBROUTINE amgx_log_gpu_status_c(zone_id, timestep, ierr) BIND(C, NAME='amgx_log_gpu_status_')
      IMPORT :: C_INT
      INTEGER(C_INT), INTENT(IN) :: zone_id
      INTEGER(C_INT), INTENT(IN) :: timestep
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE amgx_log_gpu_status_c

   !> Get all GPU stats at once
   SUBROUTINE amgx_get_gpu_stats_c(util_pct, mem_used_mb, mem_total_mb, temp_c, power_w, ierr) &
              BIND(C, NAME='amgx_get_gpu_stats_')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), INTENT(OUT) :: util_pct
      REAL(C_DOUBLE), INTENT(OUT) :: mem_used_mb
      REAL(C_DOUBLE), INTENT(OUT) :: mem_total_mb
      INTEGER(C_INT), INTENT(OUT) :: temp_c
      REAL(C_DOUBLE), INTENT(OUT) :: power_w
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE amgx_get_gpu_stats_c

   !> Upload RHS to GPU-resident buffer
   SUBROUTINE amgx_upload_rhs_gpu_c(zone_id, n, rhs, ierr) BIND(C, NAME='amgx_upload_rhs_gpu_')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), INTENT(IN) :: zone_id
      INTEGER(C_INT), INTENT(IN) :: n
      REAL(C_DOUBLE), INTENT(IN) :: rhs(*)
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE amgx_upload_rhs_gpu_c

   !> Solve using GPU-resident data (no CPU-GPU transfer during solve)
   SUBROUTINE amgx_solve_gpu_resident_c(zone_id, ierr) BIND(C, NAME='amgx_solve_gpu_resident_')
      IMPORT :: C_INT
      INTEGER(C_INT), INTENT(IN) :: zone_id
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE amgx_solve_gpu_resident_c

   !> Download solution from GPU to CPU
   SUBROUTINE amgx_download_solution_gpu_c(zone_id, n, sol, ierr) BIND(C, NAME='amgx_download_solution_gpu_')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), INTENT(IN) :: zone_id
      INTEGER(C_INT), INTENT(IN) :: n
      REAL(C_DOUBLE), INTENT(OUT) :: sol(*)
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE amgx_download_solution_gpu_c

   !> Check if GPU-resident mode is enabled
   SUBROUTINE amgx_is_gpu_resident_c(zone_id, is_enabled, ierr) BIND(C, NAME='amgx_is_gpu_resident_')
      IMPORT :: C_INT
      INTEGER(C_INT), INTENT(IN) :: zone_id
      INTEGER(C_INT), INTENT(OUT) :: is_enabled
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE amgx_is_gpu_resident_c

   !> Optimized solve with GPU-resident data
   SUBROUTINE amgx_solve_gpu_optimized_c(zone_id, n, rhs, sol, ierr) BIND(C, NAME='amgx_solve_gpu_optimized_')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), INTENT(IN) :: zone_id
      INTEGER(C_INT), INTENT(IN) :: n
      REAL(C_DOUBLE), INTENT(IN) :: rhs(*)
      REAL(C_DOUBLE), INTENT(OUT) :: sol(*)
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE amgx_solve_gpu_optimized_c

END INTERFACE

CONTAINS

!> @brief Initialize AmgX library
!> @param[out] IERR Error code (0 = success)
SUBROUTINE AMGX_INIT(IERR)
   INTEGER, INTENT(OUT) :: IERR
   INTEGER(C_INT) :: C_IERR

   CALL amgx_initialize_c(C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE AMGX_INIT

!> @brief Finalize AmgX library
!> @param[out] IERR Error code (0 = success)
SUBROUTINE AMGX_FINALIZE(IERR)
   INTEGER, INTENT(OUT) :: IERR
   INTEGER(C_INT) :: C_IERR

   CALL amgx_finalize_c(C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE AMGX_FINALIZE

!> @brief Setup AmgX zone for solving
!> @param[in]  ZONE_ID Zone identifier (typically NM*1000 + IPZ)
!> @param[in]  N       Number of unknowns
!> @param[in]  NNZ     Number of non-zeros in upper triangular part
!> @param[out] IERR    Error code (0 = success)
!> @param[in]  CONFIG  Optional solver configuration string
SUBROUTINE AMGX_SETUP_ZONE(ZONE_ID, N, NNZ, IERR, CONFIG)
   INTEGER, INTENT(IN) :: ZONE_ID
   INTEGER, INTENT(IN) :: N
   INTEGER, INTENT(IN) :: NNZ
   INTEGER, INTENT(OUT) :: IERR
   CHARACTER(LEN=*), INTENT(IN), OPTIONAL :: CONFIG

   INTEGER(C_INT) :: C_ZONE_ID, C_N, C_NNZ, C_IERR
   CHARACTER(LEN=1024, KIND=C_CHAR) :: C_CONFIG

   C_ZONE_ID = INT(ZONE_ID, C_INT)
   C_N = INT(N, C_INT)
   C_NNZ = INT(NNZ, C_INT)

   IF (PRESENT(CONFIG)) THEN
      C_CONFIG = TRIM(CONFIG) // C_NULL_CHAR
   ELSE
      C_CONFIG = C_NULL_CHAR
   ENDIF

   CALL amgx_setup_zone_c(C_ZONE_ID, C_N, C_NNZ, C_CONFIG, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE AMGX_SETUP_ZONE

!> @brief Upload sparse matrix to GPU
!>
!> This function uploads the CSR matrix to the GPU. FDS stores matrices
!> in upper triangular form with 1-based indexing. The C wrapper converts
!> to full symmetric matrix with 0-based indexing.
!>
!> @param[in]  ZONE_ID     Zone identifier
!> @param[in]  N           Number of rows/unknowns
!> @param[in]  NNZ         Number of non-zeros (upper triangular)
!> @param[in]  IA          CSR row pointers (1-based, size N+1)
!> @param[in]  JA          CSR column indices (1-based)
!> @param[in]  A           Matrix values
!> @param[out] IERR        Error code (0 = success)
SUBROUTINE AMGX_UPLOAD_MATRIX(ZONE_ID, N, NNZ, IA, JA, A, IERR)
   INTEGER, INTENT(IN) :: ZONE_ID
   INTEGER, INTENT(IN) :: N
   INTEGER, INTENT(IN) :: NNZ
   INTEGER, INTENT(IN) :: IA(:)
   INTEGER, INTENT(IN) :: JA(:)
   REAL(EB), INTENT(IN) :: A(:)
   INTEGER, INTENT(OUT) :: IERR

   INTEGER(C_INT) :: C_ZONE_ID, C_N, C_NNZ, C_IERR

   C_ZONE_ID = INT(ZONE_ID, C_INT)
   C_N = INT(N, C_INT)
   C_NNZ = INT(NNZ, C_INT)

   CALL amgx_upload_matrix_c(C_ZONE_ID, C_N, C_NNZ, IA, JA, A, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE AMGX_UPLOAD_MATRIX

!> @brief Update matrix coefficients without changing structure
!> @param[in]  ZONE_ID     Zone identifier
!> @param[in]  N           Number of rows/unknowns
!> @param[in]  NNZ         Number of non-zeros
!> @param[in]  IA          CSR row pointers
!> @param[in]  JA          CSR column indices
!> @param[in]  A           Matrix values
!> @param[out] IERR        Error code
SUBROUTINE AMGX_UPDATE_MATRIX(ZONE_ID, N, NNZ, IA, JA, A, IERR)
   INTEGER, INTENT(IN) :: ZONE_ID
   INTEGER, INTENT(IN) :: N
   INTEGER, INTENT(IN) :: NNZ
   INTEGER, INTENT(IN) :: IA(:)
   INTEGER, INTENT(IN) :: JA(:)
   REAL(EB), INTENT(IN) :: A(:)
   INTEGER, INTENT(OUT) :: IERR

   INTEGER(C_INT) :: C_ZONE_ID, C_N, C_NNZ, C_IERR

   C_ZONE_ID = INT(ZONE_ID, C_INT)
   C_N = INT(N, C_INT)
   C_NNZ = INT(NNZ, C_INT)

   CALL amgx_update_matrix_c(C_ZONE_ID, C_N, C_NNZ, IA, JA, A, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE AMGX_UPDATE_MATRIX

!> @brief Solve the linear system Ax = b
!>
!> Solves the pressure Poisson equation using AmgX on GPU.
!>
!> @param[in]     ZONE_ID Zone identifier
!> @param[in]     N       Number of unknowns
!> @param[in]     F_H     RHS vector (PRHS in FDS)
!> @param[in,out] X_H     Solution vector (initial guess on input, solution on output)
!> @param[out]    IERR    Error code (0 = success, 1 = not converged)
SUBROUTINE AMGX_SOLVE(ZONE_ID, N, F_H, X_H, IERR)
   INTEGER, INTENT(IN) :: ZONE_ID
   INTEGER, INTENT(IN) :: N
   REAL(EB), INTENT(IN) :: F_H(:)
   REAL(EB), INTENT(INOUT) :: X_H(:)
   INTEGER, INTENT(OUT) :: IERR

   INTEGER(C_INT) :: C_ZONE_ID, C_N, C_IERR

   C_ZONE_ID = INT(ZONE_ID, C_INT)
   C_N = INT(N, C_INT)

   CALL amgx_solve_c(C_ZONE_ID, C_N, F_H, X_H, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE AMGX_SOLVE

!> @brief Solve with zero initial guess
!> @param[in]  ZONE_ID Zone identifier
!> @param[in]  N       Number of unknowns
!> @param[in]  F_H     RHS vector
!> @param[out] X_H     Solution vector
!> @param[out] IERR    Error code
SUBROUTINE AMGX_SOLVE_ZERO_INIT(ZONE_ID, N, F_H, X_H, IERR)
   INTEGER, INTENT(IN) :: ZONE_ID
   INTEGER, INTENT(IN) :: N
   REAL(EB), INTENT(IN) :: F_H(:)
   REAL(EB), INTENT(OUT) :: X_H(:)
   INTEGER, INTENT(OUT) :: IERR

   INTEGER(C_INT) :: C_ZONE_ID, C_N, C_IERR

   C_ZONE_ID = INT(ZONE_ID, C_INT)
   C_N = INT(N, C_INT)

   CALL amgx_solve_zero_init_c(C_ZONE_ID, C_N, F_H, X_H, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE AMGX_SOLVE_ZERO_INIT

!> @brief Get solver iteration count
!> @param[in]  ZONE_ID Zone identifier
!> @param[out] ITERS   Number of iterations
!> @param[out] IERR    Error code
SUBROUTINE AMGX_GET_ITERATIONS(ZONE_ID, ITERS, IERR)
   INTEGER, INTENT(IN) :: ZONE_ID
   INTEGER, INTENT(OUT) :: ITERS
   INTEGER, INTENT(OUT) :: IERR

   INTEGER(C_INT) :: C_ZONE_ID, C_ITERS, C_IERR

   C_ZONE_ID = INT(ZONE_ID, C_INT)
   CALL amgx_get_iterations_c(C_ZONE_ID, C_ITERS, C_IERR)
   ITERS = INT(C_ITERS)
   IERR = INT(C_IERR)
END SUBROUTINE AMGX_GET_ITERATIONS

!> @brief Get final residual norm
!> @param[in]  ZONE_ID  Zone identifier
!> @param[out] RESIDUAL Final residual
!> @param[out] IERR     Error code
SUBROUTINE AMGX_GET_RESIDUAL(ZONE_ID, RESIDUAL, IERR)
   INTEGER, INTENT(IN) :: ZONE_ID
   REAL(EB), INTENT(OUT) :: RESIDUAL
   INTEGER, INTENT(OUT) :: IERR

   INTEGER(C_INT) :: C_ZONE_ID, C_IERR
   REAL(C_DOUBLE) :: C_RESIDUAL

   C_ZONE_ID = INT(ZONE_ID, C_INT)
   CALL amgx_get_residual_c(C_ZONE_ID, C_RESIDUAL, C_IERR)
   RESIDUAL = REAL(C_RESIDUAL, EB)
   IERR = INT(C_IERR)
END SUBROUTINE AMGX_GET_RESIDUAL

!> @brief Destroy zone resources
!> @param[in]  ZONE_ID Zone identifier
!> @param[out] IERR    Error code
SUBROUTINE AMGX_DESTROY_ZONE(ZONE_ID, IERR)
   INTEGER, INTENT(IN) :: ZONE_ID
   INTEGER, INTENT(OUT) :: IERR

   INTEGER(C_INT) :: C_ZONE_ID, C_IERR

   C_ZONE_ID = INT(ZONE_ID, C_INT)
   CALL amgx_destroy_zone_c(C_ZONE_ID, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE AMGX_DESTROY_ZONE

! ============================================================================
! GPU Monitoring Functions
! ============================================================================

!> @brief Initialize GPU monitoring (NVML)
!> @param[out] IERR Error code (0 = success)
SUBROUTINE AMGX_GPU_MONITOR_INIT(IERR)
   INTEGER, INTENT(OUT) :: IERR
   INTEGER(C_INT) :: C_IERR

   CALL amgx_gpu_monitor_init_c(C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE AMGX_GPU_MONITOR_INIT

!> @brief Shutdown GPU monitoring
!> @param[out] IERR Error code
SUBROUTINE AMGX_GPU_MONITOR_SHUTDOWN(IERR)
   INTEGER, INTENT(OUT) :: IERR
   INTEGER(C_INT) :: C_IERR

   CALL amgx_gpu_monitor_shutdown_c(C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE AMGX_GPU_MONITOR_SHUTDOWN

!> @brief Get GPU utilization percentage
!> @param[out] UTIL GPU utilization (0-100%), -1 if unavailable
!> @param[out] IERR Error code
SUBROUTINE AMGX_GET_GPU_UTILIZATION(UTIL, IERR)
   INTEGER, INTENT(OUT) :: UTIL
   INTEGER, INTENT(OUT) :: IERR
   INTEGER(C_INT) :: C_UTIL, C_IERR

   CALL amgx_get_gpu_utilization_c(C_UTIL, C_IERR)
   UTIL = INT(C_UTIL)
   IERR = INT(C_IERR)
END SUBROUTINE AMGX_GET_GPU_UTILIZATION

!> @brief Get GPU memory usage
!> @param[out] USED_MB  Used memory in MB
!> @param[out] TOTAL_MB Total memory in MB
!> @param[out] IERR     Error code
SUBROUTINE AMGX_GET_GPU_MEMORY(USED_MB, TOTAL_MB, IERR)
   REAL(EB), INTENT(OUT) :: USED_MB
   REAL(EB), INTENT(OUT) :: TOTAL_MB
   INTEGER, INTENT(OUT) :: IERR
   REAL(C_DOUBLE) :: C_USED, C_TOTAL
   INTEGER(C_INT) :: C_IERR

   CALL amgx_get_gpu_memory_c(C_USED, C_TOTAL, C_IERR)
   USED_MB = REAL(C_USED, EB)
   TOTAL_MB = REAL(C_TOTAL, EB)
   IERR = INT(C_IERR)
END SUBROUTINE AMGX_GET_GPU_MEMORY

!> @brief Get GPU temperature
!> @param[out] TEMP Temperature in Celsius, -1 if unavailable
!> @param[out] IERR Error code
SUBROUTINE AMGX_GET_GPU_TEMPERATURE(TEMP, IERR)
   INTEGER, INTENT(OUT) :: TEMP
   INTEGER, INTENT(OUT) :: IERR
   INTEGER(C_INT) :: C_TEMP, C_IERR

   CALL amgx_get_gpu_temperature_c(C_TEMP, C_IERR)
   TEMP = INT(C_TEMP)
   IERR = INT(C_IERR)
END SUBROUTINE AMGX_GET_GPU_TEMPERATURE

!> @brief Get GPU power consumption
!> @param[out] POWER_W Power in Watts, -1 if unavailable
!> @param[out] IERR    Error code
SUBROUTINE AMGX_GET_GPU_POWER(POWER_W, IERR)
   REAL(EB), INTENT(OUT) :: POWER_W
   INTEGER, INTENT(OUT) :: IERR
   REAL(C_DOUBLE) :: C_POWER
   INTEGER(C_INT) :: C_IERR

   CALL amgx_get_gpu_power_c(C_POWER, C_IERR)
   POWER_W = REAL(C_POWER, EB)
   IERR = INT(C_IERR)
END SUBROUTINE AMGX_GET_GPU_POWER

!> @brief Get GPU name/model
!> @param[out] NAME GPU name string
!> @param[out] IERR Error code
SUBROUTINE AMGX_GET_GPU_NAME(NAME, IERR)
   CHARACTER(LEN=*), INTENT(OUT) :: NAME
   INTEGER, INTENT(OUT) :: IERR
   CHARACTER(LEN=256, KIND=C_CHAR) :: C_NAME
   INTEGER(C_INT) :: C_LEN, C_IERR
   INTEGER :: I

   C_LEN = 256_C_INT
   CALL amgx_get_gpu_name_c(C_NAME, C_LEN, C_IERR)

   ! Copy and trim at null character
   NAME = ' '
   DO I = 1, MIN(LEN(NAME), 255)
      IF (C_NAME(I:I) == C_NULL_CHAR) EXIT
      NAME(I:I) = C_NAME(I:I)
   END DO
   IERR = INT(C_IERR)
END SUBROUTINE AMGX_GET_GPU_NAME

!> @brief Log GPU status to console
!> @param[in]  ZONE_ID  Zone identifier (-1 for general)
!> @param[in]  TIMESTEP Current time step
!> @param[out] IERR     Error code
SUBROUTINE AMGX_LOG_GPU_STATUS(ZONE_ID, TIMESTEP, IERR)
   INTEGER, INTENT(IN) :: ZONE_ID
   INTEGER, INTENT(IN) :: TIMESTEP
   INTEGER, INTENT(OUT) :: IERR
   INTEGER(C_INT) :: C_ZONE_ID, C_TIMESTEP, C_IERR

   C_ZONE_ID = INT(ZONE_ID, C_INT)
   C_TIMESTEP = INT(TIMESTEP, C_INT)
   CALL amgx_log_gpu_status_c(C_ZONE_ID, C_TIMESTEP, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE AMGX_LOG_GPU_STATUS

!> @brief Get all GPU stats in one call
!> @param[out] UTIL_PCT    GPU utilization percentage (-1 if unavailable)
!> @param[out] MEM_USED_MB Memory used in MB
!> @param[out] MEM_TOTAL_MB Memory total in MB
!> @param[out] TEMP_C      Temperature in Celsius (-1 if unavailable)
!> @param[out] POWER_W     Power in Watts (-1 if unavailable)
!> @param[out] IERR        Error code
SUBROUTINE AMGX_GET_GPU_STATS(UTIL_PCT, MEM_USED_MB, MEM_TOTAL_MB, TEMP_C, POWER_W, IERR)
   INTEGER, INTENT(OUT) :: UTIL_PCT
   REAL(EB), INTENT(OUT) :: MEM_USED_MB
   REAL(EB), INTENT(OUT) :: MEM_TOTAL_MB
   INTEGER, INTENT(OUT) :: TEMP_C
   REAL(EB), INTENT(OUT) :: POWER_W
   INTEGER, INTENT(OUT) :: IERR

   INTEGER(C_INT) :: C_UTIL, C_TEMP, C_IERR
   REAL(C_DOUBLE) :: C_MEM_USED, C_MEM_TOTAL, C_POWER

   CALL amgx_get_gpu_stats_c(C_UTIL, C_MEM_USED, C_MEM_TOTAL, C_TEMP, C_POWER, C_IERR)

   UTIL_PCT = INT(C_UTIL)
   MEM_USED_MB = REAL(C_MEM_USED, EB)
   MEM_TOTAL_MB = REAL(C_MEM_TOTAL, EB)
   TEMP_C = INT(C_TEMP)
   POWER_W = REAL(C_POWER, EB)
   IERR = INT(C_IERR)
END SUBROUTINE AMGX_GET_GPU_STATS

! ============================================================================
! GPU-Resident Data Functions (Zero-copy Optimization)
! ============================================================================

!> @brief Upload RHS vector to GPU-resident buffer
!>
!> This uploads the RHS vector to GPU device memory where it stays
!> for the solve. No per-timestep download needed.
!>
!> @param[in]  ZONE_ID Zone identifier
!> @param[in]  N       Number of unknowns
!> @param[in]  F_H     RHS vector
!> @param[out] IERR    Error code
SUBROUTINE AMGX_UPLOAD_RHS_GPU(ZONE_ID, N, F_H, IERR)
   INTEGER, INTENT(IN) :: ZONE_ID
   INTEGER, INTENT(IN) :: N
   REAL(EB), INTENT(IN) :: F_H(:)
   INTEGER, INTENT(OUT) :: IERR

   INTEGER(C_INT) :: C_ZONE_ID, C_N, C_IERR

   C_ZONE_ID = INT(ZONE_ID, C_INT)
   C_N = INT(N, C_INT)

   CALL amgx_upload_rhs_gpu_c(C_ZONE_ID, C_N, F_H, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE AMGX_UPLOAD_RHS_GPU

!> @brief Solve using GPU-resident data (ZERO CPU-GPU transfer!)
!>
!> This is the key optimization function:
!> - RHS is already on GPU (uploaded via AMGX_UPLOAD_RHS_GPU)
!> - Solution stays on GPU - previous solution used as initial guess
!> - NO download to CPU after solve
!>
!> Call AMGX_DOWNLOAD_SOLUTION_GPU only when FDS needs the solution on CPU.
!>
!> @param[in]  ZONE_ID Zone identifier
!> @param[out] IERR    Error code (0 = success, 1 = not converged)
SUBROUTINE AMGX_SOLVE_GPU_RESIDENT(ZONE_ID, IERR)
   INTEGER, INTENT(IN) :: ZONE_ID
   INTEGER, INTENT(OUT) :: IERR

   INTEGER(C_INT) :: C_ZONE_ID, C_IERR

   C_ZONE_ID = INT(ZONE_ID, C_INT)

   CALL amgx_solve_gpu_resident_c(C_ZONE_ID, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE AMGX_SOLVE_GPU_RESIDENT

!> @brief Download solution from GPU to CPU
!>
!> Call this only when FDS needs the pressure solution on CPU.
!> In typical FDS workflow, this is needed for:
!> - Velocity correction (u* = u - dt * grad(p))
!> - Output/visualization
!>
!> @param[in]  ZONE_ID Zone identifier
!> @param[in]  N       Number of unknowns
!> @param[out] X_H     Solution vector on CPU
!> @param[out] IERR    Error code
SUBROUTINE AMGX_DOWNLOAD_SOLUTION_GPU(ZONE_ID, N, X_H, IERR)
   INTEGER, INTENT(IN) :: ZONE_ID
   INTEGER, INTENT(IN) :: N
   REAL(EB), INTENT(OUT) :: X_H(:)
   INTEGER, INTENT(OUT) :: IERR

   INTEGER(C_INT) :: C_ZONE_ID, C_N, C_IERR

   C_ZONE_ID = INT(ZONE_ID, C_INT)
   C_N = INT(N, C_INT)

   CALL amgx_download_solution_gpu_c(C_ZONE_ID, C_N, X_H, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE AMGX_DOWNLOAD_SOLUTION_GPU

!> @brief Check if GPU-resident mode is enabled for a zone
!>
!> @param[in]  ZONE_ID    Zone identifier
!> @param[out] IS_ENABLED 1 if GPU-resident mode is enabled, 0 otherwise
!> @param[out] IERR       Error code
SUBROUTINE AMGX_IS_GPU_RESIDENT(ZONE_ID, IS_ENABLED, IERR)
   INTEGER, INTENT(IN) :: ZONE_ID
   INTEGER, INTENT(OUT) :: IS_ENABLED
   INTEGER, INTENT(OUT) :: IERR

   INTEGER(C_INT) :: C_ZONE_ID, C_ENABLED, C_IERR

   C_ZONE_ID = INT(ZONE_ID, C_INT)

   CALL amgx_is_gpu_resident_c(C_ZONE_ID, C_ENABLED, C_IERR)
   IS_ENABLED = INT(C_ENABLED)
   IERR = INT(C_IERR)
END SUBROUTINE AMGX_IS_GPU_RESIDENT

!> @brief Optimized solve with GPU-resident data
!>
!> Convenience function that combines:
!> 1. Upload RHS to GPU
!> 2. Solve on GPU (using previous solution as initial guess)
!> 3. Download solution to CPU
!>
!> This is a drop-in replacement for AMGX_SOLVE with optimized transfers.
!> If GPU-resident mode is not available, falls back to regular solve.
!>
!> @param[in]     ZONE_ID Zone identifier
!> @param[in]     N       Number of unknowns
!> @param[in]     F_H     RHS vector
!> @param[in,out] X_H     Solution vector
!> @param[out]    IERR    Error code
SUBROUTINE AMGX_SOLVE_GPU_OPTIMIZED(ZONE_ID, N, F_H, X_H, IERR)
   INTEGER, INTENT(IN) :: ZONE_ID
   INTEGER, INTENT(IN) :: N
   REAL(EB), INTENT(IN) :: F_H(:)
   REAL(EB), INTENT(INOUT) :: X_H(:)
   INTEGER, INTENT(OUT) :: IERR

   INTEGER(C_INT) :: C_ZONE_ID, C_N, C_IERR

   C_ZONE_ID = INT(ZONE_ID, C_INT)
   C_N = INT(N, C_INT)

   CALL amgx_solve_gpu_optimized_c(C_ZONE_ID, C_N, F_H, X_H, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE AMGX_SOLVE_GPU_OPTIMIZED

END MODULE AMGX_FORTRAN

!> @file amgx_validation.f90
!> @brief Runtime validation module for AmgX GPU solver
!>
!> This module provides runtime checks to detect numerical issues before
!> they cause simulation divergence. It should be used during development
!> and can be disabled in production for performance.
!>
!> Key Features:
!> - Solution bounds checking
!> - Residual monitoring
!> - NaN/Inf detection
!> - CPU vs GPU comparison (debug mode)

MODULE AMGX_VALIDATION

USE PRECISION_PARAMETERS
USE GLOBAL_CONSTANTS, ONLY: MY_RANK, LU_ERR

IMPLICIT NONE

PRIVATE

PUBLIC :: AMGX_CHECK_SOLUTION
PUBLIC :: AMGX_CHECK_RESIDUAL
PUBLIC :: AMGX_CHECK_MATRIX
PUBLIC :: AMGX_COMPARE_SOLUTIONS
PUBLIC :: AMGX_LOG_SOLVER_STATS
PUBLIC :: SET_VALIDATION_LEVEL

! Validation levels
INTEGER, PARAMETER, PUBLIC :: VALIDATION_OFF = 0       !< No validation (production)
INTEGER, PARAMETER, PUBLIC :: VALIDATION_BASIC = 1     !< NaN/Inf check only
INTEGER, PARAMETER, PUBLIC :: VALIDATION_NORMAL = 2    !< Basic + bounds check
INTEGER, PARAMETER, PUBLIC :: VALIDATION_FULL = 3      !< Full validation with CPU comparison

! Current validation level (can be set at runtime)
INTEGER :: CURRENT_VALIDATION_LEVEL = VALIDATION_NORMAL

! Validation thresholds
REAL(EB), PARAMETER :: MAX_PRESSURE_VALUE = 1.0E10_EB  !< Maximum expected pressure
REAL(EB), PARAMETER :: MIN_PRESSURE_VALUE = -1.0E10_EB !< Minimum expected pressure
REAL(EB), PARAMETER :: MAX_RESIDUAL_RATIO = 1.0E3_EB   !< Max residual increase ratio
REAL(EB), PARAMETER :: CPU_GPU_TOLERANCE = 1.0E-4_EB   !< Tolerance for CPU/GPU comparison

! Statistics tracking
TYPE :: SOLVER_STATS_TYPE
   INTEGER :: TOTAL_SOLVES = 0
   INTEGER :: FAILED_SOLVES = 0
   INTEGER :: NAN_DETECTED = 0
   INTEGER :: INF_DETECTED = 0
   INTEGER :: BOUNDS_EXCEEDED = 0
   REAL(EB) :: AVG_ITERATIONS = 0.0_EB
   REAL(EB) :: AVG_RESIDUAL = 0.0_EB
   REAL(EB) :: MAX_RESIDUAL = 0.0_EB
   REAL(EB) :: TOTAL_SOLVE_TIME = 0.0_EB
END TYPE SOLVER_STATS_TYPE

TYPE(SOLVER_STATS_TYPE), PUBLIC :: AMGX_STATS

CONTAINS

!> @brief Set validation level
!> @param[in] LEVEL Validation level (0-3)
SUBROUTINE SET_VALIDATION_LEVEL(LEVEL)
   INTEGER, INTENT(IN) :: LEVEL
   CURRENT_VALIDATION_LEVEL = MAX(0, MIN(3, LEVEL))
   IF (MY_RANK == 0) THEN
      SELECT CASE(CURRENT_VALIDATION_LEVEL)
         CASE(VALIDATION_OFF)
            WRITE(LU_ERR,'(A)') 'AmgX Validation: OFF (production mode)'
         CASE(VALIDATION_BASIC)
            WRITE(LU_ERR,'(A)') 'AmgX Validation: BASIC (NaN/Inf check)'
         CASE(VALIDATION_NORMAL)
            WRITE(LU_ERR,'(A)') 'AmgX Validation: NORMAL (NaN/Inf + bounds)'
         CASE(VALIDATION_FULL)
            WRITE(LU_ERR,'(A)') 'AmgX Validation: FULL (with CPU comparison)'
      END SELECT
   ENDIF
END SUBROUTINE SET_VALIDATION_LEVEL


!> @brief Check solution vector for numerical issues
!> @param[in]  N         Vector size
!> @param[in]  X         Solution vector
!> @param[in]  ZONE_ID   Zone identifier for error reporting
!> @param[out] IS_VALID  True if solution is valid
!> @param[out] ERROR_MSG Error message if invalid
SUBROUTINE AMGX_CHECK_SOLUTION(N, X, ZONE_ID, IS_VALID, ERROR_MSG)
   INTEGER, INTENT(IN) :: N, ZONE_ID
   REAL(EB), INTENT(IN) :: X(:)
   LOGICAL, INTENT(OUT) :: IS_VALID
   CHARACTER(LEN=*), INTENT(OUT) :: ERROR_MSG

   INTEGER :: I, NAN_COUNT, INF_COUNT, BOUNDS_COUNT
   REAL(EB) :: X_MIN, X_MAX
   LOGICAL :: HAS_NAN, HAS_INF, OUT_OF_BOUNDS

   IF (CURRENT_VALIDATION_LEVEL == VALIDATION_OFF) THEN
      IS_VALID = .TRUE.
      ERROR_MSG = ''
      RETURN
   ENDIF

   IS_VALID = .TRUE.
   ERROR_MSG = ''
   NAN_COUNT = 0
   INF_COUNT = 0
   BOUNDS_COUNT = 0
   X_MIN = HUGE(1.0_EB)
   X_MAX = -HUGE(1.0_EB)

   ! Check for NaN and Inf
   DO I = 1, N
      IF (ISNAN(X(I))) THEN
         NAN_COUNT = NAN_COUNT + 1
         IS_VALID = .FALSE.
      ELSEIF (.NOT. IEEE_IS_FINITE(X(I))) THEN
         INF_COUNT = INF_COUNT + 1
         IS_VALID = .FALSE.
      ELSE
         X_MIN = MIN(X_MIN, X(I))
         X_MAX = MAX(X_MAX, X(I))

         ! Bounds check (only in NORMAL or higher)
         IF (CURRENT_VALIDATION_LEVEL >= VALIDATION_NORMAL) THEN
            IF (X(I) < MIN_PRESSURE_VALUE .OR. X(I) > MAX_PRESSURE_VALUE) THEN
               BOUNDS_COUNT = BOUNDS_COUNT + 1
            ENDIF
         ENDIF
      ENDIF
   ENDDO

   ! Update statistics
   IF (NAN_COUNT > 0) AMGX_STATS%NAN_DETECTED = AMGX_STATS%NAN_DETECTED + 1
   IF (INF_COUNT > 0) AMGX_STATS%INF_DETECTED = AMGX_STATS%INF_DETECTED + 1
   IF (BOUNDS_COUNT > 0) AMGX_STATS%BOUNDS_EXCEEDED = AMGX_STATS%BOUNDS_EXCEEDED + 1

   ! Build error message
   IF (NAN_COUNT > 0) THEN
      WRITE(ERROR_MSG, '(A,I5,A,I8)') 'Zone ', ZONE_ID, ': NaN detected, count=', NAN_COUNT
   ELSEIF (INF_COUNT > 0) THEN
      WRITE(ERROR_MSG, '(A,I5,A,I8)') 'Zone ', ZONE_ID, ': Inf detected, count=', INF_COUNT
   ELSEIF (BOUNDS_COUNT > N/10) THEN  ! More than 10% out of bounds
      WRITE(ERROR_MSG, '(A,I5,A,ES12.5,A,ES12.5)') 'Zone ', ZONE_ID, &
            ': Solution out of expected bounds. Min=', X_MIN, ', Max=', X_MAX
      IF (CURRENT_VALIDATION_LEVEL >= VALIDATION_NORMAL) IS_VALID = .FALSE.
   ENDIF

CONTAINS

   LOGICAL FUNCTION IEEE_IS_FINITE(X)
      REAL(EB), INTENT(IN) :: X
      IEEE_IS_FINITE = (X == X) .AND. (ABS(X) < HUGE(1.0_EB))
   END FUNCTION IEEE_IS_FINITE

END SUBROUTINE AMGX_CHECK_SOLUTION


!> @brief Check residual for solver convergence issues
!> @param[in]  N              Vector size
!> @param[in]  RESIDUAL       Current residual norm
!> @param[in]  INITIAL_RES    Initial residual norm
!> @param[in]  ITERATIONS     Number of iterations
!> @param[in]  ZONE_ID        Zone identifier
!> @param[out] IS_CONVERGED   True if solver converged properly
!> @param[out] WARNING_MSG    Warning message if issues detected
SUBROUTINE AMGX_CHECK_RESIDUAL(N, RESIDUAL, INITIAL_RES, ITERATIONS, ZONE_ID, &
                                IS_CONVERGED, WARNING_MSG)
   INTEGER, INTENT(IN) :: N, ITERATIONS, ZONE_ID
   REAL(EB), INTENT(IN) :: RESIDUAL, INITIAL_RES
   LOGICAL, INTENT(OUT) :: IS_CONVERGED
   CHARACTER(LEN=*), INTENT(OUT) :: WARNING_MSG

   REAL(EB) :: REDUCTION_RATIO
   LOGICAL :: RESIDUAL_INCREASED, SLOW_CONVERGENCE

   IS_CONVERGED = .TRUE.
   WARNING_MSG = ''

   ! Update statistics
   AMGX_STATS%TOTAL_SOLVES = AMGX_STATS%TOTAL_SOLVES + 1
   AMGX_STATS%AVG_ITERATIONS = (AMGX_STATS%AVG_ITERATIONS * (AMGX_STATS%TOTAL_SOLVES - 1) + &
                                 REAL(ITERATIONS, EB)) / REAL(AMGX_STATS%TOTAL_SOLVES, EB)
   AMGX_STATS%AVG_RESIDUAL = (AMGX_STATS%AVG_RESIDUAL * (AMGX_STATS%TOTAL_SOLVES - 1) + &
                               RESIDUAL) / REAL(AMGX_STATS%TOTAL_SOLVES, EB)
   AMGX_STATS%MAX_RESIDUAL = MAX(AMGX_STATS%MAX_RESIDUAL, RESIDUAL)

   IF (CURRENT_VALIDATION_LEVEL == VALIDATION_OFF) RETURN

   ! Check for residual increase (potential divergence)
   IF (INITIAL_RES > 1.0E-15_EB) THEN
      REDUCTION_RATIO = RESIDUAL / INITIAL_RES
      RESIDUAL_INCREASED = (REDUCTION_RATIO > MAX_RESIDUAL_RATIO)
   ELSE
      RESIDUAL_INCREASED = (RESIDUAL > 1.0E-10_EB)
   ENDIF

   ! Check for slow convergence
   SLOW_CONVERGENCE = (ITERATIONS >= 100) .AND. (RESIDUAL > 1.0E-4_EB)

   IF (RESIDUAL_INCREASED) THEN
      IS_CONVERGED = .FALSE.
      AMGX_STATS%FAILED_SOLVES = AMGX_STATS%FAILED_SOLVES + 1
      WRITE(WARNING_MSG, '(A,I5,A,ES12.5,A,ES12.5)') 'Zone ', ZONE_ID, &
            ': Residual INCREASED! Initial=', INITIAL_RES, ', Final=', RESIDUAL
   ELSEIF (SLOW_CONVERGENCE) THEN
      WRITE(WARNING_MSG, '(A,I5,A,I5,A,ES12.5)') 'Zone ', ZONE_ID, &
            ': Slow convergence. Iters=', ITERATIONS, ', Residual=', RESIDUAL
   ENDIF

END SUBROUTINE AMGX_CHECK_RESIDUAL


!> @brief Check matrix for numerical issues
!> @param[in]  N         Matrix dimension
!> @param[in]  NNZ       Number of non-zeros
!> @param[in]  IA        CSR row pointers
!> @param[in]  JA        CSR column indices
!> @param[in]  A         Matrix values
!> @param[in]  ZONE_ID   Zone identifier
!> @param[out] IS_VALID  True if matrix is valid
!> @param[out] ERROR_MSG Error message if invalid
SUBROUTINE AMGX_CHECK_MATRIX(N, NNZ, IA, JA, A, ZONE_ID, IS_VALID, ERROR_MSG)
   INTEGER, INTENT(IN) :: N, NNZ, ZONE_ID
   INTEGER, INTENT(IN) :: IA(:), JA(:)
   REAL(EB), INTENT(IN) :: A(:)
   LOGICAL, INTENT(OUT) :: IS_VALID
   CHARACTER(LEN=*), INTENT(OUT) :: ERROR_MSG

   INTEGER :: I, K, J
   REAL(EB) :: DIAG, OFF_DIAG_SUM, MIN_DIAG, MAX_OFF_DIAG
   LOGICAL :: HAS_ZERO_DIAG, NOT_DIAG_DOMINANT
   INTEGER :: ZERO_DIAG_COUNT, NOT_DD_COUNT

   IF (CURRENT_VALIDATION_LEVEL < VALIDATION_NORMAL) THEN
      IS_VALID = .TRUE.
      ERROR_MSG = ''
      RETURN
   ENDIF

   IS_VALID = .TRUE.
   ERROR_MSG = ''
   ZERO_DIAG_COUNT = 0
   NOT_DD_COUNT = 0
   MIN_DIAG = HUGE(1.0_EB)

   ! Check each row
   DO I = 1, N
      DIAG = 0.0_EB
      OFF_DIAG_SUM = 0.0_EB

      DO K = IA(I), IA(I+1)-1
         J = JA(K)
         IF (J == I) THEN
            DIAG = A(K)
            MIN_DIAG = MIN(MIN_DIAG, ABS(DIAG))
         ELSE
            OFF_DIAG_SUM = OFF_DIAG_SUM + ABS(A(K))
            ! For symmetric matrix, count symmetric contribution too
            IF (J > I) OFF_DIAG_SUM = OFF_DIAG_SUM + ABS(A(K))
         ENDIF

         ! Check for NaN/Inf in matrix
         IF (ISNAN(A(K))) THEN
            IS_VALID = .FALSE.
            WRITE(ERROR_MSG, '(A,I5,A,I8,A,I8)') 'Zone ', ZONE_ID, &
                  ': NaN in matrix at row ', I, ', col ', J
            RETURN
         ENDIF
      ENDDO

      ! Check for zero diagonal
      IF (ABS(DIAG) < 1.0E-15_EB) THEN
         ZERO_DIAG_COUNT = ZERO_DIAG_COUNT + 1
      ENDIF

      ! Check diagonal dominance (weak)
      IF (ABS(DIAG) < OFF_DIAG_SUM * 0.5_EB) THEN
         NOT_DD_COUNT = NOT_DD_COUNT + 1
      ENDIF
   ENDDO

   IF (ZERO_DIAG_COUNT > 0) THEN
      IS_VALID = .FALSE.
      WRITE(ERROR_MSG, '(A,I5,A,I8)') 'Zone ', ZONE_ID, &
            ': Zero diagonal elements count=', ZERO_DIAG_COUNT
   ELSEIF (NOT_DD_COUNT > N/2) THEN
      ! More than half rows are not diagonally dominant - warning only
      WRITE(ERROR_MSG, '(A,I5,A,I8,A,ES12.5)') 'Zone ', ZONE_ID, &
            ': Many rows not diag dominant. Count=', NOT_DD_COUNT, ', MinDiag=', MIN_DIAG
   ENDIF

END SUBROUTINE AMGX_CHECK_MATRIX


!> @brief Compare CPU and GPU solutions (for debugging)
!> @param[in]  N         Vector size
!> @param[in]  X_CPU     CPU solution
!> @param[in]  X_GPU     GPU solution
!> @param[in]  ZONE_ID   Zone identifier
!> @param[out] MAX_DIFF  Maximum absolute difference
!> @param[out] REL_DIFF  Relative L2 difference
!> @param[out] IS_MATCH  True if solutions match within tolerance
SUBROUTINE AMGX_COMPARE_SOLUTIONS(N, X_CPU, X_GPU, ZONE_ID, MAX_DIFF, REL_DIFF, IS_MATCH)
   INTEGER, INTENT(IN) :: N, ZONE_ID
   REAL(EB), INTENT(IN) :: X_CPU(:), X_GPU(:)
   REAL(EB), INTENT(OUT) :: MAX_DIFF, REL_DIFF
   LOGICAL, INTENT(OUT) :: IS_MATCH

   REAL(EB) :: L2_DIFF, L2_CPU
   INTEGER :: I, MAX_LOC

   IF (CURRENT_VALIDATION_LEVEL < VALIDATION_FULL) THEN
      MAX_DIFF = 0.0_EB
      REL_DIFF = 0.0_EB
      IS_MATCH = .TRUE.
      RETURN
   ENDIF

   MAX_DIFF = 0.0_EB
   L2_DIFF = 0.0_EB
   L2_CPU = 0.0_EB
   MAX_LOC = 1

   DO I = 1, N
      IF (ABS(X_GPU(I) - X_CPU(I)) > MAX_DIFF) THEN
         MAX_DIFF = ABS(X_GPU(I) - X_CPU(I))
         MAX_LOC = I
      ENDIF
      L2_DIFF = L2_DIFF + (X_GPU(I) - X_CPU(I))**2
      L2_CPU = L2_CPU + X_CPU(I)**2
   ENDDO

   L2_DIFF = SQRT(L2_DIFF)
   L2_CPU = SQRT(L2_CPU)

   IF (L2_CPU > 1.0E-15_EB) THEN
      REL_DIFF = L2_DIFF / L2_CPU
   ELSE
      REL_DIFF = L2_DIFF
   ENDIF

   IS_MATCH = (REL_DIFF < CPU_GPU_TOLERANCE)

   IF (.NOT. IS_MATCH .AND. MY_RANK == 0) THEN
      WRITE(LU_ERR, '(A,I5)') '*** CPU/GPU MISMATCH in Zone ', ZONE_ID
      WRITE(LU_ERR, '(A,ES12.5,A,I8)') '    Max diff: ', MAX_DIFF, ' at index ', MAX_LOC
      WRITE(LU_ERR, '(A,ES12.5)') '    Rel diff: ', REL_DIFF
      WRITE(LU_ERR, '(A,ES12.5,A,ES12.5)') '    CPU value: ', X_CPU(MAX_LOC), &
            ', GPU value: ', X_GPU(MAX_LOC)
   ENDIF

END SUBROUTINE AMGX_COMPARE_SOLUTIONS


!> @brief Log solver statistics
!> @param[in] ZONE_ID Zone identifier (or -1 for summary)
SUBROUTINE AMGX_LOG_SOLVER_STATS(ZONE_ID)
   INTEGER, INTENT(IN) :: ZONE_ID

   IF (MY_RANK /= 0) RETURN

   IF (ZONE_ID < 0) THEN
      ! Print summary statistics
      WRITE(LU_ERR, '(A)') ''
      WRITE(LU_ERR, '(A)') '========== AmgX Solver Statistics =========='
      WRITE(LU_ERR, '(A,I10)')    '  Total solves:       ', AMGX_STATS%TOTAL_SOLVES
      WRITE(LU_ERR, '(A,I10)')    '  Failed solves:      ', AMGX_STATS%FAILED_SOLVES
      WRITE(LU_ERR, '(A,I10)')    '  NaN detected:       ', AMGX_STATS%NAN_DETECTED
      WRITE(LU_ERR, '(A,I10)')    '  Inf detected:       ', AMGX_STATS%INF_DETECTED
      WRITE(LU_ERR, '(A,I10)')    '  Bounds exceeded:    ', AMGX_STATS%BOUNDS_EXCEEDED
      WRITE(LU_ERR, '(A,F10.2)')  '  Avg iterations:     ', AMGX_STATS%AVG_ITERATIONS
      WRITE(LU_ERR, '(A,ES12.5)') '  Avg residual:       ', AMGX_STATS%AVG_RESIDUAL
      WRITE(LU_ERR, '(A,ES12.5)') '  Max residual:       ', AMGX_STATS%MAX_RESIDUAL
      WRITE(LU_ERR, '(A,F10.3,A)')'  Total solve time:   ', AMGX_STATS%TOTAL_SOLVE_TIME, ' s'
      WRITE(LU_ERR, '(A)') '============================================'

      IF (AMGX_STATS%FAILED_SOLVES > 0) THEN
         WRITE(LU_ERR, '(A)') '*** WARNING: Some solves failed! Check for divergence.'
      ENDIF
      IF (AMGX_STATS%NAN_DETECTED > 0 .OR. AMGX_STATS%INF_DETECTED > 0) THEN
         WRITE(LU_ERR, '(A)') '*** ERROR: NaN or Inf detected in solutions!'
      ENDIF
   ENDIF

END SUBROUTINE AMGX_LOG_SOLVER_STATS

END MODULE AMGX_VALIDATION

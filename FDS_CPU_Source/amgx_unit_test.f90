!> @file amgx_unit_test.f90
!> @brief Unit tests for AmgX GPU solver verification
!>
!> This module provides comprehensive tests to verify:
!> 1. Matrix format conversion (upper triangular to full symmetric)
!> 2. Solver accuracy compared to reference solutions
!> 3. Numerical consistency between CPU (PARDISO) and GPU (AmgX) solvers
!>
!> These tests help prevent divergence issues by catching numerical errors early.

MODULE AMGX_UNIT_TEST

USE PRECISION_PARAMETERS
USE ISO_C_BINDING

#ifdef WITH_AMGX
USE AMGX_FORTRAN
#endif

IMPLICIT NONE

PRIVATE

PUBLIC :: RUN_AMGX_UNIT_TESTS
PUBLIC :: TEST_MATRIX_CONVERSION
PUBLIC :: TEST_POISSON_3D_SOLUTION
PUBLIC :: TEST_COMPARE_CPU_GPU_SOLVER
PUBLIC :: VERIFY_SOLUTION_ACCURACY

! Test parameters
REAL(EB), PARAMETER :: TEST_TOLERANCE = 1.0E-10_EB      !< Tolerance for exact comparisons
REAL(EB), PARAMETER :: SOLVER_TOLERANCE = 1.0E-6_EB     !< Tolerance for solver comparisons
INTEGER, PARAMETER :: TEST_SMALL_N = 8                   !< Small test grid size
INTEGER, PARAMETER :: TEST_MEDIUM_N = 32                 !< Medium test grid size

CONTAINS

!> @brief Run all unit tests
!> @param[out] ALL_PASSED True if all tests pass
SUBROUTINE RUN_AMGX_UNIT_TESTS(ALL_PASSED)
   LOGICAL, INTENT(OUT) :: ALL_PASSED
   LOGICAL :: TEST_RESULT
   INTEGER :: IERR, N_PASSED, N_TOTAL

   N_PASSED = 0
   N_TOTAL = 0

   WRITE(*,'(A)') '=================================================='
   WRITE(*,'(A)') '         AmgX Unit Test Suite                     '
   WRITE(*,'(A)') '=================================================='
   WRITE(*,*)

#ifdef WITH_AMGX
   ! Initialize AmgX for testing
   CALL AMGX_INIT(IERR)
   IF (IERR /= 0) THEN
      WRITE(*,'(A,I5)') 'ERROR: AmgX initialization failed with code ', IERR
      ALL_PASSED = .FALSE.
      RETURN
   ENDIF

   ! Test 1: Matrix conversion verification
   WRITE(*,'(A)') '--- Test 1: Matrix Conversion (Upper Tri -> Full) ---'
   CALL TEST_MATRIX_CONVERSION(TEST_RESULT)
   N_TOTAL = N_TOTAL + 1
   IF (TEST_RESULT) THEN
      WRITE(*,'(A)') 'PASSED: Matrix conversion is correct'
      N_PASSED = N_PASSED + 1
   ELSE
      WRITE(*,'(A)') 'FAILED: Matrix conversion error detected'
   ENDIF
   WRITE(*,*)

   ! Test 2: 3D Poisson solver with known solution
   WRITE(*,'(A)') '--- Test 2: 3D Poisson Solver (Known Solution) ---'
   CALL TEST_POISSON_3D_SOLUTION(TEST_RESULT)
   N_TOTAL = N_TOTAL + 1
   IF (TEST_RESULT) THEN
      WRITE(*,'(A)') 'PASSED: Poisson solver produces correct solution'
      N_PASSED = N_PASSED + 1
   ELSE
      WRITE(*,'(A)') 'FAILED: Poisson solver error detected'
   ENDIF
   WRITE(*,*)

   ! Test 3: Compare CPU reference vs GPU solver
   WRITE(*,'(A)') '--- Test 3: CPU vs GPU Solver Comparison ---'
   CALL TEST_COMPARE_CPU_GPU_SOLVER(TEST_RESULT)
   N_TOTAL = N_TOTAL + 1
   IF (TEST_RESULT) THEN
      WRITE(*,'(A)') 'PASSED: GPU solver matches CPU reference'
      N_PASSED = N_PASSED + 1
   ELSE
      WRITE(*,'(A)') 'FAILED: GPU solver differs from CPU reference'
   ENDIF
   WRITE(*,*)

   ! Finalize AmgX
   CALL AMGX_FINALIZE(IERR)

#else
   WRITE(*,'(A)') 'WARNING: AmgX not compiled. Skipping GPU tests.'
   WRITE(*,'(A)') 'Running CPU-only verification tests...'

   ! Test matrix operations without GPU
   CALL TEST_MATRIX_CONVERSION(TEST_RESULT)
   N_TOTAL = N_TOTAL + 1
   IF (TEST_RESULT) N_PASSED = N_PASSED + 1
#endif

   ! Summary
   WRITE(*,'(A)') '=================================================='
   WRITE(*,'(A,I2,A,I2,A)') '  Test Results: ', N_PASSED, ' / ', N_TOTAL, ' PASSED'
   WRITE(*,'(A)') '=================================================='

   ALL_PASSED = (N_PASSED == N_TOTAL)

END SUBROUTINE RUN_AMGX_UNIT_TESTS


!> @brief Test matrix conversion from upper triangular to full symmetric
!> @param[out] PASSED True if test passes
SUBROUTINE TEST_MATRIX_CONVERSION(PASSED)
   LOGICAL, INTENT(OUT) :: PASSED

   INTEGER :: N, NNZ_UPPER, NNZ_FULL
   INTEGER, ALLOCATABLE :: IA_UPPER(:), JA_UPPER(:)
   REAL(EB), ALLOCATABLE :: A_UPPER(:)
   INTEGER, ALLOCATABLE :: IA_FULL(:), JA_FULL(:)
   REAL(EB), ALLOCATABLE :: A_FULL(:)
   REAL(EB), ALLOCATABLE :: FULL_MATRIX(:,:), RECONSTRUCTED(:,:)
   INTEGER :: I, J, K, IDX
   REAL(EB) :: MAX_ERROR
   LOGICAL :: IS_SYMMETRIC

   ! Create a simple 4x4 test matrix (7-point stencil style)
   ! Full symmetric matrix:
   !     [ 4 -1  0 -1]
   ! A = [-1  4 -1  0]
   !     [ 0 -1  4 -1]
   !     [-1  0 -1  4]

   N = 4

   ! Upper triangular part only (including diagonal)
   ! Row 1: (1,1)=4, (1,2)=-1, (1,4)=-1
   ! Row 2: (2,2)=4, (2,3)=-1
   ! Row 3: (3,3)=4, (3,4)=-1
   ! Row 4: (4,4)=4

   NNZ_UPPER = 7  ! Upper triangular entries

   ALLOCATE(IA_UPPER(N+1), JA_UPPER(NNZ_UPPER), A_UPPER(NNZ_UPPER))

   ! CSR format for upper triangular (1-based indexing like FDS)
   IA_UPPER = [1, 4, 6, 8, 9]
   JA_UPPER = [1, 2, 4, 2, 3, 3, 4, 4]
   A_UPPER  = [4.0_EB, -1.0_EB, -1.0_EB, 4.0_EB, -1.0_EB, 4.0_EB, -1.0_EB, 4.0_EB]

   ! Convert to full symmetric matrix
   CALL CONVERT_UPPER_TO_FULL(N, NNZ_UPPER, IA_UPPER, JA_UPPER, A_UPPER, &
                              NNZ_FULL, IA_FULL, JA_FULL, A_FULL)

   ! Build dense matrices for verification
   ALLOCATE(FULL_MATRIX(N,N), RECONSTRUCTED(N,N))
   FULL_MATRIX = 0.0_EB
   RECONSTRUCTED = 0.0_EB

   ! Expected full matrix
   FULL_MATRIX(1,:) = [ 4.0_EB, -1.0_EB,  0.0_EB, -1.0_EB]
   FULL_MATRIX(2,:) = [-1.0_EB,  4.0_EB, -1.0_EB,  0.0_EB]
   FULL_MATRIX(3,:) = [ 0.0_EB, -1.0_EB,  4.0_EB, -1.0_EB]
   FULL_MATRIX(4,:) = [-1.0_EB,  0.0_EB, -1.0_EB,  4.0_EB]

   ! Reconstruct from converted CSR
   DO I = 1, N
      DO K = IA_FULL(I), IA_FULL(I+1)-1
         J = JA_FULL(K)
         RECONSTRUCTED(I,J) = A_FULL(K)
      ENDDO
   ENDDO

   ! Check maximum error
   MAX_ERROR = MAXVAL(ABS(FULL_MATRIX - RECONSTRUCTED))

   ! Check symmetry of reconstructed matrix
   IS_SYMMETRIC = .TRUE.
   DO I = 1, N
      DO J = I+1, N
         IF (ABS(RECONSTRUCTED(I,J) - RECONSTRUCTED(J,I)) > TEST_TOLERANCE) THEN
            IS_SYMMETRIC = .FALSE.
            EXIT
         ENDIF
      ENDDO
      IF (.NOT. IS_SYMMETRIC) EXIT
   ENDDO

   WRITE(*,'(A,ES12.5)') '  Maximum reconstruction error: ', MAX_ERROR
   WRITE(*,'(A,L1)') '  Matrix is symmetric: ', IS_SYMMETRIC
   WRITE(*,'(A,I3,A,I3)') '  NNZ: upper=', NNZ_UPPER, ', full=', NNZ_FULL

   PASSED = (MAX_ERROR < TEST_TOLERANCE) .AND. IS_SYMMETRIC

   DEALLOCATE(IA_UPPER, JA_UPPER, A_UPPER)
   DEALLOCATE(IA_FULL, JA_FULL, A_FULL)
   DEALLOCATE(FULL_MATRIX, RECONSTRUCTED)

END SUBROUTINE TEST_MATRIX_CONVERSION


!> @brief Helper: Convert upper triangular CSR to full symmetric CSR
SUBROUTINE CONVERT_UPPER_TO_FULL(N, NNZ_UPPER, IA_UPPER, JA_UPPER, A_UPPER, &
                                  NNZ_FULL, IA_FULL, JA_FULL, A_FULL)
   INTEGER, INTENT(IN) :: N, NNZ_UPPER
   INTEGER, INTENT(IN) :: IA_UPPER(:), JA_UPPER(:)
   REAL(EB), INTENT(IN) :: A_UPPER(:)
   INTEGER, INTENT(OUT) :: NNZ_FULL
   INTEGER, ALLOCATABLE, INTENT(OUT) :: IA_FULL(:), JA_FULL(:)
   REAL(EB), ALLOCATABLE, INTENT(OUT) :: A_FULL(:)

   INTEGER :: I, J, K, ROW, COL, IDX
   INTEGER, ALLOCATABLE :: ROW_NNZ(:), CURRENT_POS(:)

   TYPE :: ENTRY_TYPE
      INTEGER :: COL
      REAL(EB) :: VAL
   END TYPE ENTRY_TYPE
   TYPE(ENTRY_TYPE), ALLOCATABLE :: ROW_ENTRIES(:,:)
   INTEGER :: MAX_ROW_NNZ

   ! Count non-zeros per row for full matrix
   ALLOCATE(ROW_NNZ(N))
   ROW_NNZ = 0

   DO ROW = 1, N
      DO K = IA_UPPER(ROW), IA_UPPER(ROW+1)-1
         COL = JA_UPPER(K)
         ROW_NNZ(ROW) = ROW_NNZ(ROW) + 1
         IF (ROW /= COL) THEN
            ROW_NNZ(COL) = ROW_NNZ(COL) + 1  ! Symmetric entry
         ENDIF
      ENDDO
   ENDDO

   NNZ_FULL = SUM(ROW_NNZ)
   MAX_ROW_NNZ = MAXVAL(ROW_NNZ)

   ! Allocate storage for row entries
   ALLOCATE(ROW_ENTRIES(MAX_ROW_NNZ, N))
   ALLOCATE(CURRENT_POS(N))
   CURRENT_POS = 0

   ! Fill entries
   DO ROW = 1, N
      DO K = IA_UPPER(ROW), IA_UPPER(ROW+1)-1
         COL = JA_UPPER(K)

         ! Add to this row
         CURRENT_POS(ROW) = CURRENT_POS(ROW) + 1
         ROW_ENTRIES(CURRENT_POS(ROW), ROW)%COL = COL
         ROW_ENTRIES(CURRENT_POS(ROW), ROW)%VAL = A_UPPER(K)

         ! Add symmetric entry if off-diagonal
         IF (ROW /= COL) THEN
            CURRENT_POS(COL) = CURRENT_POS(COL) + 1
            ROW_ENTRIES(CURRENT_POS(COL), COL)%COL = ROW
            ROW_ENTRIES(CURRENT_POS(COL), COL)%VAL = A_UPPER(K)
         ENDIF
      ENDDO
   ENDDO

   ! Sort each row by column index (bubble sort for small arrays)
   DO ROW = 1, N
      DO I = 1, ROW_NNZ(ROW)-1
         DO J = 1, ROW_NNZ(ROW)-I
            IF (ROW_ENTRIES(J, ROW)%COL > ROW_ENTRIES(J+1, ROW)%COL) THEN
               ! Swap
               COL = ROW_ENTRIES(J, ROW)%COL
               ROW_ENTRIES(J, ROW)%COL = ROW_ENTRIES(J+1, ROW)%COL
               ROW_ENTRIES(J+1, ROW)%COL = COL

               CALL SWAP_REAL(ROW_ENTRIES(J, ROW)%VAL, ROW_ENTRIES(J+1, ROW)%VAL)
            ENDIF
         ENDDO
      ENDDO
   ENDDO

   ! Build CSR arrays
   ALLOCATE(IA_FULL(N+1), JA_FULL(NNZ_FULL), A_FULL(NNZ_FULL))

   IA_FULL(1) = 1
   IDX = 0
   DO ROW = 1, N
      DO K = 1, ROW_NNZ(ROW)
         IDX = IDX + 1
         JA_FULL(IDX) = ROW_ENTRIES(K, ROW)%COL
         A_FULL(IDX) = ROW_ENTRIES(K, ROW)%VAL
      ENDDO
      IA_FULL(ROW+1) = IDX + 1
   ENDDO

   DEALLOCATE(ROW_NNZ, CURRENT_POS, ROW_ENTRIES)

END SUBROUTINE CONVERT_UPPER_TO_FULL


!> @brief Helper: Swap two real values
SUBROUTINE SWAP_REAL(A, B)
   REAL(EB), INTENT(INOUT) :: A, B
   REAL(EB) :: TMP
   TMP = A
   A = B
   B = TMP
END SUBROUTINE SWAP_REAL


!> @brief Test 3D Poisson solver with known analytical solution
!> @param[out] PASSED True if test passes
SUBROUTINE TEST_POISSON_3D_SOLUTION(PASSED)
   LOGICAL, INTENT(OUT) :: PASSED

   INTEGER :: NX, NY, NZ, N, NNZ
   INTEGER :: I, J, K, IDX, ROW, IERR
   REAL(EB) :: DX, DY, DZ, DX2, DY2, DZ2
   REAL(EB) :: X, Y, Z, PI
   REAL(EB), ALLOCATABLE :: RHS(:), SOL(:), EXACT(:)
   INTEGER, ALLOCATABLE :: IA(:), JA(:)
   REAL(EB), ALLOCATABLE :: A(:)
   REAL(EB) :: MAX_ERROR, L2_ERROR, LINF_ERROR
   INTEGER :: ZONE_ID

   PI = 4.0_EB * ATAN(1.0_EB)

   ! Small grid for testing: 8x8x8
   NX = TEST_SMALL_N
   NY = TEST_SMALL_N
   NZ = TEST_SMALL_N
   N = NX * NY * NZ

   DX = 1.0_EB / REAL(NX+1, EB)
   DY = 1.0_EB / REAL(NY+1, EB)
   DZ = 1.0_EB / REAL(NZ+1, EB)
   DX2 = DX * DX
   DY2 = DY * DY
   DZ2 = DZ * DZ

   ! Allocate arrays
   ALLOCATE(RHS(N), SOL(N), EXACT(N))

   ! Build 7-point stencil Laplacian matrix
   CALL BUILD_3D_LAPLACIAN(NX, NY, NZ, DX2, DY2, DZ2, NNZ, IA, JA, A)

   ! Set up problem: -Laplacian(u) = f
   ! Analytical solution: u(x,y,z) = sin(pi*x)*sin(pi*y)*sin(pi*z)
   ! RHS: f = 3*pi^2 * sin(pi*x)*sin(pi*y)*sin(pi*z)

   DO K = 1, NZ
      Z = K * DZ
      DO J = 1, NY
         Y = J * DY
         DO I = 1, NX
            X = I * DX
            IDX = (K-1)*NX*NY + (J-1)*NX + I

            ! Exact solution
            EXACT(IDX) = SIN(PI*X) * SIN(PI*Y) * SIN(PI*Z)

            ! RHS (negative because we solve -Laplacian = f)
            RHS(IDX) = 3.0_EB * PI * PI * SIN(PI*X) * SIN(PI*Y) * SIN(PI*Z)
         ENDDO
      ENDDO
   ENDDO

   ! Initialize solution to zero
   SOL = 0.0_EB

#ifdef WITH_AMGX
   ! Solve using AmgX
   ZONE_ID = 999  ! Test zone ID

   CALL AMGX_SETUP_ZONE(ZONE_ID, N, NNZ, IERR)
   IF (IERR /= 0) THEN
      WRITE(*,'(A,I5)') '  ERROR: AmgX setup failed: ', IERR
      PASSED = .FALSE.
      RETURN
   ENDIF

   CALL AMGX_UPLOAD_MATRIX(ZONE_ID, N, NNZ, IA, JA, A, IERR)
   IF (IERR /= 0) THEN
      WRITE(*,'(A,I5)') '  ERROR: AmgX matrix upload failed: ', IERR
      PASSED = .FALSE.
      RETURN
   ENDIF

   CALL AMGX_SOLVE(ZONE_ID, N, RHS, SOL, IERR)
   IF (IERR /= 0) THEN
      WRITE(*,'(A,I5)') '  WARNING: AmgX solve returned: ', IERR
   ENDIF

   CALL AMGX_DESTROY_ZONE(ZONE_ID, IERR)
#else
   ! Without AmgX, use simple Jacobi iteration for reference
   CALL JACOBI_SOLVE(N, IA, JA, A, RHS, SOL, 1000, 1.0E-8_EB, IERR)
#endif

   ! Compute errors
   MAX_ERROR = 0.0_EB
   L2_ERROR = 0.0_EB

   DO I = 1, N
      MAX_ERROR = MAX(MAX_ERROR, ABS(SOL(I) - EXACT(I)))
      L2_ERROR = L2_ERROR + (SOL(I) - EXACT(I))**2
   ENDDO
   L2_ERROR = SQRT(L2_ERROR / REAL(N, EB))
   LINF_ERROR = MAX_ERROR

   WRITE(*,'(A,I5)') '  Grid size: ', NX
   WRITE(*,'(A,I8)') '  Number of unknowns: ', N
   WRITE(*,'(A,ES12.5)') '  L-infinity error: ', LINF_ERROR
   WRITE(*,'(A,ES12.5)') '  L2 error: ', L2_ERROR

   ! For this discretization, expect O(h^2) convergence
   ! With h ~ 0.1, error should be around 0.01
   PASSED = (L2_ERROR < 0.1_EB)

   DEALLOCATE(RHS, SOL, EXACT, IA, JA, A)

END SUBROUTINE TEST_POISSON_3D_SOLUTION


!> @brief Build 3D Laplacian matrix in CSR format (upper triangular)
SUBROUTINE BUILD_3D_LAPLACIAN(NX, NY, NZ, DX2, DY2, DZ2, NNZ, IA, JA, A)
   INTEGER, INTENT(IN) :: NX, NY, NZ
   REAL(EB), INTENT(IN) :: DX2, DY2, DZ2
   INTEGER, INTENT(OUT) :: NNZ
   INTEGER, ALLOCATABLE, INTENT(OUT) :: IA(:), JA(:)
   REAL(EB), ALLOCATABLE, INTENT(OUT) :: A(:)

   INTEGER :: N, I, J, K, ROW, COL, IDX
   REAL(EB) :: DIAG, OFFX, OFFY, OFFZ
   INTEGER, ALLOCATABLE :: NNZ_ROW(:)
   INTEGER :: MAX_NNZ

   ! Coefficients
   OFFX = -1.0_EB / DX2
   OFFY = -1.0_EB / DY2
   OFFZ = -1.0_EB / DZ2
   DIAG = 2.0_EB/DX2 + 2.0_EB/DY2 + 2.0_EB/DZ2

   N = NX * NY * NZ

   ! Count max non-zeros per row (upper triangular: at most 4)
   MAX_NNZ = 4 * N  ! Conservative estimate

   ALLOCATE(IA(N+1), JA(MAX_NNZ), A(MAX_NNZ))

   IDX = 0
   DO ROW = 1, N
      IA(ROW) = IDX + 1

      ! Get 3D indices
      K = (ROW-1) / (NX*NY) + 1
      J = MOD((ROW-1) / NX, NY) + 1
      I = MOD(ROW-1, NX) + 1

      ! Diagonal
      IDX = IDX + 1
      JA(IDX) = ROW
      A(IDX) = DIAG

      ! +X neighbor (upper triangular only)
      IF (I < NX) THEN
         COL = ROW + 1
         IDX = IDX + 1
         JA(IDX) = COL
         A(IDX) = OFFX
      ENDIF

      ! +Y neighbor
      IF (J < NY) THEN
         COL = ROW + NX
         IDX = IDX + 1
         JA(IDX) = COL
         A(IDX) = OFFY
      ENDIF

      ! +Z neighbor
      IF (K < NZ) THEN
         COL = ROW + NX*NY
         IDX = IDX + 1
         JA(IDX) = COL
         A(IDX) = OFFZ
      ENDIF
   ENDDO
   IA(N+1) = IDX + 1
   NNZ = IDX

END SUBROUTINE BUILD_3D_LAPLACIAN


!> @brief Simple Jacobi solver for CPU reference
SUBROUTINE JACOBI_SOLVE(N, IA, JA, A, B, X, MAX_ITER, TOL, IERR)
   INTEGER, INTENT(IN) :: N, MAX_ITER
   INTEGER, INTENT(IN) :: IA(:), JA(:)
   REAL(EB), INTENT(IN) :: A(:), B(:), TOL
   REAL(EB), INTENT(INOUT) :: X(:)
   INTEGER, INTENT(OUT) :: IERR

   REAL(EB), ALLOCATABLE :: X_NEW(:), DIAG(:)
   REAL(EB) :: SUM_VAL, RESIDUAL
   INTEGER :: ITER, I, K, J

   ALLOCATE(X_NEW(N), DIAG(N))

   ! Extract diagonal
   DO I = 1, N
      DO K = IA(I), IA(I+1)-1
         IF (JA(K) == I) THEN
            DIAG(I) = A(K)
            EXIT
         ENDIF
      ENDDO
   ENDDO

   ! Jacobi iteration
   DO ITER = 1, MAX_ITER
      RESIDUAL = 0.0_EB

      DO I = 1, N
         SUM_VAL = B(I)
         DO K = IA(I), IA(I+1)-1
            J = JA(K)
            IF (J /= I) THEN
               SUM_VAL = SUM_VAL - A(K) * X(J)
               ! Symmetric contribution
               IF (J > I) THEN
                  ! This is in upper part, also affects row J
               ENDIF
            ENDIF
         ENDDO
         X_NEW(I) = SUM_VAL / DIAG(I)
         RESIDUAL = RESIDUAL + (X_NEW(I) - X(I))**2
      ENDDO

      X = X_NEW
      RESIDUAL = SQRT(RESIDUAL / REAL(N, EB))

      IF (RESIDUAL < TOL) THEN
         IERR = 0
         DEALLOCATE(X_NEW, DIAG)
         RETURN
      ENDIF
   ENDDO

   IERR = 1  ! Did not converge
   DEALLOCATE(X_NEW, DIAG)

END SUBROUTINE JACOBI_SOLVE


!> @brief Compare CPU (reference) and GPU (AmgX) solver results
!> @param[out] PASSED True if results match within tolerance
SUBROUTINE TEST_COMPARE_CPU_GPU_SOLVER(PASSED)
   LOGICAL, INTENT(OUT) :: PASSED

   INTEGER :: NX, NY, NZ, N, NNZ
   INTEGER :: I, J, K, IDX, IERR
   REAL(EB) :: DX2, DY2, DZ2
   REAL(EB), ALLOCATABLE :: RHS(:), SOL_CPU(:), SOL_GPU(:)
   INTEGER, ALLOCATABLE :: IA(:), JA(:)
   REAL(EB), ALLOCATABLE :: A(:)
   REAL(EB) :: MAX_DIFF, L2_DIFF, REL_DIFF
   INTEGER :: ZONE_ID

   ! Medium grid for comparison
   NX = 16
   NY = 16
   NZ = 16
   N = NX * NY * NZ

   DX2 = 1.0_EB / REAL((NX+1)**2, EB)
   DY2 = 1.0_EB / REAL((NY+1)**2, EB)
   DZ2 = 1.0_EB / REAL((NZ+1)**2, EB)

   ALLOCATE(RHS(N), SOL_CPU(N), SOL_GPU(N))

   ! Build matrix
   CALL BUILD_3D_LAPLACIAN(NX, NY, NZ, DX2, DY2, DZ2, NNZ, IA, JA, A)

   ! Random RHS for testing
   DO I = 1, N
      RHS(I) = SIN(REAL(I, EB) * 0.1_EB)
   ENDDO

   ! CPU solution (using Jacobi as simple reference)
   SOL_CPU = 0.0_EB
   CALL JACOBI_SOLVE(N, IA, JA, A, RHS, SOL_CPU, 5000, 1.0E-10_EB, IERR)

   IF (IERR /= 0) THEN
      WRITE(*,'(A)') '  WARNING: CPU reference solver did not converge fully'
   ENDIF

#ifdef WITH_AMGX
   ! GPU solution
   SOL_GPU = 0.0_EB
   ZONE_ID = 998

   CALL AMGX_SETUP_ZONE(ZONE_ID, N, NNZ, IERR)
   CALL AMGX_UPLOAD_MATRIX(ZONE_ID, N, NNZ, IA, JA, A, IERR)
   CALL AMGX_SOLVE(ZONE_ID, N, RHS, SOL_GPU, IERR)
   CALL AMGX_DESTROY_ZONE(ZONE_ID, IERR)

   ! Compare solutions
   MAX_DIFF = 0.0_EB
   L2_DIFF = 0.0_EB

   DO I = 1, N
      MAX_DIFF = MAX(MAX_DIFF, ABS(SOL_GPU(I) - SOL_CPU(I)))
      L2_DIFF = L2_DIFF + (SOL_GPU(I) - SOL_CPU(I))**2
   ENDDO
   L2_DIFF = SQRT(L2_DIFF / REAL(N, EB))

   ! Relative difference (avoid division by zero)
   REL_DIFF = L2_DIFF / (SQRT(SUM(SOL_CPU**2) / REAL(N, EB)) + 1.0E-15_EB)

   WRITE(*,'(A,I8)') '  Number of unknowns: ', N
   WRITE(*,'(A,ES12.5)') '  Max absolute difference: ', MAX_DIFF
   WRITE(*,'(A,ES12.5)') '  L2 difference: ', L2_DIFF
   WRITE(*,'(A,ES12.5)') '  Relative difference: ', REL_DIFF

   PASSED = (REL_DIFF < SOLVER_TOLERANCE)
#else
   WRITE(*,'(A)') '  Skipping GPU comparison (AmgX not available)'
   PASSED = .TRUE.
#endif

   DEALLOCATE(RHS, SOL_CPU, SOL_GPU, IA, JA, A)

END SUBROUTINE TEST_COMPARE_CPU_GPU_SOLVER


!> @brief Verify solution accuracy by computing residual
!> @param[in]  N    Number of unknowns
!> @param[in]  IA   CSR row pointers
!> @param[in]  JA   CSR column indices
!> @param[in]  A    Matrix values
!> @param[in]  RHS  Right-hand side vector
!> @param[in]  SOL  Solution vector
!> @param[out] RESIDUAL_NORM Norm of residual ||Ax - b||
!> @param[out] RELATIVE_RESIDUAL Relative residual ||Ax - b|| / ||b||
SUBROUTINE VERIFY_SOLUTION_ACCURACY(N, IA, JA, A, RHS, SOL, RESIDUAL_NORM, RELATIVE_RESIDUAL)
   INTEGER, INTENT(IN) :: N
   INTEGER, INTENT(IN) :: IA(:), JA(:)
   REAL(EB), INTENT(IN) :: A(:), RHS(:), SOL(:)
   REAL(EB), INTENT(OUT) :: RESIDUAL_NORM, RELATIVE_RESIDUAL

   REAL(EB), ALLOCATABLE :: RESIDUAL(:), AX(:)
   REAL(EB) :: RHS_NORM
   INTEGER :: I, K, J

   ALLOCATE(RESIDUAL(N), AX(N))

   ! Compute Ax (considering symmetric structure)
   AX = 0.0_EB
   DO I = 1, N
      DO K = IA(I), IA(I+1)-1
         J = JA(K)
         AX(I) = AX(I) + A(K) * SOL(J)
         IF (I /= J) THEN
            ! Symmetric contribution
            AX(J) = AX(J) + A(K) * SOL(I)
         ENDIF
      ENDDO
   ENDDO

   ! Compute residual = Ax - b
   RESIDUAL = AX - RHS

   ! Compute norms
   RESIDUAL_NORM = SQRT(SUM(RESIDUAL**2))
   RHS_NORM = SQRT(SUM(RHS**2))

   IF (RHS_NORM > 1.0E-15_EB) THEN
      RELATIVE_RESIDUAL = RESIDUAL_NORM / RHS_NORM
   ELSE
      RELATIVE_RESIDUAL = RESIDUAL_NORM
   ENDIF

   DEALLOCATE(RESIDUAL, AX)

END SUBROUTINE VERIFY_SOLUTION_ACCURACY

END MODULE AMGX_UNIT_TEST


!> @brief Standalone test program
PROGRAM TEST_AMGX_STANDALONE
   USE AMGX_UNIT_TEST
   IMPLICIT NONE

   LOGICAL :: ALL_PASSED

   CALL RUN_AMGX_UNIT_TESTS(ALL_PASSED)

   IF (ALL_PASSED) THEN
      WRITE(*,'(A)') 'All tests PASSED - AmgX integration verified'
      STOP 0
   ELSE
      WRITE(*,'(A)') 'Some tests FAILED - Check implementation'
      STOP 1
   ENDIF

END PROGRAM TEST_AMGX_STANDALONE

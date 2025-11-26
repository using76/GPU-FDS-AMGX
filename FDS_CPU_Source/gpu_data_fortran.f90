!> @file gpu_data_fortran.f90
!> @brief Fortran interface module for GPU data manager
!>
!> This module provides Fortran bindings to the gpu_data_manager.c functions.
!> It handles allocation and transfer of FDS field data to/from GPU.
!>
!> Usage in FDS:
!>   1. Call GPU_DATA_MANAGER_INIT() at program start
!>   2. Call GPU_ALLOCATE_MESH() for each mesh during setup
!>   3. Call GPU_UPLOAD_GEOMETRY() once per mesh
!>   4. Call GPU_UPLOAD_VELOCITY(), GPU_UPLOAD_SCALARS() each timestep
!>   5. Call GPU_DOWNLOAD_VELOCITY_FLUX() after GPU computation
!>   6. Call GPU_DATA_MANAGER_FINALIZE() at program end

MODULE GPU_DATA_FORTRAN

USE ISO_C_BINDING
USE PRECISION_PARAMETERS, ONLY: EB

IMPLICIT NONE

PRIVATE

! Public procedures - Initialization/Finalization
PUBLIC :: GPU_DATA_MANAGER_INIT
PUBLIC :: GPU_DATA_MANAGER_FINALIZE

! Public procedures - Memory Management
PUBLIC :: GPU_ALLOCATE_MESH
PUBLIC :: GPU_DEALLOCATE_MESH
PUBLIC :: GPU_IS_MESH_READY

! Public procedures - Geometry Upload
PUBLIC :: GPU_UPLOAD_GEOMETRY

! Public procedures - Field Uploads (CPU -> GPU)
PUBLIC :: GPU_UPLOAD_VELOCITY
PUBLIC :: GPU_UPLOAD_SCALARS
PUBLIC :: GPU_UPLOAD_SPECIES

! Public procedures - Result Downloads (GPU -> CPU)
PUBLIC :: GPU_DOWNLOAD_VELOCITY_FLUX
PUBLIC :: GPU_DOWNLOAD_DIFFUSION_FLUX

! Public procedures - Synchronization
PUBLIC :: GPU_SYNC_COMPUTE

! Public procedures - Status/Diagnostics
PUBLIC :: GPU_GET_MEMORY_USAGE
PUBLIC :: GPU_GET_MESH_DIMS

! C interface declarations
INTERFACE

   !> Initialize GPU data manager
   SUBROUTINE gpu_data_manager_init_c(ierr) BIND(C, NAME='gpu_data_manager_init_')
      IMPORT :: C_INT
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE gpu_data_manager_init_c

   !> Finalize GPU data manager
   SUBROUTINE gpu_data_manager_finalize_c(ierr) BIND(C, NAME='gpu_data_manager_finalize_')
      IMPORT :: C_INT
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE gpu_data_manager_finalize_c

   !> Allocate GPU memory for a mesh
   SUBROUTINE gpu_allocate_mesh_c(mesh_id, ibar, jbar, kbar, n_species, ierr) &
              BIND(C, NAME='gpu_allocate_mesh_')
      IMPORT :: C_INT
      INTEGER(C_INT), INTENT(IN) :: mesh_id
      INTEGER(C_INT), INTENT(IN) :: ibar, jbar, kbar
      INTEGER(C_INT), INTENT(IN) :: n_species
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE gpu_allocate_mesh_c

   !> Deallocate GPU memory for a mesh
   SUBROUTINE gpu_deallocate_mesh_c(mesh_id, ierr) BIND(C, NAME='gpu_deallocate_mesh_')
      IMPORT :: C_INT
      INTEGER(C_INT), INTENT(IN) :: mesh_id
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE gpu_deallocate_mesh_c

   !> Check if mesh GPU data is ready
   SUBROUTINE gpu_is_mesh_ready_c(mesh_id, is_valid, ierr) BIND(C, NAME='gpu_is_mesh_ready_')
      IMPORT :: C_INT
      INTEGER(C_INT), INTENT(IN) :: mesh_id
      INTEGER(C_INT), INTENT(OUT) :: is_valid
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE gpu_is_mesh_ready_c

   !> Upload grid geometry to GPU
   SUBROUTINE gpu_upload_geometry_c(mesh_id, dx, dy, dz, rdx, rdy, rdz, &
                                     rdxn, rdyn, rdzn, rho_0, ierr) &
              BIND(C, NAME='gpu_upload_geometry_')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), INTENT(IN) :: mesh_id
      REAL(C_DOUBLE), INTENT(IN) :: dx(*), dy(*), dz(*)
      REAL(C_DOUBLE), INTENT(IN) :: rdx(*), rdy(*), rdz(*)
      REAL(C_DOUBLE), INTENT(IN) :: rdxn(*), rdyn(*), rdzn(*)
      REAL(C_DOUBLE), INTENT(IN) :: rho_0(*)
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE gpu_upload_geometry_c

   !> Upload velocity fields to GPU
   SUBROUTINE gpu_upload_velocity_c(mesh_id, u, v, w, ierr) &
              BIND(C, NAME='gpu_upload_velocity_')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), INTENT(IN) :: mesh_id
      REAL(C_DOUBLE), INTENT(IN) :: u(*), v(*), w(*)
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE gpu_upload_velocity_c

   !> Upload scalar fields to GPU
   SUBROUTINE gpu_upload_scalars_c(mesh_id, rho, tmp, mu, ierr) &
              BIND(C, NAME='gpu_upload_scalars_')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), INTENT(IN) :: mesh_id
      REAL(C_DOUBLE), INTENT(IN) :: rho(*), tmp(*), mu(*)
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE gpu_upload_scalars_c

   !> Upload species mass fractions to GPU
   SUBROUTINE gpu_upload_species_c(mesh_id, zz, n_species, ierr) &
              BIND(C, NAME='gpu_upload_species_')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), INTENT(IN) :: mesh_id
      REAL(C_DOUBLE), INTENT(IN) :: zz(*)
      INTEGER(C_INT), INTENT(IN) :: n_species
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE gpu_upload_species_c

   !> Download velocity flux results from GPU
   SUBROUTINE gpu_download_velocity_flux_c(mesh_id, fvx, fvy, fvz, ierr) &
              BIND(C, NAME='gpu_download_velocity_flux_')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), INTENT(IN) :: mesh_id
      REAL(C_DOUBLE), INTENT(OUT) :: fvx(*), fvy(*), fvz(*)
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE gpu_download_velocity_flux_c

   !> Download diffusion flux results from GPU
   SUBROUTINE gpu_download_diffusion_flux_c(mesh_id, kdtdx, kdtdy, kdtdz, ierr) &
              BIND(C, NAME='gpu_download_diffusion_flux_')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), INTENT(IN) :: mesh_id
      REAL(C_DOUBLE), INTENT(OUT) :: kdtdx(*), kdtdy(*), kdtdz(*)
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE gpu_download_diffusion_flux_c

   !> Synchronize GPU compute stream
   SUBROUTINE gpu_sync_compute_c(mesh_id, ierr) BIND(C, NAME='gpu_sync_compute_')
      IMPORT :: C_INT
      INTEGER(C_INT), INTENT(IN) :: mesh_id
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE gpu_sync_compute_c

   !> Get GPU memory usage
   SUBROUTINE gpu_get_memory_usage_c(used_mb, total_mb, ierr) &
              BIND(C, NAME='gpu_get_memory_usage_')
      IMPORT :: C_INT, C_DOUBLE
      REAL(C_DOUBLE), INTENT(OUT) :: used_mb, total_mb
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE gpu_get_memory_usage_c

   !> Get mesh dimensions
   SUBROUTINE gpu_get_mesh_dims_c(mesh_id, ibar, jbar, kbar, ierr) &
              BIND(C, NAME='gpu_get_mesh_dims_')
      IMPORT :: C_INT
      INTEGER(C_INT), INTENT(IN) :: mesh_id
      INTEGER(C_INT), INTENT(OUT) :: ibar, jbar, kbar
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE gpu_get_mesh_dims_c

END INTERFACE

CONTAINS

! ============================================================================
! Initialization / Finalization
! ============================================================================

!> @brief Initialize GPU data manager
!> @param[out] IERR Error code (0 = success)
SUBROUTINE GPU_DATA_MANAGER_INIT(IERR)
   INTEGER, INTENT(OUT) :: IERR
   INTEGER(C_INT) :: C_IERR

   CALL gpu_data_manager_init_c(C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_DATA_MANAGER_INIT

!> @brief Finalize GPU data manager
!> @param[out] IERR Error code (0 = success)
SUBROUTINE GPU_DATA_MANAGER_FINALIZE(IERR)
   INTEGER, INTENT(OUT) :: IERR
   INTEGER(C_INT) :: C_IERR

   CALL gpu_data_manager_finalize_c(C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_DATA_MANAGER_FINALIZE

! ============================================================================
! Memory Management
! ============================================================================

!> @brief Allocate GPU memory for a mesh
!> @param[in]  NM        Mesh number (1-based)
!> @param[in]  IBAR      Number of cells in X
!> @param[in]  JBAR      Number of cells in Y
!> @param[in]  KBAR      Number of cells in Z
!> @param[in]  N_SPECIES Number of tracked species
!> @param[out] IERR      Error code (0 = success)
SUBROUTINE GPU_ALLOCATE_MESH(NM, IBAR, JBAR, KBAR, N_SPECIES, IERR)
   INTEGER, INTENT(IN) :: NM
   INTEGER, INTENT(IN) :: IBAR, JBAR, KBAR
   INTEGER, INTENT(IN) :: N_SPECIES
   INTEGER, INTENT(OUT) :: IERR

   INTEGER(C_INT) :: C_NM, C_IBAR, C_JBAR, C_KBAR, C_NSPEC, C_IERR

   C_NM = INT(NM, C_INT)
   C_IBAR = INT(IBAR, C_INT)
   C_JBAR = INT(JBAR, C_INT)
   C_KBAR = INT(KBAR, C_INT)
   C_NSPEC = INT(N_SPECIES, C_INT)

   CALL gpu_allocate_mesh_c(C_NM, C_IBAR, C_JBAR, C_KBAR, C_NSPEC, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_ALLOCATE_MESH

!> @brief Deallocate GPU memory for a mesh
!> @param[in]  NM   Mesh number
!> @param[out] IERR Error code
SUBROUTINE GPU_DEALLOCATE_MESH(NM, IERR)
   INTEGER, INTENT(IN) :: NM
   INTEGER, INTENT(OUT) :: IERR

   INTEGER(C_INT) :: C_NM, C_IERR

   C_NM = INT(NM, C_INT)
   CALL gpu_deallocate_mesh_c(C_NM, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_DEALLOCATE_MESH

!> @brief Check if mesh GPU data is ready
!> @param[in]  NM       Mesh number
!> @param[out] IS_READY True if GPU data is allocated and ready
!> @param[out] IERR     Error code
SUBROUTINE GPU_IS_MESH_READY(NM, IS_READY, IERR)
   INTEGER, INTENT(IN) :: NM
   LOGICAL, INTENT(OUT) :: IS_READY
   INTEGER, INTENT(OUT) :: IERR

   INTEGER(C_INT) :: C_NM, C_VALID, C_IERR

   C_NM = INT(NM, C_INT)
   CALL gpu_is_mesh_ready_c(C_NM, C_VALID, C_IERR)

   IS_READY = (C_VALID == 1)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_IS_MESH_READY

! ============================================================================
! Geometry Upload
! ============================================================================

!> @brief Upload grid geometry to GPU (call once during setup)
!>
!> Uploads cell widths and their reciprocals to GPU constant memory.
!> These are used by diffusion and advection kernels.
!>
!> @param[in]  NM   Mesh number
!> @param[in]  DX   Cell widths in X (0:IBAR+1)
!> @param[in]  DY   Cell widths in Y (0:JBAR+1)
!> @param[in]  DZ   Cell widths in Z (0:KBAR+1)
!> @param[in]  RDX  1/DX at cell centers
!> @param[in]  RDY  1/DY at cell centers
!> @param[in]  RDZ  1/DZ at cell centers
!> @param[in]  RDXN 1/DX at cell faces
!> @param[in]  RDYN 1/DY at cell faces
!> @param[in]  RDZN 1/DZ at cell faces
!> @param[in]  RHO_0 Background density profile
!> @param[out] IERR Error code
SUBROUTINE GPU_UPLOAD_GEOMETRY(NM, DX, DY, DZ, RDX, RDY, RDZ, RDXN, RDYN, RDZN, RHO_0, IERR)
   INTEGER, INTENT(IN) :: NM
   REAL(EB), INTENT(IN) :: DX(:), DY(:), DZ(:)
   REAL(EB), INTENT(IN) :: RDX(:), RDY(:), RDZ(:)
   REAL(EB), INTENT(IN) :: RDXN(:), RDYN(:), RDZN(:)
   REAL(EB), INTENT(IN) :: RHO_0(:)
   INTEGER, INTENT(OUT) :: IERR

   INTEGER(C_INT) :: C_NM, C_IERR

   C_NM = INT(NM, C_INT)

   CALL gpu_upload_geometry_c(C_NM, DX, DY, DZ, RDX, RDY, RDZ, RDXN, RDYN, RDZN, RHO_0, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_UPLOAD_GEOMETRY

! ============================================================================
! Field Uploads (CPU -> GPU)
! ============================================================================

!> @brief Upload velocity fields to GPU
!>
!> Uploads U, V, W velocity components from FDS arrays to GPU.
!> Call this before running velocity flux GPU kernel.
!>
!> @param[in]  NM   Mesh number
!> @param[in]  U    X-velocity at X-faces
!> @param[in]  V    Y-velocity at Y-faces
!> @param[in]  W    Z-velocity at Z-faces
!> @param[out] IERR Error code
SUBROUTINE GPU_UPLOAD_VELOCITY(NM, U, V, W, IERR)
   INTEGER, INTENT(IN) :: NM
   REAL(EB), INTENT(IN) :: U(:,:,:)
   REAL(EB), INTENT(IN) :: V(:,:,:)
   REAL(EB), INTENT(IN) :: W(:,:,:)
   INTEGER, INTENT(OUT) :: IERR

   INTEGER(C_INT) :: C_NM, C_IERR

   C_NM = INT(NM, C_INT)

   ! Note: Fortran arrays are passed directly; C wrapper handles layout
   CALL gpu_upload_velocity_c(C_NM, U, V, W, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_UPLOAD_VELOCITY

!> @brief Upload scalar fields (RHO, TMP, MU) to GPU
!>
!> Uploads density, temperature, and viscosity from FDS to GPU.
!> Call this before running diffusion GPU kernel.
!>
!> @param[in]  NM   Mesh number
!> @param[in]  RHO  Density
!> @param[in]  TMP  Temperature
!> @param[in]  MU   Dynamic viscosity
!> @param[out] IERR Error code
SUBROUTINE GPU_UPLOAD_SCALARS(NM, RHO, TMP, MU, IERR)
   INTEGER, INTENT(IN) :: NM
   REAL(EB), INTENT(IN) :: RHO(:,:,:)
   REAL(EB), INTENT(IN) :: TMP(:,:,:)
   REAL(EB), INTENT(IN) :: MU(:,:,:)
   INTEGER, INTENT(OUT) :: IERR

   INTEGER(C_INT) :: C_NM, C_IERR

   C_NM = INT(NM, C_INT)

   CALL gpu_upload_scalars_c(C_NM, RHO, TMP, MU, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_UPLOAD_SCALARS

!> @brief Upload species mass fractions to GPU
!>
!> Uploads ZZ array from FDS to GPU for species diffusion calculation.
!>
!> @param[in]  NM        Mesh number
!> @param[in]  ZZ        Species mass fractions (IBAR+2, JBAR+2, KBAR+2, N_SPECIES)
!> @param[in]  N_SPECIES Number of species
!> @param[out] IERR      Error code
SUBROUTINE GPU_UPLOAD_SPECIES(NM, ZZ, N_SPECIES, IERR)
   INTEGER, INTENT(IN) :: NM
   REAL(EB), INTENT(IN) :: ZZ(:,:,:,:)
   INTEGER, INTENT(IN) :: N_SPECIES
   INTEGER, INTENT(OUT) :: IERR

   INTEGER(C_INT) :: C_NM, C_NSPEC, C_IERR

   C_NM = INT(NM, C_INT)
   C_NSPEC = INT(N_SPECIES, C_INT)

   CALL gpu_upload_species_c(C_NM, ZZ, C_NSPEC, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_UPLOAD_SPECIES

! ============================================================================
! Result Downloads (GPU -> CPU)
! ============================================================================

!> @brief Download velocity flux results from GPU
!>
!> Downloads FVX, FVY, FVZ computed on GPU back to FDS arrays.
!> Call this after velocity flux GPU kernel completes.
!>
!> @param[in]  NM   Mesh number
!> @param[out] FVX  X-momentum flux
!> @param[out] FVY  Y-momentum flux
!> @param[out] FVZ  Z-momentum flux
!> @param[out] IERR Error code
SUBROUTINE GPU_DOWNLOAD_VELOCITY_FLUX(NM, FVX, FVY, FVZ, IERR)
   INTEGER, INTENT(IN) :: NM
   REAL(EB), INTENT(OUT) :: FVX(:,:,:)
   REAL(EB), INTENT(OUT) :: FVY(:,:,:)
   REAL(EB), INTENT(OUT) :: FVZ(:,:,:)
   INTEGER, INTENT(OUT) :: IERR

   INTEGER(C_INT) :: C_NM, C_IERR

   C_NM = INT(NM, C_INT)

   CALL gpu_download_velocity_flux_c(C_NM, FVX, FVY, FVZ, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_DOWNLOAD_VELOCITY_FLUX

!> @brief Download diffusion flux results from GPU
!>
!> Downloads KDTDX, KDTDY, KDTDZ computed on GPU back to FDS arrays.
!> Call this after thermal diffusion GPU kernel completes.
!>
!> @param[in]  NM    Mesh number
!> @param[out] KDTDX Thermal flux X: k * dT/dx
!> @param[out] KDTDY Thermal flux Y: k * dT/dy
!> @param[out] KDTDZ Thermal flux Z: k * dT/dz
!> @param[out] IERR  Error code
SUBROUTINE GPU_DOWNLOAD_DIFFUSION_FLUX(NM, KDTDX, KDTDY, KDTDZ, IERR)
   INTEGER, INTENT(IN) :: NM
   REAL(EB), INTENT(OUT) :: KDTDX(:,:,:)
   REAL(EB), INTENT(OUT) :: KDTDY(:,:,:)
   REAL(EB), INTENT(OUT) :: KDTDZ(:,:,:)
   INTEGER, INTENT(OUT) :: IERR

   INTEGER(C_INT) :: C_NM, C_IERR

   C_NM = INT(NM, C_INT)

   CALL gpu_download_diffusion_flux_c(C_NM, KDTDX, KDTDY, KDTDZ, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_DOWNLOAD_DIFFUSION_FLUX

! ============================================================================
! Synchronization
! ============================================================================

!> @brief Synchronize GPU compute stream
!>
!> Blocks until all GPU operations on the compute stream complete.
!> Call this before downloading results that depend on GPU computation.
!>
!> @param[in]  NM   Mesh number
!> @param[out] IERR Error code
SUBROUTINE GPU_SYNC_COMPUTE(NM, IERR)
   INTEGER, INTENT(IN) :: NM
   INTEGER, INTENT(OUT) :: IERR

   INTEGER(C_INT) :: C_NM, C_IERR

   C_NM = INT(NM, C_INT)

   CALL gpu_sync_compute_c(C_NM, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_SYNC_COMPUTE

! ============================================================================
! Status / Diagnostics
! ============================================================================

!> @brief Get GPU memory usage statistics
!>
!> Reports current GPU memory usage for monitoring and diagnostics.
!>
!> @param[out] USED_MB  Used GPU memory in MB
!> @param[out] TOTAL_MB Total GPU memory in MB
!> @param[out] IERR     Error code
SUBROUTINE GPU_GET_MEMORY_USAGE(USED_MB, TOTAL_MB, IERR)
   REAL(EB), INTENT(OUT) :: USED_MB
   REAL(EB), INTENT(OUT) :: TOTAL_MB
   INTEGER, INTENT(OUT) :: IERR

   REAL(C_DOUBLE) :: C_USED, C_TOTAL
   INTEGER(C_INT) :: C_IERR

   CALL gpu_get_memory_usage_c(C_USED, C_TOTAL, C_IERR)

   USED_MB = REAL(C_USED, EB)
   TOTAL_MB = REAL(C_TOTAL, EB)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_GET_MEMORY_USAGE

!> @brief Get mesh dimensions stored on GPU
!>
!> @param[in]  NM   Mesh number
!> @param[out] IBAR Number of cells in X
!> @param[out] JBAR Number of cells in Y
!> @param[out] KBAR Number of cells in Z
!> @param[out] IERR Error code
SUBROUTINE GPU_GET_MESH_DIMS(NM, IBAR, JBAR, KBAR, IERR)
   INTEGER, INTENT(IN) :: NM
   INTEGER, INTENT(OUT) :: IBAR, JBAR, KBAR
   INTEGER, INTENT(OUT) :: IERR

   INTEGER(C_INT) :: C_NM, C_IBAR, C_JBAR, C_KBAR, C_IERR

   C_NM = INT(NM, C_INT)

   CALL gpu_get_mesh_dims_c(C_NM, C_IBAR, C_JBAR, C_KBAR, C_IERR)

   IBAR = INT(C_IBAR)
   JBAR = INT(C_JBAR)
   KBAR = INT(C_KBAR)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_GET_MESH_DIMS

END MODULE GPU_DATA_FORTRAN

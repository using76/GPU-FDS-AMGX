!> @file gpu_fortran.f90
!> @brief Fortran Interface for FDS GPU Kernels
!>
!> Provides high-level Fortran interface to GPU-accelerated diffusion
!> and advection calculations via ISO_C_BINDING.
!>
!> Part of FDS GPU Acceleration Project
!> Copyright 2024, FDS-AmgX Project

MODULE GPU_FORTRAN

USE ISO_C_BINDING
USE PRECISION_PARAMETERS, ONLY: EB, FB

IMPLICIT NONE

PRIVATE

!============================================================================
! Public Procedures
!============================================================================

! Initialization/Finalization
PUBLIC :: GPU_KERNEL_INIT
PUBLIC :: GPU_KERNEL_FINALIZE

! Mesh Setup
PUBLIC :: GPU_KERNEL_ALLOCATE_MESH
PUBLIC :: GPU_KERNEL_UPLOAD_GRID
PUBLIC :: GPU_KERNEL_UPLOAD_GRAVITY

! Diffusion Kernels
PUBLIC :: GPU_COMPUTE_SPECIES_DIFFUSION
PUBLIC :: GPU_COMPUTE_SPECIES_DIFFUSION_SINGLE
PUBLIC :: GPU_COMPUTE_THERMAL_DIFFUSION

! Velocity Flux Kernels
PUBLIC :: GPU_COMPUTE_VORTICITY_STRESS
PUBLIC :: GPU_COMPUTE_VELOCITY_FLUX

! Advection Kernels
PUBLIC :: GPU_COMPUTE_ADVECTION
PUBLIC :: GPU_COMPUTE_ENTHALPY_ADVECTION
PUBLIC :: GPU_COMPUTE_SPECIES_ADVECTION

! Density Update Kernels
PUBLIC :: GPU_COMPUTE_DENSITY_UPDATE

! Query Functions
PUBLIC :: GPU_KERNEL_AVAILABLE
PUBLIC :: GPU_GET_MESH_CELLS

!============================================================================
! Flux Limiter Constants
!============================================================================

INTEGER, PARAMETER, PUBLIC :: FLUX_LIMITER_UPWIND   = 0
INTEGER, PARAMETER, PUBLIC :: FLUX_LIMITER_SUPERBEE = 1
INTEGER, PARAMETER, PUBLIC :: FLUX_LIMITER_CHARM    = 2

!============================================================================
! C Interface Declarations
!============================================================================

INTERFACE

   !-------------------------------------------------------------------------
   ! Initialization/Finalization
   !-------------------------------------------------------------------------

   SUBROUTINE gpu_kernel_init_c(ierr) BIND(C, NAME='gpu_kernel_init_')
      IMPORT :: C_INT
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE gpu_kernel_init_c

   SUBROUTINE gpu_kernel_finalize_c(ierr) BIND(C, NAME='gpu_kernel_finalize_')
      IMPORT :: C_INT
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE gpu_kernel_finalize_c

   !-------------------------------------------------------------------------
   ! Mesh Setup
   !-------------------------------------------------------------------------

   SUBROUTINE gpu_kernel_allocate_mesh_c(mesh_id, ibar, jbar, kbar, n_species, ierr) &
              BIND(C, NAME='gpu_kernel_allocate_mesh_')
      IMPORT :: C_INT
      INTEGER(C_INT), INTENT(IN)  :: mesh_id, ibar, jbar, kbar, n_species
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE gpu_kernel_allocate_mesh_c

   SUBROUTINE gpu_kernel_upload_grid_c(mesh_id, rdx, rdy, rdz, rdxn, rdyn, rdzn, rho_0, ierr) &
              BIND(C, NAME='gpu_kernel_upload_grid_')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), INTENT(IN)   :: mesh_id
      REAL(C_DOUBLE), INTENT(IN)   :: rdx(*), rdy(*), rdz(*)
      REAL(C_DOUBLE), INTENT(IN)   :: rdxn(*), rdyn(*), rdzn(*)
      REAL(C_DOUBLE), INTENT(IN)   :: rho_0(*)
      INTEGER(C_INT), INTENT(OUT)  :: ierr
   END SUBROUTINE gpu_kernel_upload_grid_c

   SUBROUTINE gpu_kernel_upload_gravity_c(mesh_id, gx, gy, gz, ierr) &
              BIND(C, NAME='gpu_kernel_upload_gravity_')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), INTENT(IN)  :: mesh_id
      REAL(C_DOUBLE), INTENT(IN)  :: gx(*), gy(*), gz(*)
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE gpu_kernel_upload_gravity_c

   !-------------------------------------------------------------------------
   ! Diffusion Kernels
   !-------------------------------------------------------------------------

   SUBROUTINE gpu_compute_species_diffusion_c(mesh_id, zzp, rho_d, &
              rho_d_dzdx, rho_d_dzdy, rho_d_dzdz, n_species, ierr) &
              BIND(C, NAME='gpu_compute_species_diffusion_')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), INTENT(IN)    :: mesh_id
      REAL(C_DOUBLE), INTENT(IN)    :: zzp(*), rho_d(*)
      REAL(C_DOUBLE), INTENT(INOUT) :: rho_d_dzdx(*), rho_d_dzdy(*), rho_d_dzdz(*)
      INTEGER(C_INT), INTENT(IN)    :: n_species
      INTEGER(C_INT), INTENT(OUT)   :: ierr
   END SUBROUTINE gpu_compute_species_diffusion_c

   SUBROUTINE gpu_compute_species_diffusion_single_c(mesh_id, zz_n, rho_d, &
              rho_d_dzdx_n, rho_d_dzdy_n, rho_d_dzdz_n, ierr) &
              BIND(C, NAME='gpu_compute_species_diffusion_single_')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), INTENT(IN)    :: mesh_id
      REAL(C_DOUBLE), INTENT(IN)    :: zz_n(*), rho_d(*)
      REAL(C_DOUBLE), INTENT(INOUT) :: rho_d_dzdx_n(*), rho_d_dzdy_n(*), rho_d_dzdz_n(*)
      INTEGER(C_INT), INTENT(OUT)   :: ierr
   END SUBROUTINE gpu_compute_species_diffusion_single_c

   SUBROUTINE gpu_compute_thermal_diffusion_c(mesh_id, tmp, kp, &
              kdtdx, kdtdy, kdtdz, ierr) &
              BIND(C, NAME='gpu_compute_thermal_diffusion_')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), INTENT(IN)    :: mesh_id
      REAL(C_DOUBLE), INTENT(IN)    :: tmp(*), kp(*)
      REAL(C_DOUBLE), INTENT(INOUT) :: kdtdx(*), kdtdy(*), kdtdz(*)
      INTEGER(C_INT), INTENT(OUT)   :: ierr
   END SUBROUTINE gpu_compute_thermal_diffusion_c

   !-------------------------------------------------------------------------
   ! Velocity Flux Kernels
   !-------------------------------------------------------------------------

   SUBROUTINE gpu_compute_vorticity_stress_c(mesh_id, uu, vv, ww, mu, &
              omx, omy, omz, txy, txz, tyz, ierr) &
              BIND(C, NAME='gpu_compute_vorticity_stress_')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), INTENT(IN)    :: mesh_id
      REAL(C_DOUBLE), INTENT(IN)    :: uu(*), vv(*), ww(*), mu(*)
      REAL(C_DOUBLE), INTENT(INOUT) :: omx(*), omy(*), omz(*)
      REAL(C_DOUBLE), INTENT(INOUT) :: txy(*), txz(*), tyz(*)
      INTEGER(C_INT), INTENT(OUT)   :: ierr
   END SUBROUTINE gpu_compute_vorticity_stress_c

   SUBROUTINE gpu_compute_velocity_flux_c(mesh_id, uu, vv, ww, &
              rhop, mu, dp, fvx, fvy, fvz, ierr) &
              BIND(C, NAME='gpu_compute_velocity_flux_')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), INTENT(IN)    :: mesh_id
      REAL(C_DOUBLE), INTENT(IN)    :: uu(*), vv(*), ww(*)
      REAL(C_DOUBLE), INTENT(IN)    :: rhop(*), mu(*), dp(*)
      REAL(C_DOUBLE), INTENT(INOUT) :: fvx(*), fvy(*), fvz(*)
      INTEGER(C_INT), INTENT(OUT)   :: ierr
   END SUBROUTINE gpu_compute_velocity_flux_c

   !-------------------------------------------------------------------------
   ! Advection Kernels
   !-------------------------------------------------------------------------

   SUBROUTINE gpu_compute_advection_c(mesh_id, rho_scalar, uu, vv, ww, &
              u_dot_del, flux_limiter_type, ierr) &
              BIND(C, NAME='gpu_compute_advection_')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), INTENT(IN)    :: mesh_id
      REAL(C_DOUBLE), INTENT(IN)    :: rho_scalar(*), uu(*), vv(*), ww(*)
      REAL(C_DOUBLE), INTENT(INOUT) :: u_dot_del(*)
      INTEGER(C_INT), INTENT(IN)    :: flux_limiter_type
      INTEGER(C_INT), INTENT(OUT)   :: ierr
   END SUBROUTINE gpu_compute_advection_c

   SUBROUTINE gpu_compute_enthalpy_advection_c(mesh_id, rho_h_s_p, &
              uu, vv, ww, u_dot_del_rho_h_s, flux_limiter_type, ierr) &
              BIND(C, NAME='gpu_compute_enthalpy_advection_')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), INTENT(IN)    :: mesh_id
      REAL(C_DOUBLE), INTENT(IN)    :: rho_h_s_p(*)
      REAL(C_DOUBLE), INTENT(IN)    :: uu(*), vv(*), ww(*)
      REAL(C_DOUBLE), INTENT(INOUT) :: u_dot_del_rho_h_s(*)
      INTEGER(C_INT), INTENT(IN)    :: flux_limiter_type
      INTEGER(C_INT), INTENT(OUT)   :: ierr
   END SUBROUTINE gpu_compute_enthalpy_advection_c

   SUBROUTINE gpu_compute_species_advection_c(mesh_id, rho_z_p, &
              uu, vv, ww, u_dot_del_rho_z, flux_limiter_type, ierr) &
              BIND(C, NAME='gpu_compute_species_advection_')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), INTENT(IN)    :: mesh_id
      REAL(C_DOUBLE), INTENT(IN)    :: rho_z_p(*)
      REAL(C_DOUBLE), INTENT(IN)    :: uu(*), vv(*), ww(*)
      REAL(C_DOUBLE), INTENT(INOUT) :: u_dot_del_rho_z(*)
      INTEGER(C_INT), INTENT(IN)    :: flux_limiter_type
      INTEGER(C_INT), INTENT(OUT)   :: ierr
   END SUBROUTINE gpu_compute_species_advection_c

   !-------------------------------------------------------------------------
   ! Density Update Kernels
   !-------------------------------------------------------------------------

   SUBROUTINE gpu_compute_density_update_c(mesh_id, rho, zz, del_rho_d_del_z, &
              fx, fy, fz, uu, vv, ww, dt, zzs, ierr) &
              BIND(C, NAME='gpu_compute_density_update_')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), INTENT(IN)    :: mesh_id
      REAL(C_DOUBLE), INTENT(IN)    :: rho(*), zz(*), del_rho_d_del_z(*)
      REAL(C_DOUBLE), INTENT(IN)    :: fx(*), fy(*), fz(*)
      REAL(C_DOUBLE), INTENT(IN)    :: uu(*), vv(*), ww(*)
      REAL(C_DOUBLE), INTENT(IN)    :: dt
      REAL(C_DOUBLE), INTENT(INOUT) :: zzs(*)
      INTEGER(C_INT), INTENT(OUT)   :: ierr
   END SUBROUTINE gpu_compute_density_update_c

   !-------------------------------------------------------------------------
   ! Query Functions
   !-------------------------------------------------------------------------

   SUBROUTINE gpu_kernel_available_c(available) BIND(C, NAME='gpu_kernel_available_')
      IMPORT :: C_INT
      INTEGER(C_INT), INTENT(OUT) :: available
   END SUBROUTINE gpu_kernel_available_c

   SUBROUTINE gpu_get_mesh_cells_c(mesh_id, n_cells) BIND(C, NAME='gpu_get_mesh_cells_')
      IMPORT :: C_INT
      INTEGER(C_INT), INTENT(IN)  :: mesh_id
      INTEGER(C_INT), INTENT(OUT) :: n_cells
   END SUBROUTINE gpu_get_mesh_cells_c

END INTERFACE

CONTAINS

!============================================================================
! Initialization/Finalization
!============================================================================

!> Initialize GPU kernel system
SUBROUTINE GPU_KERNEL_INIT(IERR)
   INTEGER, INTENT(OUT) :: IERR
   INTEGER(C_INT) :: C_IERR

   CALL gpu_kernel_init_c(C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_KERNEL_INIT

!> Finalize GPU kernel system
SUBROUTINE GPU_KERNEL_FINALIZE(IERR)
   INTEGER, INTENT(OUT) :: IERR
   INTEGER(C_INT) :: C_IERR

   CALL gpu_kernel_finalize_c(C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_KERNEL_FINALIZE

!============================================================================
! Mesh Setup
!============================================================================

!> Allocate GPU resources for a mesh
SUBROUTINE GPU_KERNEL_ALLOCATE_MESH(NM, IBAR, JBAR, KBAR, N_SPECIES, IERR)
   INTEGER, INTENT(IN)  :: NM, IBAR, JBAR, KBAR, N_SPECIES
   INTEGER, INTENT(OUT) :: IERR
   INTEGER(C_INT) :: C_IERR

   CALL gpu_kernel_allocate_mesh_c(INT(NM, C_INT), INT(IBAR, C_INT), &
        INT(JBAR, C_INT), INT(KBAR, C_INT), INT(N_SPECIES, C_INT), C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_KERNEL_ALLOCATE_MESH

!> Upload grid spacing and background density to GPU
SUBROUTINE GPU_KERNEL_UPLOAD_GRID(NM, RDX, RDY, RDZ, RDXN, RDYN, RDZN, RHO_0, IERR)
   INTEGER, INTENT(IN) :: NM
   REAL(EB), INTENT(IN) :: RDX(0:), RDY(0:), RDZ(0:)
   REAL(EB), INTENT(IN) :: RDXN(0:), RDYN(0:), RDZN(0:)
   REAL(EB), INTENT(IN) :: RHO_0(0:)
   INTEGER, INTENT(OUT) :: IERR
   INTEGER(C_INT) :: C_IERR

   CALL gpu_kernel_upload_grid_c(INT(NM, C_INT), RDX, RDY, RDZ, &
        RDXN, RDYN, RDZN, RHO_0, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_KERNEL_UPLOAD_GRID

!> Upload gravity values to GPU
SUBROUTINE GPU_KERNEL_UPLOAD_GRAVITY(NM, GX, GY, GZ, IERR)
   INTEGER, INTENT(IN) :: NM
   REAL(EB), INTENT(IN) :: GX(0:), GY(0:), GZ(0:)
   INTEGER, INTENT(OUT) :: IERR
   INTEGER(C_INT) :: C_IERR

   CALL gpu_kernel_upload_gravity_c(INT(NM, C_INT), GX, GY, GZ, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_KERNEL_UPLOAD_GRAVITY

!============================================================================
! Diffusion Kernels
!============================================================================

!> Compute species diffusion flux on GPU
!>
!> Calculates: RHO_D_DZDX(I,J,K,N) = 0.5*(RHO_D(I+1)+RHO_D(I)) * dZ/dx
!>
!> @param[in]     NM           Mesh number
!> @param[in]     ZZP          Species mass fraction array (0:IBAR+1,0:JBAR+1,0:KBAR+1,N_SPEC)
!> @param[in]     RHO_D        Diffusion coefficient (0:IBAR+1,0:JBAR+1,0:KBAR+1)
!> @param[inout]  RHO_D_DZDX   X-direction diffusion flux (0:IBAR,0:JBAR,0:KBAR,N_SPEC)
!> @param[inout]  RHO_D_DZDY   Y-direction diffusion flux
!> @param[inout]  RHO_D_DZDZ   Z-direction diffusion flux
!> @param[in]     N_SPECIES    Number of species
!> @param[out]    IERR         Error code (0=success)
SUBROUTINE GPU_COMPUTE_SPECIES_DIFFUSION(NM, ZZP, RHO_D, &
           RHO_D_DZDX, RHO_D_DZDY, RHO_D_DZDZ, N_SPECIES, IERR)
   INTEGER, INTENT(IN) :: NM, N_SPECIES
   REAL(EB), INTENT(IN) :: ZZP(0:,0:,0:,:)
   REAL(EB), INTENT(IN) :: RHO_D(0:,0:,0:)
   REAL(EB), INTENT(INOUT) :: RHO_D_DZDX(0:,0:,0:,:)
   REAL(EB), INTENT(INOUT) :: RHO_D_DZDY(0:,0:,0:,:)
   REAL(EB), INTENT(INOUT) :: RHO_D_DZDZ(0:,0:,0:,:)
   INTEGER, INTENT(OUT) :: IERR
   INTEGER(C_INT) :: C_IERR

   CALL gpu_compute_species_diffusion_c(INT(NM, C_INT), ZZP, RHO_D, &
        RHO_D_DZDX, RHO_D_DZDY, RHO_D_DZDZ, INT(N_SPECIES, C_INT), C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_COMPUTE_SPECIES_DIFFUSION

!> Compute single-species diffusion flux on GPU
!>
!> For use inside DIFFUSIVE_FLUX_LOOP where RHO_D is computed per-species.
!> Calculates: RHO_D_DZDX_N(I,J,K) = 0.5*(RHO_D(I+1)+RHO_D(I)) * dZ_N/dx
!>
!> @param[in]     NM            Mesh number
!> @param[in]     ZZ_N          Single species mass fraction array (0:IBAR+1,0:JBAR+1,0:KBAR+1)
!> @param[in]     RHO_D         Diffusion coefficient for this species
!> @param[inout]  RHO_D_DZDX_N  X-direction flux for this species
!> @param[inout]  RHO_D_DZDY_N  Y-direction flux for this species
!> @param[inout]  RHO_D_DZDZ_N  Z-direction flux for this species
!> @param[out]    IERR          Error code (0=success)
SUBROUTINE GPU_COMPUTE_SPECIES_DIFFUSION_SINGLE(NM, ZZ_N, RHO_D, &
           RHO_D_DZDX_N, RHO_D_DZDY_N, RHO_D_DZDZ_N, IERR)
   INTEGER, INTENT(IN) :: NM
   REAL(EB), INTENT(IN) :: ZZ_N(0:,0:,0:)
   REAL(EB), INTENT(IN) :: RHO_D(0:,0:,0:)
   REAL(EB), INTENT(INOUT) :: RHO_D_DZDX_N(0:,0:,0:)
   REAL(EB), INTENT(INOUT) :: RHO_D_DZDY_N(0:,0:,0:)
   REAL(EB), INTENT(INOUT) :: RHO_D_DZDZ_N(0:,0:,0:)
   INTEGER, INTENT(OUT) :: IERR
   INTEGER(C_INT) :: C_IERR

   CALL gpu_compute_species_diffusion_single_c(INT(NM, C_INT), ZZ_N, RHO_D, &
        RHO_D_DZDX_N, RHO_D_DZDY_N, RHO_D_DZDZ_N, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_COMPUTE_SPECIES_DIFFUSION_SINGLE

!> Compute thermal diffusion flux on GPU
!>
!> Calculates: KDTDX(I,J,K) = 0.5*(KP(I+1)+KP(I)) * dT/dx
!>
!> @param[in]     NM      Mesh number
!> @param[in]     TMP     Temperature array (0:IBAR+1,0:JBAR+1,0:KBAR+1)
!> @param[in]     KP      Thermal conductivity (0:IBAR+1,0:JBAR+1,0:KBAR+1)
!> @param[inout]  KDTDX   X-direction thermal flux (0:IBAR,0:JBAR,0:KBAR)
!> @param[inout]  KDTDY   Y-direction thermal flux
!> @param[inout]  KDTDZ   Z-direction thermal flux
!> @param[out]    IERR    Error code (0=success)
SUBROUTINE GPU_COMPUTE_THERMAL_DIFFUSION(NM, TMP, KP, KDTDX, KDTDY, KDTDZ, IERR)
   INTEGER, INTENT(IN) :: NM
   REAL(EB), INTENT(IN) :: TMP(0:,0:,0:)
   REAL(EB), INTENT(IN) :: KP(0:,0:,0:)
   REAL(EB), INTENT(INOUT) :: KDTDX(0:,0:,0:)
   REAL(EB), INTENT(INOUT) :: KDTDY(0:,0:,0:)
   REAL(EB), INTENT(INOUT) :: KDTDZ(0:,0:,0:)
   INTEGER, INTENT(OUT) :: IERR
   INTEGER(C_INT) :: C_IERR

   CALL gpu_compute_thermal_diffusion_c(INT(NM, C_INT), TMP, KP, &
        KDTDX, KDTDY, KDTDZ, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_COMPUTE_THERMAL_DIFFUSION

!============================================================================
! Velocity Flux Kernels
!============================================================================

!> Compute vorticity and stress tensor on GPU
!>
!> Calculates:
!>   OMX = dW/dy - dV/dz, OMY = dU/dz - dW/dx, OMZ = dV/dx - dU/dy
!>   TXY = MU*(dV/dx + dU/dy), TXZ = MU*(dU/dz + dW/dx), TYZ = MU*(dV/dz + dW/dy)
!>
!> This function computes interior cell values only. No EDGE boundary handling.
!>
!> @param[in]     NM    Mesh number
!> @param[in]     UU    X-velocity (0:IBAR+1,0:JBAR+1,0:KBAR+1)
!> @param[in]     VV    Y-velocity
!> @param[in]     WW    Z-velocity
!> @param[in]     MU    Dynamic viscosity
!> @param[inout]  OMX   X-component of vorticity
!> @param[inout]  OMY   Y-component of vorticity
!> @param[inout]  OMZ   Z-component of vorticity
!> @param[inout]  TXY   XY stress tensor component
!> @param[inout]  TXZ   XZ stress tensor component
!> @param[inout]  TYZ   YZ stress tensor component
!> @param[out]    IERR  Error code (0=success)
SUBROUTINE GPU_COMPUTE_VORTICITY_STRESS(NM, UU, VV, WW, MU, OMX, OMY, OMZ, TXY, TXZ, TYZ, IERR)
   INTEGER, INTENT(IN) :: NM
   REAL(EB), INTENT(IN) :: UU(0:,0:,0:)
   REAL(EB), INTENT(IN) :: VV(0:,0:,0:)
   REAL(EB), INTENT(IN) :: WW(0:,0:,0:)
   REAL(EB), INTENT(IN) :: MU(0:,0:,0:)
   REAL(EB), INTENT(INOUT) :: OMX(0:,0:,0:)
   REAL(EB), INTENT(INOUT) :: OMY(0:,0:,0:)
   REAL(EB), INTENT(INOUT) :: OMZ(0:,0:,0:)
   REAL(EB), INTENT(INOUT) :: TXY(0:,0:,0:)
   REAL(EB), INTENT(INOUT) :: TXZ(0:,0:,0:)
   REAL(EB), INTENT(INOUT) :: TYZ(0:,0:,0:)
   INTEGER, INTENT(OUT) :: IERR
   INTEGER(C_INT) :: C_IERR

   CALL gpu_compute_vorticity_stress_c(INT(NM, C_INT), UU, VV, WW, MU, &
        OMX, OMY, OMZ, TXY, TXZ, TYZ, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_COMPUTE_VORTICITY_STRESS

!> Compute velocity flux (FVX, FVY, FVZ) on GPU
!>
!> Calculates momentum flux including vorticity, stress tensor, and gravity terms
!>
!> @param[in]     NM    Mesh number
!> @param[in]     UU    X-velocity (0:IBAR+1,0:JBAR+1,0:KBAR+1)
!> @param[in]     VV    Y-velocity
!> @param[in]     WW    Z-velocity
!> @param[in]     RHOP  Density
!> @param[in]     MU    Dynamic viscosity
!> @param[in]     DP    Divergence
!> @param[inout]  FVX   X-direction velocity flux
!> @param[inout]  FVY   Y-direction velocity flux
!> @param[inout]  FVZ   Z-direction velocity flux
!> @param[out]    IERR  Error code (0=success)
SUBROUTINE GPU_COMPUTE_VELOCITY_FLUX(NM, UU, VV, WW, RHOP, MU, DP, FVX, FVY, FVZ, IERR)
   INTEGER, INTENT(IN) :: NM
   REAL(EB), INTENT(IN) :: UU(0:,0:,0:)
   REAL(EB), INTENT(IN) :: VV(0:,0:,0:)
   REAL(EB), INTENT(IN) :: WW(0:,0:,0:)
   REAL(EB), INTENT(IN) :: RHOP(0:,0:,0:)
   REAL(EB), INTENT(IN) :: MU(0:,0:,0:)
   REAL(EB), INTENT(IN) :: DP(0:,0:,0:)
   REAL(EB), INTENT(INOUT) :: FVX(0:,0:,0:)
   REAL(EB), INTENT(INOUT) :: FVY(0:,0:,0:)
   REAL(EB), INTENT(INOUT) :: FVZ(0:,0:,0:)
   INTEGER, INTENT(OUT) :: IERR
   INTEGER(C_INT) :: C_IERR

   CALL gpu_compute_velocity_flux_c(INT(NM, C_INT), UU, VV, WW, &
        RHOP, MU, DP, FVX, FVY, FVZ, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_COMPUTE_VELOCITY_FLUX

!============================================================================
! Advection Kernels
!============================================================================

!> Compute scalar advection on GPU
!>
!> Calculates: U_DOT_DEL = div(u * rho * scalar)
!> using flux limiters for stability
!>
!> @param[in]     NM                  Mesh number
!> @param[in]     RHO_SCALAR          rho * scalar (0:IBAR+1,0:JBAR+1,0:KBAR+1)
!> @param[in]     UU                  X-velocity
!> @param[in]     VV                  Y-velocity
!> @param[in]     WW                  Z-velocity
!> @param[inout]  U_DOT_DEL           Advection term output
!> @param[in]     FLUX_LIMITER_TYPE   0=upwind, 1=superbee, 2=charm
!> @param[out]    IERR                Error code (0=success)
SUBROUTINE GPU_COMPUTE_ADVECTION(NM, RHO_SCALAR, UU, VV, WW, U_DOT_DEL, &
           FLUX_LIMITER_TYPE, IERR)
   INTEGER, INTENT(IN) :: NM, FLUX_LIMITER_TYPE
   REAL(EB), INTENT(IN) :: RHO_SCALAR(0:,0:,0:)
   REAL(EB), INTENT(IN) :: UU(0:,0:,0:)
   REAL(EB), INTENT(IN) :: VV(0:,0:,0:)
   REAL(EB), INTENT(IN) :: WW(0:,0:,0:)
   REAL(EB), INTENT(INOUT) :: U_DOT_DEL(0:,0:,0:)
   INTEGER, INTENT(OUT) :: IERR
   INTEGER(C_INT) :: C_IERR

   CALL gpu_compute_advection_c(INT(NM, C_INT), RHO_SCALAR, UU, VV, WW, &
        U_DOT_DEL, INT(FLUX_LIMITER_TYPE, C_INT), C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_COMPUTE_ADVECTION

!> Compute enthalpy advection on GPU (FDS-style)
!>
!> Calculates: U_DOT_DEL_RHO_H_S = div((FX - rho*h_s)*u)
!> using FDS Tech Guide B.12-B.14 formulation with flux limiters
!>
!> This is the primary enthalpy advection for ENTHALPY_ADVECTION_NEW.
!> Uses FDS-style divergence: ((FX-RHO)*U - (FX_m1-RHO)*U_m1)*RDX
!>
!> @param[in]     NM                  Mesh number
!> @param[in]     RHO_H_S_P           rho * h_s (sensible enthalpy density)
!> @param[in]     UU                  X-velocity
!> @param[in]     VV                  Y-velocity
!> @param[in]     WW                  Z-velocity
!> @param[inout]  U_DOT_DEL_RHO_H_S   Advection term output
!> @param[in]     FLUX_LIMITER_TYPE   0=upwind, 1=superbee, 2=charm
!> @param[out]    IERR                Error code (0=success)
SUBROUTINE GPU_COMPUTE_ENTHALPY_ADVECTION(NM, RHO_H_S_P, UU, VV, WW, &
           U_DOT_DEL_RHO_H_S, FLUX_LIMITER_TYPE, IERR)
   INTEGER, INTENT(IN) :: NM, FLUX_LIMITER_TYPE
   REAL(EB), INTENT(IN) :: RHO_H_S_P(0:,0:,0:)
   REAL(EB), INTENT(IN) :: UU(0:,0:,0:)
   REAL(EB), INTENT(IN) :: VV(0:,0:,0:)
   REAL(EB), INTENT(IN) :: WW(0:,0:,0:)
   REAL(EB), INTENT(INOUT) :: U_DOT_DEL_RHO_H_S(0:,0:,0:)
   INTEGER, INTENT(OUT) :: IERR
   INTEGER(C_INT) :: C_IERR

   CALL gpu_compute_enthalpy_advection_c(INT(NM, C_INT), RHO_H_S_P, UU, VV, WW, &
        U_DOT_DEL_RHO_H_S, INT(FLUX_LIMITER_TYPE, C_INT), C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_COMPUTE_ENTHALPY_ADVECTION

!> Compute species advection on GPU (FDS-style)
!>
!> Calculates: U_DOT_DEL_RHO_Z = div((FX_Z - rho*z)*u)
!> using FDS Tech Guide B.12-B.14 formulation with flux limiters
!>
!> @param[in]     NM                  Mesh number
!> @param[in]     RHO_Z_P             rho * Z (species mass fraction density)
!> @param[in]     UU                  X-velocity
!> @param[in]     VV                  Y-velocity
!> @param[in]     WW                  Z-velocity
!> @param[inout]  U_DOT_DEL_RHO_Z     Advection term output
!> @param[in]     FLUX_LIMITER_TYPE   0=upwind, 1=superbee, 2=charm
!> @param[out]    IERR                Error code (0=success)
SUBROUTINE GPU_COMPUTE_SPECIES_ADVECTION(NM, RHO_Z_P, UU, VV, WW, &
           U_DOT_DEL_RHO_Z, FLUX_LIMITER_TYPE, IERR)
   INTEGER, INTENT(IN) :: NM, FLUX_LIMITER_TYPE
   REAL(EB), INTENT(IN) :: RHO_Z_P(0:,0:,0:)
   REAL(EB), INTENT(IN) :: UU(0:,0:,0:)
   REAL(EB), INTENT(IN) :: VV(0:,0:,0:)
   REAL(EB), INTENT(IN) :: WW(0:,0:,0:)
   REAL(EB), INTENT(INOUT) :: U_DOT_DEL_RHO_Z(0:,0:,0:)
   INTEGER, INTENT(OUT) :: IERR
   INTEGER(C_INT) :: C_IERR

   CALL gpu_compute_species_advection_c(INT(NM, C_INT), RHO_Z_P, UU, VV, WW, &
        U_DOT_DEL_RHO_Z, INT(FLUX_LIMITER_TYPE, C_INT), C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_COMPUTE_SPECIES_ADVECTION

!============================================================================
! Density Update Kernels
!============================================================================

!> Compute density update on GPU
!>
!> Calculates: ZZS = RHO*ZZ - DT*RHS
!> where RHS = -DEL_RHO_D_DEL_Z + div(FX*U, FY*V, FZ*W)
!>
!> For Cartesian meshes only (assumes R=RRN=1).
!>
!> @param[in]     NM               Mesh number
!> @param[in]     RHO              Density (0:IBAR+1,0:JBAR+1,0:KBAR+1)
!> @param[in]     ZZ               Species mass fraction
!> @param[in]     DEL_RHO_D_DEL_Z  Diffusion term (laplacian of species)
!> @param[in]     FX               Face interpolation factor X
!> @param[in]     FY               Face interpolation factor Y
!> @param[in]     FZ               Face interpolation factor Z
!> @param[in]     UU               X-velocity
!> @param[in]     VV               Y-velocity
!> @param[in]     WW               Z-velocity
!> @param[in]     DT               Time step
!> @param[inout]  ZZS              Updated species mass fraction * density
!> @param[out]    IERR             Error code (0=success)
SUBROUTINE GPU_COMPUTE_DENSITY_UPDATE(NM, RHO, ZZ, DEL_RHO_D_DEL_Z, &
           FX, FY, FZ, UU, VV, WW, DT, ZZS, IERR)
   INTEGER, INTENT(IN) :: NM
   REAL(EB), INTENT(IN) :: RHO(0:,0:,0:)
   REAL(EB), INTENT(IN) :: ZZ(0:,0:,0:)
   REAL(EB), INTENT(IN) :: DEL_RHO_D_DEL_Z(0:,0:,0:)
   REAL(EB), INTENT(IN) :: FX(0:,0:,0:)
   REAL(EB), INTENT(IN) :: FY(0:,0:,0:)
   REAL(EB), INTENT(IN) :: FZ(0:,0:,0:)
   REAL(EB), INTENT(IN) :: UU(0:,0:,0:)
   REAL(EB), INTENT(IN) :: VV(0:,0:,0:)
   REAL(EB), INTENT(IN) :: WW(0:,0:,0:)
   REAL(EB), INTENT(IN) :: DT
   REAL(EB), INTENT(INOUT) :: ZZS(0:,0:,0:)
   INTEGER, INTENT(OUT) :: IERR
   INTEGER(C_INT) :: C_IERR

   CALL gpu_compute_density_update_c(INT(NM, C_INT), RHO, ZZ, DEL_RHO_D_DEL_Z, &
        FX, FY, FZ, UU, VV, WW, DT, ZZS, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE GPU_COMPUTE_DENSITY_UPDATE

!============================================================================
! Query Functions
!============================================================================

!> Check if GPU kernels are available
FUNCTION GPU_KERNEL_AVAILABLE() RESULT(AVAILABLE)
   LOGICAL :: AVAILABLE
   INTEGER(C_INT) :: C_AVAILABLE

   CALL gpu_kernel_available_c(C_AVAILABLE)
   AVAILABLE = (C_AVAILABLE /= 0)
END FUNCTION GPU_KERNEL_AVAILABLE

!> Get number of cells in a mesh
FUNCTION GPU_GET_MESH_CELLS(NM) RESULT(N_CELLS)
   INTEGER, INTENT(IN) :: NM
   INTEGER :: N_CELLS
   INTEGER(C_INT) :: C_N_CELLS

   CALL gpu_get_mesh_cells_c(INT(NM, C_INT), C_N_CELLS)
   N_CELLS = INT(C_N_CELLS)
END FUNCTION GPU_GET_MESH_CELLS

END MODULE GPU_FORTRAN

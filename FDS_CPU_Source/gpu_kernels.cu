/*
 * gpu_kernels.cu - CUDA Kernels for FDS Advection & Diffusion
 *
 * Part of FDS GPU Acceleration Project
 * Implements GPU-accelerated diffusion and velocity flux calculations
 *
 * Copyright 2024, FDS-AmgX Project
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>

/* ============================================================================
 * Constants and Macros
 * ============================================================================ */

#define BLOCK_X 8
#define BLOCK_Y 8
#define BLOCK_Z 4

#define TWO_EPSILON_EB 1.0e-15
#define FOTH (4.0/3.0)

/* 3D array indexing macros for FDS memory layout */
/* FDS arrays: (0:IBAR+1, 0:JBAR+1, 0:KBAR+1) with ghost cells */
#define IDX3D(i, j, k, ni, nj)     ((i) + (ni)*(j) + (ni)*(nj)*(k))
#define IDX4D(i, j, k, n, ni, nj, nk) ((i) + (ni)*(j) + (ni)*(nj)*(k) + (ni)*(nj)*(nk)*(n))

/* Stencil size with ghost cells */
#define NI(ibar) ((ibar) + 2)
#define NJ(jbar) ((jbar) + 2)
#define NK(kbar) ((kbar) + 2)

/* ============================================================================
 * Device Helper Functions
 * ============================================================================ */

__device__ __forceinline__ double fds_sign(double val) {
    return (val >= 0.0) ? 1.0 : -1.0;
}

__device__ __forceinline__ double fds_max(double a, double b) {
    return (a > b) ? a : b;
}

__device__ __forceinline__ double fds_min(double a, double b) {
    return (a < b) ? a : b;
}

/* ============================================================================
 * Species Diffusion Flux Kernel
 *
 * Computes: RHO_D_DZDX(I,J,K,N) = 0.5*(RHO_D(I+1,J,K)+RHO_D(I,J,K)) *
 *                                  (ZZP(I+1,J,K,N)-ZZP(I,J,K,N)) * RDXN(I)
 * (same for Y and Z directions)
 *
 * Grid dimensions: (0:IBAR, 0:JBAR, 0:KBAR) for each species
 * ============================================================================ */

__global__ void species_diffusion_flux_kernel(
    const double* __restrict__ ZZP,        // Species mass fraction (0:IBAR+1, 0:JBAR+1, 0:KBAR+1, N_SPEC)
    const double* __restrict__ RHO_D,      // Diffusion coefficient (0:IBAR+1, 0:JBAR+1, 0:KBAR+1)
    const double* __restrict__ RDXN,       // 1/dx at nodes
    const double* __restrict__ RDYN,       // 1/dy at nodes
    const double* __restrict__ RDZN,       // 1/dz at nodes
    double* __restrict__ RHO_D_DZDX,       // X-flux output
    double* __restrict__ RHO_D_DZDY,       // Y-flux output
    double* __restrict__ RHO_D_DZDZ,       // Z-flux output
    int IBAR, int JBAR, int KBAR,
    int n_species,
    int species_idx)                        // Current species index
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int ni = NI(IBAR);
    int nj = NJ(JBAR);
    int nk = NK(KBAR);

    // Valid range: i in [0,IBAR], j in [0,JBAR], k in [0,KBAR]
    if (i > IBAR || j > JBAR || k > KBAR) return;

    // 3D index for diffusion coefficient (no species dimension)
    int idx = IDX3D(i, j, k, ni, nj);
    int idx_ip1 = IDX3D(i+1, j, k, ni, nj);
    int idx_jp1 = IDX3D(i, j+1, k, ni, nj);
    int idx_kp1 = IDX3D(i, j, k+1, ni, nj);

    // 4D index for species array
    int idx4d = IDX4D(i, j, k, species_idx, ni, nj, nk);
    int idx4d_ip1 = IDX4D(i+1, j, k, species_idx, ni, nj, nk);
    int idx4d_jp1 = IDX4D(i, j+1, k, species_idx, ni, nj, nk);
    int idx4d_kp1 = IDX4D(i, j, k+1, species_idx, ni, nj, nk);

    // X-direction flux
    double DZDX = (ZZP[idx4d_ip1] - ZZP[idx4d]) * RDXN[i];
    double RHO_D_avg_x = 0.5 * (RHO_D[idx_ip1] + RHO_D[idx]);
    RHO_D_DZDX[idx4d] = RHO_D_avg_x * DZDX;

    // Y-direction flux
    double DZDY = (ZZP[idx4d_jp1] - ZZP[idx4d]) * RDYN[j];
    double RHO_D_avg_y = 0.5 * (RHO_D[idx_jp1] + RHO_D[idx]);
    RHO_D_DZDY[idx4d] = RHO_D_avg_y * DZDY;

    // Z-direction flux
    double DZDZ = (ZZP[idx4d_kp1] - ZZP[idx4d]) * RDZN[k];
    double RHO_D_avg_z = 0.5 * (RHO_D[idx_kp1] + RHO_D[idx]);
    RHO_D_DZDZ[idx4d] = RHO_D_avg_z * DZDZ;
}

/* ============================================================================
 * Thermal Diffusion Flux Kernel
 *
 * Computes: KDTDX(I,J,K) = 0.5*(KP(I+1,J,K)+KP(I,J,K)) *
 *                          (TMP(I+1,J,K)-TMP(I,J,K)) * RDXN(I)
 * (same for Y and Z directions)
 *
 * Grid dimensions: (0:IBAR, 0:JBAR, 0:KBAR)
 * ============================================================================ */

__global__ void thermal_diffusion_flux_kernel(
    const double* __restrict__ TMP,        // Temperature (0:IBAR+1, 0:JBAR+1, 0:KBAR+1)
    const double* __restrict__ KP,         // Thermal conductivity
    const double* __restrict__ RDXN,       // 1/dx at nodes
    const double* __restrict__ RDYN,       // 1/dy at nodes
    const double* __restrict__ RDZN,       // 1/dz at nodes
    double* __restrict__ KDTDX,            // X-flux output
    double* __restrict__ KDTDY,            // Y-flux output
    double* __restrict__ KDTDZ,            // Z-flux output
    int IBAR, int JBAR, int KBAR)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int ni = NI(IBAR);
    int nj = NJ(JBAR);

    // Valid range: i in [0,IBAR], j in [0,JBAR], k in [0,KBAR]
    if (i > IBAR || j > JBAR || k > KBAR) return;

    int idx = IDX3D(i, j, k, ni, nj);
    int idx_ip1 = IDX3D(i+1, j, k, ni, nj);
    int idx_jp1 = IDX3D(i, j+1, k, ni, nj);
    int idx_kp1 = IDX3D(i, j, k+1, ni, nj);

    // X-direction thermal flux: KDTDX = 0.5*(KP(i+1)+KP(i)) * dT/dx
    double DTDX = (TMP[idx_ip1] - TMP[idx]) * RDXN[i];
    KDTDX[idx] = 0.5 * (KP[idx_ip1] + KP[idx]) * DTDX;

    // Y-direction thermal flux
    double DTDY = (TMP[idx_jp1] - TMP[idx]) * RDYN[j];
    KDTDY[idx] = 0.5 * (KP[idx_jp1] + KP[idx]) * DTDY;

    // Z-direction thermal flux
    double DTDZ = (TMP[idx_kp1] - TMP[idx]) * RDZN[k];
    KDTDZ[idx] = 0.5 * (KP[idx_kp1] + KP[idx]) * DTDZ;
}

/* ============================================================================
 * Vorticity and Stress Tensor Kernel
 *
 * Computes:
 *   OMX = dW/dy - dV/dz  (vorticity x-component)
 *   OMY = dU/dz - dW/dx  (vorticity y-component)
 *   OMZ = dV/dx - dU/dy  (vorticity z-component)
 *   TXY = MU * (dU/dy + dV/dx)
 *   TXZ = MU * (dU/dz + dW/dx)
 *   TYZ = MU * (dV/dz + dW/dy)
 *
 * Grid dimensions: (0:IBAR, 0:JBAR, 0:KBAR)
 * ============================================================================ */

__global__ void vorticity_stress_kernel(
    const double* __restrict__ UU,         // X-velocity
    const double* __restrict__ VV,         // Y-velocity
    const double* __restrict__ WW,         // Z-velocity
    const double* __restrict__ MU,         // Dynamic viscosity
    const double* __restrict__ RDXN,       // 1/dx at nodes
    const double* __restrict__ RDYN,       // 1/dy at nodes
    const double* __restrict__ RDZN,       // 1/dz at nodes
    double* __restrict__ OMX,              // Vorticity X
    double* __restrict__ OMY,              // Vorticity Y
    double* __restrict__ OMZ,              // Vorticity Z
    double* __restrict__ TXY,              // Stress tensor XY
    double* __restrict__ TXZ,              // Stress tensor XZ
    double* __restrict__ TYZ,              // Stress tensor YZ
    int IBAR, int JBAR, int KBAR)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int ni = NI(IBAR);
    int nj = NJ(JBAR);

    // Valid range: (0:IBAR, 0:JBAR, 0:KBAR)
    if (i > IBAR || j > JBAR || k > KBAR) return;

    // Indices
    int idx = IDX3D(i, j, k, ni, nj);
    int idx_ip1 = IDX3D(i+1, j, k, ni, nj);
    int idx_jp1 = IDX3D(i, j+1, k, ni, nj);
    int idx_kp1 = IDX3D(i, j, k+1, ni, nj);
    int idx_ip1_jp1 = IDX3D(i+1, j+1, k, ni, nj);
    int idx_ip1_kp1 = IDX3D(i+1, j, k+1, ni, nj);
    int idx_jp1_kp1 = IDX3D(i, j+1, k+1, ni, nj);
    int idx_ip1_jp1_kp1 = IDX3D(i+1, j+1, k+1, ni, nj);

    // Velocity gradients
    double DUDY = RDYN[j] * (UU[idx_jp1] - UU[idx]);
    double DVDX = RDXN[i] * (VV[idx_ip1] - VV[idx]);
    double DUDZ = RDZN[k] * (UU[idx_kp1] - UU[idx]);
    double DWDX = RDXN[i] * (WW[idx_ip1] - WW[idx]);
    double DVDZ = RDZN[k] * (VV[idx_kp1] - VV[idx]);
    double DWDY = RDYN[j] * (WW[idx_jp1] - WW[idx]);

    // Vorticity components
    OMX[idx] = DWDY - DVDZ;
    OMY[idx] = DUDZ - DWDX;
    OMZ[idx] = DVDX - DUDY;

    // Average viscosity at cell edges
    double MUX = 0.25 * (MU[idx] + MU[idx_jp1] + MU[idx_kp1] + MU[idx_jp1_kp1]);
    double MUY = 0.25 * (MU[idx] + MU[idx_ip1] + MU[idx_kp1] + MU[idx_ip1_kp1]);
    double MUZ = 0.25 * (MU[idx] + MU[idx_ip1] + MU[idx_jp1] + MU[idx_ip1_jp1]);

    // Stress tensor components
    TXY[idx] = MUZ * (DVDX + DUDY);
    TXZ[idx] = MUY * (DUDZ + DWDX);
    TYZ[idx] = MUX * (DVDZ + DWDY);
}

/* ============================================================================
 * Velocity Flux X-Direction Kernel (FVX)
 *
 * Computes: FVX = 0.25*(W*OMY - V*OMZ) - GX + RRHO*(GX*RHO_0 - VTRM)
 *
 * Grid dimensions: (0:IBAR, 1:JBAR, 1:KBAR)
 * ============================================================================ */

__global__ void velocity_flux_x_kernel(
    const double* __restrict__ UU,
    const double* __restrict__ VV,
    const double* __restrict__ WW,
    const double* __restrict__ RHOP,
    const double* __restrict__ MU,
    const double* __restrict__ DP,          // Divergence
    const double* __restrict__ OMY,
    const double* __restrict__ OMZ,
    const double* __restrict__ TXY,
    const double* __restrict__ TXZ,
    const double* __restrict__ RDX,
    const double* __restrict__ RDXN,
    const double* __restrict__ RDY,
    const double* __restrict__ RDZ,
    const double* __restrict__ RHO_0,       // Background density
    const double* __restrict__ GX_arr,      // Gravity X (can vary with position)
    double* __restrict__ FVX,
    int IBAR, int JBAR, int KBAR)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;  // j starts at 1
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;  // k starts at 1

    int ni = NI(IBAR);
    int nj = NJ(JBAR);

    // Valid range: i in [0,IBAR], j in [1,JBAR], k in [1,KBAR]
    if (i > IBAR || j > JBAR || k > KBAR) return;

    // Indices
    int idx = IDX3D(i, j, k, ni, nj);
    int idx_ip1 = IDX3D(i+1, j, k, ni, nj);
    int idx_jm1 = IDX3D(i, j-1, k, ni, nj);
    int idx_km1 = IDX3D(i, j, k-1, ni, nj);
    int idx_ip1_jm1 = IDX3D(i+1, j-1, k, ni, nj);
    int idx_ip1_km1 = IDX3D(i+1, j, k-1, ni, nj);

    // Interpolated velocities for convective terms
    double WP = WW[idx] + WW[idx_ip1];
    double WM = WW[idx_km1] + WW[idx_ip1_km1];
    double VP = VV[idx] + VV[idx_ip1];
    double VM = VV[idx_jm1] + VV[idx_ip1_jm1];

    // Vorticity at cell faces (simplified, no edge handling)
    double OMYP = OMY[idx];
    double OMYM = OMY[idx_km1];
    double OMZP = OMZ[idx];
    double OMZM = OMZ[idx_jm1];

    // Stress tensor at cell faces
    double TXZP = TXZ[idx];
    double TXZM = TXZ[idx_km1];
    double TXYP = TXY[idx];
    double TXYM = TXY[idx_jm1];

    // Convective terms: W*OMY and V*OMZ
    double WOMY = WP * OMYP + WM * OMYM;
    double VOMZ = VP * OMZP + VM * OMZM;

    // Density interpolation
    double RRHO = 2.0 / (RHOP[idx] + RHOP[idx_ip1]);

    // Diagonal stress TXX
    double DVDY_p = (VV[idx_ip1] - VV[idx_ip1_jm1]) * RDY[j];
    double DWDZ_p = (WW[idx_ip1] - WW[idx_ip1_km1]) * RDZ[k];
    double TXXP = MU[idx_ip1] * (FOTH * DP[idx_ip1] - 2.0 * (DVDY_p + DWDZ_p));

    double DVDY_m = (VV[idx] - VV[idx_jm1]) * RDY[j];
    double DWDZ_m = (WW[idx] - WW[idx_km1]) * RDZ[k];
    double TXXM = MU[idx] * (FOTH * DP[idx] - 2.0 * (DVDY_m + DWDZ_m));

    // Stress tensor gradients
    double DTXXDX = RDXN[i] * (TXXP - TXXM);
    double DTXYDY = RDY[j] * (TXYP - TXYM);
    double DTXZDZ = RDZ[k] * (TXZP - TXZM);
    double VTRM = DTXXDX + DTXYDY + DTXZDZ;

    // Gravity
    double GX = GX_arr[i];

    // Final FVX calculation
    FVX[idx] = 0.25 * (WOMY - VOMZ) - GX + RRHO * (GX * RHO_0[k] - VTRM);
}

/* ============================================================================
 * Velocity Flux Y-Direction Kernel (FVY)
 *
 * Computes: FVY = 0.25*(U*OMZ - W*OMX) - GY + RRHO*(GY*RHO_0 - VTRM)
 *
 * Grid dimensions: (1:IBAR, 0:JBAR, 1:KBAR)
 * ============================================================================ */

__global__ void velocity_flux_y_kernel(
    const double* __restrict__ UU,
    const double* __restrict__ VV,
    const double* __restrict__ WW,
    const double* __restrict__ RHOP,
    const double* __restrict__ MU,
    const double* __restrict__ DP,
    const double* __restrict__ OMX,
    const double* __restrict__ OMZ,
    const double* __restrict__ TXY,
    const double* __restrict__ TYZ,
    const double* __restrict__ RDX,
    const double* __restrict__ RDYN,
    const double* __restrict__ RDZ,
    const double* __restrict__ RHO_0,
    const double* __restrict__ GY_arr,
    double* __restrict__ FVY,
    int IBAR, int JBAR, int KBAR)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;  // i starts at 1
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;  // k starts at 1

    int ni = NI(IBAR);
    int nj = NJ(JBAR);

    // Valid range: i in [1,IBAR], j in [0,JBAR], k in [1,KBAR]
    if (i > IBAR || j > JBAR || k > KBAR) return;

    // Indices
    int idx = IDX3D(i, j, k, ni, nj);
    int idx_im1 = IDX3D(i-1, j, k, ni, nj);
    int idx_jp1 = IDX3D(i, j+1, k, ni, nj);
    int idx_km1 = IDX3D(i, j, k-1, ni, nj);
    int idx_im1_jp1 = IDX3D(i-1, j+1, k, ni, nj);
    int idx_jp1_km1 = IDX3D(i, j+1, k-1, ni, nj);

    // Interpolated velocities
    double UP = UU[idx] + UU[idx_jp1];
    double UM = UU[idx_im1] + UU[idx_im1_jp1];
    double WP = WW[idx] + WW[idx_jp1];
    double WM = WW[idx_km1] + WW[idx_jp1_km1];

    // Vorticity and stress
    double OMXP = OMX[idx];
    double OMXM = OMX[idx_km1];
    double OMZP = OMZ[idx];
    double OMZM = OMZ[idx_im1];

    double TYZP = TYZ[idx];
    double TYZM = TYZ[idx_km1];
    double TXYP = TXY[idx];
    double TXYM = TXY[idx_im1];

    // Convective terms
    double WOMX = WP * OMXP + WM * OMXM;
    double UOMZ = UP * OMZP + UM * OMZM;

    // Density
    double RRHO = 2.0 / (RHOP[idx] + RHOP[idx_jp1]);

    // Diagonal stress TYY
    double DUDX_p = (UU[idx_jp1] - UU[idx_im1_jp1]) * RDX[i];
    double DWDZ_p = (WW[idx_jp1] - WW[idx_jp1_km1]) * RDZ[k];
    double TYYP = MU[idx_jp1] * (FOTH * DP[idx_jp1] - 2.0 * (DUDX_p + DWDZ_p));

    double DUDX_m = (UU[idx] - UU[idx_im1]) * RDX[i];
    double DWDZ_m = (WW[idx] - WW[idx_km1]) * RDZ[k];
    double TYYM = MU[idx] * (FOTH * DP[idx] - 2.0 * (DUDX_m + DWDZ_m));

    // Stress gradients
    double DTXYDX = RDX[i] * (TXYP - TXYM);
    double DTYYDY = RDYN[j] * (TYYP - TYYM);
    double DTYZDZ = RDZ[k] * (TYZP - TYZM);
    double VTRM = DTXYDX + DTYYDY + DTYZDZ;

    // Gravity (using same array index pattern as GX)
    double GY = GY_arr[i];

    // Final FVY
    FVY[idx] = 0.25 * (UOMZ - WOMX) - GY + RRHO * (GY * RHO_0[k] - VTRM);
}

/* ============================================================================
 * Velocity Flux Z-Direction Kernel (FVZ)
 *
 * Computes: FVZ = 0.25*(V*OMX - U*OMY) - GZ + RRHO*(GZ*RHO_0_avg - VTRM)
 *
 * Grid dimensions: (1:IBAR, 1:JBAR, 0:KBAR)
 * ============================================================================ */

__global__ void velocity_flux_z_kernel(
    const double* __restrict__ UU,
    const double* __restrict__ VV,
    const double* __restrict__ WW,
    const double* __restrict__ RHOP,
    const double* __restrict__ MU,
    const double* __restrict__ DP,
    const double* __restrict__ OMX,
    const double* __restrict__ OMY,
    const double* __restrict__ TXZ,
    const double* __restrict__ TYZ,
    const double* __restrict__ RDX,
    const double* __restrict__ RDY,
    const double* __restrict__ RDZN,
    const double* __restrict__ RHO_0,
    const double* __restrict__ GZ_arr,
    double* __restrict__ FVZ,
    int IBAR, int JBAR, int KBAR)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;  // i starts at 1
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;  // j starts at 1
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int ni = NI(IBAR);
    int nj = NJ(JBAR);

    // Valid range: i in [1,IBAR], j in [1,JBAR], k in [0,KBAR]
    if (i > IBAR || j > JBAR || k > KBAR) return;

    // Indices
    int idx = IDX3D(i, j, k, ni, nj);
    int idx_im1 = IDX3D(i-1, j, k, ni, nj);
    int idx_jm1 = IDX3D(i, j-1, k, ni, nj);
    int idx_kp1 = IDX3D(i, j, k+1, ni, nj);
    int idx_im1_kp1 = IDX3D(i-1, j, k+1, ni, nj);
    int idx_jm1_kp1 = IDX3D(i, j-1, k+1, ni, nj);

    // Interpolated velocities
    double UP = UU[idx] + UU[idx_kp1];
    double UM = UU[idx_im1] + UU[idx_im1_kp1];
    double VP = VV[idx] + VV[idx_kp1];
    double VM = VV[idx_jm1] + VV[idx_jm1_kp1];

    // Vorticity and stress
    double OMYP = OMY[idx];
    double OMYM = OMY[idx_im1];
    double OMXP = OMX[idx];
    double OMXM = OMX[idx_jm1];

    double TXZP = TXZ[idx];
    double TXZM = TXZ[idx_im1];
    double TYZP = TYZ[idx];
    double TYZM = TYZ[idx_jm1];

    // Convective terms
    double UOMY = UP * OMYP + UM * OMYM;
    double VOMX = VP * OMXP + VM * OMXM;

    // Density
    double RRHO = 2.0 / (RHOP[idx] + RHOP[idx_kp1]);

    // Diagonal stress TZZ
    double DUDX_p = (UU[idx_kp1] - UU[idx_im1_kp1]) * RDX[i];
    double DVDY_p = (VV[idx_kp1] - VV[idx_jm1_kp1]) * RDY[j];
    double TZZP = MU[idx_kp1] * (FOTH * DP[idx_kp1] - 2.0 * (DUDX_p + DVDY_p));

    double DUDX_m = (UU[idx] - UU[idx_im1]) * RDX[i];
    double DVDY_m = (VV[idx] - VV[idx_jm1]) * RDY[j];
    double TZZM = MU[idx] * (FOTH * DP[idx] - 2.0 * (DUDX_m + DVDY_m));

    // Stress gradients
    double DTXZDX = RDX[i] * (TXZP - TXZM);
    double DTYZDY = RDY[j] * (TYZP - TYZM);
    double DTZZDZ = RDZN[k] * (TZZP - TZZM);
    double VTRM = DTXZDX + DTYZDY + DTZZDZ;

    // Gravity with averaged RHO_0
    double GZ = GZ_arr[i];
    double RHO_0_avg = 0.5 * (RHO_0[k] + RHO_0[k+1]);

    // Final FVZ
    FVZ[idx] = 0.25 * (VOMX - UOMY) - GZ + RRHO * (GZ * RHO_0_avg - VTRM);
}

/* ============================================================================
 * Advection Kernel with SUPERBEE Flux Limiter
 *
 * Computes scalar advection flux for species/enthalpy transport
 * Uses SUPERBEE flux limiter for monotonicity
 * ============================================================================ */

__device__ double superbee_limiter(double r) {
    // SUPERBEE: max(0, min(2r, 1), min(r, 2))
    if (r <= 0.0) return 0.0;
    double a = fmin(2.0 * r, 1.0);
    double b = fmin(r, 2.0);
    return fmax(a, b);
}

__device__ double charm_limiter(double r) {
    // CHARM limiter
    if (r <= 0.0) return 0.0;
    return r * (3.0 * r + 1.0) / ((r + 1.0) * (r + 1.0));
}

__global__ void advection_flux_x_kernel(
    const double* __restrict__ RHO_SCALAR,  // rho * scalar (e.g., RHO*ZZ)
    const double* __restrict__ UU,           // X-velocity
    double* __restrict__ FX,                 // X-direction flux output
    int flux_limiter_type,                   // 0=upwind, 1=superbee, 2=charm
    int IBAR, int JBAR, int KBAR)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    int ni = NI(IBAR);
    int nj = NJ(JBAR);

    // Valid range for flux: i in [0,IBAR], j in [1,JBAR], k in [1,KBAR]
    if (i > IBAR || j > JBAR || k > KBAR) return;

    int idx = IDX3D(i, j, k, ni, nj);
    int idx_ip1 = IDX3D(i+1, j, k, ni, nj);
    int idx_im1 = (i > 0) ? IDX3D(i-1, j, k, ni, nj) : idx;
    int idx_ip2 = (i < IBAR) ? IDX3D(i+2, j, k, ni, nj) : idx_ip1;

    double u = UU[idx];  // Velocity at face i

    // Upwind differences
    double du_loc = RHO_SCALAR[idx_ip1] - RHO_SCALAR[idx];
    double du_up;

    if (u >= 0.0) {
        // Flow from left to right
        du_up = RHO_SCALAR[idx] - RHO_SCALAR[idx_im1];
    } else {
        // Flow from right to left
        du_up = RHO_SCALAR[idx_ip2] - RHO_SCALAR[idx_ip1];
    }

    // Flux limiter
    double b = 0.0;
    if (flux_limiter_type > 0 && fabs(du_loc) > TWO_EPSILON_EB) {
        double r = du_up / du_loc;
        if (flux_limiter_type == 1) {
            b = superbee_limiter(r);
        } else if (flux_limiter_type == 2) {
            b = charm_limiter(r);
        }
    }

    // Reconstruct face value
    double scalar_face;
    if (u >= 0.0) {
        scalar_face = RHO_SCALAR[idx] + 0.5 * b * du_loc;
    } else {
        scalar_face = RHO_SCALAR[idx_ip1] - 0.5 * b * du_loc;
    }

    FX[idx] = u * scalar_face;
}

__global__ void advection_flux_y_kernel(
    const double* __restrict__ RHO_SCALAR,
    const double* __restrict__ VV,
    double* __restrict__ FY,
    int flux_limiter_type,
    int IBAR, int JBAR, int KBAR)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    int ni = NI(IBAR);
    int nj = NJ(JBAR);

    if (i > IBAR || j > JBAR || k > KBAR) return;

    int idx = IDX3D(i, j, k, ni, nj);
    int idx_jp1 = IDX3D(i, j+1, k, ni, nj);
    int idx_jm1 = (j > 0) ? IDX3D(i, j-1, k, ni, nj) : idx;
    int idx_jp2 = (j < JBAR) ? IDX3D(i, j+2, k, ni, nj) : idx_jp1;

    double v = VV[idx];

    double dv_loc = RHO_SCALAR[idx_jp1] - RHO_SCALAR[idx];
    double dv_up;

    if (v >= 0.0) {
        dv_up = RHO_SCALAR[idx] - RHO_SCALAR[idx_jm1];
    } else {
        dv_up = RHO_SCALAR[idx_jp2] - RHO_SCALAR[idx_jp1];
    }

    double b = 0.0;
    if (flux_limiter_type > 0 && fabs(dv_loc) > TWO_EPSILON_EB) {
        double r = dv_up / dv_loc;
        if (flux_limiter_type == 1) {
            b = superbee_limiter(r);
        } else if (flux_limiter_type == 2) {
            b = charm_limiter(r);
        }
    }

    double scalar_face;
    if (v >= 0.0) {
        scalar_face = RHO_SCALAR[idx] + 0.5 * b * dv_loc;
    } else {
        scalar_face = RHO_SCALAR[idx_jp1] - 0.5 * b * dv_loc;
    }

    FY[idx] = v * scalar_face;
}

__global__ void advection_flux_z_kernel(
    const double* __restrict__ RHO_SCALAR,
    const double* __restrict__ WW,
    double* __restrict__ FZ,
    int flux_limiter_type,
    int IBAR, int JBAR, int KBAR)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int ni = NI(IBAR);
    int nj = NJ(JBAR);

    if (i > IBAR || j > JBAR || k > KBAR) return;

    int idx = IDX3D(i, j, k, ni, nj);
    int idx_kp1 = IDX3D(i, j, k+1, ni, nj);
    int idx_km1 = (k > 0) ? IDX3D(i, j, k-1, ni, nj) : idx;
    int idx_kp2 = (k < KBAR) ? IDX3D(i, j, k+2, ni, nj) : idx_kp1;

    double w = WW[idx];

    double dw_loc = RHO_SCALAR[idx_kp1] - RHO_SCALAR[idx];
    double dw_up;

    if (w >= 0.0) {
        dw_up = RHO_SCALAR[idx] - RHO_SCALAR[idx_km1];
    } else {
        dw_up = RHO_SCALAR[idx_kp2] - RHO_SCALAR[idx_kp1];
    }

    double b = 0.0;
    if (flux_limiter_type > 0 && fabs(dw_loc) > TWO_EPSILON_EB) {
        double r = dw_up / dw_loc;
        if (flux_limiter_type == 1) {
            b = superbee_limiter(r);
        } else if (flux_limiter_type == 2) {
            b = charm_limiter(r);
        }
    }

    double scalar_face;
    if (w >= 0.0) {
        scalar_face = RHO_SCALAR[idx] + 0.5 * b * dw_loc;
    } else {
        scalar_face = RHO_SCALAR[idx_kp1] - 0.5 * b * dw_loc;
    }

    FZ[idx] = w * scalar_face;
}

/* ============================================================================
 * FDS-style Advection Divergence Kernel
 *
 * Computes: U_DOT_DEL = ((FX-RHO)*U - (FX_m1-RHO)*U_m1)*RDX + ... (for Y and Z)
 * This matches the FDS ENTHALPY_ADVECTION_NEW formulation (Tech Guide B.12-B.14)
 * ============================================================================ */

__global__ void fds_advection_divergence_kernel(
    const double* __restrict__ FX,       // Face values in X
    const double* __restrict__ FY,       // Face values in Y
    const double* __restrict__ FZ,       // Face values in Z
    const double* __restrict__ RHO_SCALAR, // Cell-centered scalar (rho*h_s)
    const double* __restrict__ UU,       // X-velocity
    const double* __restrict__ VV,       // Y-velocity
    const double* __restrict__ WW,       // Z-velocity
    const double* __restrict__ RDX,
    const double* __restrict__ RDY,
    const double* __restrict__ RDZ,
    double* __restrict__ U_DOT_DEL,
    int IBAR, int JBAR, int KBAR)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    int ni = NI(IBAR);
    int nj = NJ(JBAR);

    if (i > IBAR || j > JBAR || k > KBAR) return;

    int idx = IDX3D(i, j, k, ni, nj);
    int idx_im1 = IDX3D(i-1, j, k, ni, nj);
    int idx_jm1 = IDX3D(i, j-1, k, ni, nj);
    int idx_km1 = IDX3D(i, j, k-1, ni, nj);

    double rho_s = RHO_SCALAR[idx];

    // FDS formulation: DU_P = (FX(i) - RHO_SCALAR(i)) * UU(i)
    double du_p = (FX[idx] - rho_s) * UU[idx];
    double du_m = (FX[idx_im1] - rho_s) * UU[idx_im1];
    double dv_p = (FY[idx] - rho_s) * VV[idx];
    double dv_m = (FY[idx_jm1] - rho_s) * VV[idx_jm1];
    double dw_p = (FZ[idx] - rho_s) * WW[idx];
    double dw_m = (FZ[idx_km1] - rho_s) * WW[idx_km1];

    U_DOT_DEL[idx] = (du_p - du_m) * RDX[i] +
                     (dv_p - dv_m) * RDY[j] +
                     (dw_p - dw_m) * RDZ[k];
}

/* ============================================================================
 * Divergence of Flux Kernel
 *
 * Computes: U_DOT_DEL = (FX(i)-FX(i-1))/dx + (FY(j)-FY(j-1))/dy + (FZ(k)-FZ(k-1))/dz
 * ============================================================================ */

__global__ void flux_divergence_kernel(
    const double* __restrict__ FX,
    const double* __restrict__ FY,
    const double* __restrict__ FZ,
    const double* __restrict__ RDX,
    const double* __restrict__ RDY,
    const double* __restrict__ RDZ,
    double* __restrict__ U_DOT_DEL,
    int IBAR, int JBAR, int KBAR)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    int ni = NI(IBAR);
    int nj = NJ(JBAR);

    // Valid range: i in [1,IBAR], j in [1,JBAR], k in [1,KBAR]
    if (i > IBAR || j > JBAR || k > KBAR) return;

    int idx = IDX3D(i, j, k, ni, nj);
    int idx_im1 = IDX3D(i-1, j, k, ni, nj);
    int idx_jm1 = IDX3D(i, j-1, k, ni, nj);
    int idx_km1 = IDX3D(i, j, k-1, ni, nj);

    double div_x = (FX[idx] - FX[idx_im1]) * RDX[i];
    double div_y = (FY[idx] - FY[idx_jm1]) * RDY[j];
    double div_z = (FZ[idx] - FZ[idx_km1]) * RDZ[k];

    U_DOT_DEL[idx] = div_x + div_y + div_z;
}

/* ============================================================================
 * C Wrapper Functions (extern "C" for Fortran linkage)
 * ============================================================================ */

extern "C" {

/* Launch species diffusion flux kernel */
int gpu_species_diffusion_flux(
    const double* d_ZZP,
    const double* d_RHO_D,
    const double* d_RDXN,
    const double* d_RDYN,
    const double* d_RDZN,
    double* d_RHO_D_DZDX,
    double* d_RHO_D_DZDY,
    double* d_RHO_D_DZDZ,
    int IBAR, int JBAR, int KBAR,
    int n_species,
    int species_idx,
    cudaStream_t stream)
{
    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 grid(
        (IBAR + 1 + BLOCK_X - 1) / BLOCK_X,
        (JBAR + 1 + BLOCK_Y - 1) / BLOCK_Y,
        (KBAR + 1 + BLOCK_Z - 1) / BLOCK_Z
    );

    species_diffusion_flux_kernel<<<grid, block, 0, stream>>>(
        d_ZZP, d_RHO_D, d_RDXN, d_RDYN, d_RDZN,
        d_RHO_D_DZDX, d_RHO_D_DZDY, d_RHO_D_DZDZ,
        IBAR, JBAR, KBAR, n_species, species_idx
    );

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/* Launch thermal diffusion flux kernel */
int gpu_thermal_diffusion_flux(
    const double* d_TMP,
    const double* d_KP,
    const double* d_RDXN,
    const double* d_RDYN,
    const double* d_RDZN,
    double* d_KDTDX,
    double* d_KDTDY,
    double* d_KDTDZ,
    int IBAR, int JBAR, int KBAR,
    cudaStream_t stream)
{
    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 grid(
        (IBAR + 1 + BLOCK_X - 1) / BLOCK_X,
        (JBAR + 1 + BLOCK_Y - 1) / BLOCK_Y,
        (KBAR + 1 + BLOCK_Z - 1) / BLOCK_Z
    );

    thermal_diffusion_flux_kernel<<<grid, block, 0, stream>>>(
        d_TMP, d_KP, d_RDXN, d_RDYN, d_RDZN,
        d_KDTDX, d_KDTDY, d_KDTDZ,
        IBAR, JBAR, KBAR
    );

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/* Launch vorticity and stress tensor kernel */
int gpu_vorticity_stress(
    const double* d_UU,
    const double* d_VV,
    const double* d_WW,
    const double* d_MU,
    const double* d_RDXN,
    const double* d_RDYN,
    const double* d_RDZN,
    double* d_OMX,
    double* d_OMY,
    double* d_OMZ,
    double* d_TXY,
    double* d_TXZ,
    double* d_TYZ,
    int IBAR, int JBAR, int KBAR,
    cudaStream_t stream)
{
    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 grid(
        (IBAR + 1 + BLOCK_X - 1) / BLOCK_X,
        (JBAR + 1 + BLOCK_Y - 1) / BLOCK_Y,
        (KBAR + 1 + BLOCK_Z - 1) / BLOCK_Z
    );

    vorticity_stress_kernel<<<grid, block, 0, stream>>>(
        d_UU, d_VV, d_WW, d_MU, d_RDXN, d_RDYN, d_RDZN,
        d_OMX, d_OMY, d_OMZ, d_TXY, d_TXZ, d_TYZ,
        IBAR, JBAR, KBAR
    );

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/* Launch velocity flux kernels (FVX, FVY, FVZ) */
int gpu_velocity_flux(
    const double* d_UU,
    const double* d_VV,
    const double* d_WW,
    const double* d_RHOP,
    const double* d_MU,
    const double* d_DP,
    const double* d_OMX,
    const double* d_OMY,
    const double* d_OMZ,
    const double* d_TXY,
    const double* d_TXZ,
    const double* d_TYZ,
    const double* d_RDX,
    const double* d_RDXN,
    const double* d_RDY,
    const double* d_RDYN,
    const double* d_RDZ,
    const double* d_RDZN,
    const double* d_RHO_0,
    const double* d_GX,
    const double* d_GY,
    const double* d_GZ,
    double* d_FVX,
    double* d_FVY,
    double* d_FVZ,
    int IBAR, int JBAR, int KBAR,
    cudaStream_t stream)
{
    // FVX: (0:IBAR, 1:JBAR, 1:KBAR)
    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 grid_x(
        (IBAR + 1 + BLOCK_X - 1) / BLOCK_X,
        (JBAR + BLOCK_Y - 1) / BLOCK_Y,
        (KBAR + BLOCK_Z - 1) / BLOCK_Z
    );

    velocity_flux_x_kernel<<<grid_x, block, 0, stream>>>(
        d_UU, d_VV, d_WW, d_RHOP, d_MU, d_DP,
        d_OMY, d_OMZ, d_TXY, d_TXZ,
        d_RDX, d_RDXN, d_RDY, d_RDZ, d_RHO_0, d_GX,
        d_FVX, IBAR, JBAR, KBAR
    );

    // FVY: (1:IBAR, 0:JBAR, 1:KBAR)
    dim3 grid_y(
        (IBAR + BLOCK_X - 1) / BLOCK_X,
        (JBAR + 1 + BLOCK_Y - 1) / BLOCK_Y,
        (KBAR + BLOCK_Z - 1) / BLOCK_Z
    );

    velocity_flux_y_kernel<<<grid_y, block, 0, stream>>>(
        d_UU, d_VV, d_WW, d_RHOP, d_MU, d_DP,
        d_OMX, d_OMZ, d_TXY, d_TYZ,
        d_RDX, d_RDYN, d_RDZ, d_RHO_0, d_GY,
        d_FVY, IBAR, JBAR, KBAR
    );

    // FVZ: (1:IBAR, 1:JBAR, 0:KBAR)
    dim3 grid_z(
        (IBAR + BLOCK_X - 1) / BLOCK_X,
        (JBAR + BLOCK_Y - 1) / BLOCK_Y,
        (KBAR + 1 + BLOCK_Z - 1) / BLOCK_Z
    );

    velocity_flux_z_kernel<<<grid_z, block, 0, stream>>>(
        d_UU, d_VV, d_WW, d_RHOP, d_MU, d_DP,
        d_OMX, d_OMY, d_TXZ, d_TYZ,
        d_RDX, d_RDY, d_RDZN, d_RHO_0, d_GZ,
        d_FVZ, IBAR, JBAR, KBAR
    );

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/* Launch advection flux kernels */
int gpu_advection_flux(
    const double* d_RHO_SCALAR,
    const double* d_UU,
    const double* d_VV,
    const double* d_WW,
    double* d_FX,
    double* d_FY,
    double* d_FZ,
    int flux_limiter_type,
    int IBAR, int JBAR, int KBAR,
    cudaStream_t stream)
{
    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);

    // FX: (0:IBAR, 1:JBAR, 1:KBAR)
    dim3 grid_x(
        (IBAR + 1 + BLOCK_X - 1) / BLOCK_X,
        (JBAR + BLOCK_Y - 1) / BLOCK_Y,
        (KBAR + BLOCK_Z - 1) / BLOCK_Z
    );

    advection_flux_x_kernel<<<grid_x, block, 0, stream>>>(
        d_RHO_SCALAR, d_UU, d_FX, flux_limiter_type, IBAR, JBAR, KBAR
    );

    // FY: (1:IBAR, 0:JBAR, 1:KBAR)
    dim3 grid_y(
        (IBAR + BLOCK_X - 1) / BLOCK_X,
        (JBAR + 1 + BLOCK_Y - 1) / BLOCK_Y,
        (KBAR + BLOCK_Z - 1) / BLOCK_Z
    );

    advection_flux_y_kernel<<<grid_y, block, 0, stream>>>(
        d_RHO_SCALAR, d_VV, d_FY, flux_limiter_type, IBAR, JBAR, KBAR
    );

    // FZ: (1:IBAR, 1:JBAR, 0:KBAR)
    dim3 grid_z(
        (IBAR + BLOCK_X - 1) / BLOCK_X,
        (JBAR + BLOCK_Y - 1) / BLOCK_Y,
        (KBAR + 1 + BLOCK_Z - 1) / BLOCK_Z
    );

    advection_flux_z_kernel<<<grid_z, block, 0, stream>>>(
        d_RHO_SCALAR, d_WW, d_FZ, flux_limiter_type, IBAR, JBAR, KBAR
    );

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/* Launch flux divergence kernel */
int gpu_flux_divergence(
    const double* d_FX,
    const double* d_FY,
    const double* d_FZ,
    const double* d_RDX,
    const double* d_RDY,
    const double* d_RDZ,
    double* d_U_DOT_DEL,
    int IBAR, int JBAR, int KBAR,
    cudaStream_t stream)
{
    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 grid(
        (IBAR + BLOCK_X - 1) / BLOCK_X,
        (JBAR + BLOCK_Y - 1) / BLOCK_Y,
        (KBAR + BLOCK_Z - 1) / BLOCK_Z
    );

    flux_divergence_kernel<<<grid, block, 0, stream>>>(
        d_FX, d_FY, d_FZ, d_RDX, d_RDY, d_RDZ, d_U_DOT_DEL,
        IBAR, JBAR, KBAR
    );

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/* Launch FDS-style advection divergence kernel */
int gpu_fds_advection_divergence(
    const double* d_FX,
    const double* d_FY,
    const double* d_FZ,
    const double* d_RHO_SCALAR,
    const double* d_UU,
    const double* d_VV,
    const double* d_WW,
    const double* d_RDX,
    const double* d_RDY,
    const double* d_RDZ,
    double* d_U_DOT_DEL,
    int IBAR, int JBAR, int KBAR,
    cudaStream_t stream)
{
    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 grid(
        (IBAR + BLOCK_X - 1) / BLOCK_X,
        (JBAR + BLOCK_Y - 1) / BLOCK_Y,
        (KBAR + BLOCK_Z - 1) / BLOCK_Z
    );

    fds_advection_divergence_kernel<<<grid, block, 0, stream>>>(
        d_FX, d_FY, d_FZ, d_RHO_SCALAR, d_UU, d_VV, d_WW,
        d_RDX, d_RDY, d_RDZ, d_U_DOT_DEL,
        IBAR, JBAR, KBAR
    );

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/* Synchronize stream */
int gpu_stream_synchronize(cudaStream_t stream) {
    cudaError_t err = cudaStreamSynchronize(stream);
    return err == cudaSuccess ? 0 : -1;
}

/* ============================================================================
 * Density Update Kernels
 * ============================================================================ */

/* Kernel for single species density update
 * Computes: ZZS(I,J,K) = RHO*ZZ - DT * RHS
 * where RHS = -DEL_RHO_D_DEL_Z + div(FX*U, FY*V, FZ*W)
 * For Cartesian meshes (R=RRN=1)
 */
__global__ void density_update_kernel(
    const double* __restrict__ RHO,
    const double* __restrict__ ZZ,
    const double* __restrict__ DEL_RHO_D_DEL_Z,
    const double* __restrict__ FX,     // Face values (0:IBAR, 0:JBAR+1, 0:KBAR+1)
    const double* __restrict__ FY,
    const double* __restrict__ FZ,
    const double* __restrict__ UU,
    const double* __restrict__ VV,
    const double* __restrict__ WW,
    const double* __restrict__ RDX,
    const double* __restrict__ RDY,
    const double* __restrict__ RDZ,
    double DT,
    double* __restrict__ ZZS,
    int IBAR, int JBAR, int KBAR)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;  // 1:IBAR
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;  // 1:JBAR
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;  // 1:KBAR

    if (i > IBAR || j > JBAR || k > KBAR) return;

    // 3D array indexing: (0:IBAR+1, 0:JBAR+1, 0:KBAR+1)
    int NX = IBAR + 2;
    int NY = JBAR + 2;
    int idx = i + j * NX + k * NX * NY;
    int idx_im1 = (i-1) + j * NX + k * NX * NY;
    int idx_jm1 = i + (j-1) * NX + k * NX * NY;
    int idx_km1 = i + j * NX + (k-1) * NX * NY;

    // Compute RHS = -DEL_RHO_D_DEL_Z + div(FX*U, FY*V, FZ*W)
    // For Cartesian: R(I) = RRN(I) = 1
    double flux_x = (FX[idx] * UU[idx] - FX[idx_im1] * UU[idx_im1]) * RDX[i];
    double flux_y = (FY[idx] * VV[idx] - FY[idx_jm1] * VV[idx_jm1]) * RDY[j];
    double flux_z = (FZ[idx] * WW[idx] - FZ[idx_km1] * WW[idx_km1]) * RDZ[k];

    double RHS = -DEL_RHO_D_DEL_Z[idx] + flux_x + flux_y + flux_z;

    ZZS[idx] = RHO[idx] * ZZ[idx] - DT * RHS;
}

/* Launch density update kernel for a single species */
int gpu_density_update(
    const double* d_RHO,
    const double* d_ZZ,
    const double* d_DEL_RHO_D_DEL_Z,
    const double* d_FX,
    const double* d_FY,
    const double* d_FZ,
    const double* d_UU,
    const double* d_VV,
    const double* d_WW,
    const double* d_RDX,
    const double* d_RDY,
    const double* d_RDZ,
    double DT,
    double* d_ZZS,
    int IBAR, int JBAR, int KBAR,
    cudaStream_t stream)
{
    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 grid(
        (IBAR + BLOCK_X - 1) / BLOCK_X,
        (JBAR + BLOCK_Y - 1) / BLOCK_Y,
        (KBAR + BLOCK_Z - 1) / BLOCK_Z
    );

    density_update_kernel<<<grid, block, 0, stream>>>(
        d_RHO, d_ZZ, d_DEL_RHO_D_DEL_Z,
        d_FX, d_FY, d_FZ,
        d_UU, d_VV, d_WW,
        d_RDX, d_RDY, d_RDZ,
        DT, d_ZZS,
        IBAR, JBAR, KBAR
    );

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

} /* extern "C" */

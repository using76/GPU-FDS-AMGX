/*
 * gpu_c_wrapper.c - C Wrapper for FDS GPU Kernels
 *
 * Part of FDS GPU Acceleration Project
 * Provides the interface between Fortran and CUDA kernels
 *
 * Copyright 2024, FDS-AmgX Project
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

/* ============================================================================
 * External declarations for GPU data manager
 * ============================================================================ */

/* From gpu_data_manager.c */
extern int gpu_get_mesh_data(int mesh_id, void** data_ptr);
extern void* gpu_get_device_pointer(int mesh_id, const char* field_name);
extern cudaStream_t gpu_get_compute_stream(int mesh_id);

/* From gpu_kernels.cu */
extern int gpu_species_diffusion_flux(
    const double* d_ZZP, const double* d_RHO_D,
    const double* d_RDXN, const double* d_RDYN, const double* d_RDZN,
    double* d_RHO_D_DZDX, double* d_RHO_D_DZDY, double* d_RHO_D_DZDZ,
    int IBAR, int JBAR, int KBAR, int n_species, int species_idx,
    cudaStream_t stream);

extern int gpu_thermal_diffusion_flux(
    const double* d_TMP, const double* d_KP,
    const double* d_RDXN, const double* d_RDYN, const double* d_RDZN,
    double* d_KDTDX, double* d_KDTDY, double* d_KDTDZ,
    int IBAR, int JBAR, int KBAR,
    cudaStream_t stream);

extern int gpu_vorticity_stress(
    const double* d_UU, const double* d_VV, const double* d_WW,
    const double* d_MU,
    const double* d_RDXN, const double* d_RDYN, const double* d_RDZN,
    double* d_OMX, double* d_OMY, double* d_OMZ,
    double* d_TXY, double* d_TXZ, double* d_TYZ,
    int IBAR, int JBAR, int KBAR,
    cudaStream_t stream);

extern int gpu_velocity_flux(
    const double* d_UU, const double* d_VV, const double* d_WW,
    const double* d_RHOP, const double* d_MU, const double* d_DP,
    const double* d_OMX, const double* d_OMY, const double* d_OMZ,
    const double* d_TXY, const double* d_TXZ, const double* d_TYZ,
    const double* d_RDX, const double* d_RDXN,
    const double* d_RDY, const double* d_RDYN,
    const double* d_RDZ, const double* d_RDZN,
    const double* d_RHO_0,
    const double* d_GX, const double* d_GY, const double* d_GZ,
    double* d_FVX, double* d_FVY, double* d_FVZ,
    int IBAR, int JBAR, int KBAR,
    cudaStream_t stream);

extern int gpu_advection_flux(
    const double* d_RHO_SCALAR,
    const double* d_UU, const double* d_VV, const double* d_WW,
    double* d_FX, double* d_FY, double* d_FZ,
    int flux_limiter_type,
    int IBAR, int JBAR, int KBAR,
    cudaStream_t stream);

extern int gpu_flux_divergence(
    const double* d_FX, const double* d_FY, const double* d_FZ,
    const double* d_RDX, const double* d_RDY, const double* d_RDZ,
    double* d_U_DOT_DEL,
    int IBAR, int JBAR, int KBAR,
    cudaStream_t stream);

extern int gpu_fds_advection_divergence(
    const double* d_FX, const double* d_FY, const double* d_FZ,
    const double* d_RHO_SCALAR,
    const double* d_UU, const double* d_VV, const double* d_WW,
    const double* d_RDX, const double* d_RDY, const double* d_RDZ,
    double* d_U_DOT_DEL,
    int IBAR, int JBAR, int KBAR,
    cudaStream_t stream);

extern int gpu_stream_synchronize(cudaStream_t stream);

extern int gpu_density_update(
    const double* d_RHO, const double* d_ZZ, const double* d_DEL_RHO_D_DEL_Z,
    const double* d_FX, const double* d_FY, const double* d_FZ,
    const double* d_UU, const double* d_VV, const double* d_WW,
    const double* d_RDX, const double* d_RDY, const double* d_RDZ,
    double DT, double* d_ZZS,
    int IBAR, int JBAR, int KBAR,
    cudaStream_t stream);

/* ============================================================================
 * Constants
 * ============================================================================ */

#define MAX_GPU_MESHES 256
#define GPU_SUCCESS 0
#define GPU_ERROR -1

/* Flux limiter types */
#define FLUX_LIMITER_UPWIND   0
#define FLUX_LIMITER_SUPERBEE 1
#define FLUX_LIMITER_CHARM    2

/* ============================================================================
 * Internal Data Structures
 * ============================================================================ */

typedef struct {
    int mesh_id;
    int initialized;
    int ibar, jbar, kbar;
    int n_species;

    /* Grid spacing arrays (GPU device memory) */
    double *d_RDX, *d_RDY, *d_RDZ;
    double *d_RDXN, *d_RDYN, *d_RDZN;

    /* Background density profile */
    double *d_RHO_0;

    /* Gravity arrays */
    double *d_GX, *d_GY, *d_GZ;

    /* Vorticity and stress tensor (intermediate storage) */
    double *d_OMX, *d_OMY, *d_OMZ;
    double *d_TXY, *d_TXZ, *d_TYZ;

    /* Advection work arrays */
    double *d_FX, *d_FY, *d_FZ;
    double *d_U_DOT_DEL;

    /* ========== PERSISTENT FIELD ARRAYS (Performance Optimization) ========== */
    /* Velocity components - uploaded once per timestep, reused across kernels */
    double *d_UU, *d_VV, *d_WW;
    int velocity_uploaded;  /* Flag: 1 if velocity is current */

    /* Thermodynamic properties - uploaded once, reused across kernels */
    double *d_RHOP;         /* Density perturbation */
    double *d_MU;           /* Dynamic viscosity */
    double *d_DP;           /* Pressure gradient / Divergence */
    double *d_TMP;          /* Temperature */
    double *d_KP;           /* Thermal conductivity */
    int thermo_uploaded;    /* Flag: 1 if thermodynamic properties are current */

    /* Species arrays - for multi-species calculations */
    double *d_ZZP;          /* Species mass fraction (single species buffer) */
    double *d_RHO_D;        /* Species diffusivity */
    double *d_DEL_RHO_D_DEL_Z; /* Diffusion flux */

    /* ========== PINNED MEMORY BUFFERS (Fast DMA Transfers) ========== */
    double *pinned_upload;   /* Page-locked buffer for host->device */
    double *pinned_download; /* Page-locked buffer for device->host */
    size_t pinned_size;      /* Size of pinned buffers in bytes */

    /* ========== CUDA RESOURCES ========== */
    cudaStream_t stream;
    cudaStream_t stream_upload;  /* Secondary stream for async uploads */

} GPUKernelContext;

static GPUKernelContext kernel_contexts[MAX_GPU_MESHES];
static int gpu_kernel_initialized = 0;

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

static int find_context_slot(int mesh_id) {
    for (int i = 0; i < MAX_GPU_MESHES; i++) {
        if (kernel_contexts[i].mesh_id == mesh_id && kernel_contexts[i].initialized) {
            return i;
        }
    }
    return -1;
}

static int find_free_slot(void) {
    for (int i = 0; i < MAX_GPU_MESHES; i++) {
        if (!kernel_contexts[i].initialized) {
            return i;
        }
    }
    return -1;
}

static size_t calc_3d_size(int ibar, int jbar, int kbar) {
    return (size_t)(ibar + 2) * (jbar + 2) * (kbar + 2);
}

/* ============================================================================
 * Fortran-Callable Functions (extern names with trailing underscore)
 * ============================================================================ */

/* Initialize GPU kernel system */
void gpu_kernel_init_(int* ierr) {
    if (gpu_kernel_initialized) {
        *ierr = GPU_SUCCESS;
        return;
    }

    memset(kernel_contexts, 0, sizeof(kernel_contexts));
    gpu_kernel_initialized = 1;

    fprintf(stderr, " GPU Kernel System: Initialized\n");
    *ierr = GPU_SUCCESS;
}

/* Finalize GPU kernel system */
void gpu_kernel_finalize_(int* ierr) {
    if (!gpu_kernel_initialized) {
        *ierr = GPU_SUCCESS;
        return;
    }

    for (int i = 0; i < MAX_GPU_MESHES; i++) {
        if (kernel_contexts[i].initialized) {
            GPUKernelContext* ctx = &kernel_contexts[i];

            /* Free grid spacing arrays */
            if (ctx->d_RDX) cudaFree(ctx->d_RDX);
            if (ctx->d_RDY) cudaFree(ctx->d_RDY);
            if (ctx->d_RDZ) cudaFree(ctx->d_RDZ);
            if (ctx->d_RDXN) cudaFree(ctx->d_RDXN);
            if (ctx->d_RDYN) cudaFree(ctx->d_RDYN);
            if (ctx->d_RDZN) cudaFree(ctx->d_RDZN);

            /* Free background density */
            if (ctx->d_RHO_0) cudaFree(ctx->d_RHO_0);

            /* Free gravity arrays */
            if (ctx->d_GX) cudaFree(ctx->d_GX);
            if (ctx->d_GY) cudaFree(ctx->d_GY);
            if (ctx->d_GZ) cudaFree(ctx->d_GZ);

            /* Free intermediate arrays */
            if (ctx->d_OMX) cudaFree(ctx->d_OMX);
            if (ctx->d_OMY) cudaFree(ctx->d_OMY);
            if (ctx->d_OMZ) cudaFree(ctx->d_OMZ);
            if (ctx->d_TXY) cudaFree(ctx->d_TXY);
            if (ctx->d_TXZ) cudaFree(ctx->d_TXZ);
            if (ctx->d_TYZ) cudaFree(ctx->d_TYZ);

            /* Free advection work arrays */
            if (ctx->d_FX) cudaFree(ctx->d_FX);
            if (ctx->d_FY) cudaFree(ctx->d_FY);
            if (ctx->d_FZ) cudaFree(ctx->d_FZ);
            if (ctx->d_U_DOT_DEL) cudaFree(ctx->d_U_DOT_DEL);

            /* Free persistent velocity arrays */
            if (ctx->d_UU) cudaFree(ctx->d_UU);
            if (ctx->d_VV) cudaFree(ctx->d_VV);
            if (ctx->d_WW) cudaFree(ctx->d_WW);

            /* Free persistent thermodynamic arrays */
            if (ctx->d_RHOP) cudaFree(ctx->d_RHOP);
            if (ctx->d_MU) cudaFree(ctx->d_MU);
            if (ctx->d_DP) cudaFree(ctx->d_DP);
            if (ctx->d_TMP) cudaFree(ctx->d_TMP);
            if (ctx->d_KP) cudaFree(ctx->d_KP);

            /* Free persistent species arrays */
            if (ctx->d_ZZP) cudaFree(ctx->d_ZZP);
            if (ctx->d_RHO_D) cudaFree(ctx->d_RHO_D);
            if (ctx->d_DEL_RHO_D_DEL_Z) cudaFree(ctx->d_DEL_RHO_D_DEL_Z);

            /* Free pinned memory buffers */
            if (ctx->pinned_upload) cudaFreeHost(ctx->pinned_upload);
            if (ctx->pinned_download) cudaFreeHost(ctx->pinned_download);

            /* Destroy streams */
            if (ctx->stream) cudaStreamDestroy(ctx->stream);
            if (ctx->stream_upload) cudaStreamDestroy(ctx->stream_upload);

            ctx->initialized = 0;
        }
    }

    gpu_kernel_initialized = 0;
    fprintf(stderr, " GPU Kernel System: Finalized\n");
    *ierr = GPU_SUCCESS;
}

/* Allocate kernel context for a mesh */
void gpu_kernel_allocate_mesh_(int* mesh_id, int* ibar, int* jbar, int* kbar,
                                int* n_species, int* ierr) {
    if (!gpu_kernel_initialized) {
        fprintf(stderr, "ERROR: GPU kernel system not initialized\n");
        *ierr = GPU_ERROR;
        return;
    }

    int slot = find_context_slot(*mesh_id);
    if (slot >= 0) {
        /* Already allocated */
        *ierr = GPU_SUCCESS;
        return;
    }

    slot = find_free_slot();
    if (slot < 0) {
        fprintf(stderr, "ERROR: No free kernel context slots\n");
        *ierr = GPU_ERROR;
        return;
    }

    GPUKernelContext* ctx = &kernel_contexts[slot];
    ctx->mesh_id = *mesh_id;
    ctx->ibar = *ibar;
    ctx->jbar = *jbar;
    ctx->kbar = *kbar;
    ctx->n_species = *n_species;

    size_t n3d = calc_3d_size(*ibar, *jbar, *kbar);
    size_t size_3d = n3d * sizeof(double);

    cudaError_t err;

    /* Allocate grid spacing arrays */
    err = cudaMalloc(&ctx->d_RDX, (*ibar + 2) * sizeof(double));
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&ctx->d_RDY, (*jbar + 2) * sizeof(double));
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&ctx->d_RDZ, (*kbar + 2) * sizeof(double));
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&ctx->d_RDXN, (*ibar + 2) * sizeof(double));
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&ctx->d_RDYN, (*jbar + 2) * sizeof(double));
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&ctx->d_RDZN, (*kbar + 2) * sizeof(double));
    if (err != cudaSuccess) goto cuda_error;

    /* Allocate background density */
    err = cudaMalloc(&ctx->d_RHO_0, (*kbar + 2) * sizeof(double));
    if (err != cudaSuccess) goto cuda_error;

    /* Allocate gravity arrays */
    err = cudaMalloc(&ctx->d_GX, (*ibar + 2) * sizeof(double));
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&ctx->d_GY, (*ibar + 2) * sizeof(double));
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&ctx->d_GZ, (*ibar + 2) * sizeof(double));
    if (err != cudaSuccess) goto cuda_error;

    /* Allocate vorticity and stress tensor arrays */
    err = cudaMalloc(&ctx->d_OMX, size_3d);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&ctx->d_OMY, size_3d);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&ctx->d_OMZ, size_3d);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&ctx->d_TXY, size_3d);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&ctx->d_TXZ, size_3d);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&ctx->d_TYZ, size_3d);
    if (err != cudaSuccess) goto cuda_error;

    /* Allocate advection work arrays */
    err = cudaMalloc(&ctx->d_FX, size_3d);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&ctx->d_FY, size_3d);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&ctx->d_FZ, size_3d);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&ctx->d_U_DOT_DEL, size_3d);
    if (err != cudaSuccess) goto cuda_error;

    /* ========== PERSISTENT FIELD ARRAYS ========== */
    /* Allocate velocity arrays */
    err = cudaMalloc(&ctx->d_UU, size_3d);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&ctx->d_VV, size_3d);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&ctx->d_WW, size_3d);
    if (err != cudaSuccess) goto cuda_error;
    ctx->velocity_uploaded = 0;

    /* Allocate thermodynamic property arrays */
    err = cudaMalloc(&ctx->d_RHOP, size_3d);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&ctx->d_MU, size_3d);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&ctx->d_DP, size_3d);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&ctx->d_TMP, size_3d);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&ctx->d_KP, size_3d);
    if (err != cudaSuccess) goto cuda_error;
    ctx->thermo_uploaded = 0;

    /* Allocate species arrays */
    err = cudaMalloc(&ctx->d_ZZP, size_3d);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&ctx->d_RHO_D, size_3d);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&ctx->d_DEL_RHO_D_DEL_Z, size_3d);
    if (err != cudaSuccess) goto cuda_error;

    /* ========== PINNED MEMORY FOR FAST TRANSFERS ========== */
    ctx->pinned_size = size_3d;
    err = cudaMallocHost(&ctx->pinned_upload, size_3d);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMallocHost(&ctx->pinned_download, size_3d);
    if (err != cudaSuccess) goto cuda_error;

    /* ========== CREATE CUDA STREAMS ========== */
    err = cudaStreamCreate(&ctx->stream);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaStreamCreate(&ctx->stream_upload);
    if (err != cudaSuccess) goto cuda_error;

    ctx->initialized = 1;

    fprintf(stderr, " GPU Kernel Context: Allocated mesh %d (%dx%dx%d) with persistent memory\n",
            *mesh_id, *ibar, *jbar, *kbar);
    *ierr = GPU_SUCCESS;
    return;

cuda_error:
    fprintf(stderr, "ERROR: CUDA allocation failed: %s\n", cudaGetErrorString(err));
    *ierr = GPU_ERROR;
}

/* Upload grid parameters */
void gpu_kernel_upload_grid_(int* mesh_id,
                              double* RDX, double* RDY, double* RDZ,
                              double* RDXN, double* RDYN, double* RDZN,
                              double* RHO_0, int* ierr) {
    int slot = find_context_slot(*mesh_id);
    if (slot < 0) {
        fprintf(stderr, "ERROR: Mesh %d not found in kernel contexts\n", *mesh_id);
        *ierr = GPU_ERROR;
        return;
    }

    GPUKernelContext* ctx = &kernel_contexts[slot];
    cudaError_t err;

    /* Upload grid spacing arrays
     * FDS array bounds:
     * - RDX/RDY/RDZ are (0:IBAR+1), (0:JBAR+1), (0:KBAR+1) = ibar+2, jbar+2, kbar+2 elements
     * - RDXN/RDYN/RDZN are (0:IBAR), (0:JBAR), (0:KBAR) = ibar+1, jbar+1, kbar+1 elements
     */
    err = cudaMemcpy(ctx->d_RDX, RDX, (ctx->ibar + 2) * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMemcpy(ctx->d_RDY, RDY, (ctx->jbar + 2) * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMemcpy(ctx->d_RDZ, RDZ, (ctx->kbar + 2) * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMemcpy(ctx->d_RDXN, RDXN, (ctx->ibar + 1) * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMemcpy(ctx->d_RDYN, RDYN, (ctx->jbar + 1) * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMemcpy(ctx->d_RDZN, RDZN, (ctx->kbar + 1) * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cuda_error;

    /* Upload background density */
    err = cudaMemcpy(ctx->d_RHO_0, RHO_0, (ctx->kbar + 2) * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cuda_error;

    *ierr = GPU_SUCCESS;
    return;

cuda_error:
    fprintf(stderr, "ERROR: Grid upload failed: %s\n", cudaGetErrorString(err));
    *ierr = GPU_ERROR;
}

/* Upload gravity values */
void gpu_kernel_upload_gravity_(int* mesh_id, double* GX, double* GY, double* GZ, int* ierr) {
    int slot = find_context_slot(*mesh_id);
    if (slot < 0) {
        *ierr = GPU_ERROR;
        return;
    }

    GPUKernelContext* ctx = &kernel_contexts[slot];
    cudaError_t err;

    err = cudaMemcpy(ctx->d_GX, GX, (ctx->ibar + 2) * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMemcpy(ctx->d_GY, GY, (ctx->ibar + 2) * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMemcpy(ctx->d_GZ, GZ, (ctx->ibar + 2) * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cuda_error;

    *ierr = GPU_SUCCESS;
    return;

cuda_error:
    fprintf(stderr, "ERROR: Gravity upload failed: %s\n", cudaGetErrorString(err));
    *ierr = GPU_ERROR;
}

/* ============================================================================
 * Persistent Field Upload Functions (Performance Optimization)
 * ============================================================================ */

/* Upload velocity components to persistent GPU storage
 * Call once per timestep, reused across multiple kernel calls */
void gpu_kernel_upload_velocity_(int* mesh_id, double* UU, double* VV, double* WW, int* ierr) {
    int slot = find_context_slot(*mesh_id);
    if (slot < 0) {
        *ierr = GPU_ERROR;
        return;
    }

    GPUKernelContext* ctx = &kernel_contexts[slot];
    size_t n3d = calc_3d_size(ctx->ibar, ctx->jbar, ctx->kbar);
    size_t size_3d = n3d * sizeof(double);

    /* Use async transfers with pinned memory staging if possible */
    cudaMemcpyAsync(ctx->d_UU, UU, size_3d, cudaMemcpyHostToDevice, ctx->stream_upload);
    cudaMemcpyAsync(ctx->d_VV, VV, size_3d, cudaMemcpyHostToDevice, ctx->stream_upload);
    cudaMemcpyAsync(ctx->d_WW, WW, size_3d, cudaMemcpyHostToDevice, ctx->stream_upload);

    /* Synchronize upload stream to ensure data is ready */
    cudaStreamSynchronize(ctx->stream_upload);

    ctx->velocity_uploaded = 1;
    *ierr = GPU_SUCCESS;
}

/* Upload thermodynamic properties to persistent GPU storage
 * Call once per timestep, reused across multiple kernel calls */
void gpu_kernel_upload_thermo_(int* mesh_id, double* RHOP, double* MU, double* DP,
                                double* TMP, double* KP, int* ierr) {
    int slot = find_context_slot(*mesh_id);
    if (slot < 0) {
        *ierr = GPU_ERROR;
        return;
    }

    GPUKernelContext* ctx = &kernel_contexts[slot];
    size_t n3d = calc_3d_size(ctx->ibar, ctx->jbar, ctx->kbar);
    size_t size_3d = n3d * sizeof(double);

    /* Use async transfers */
    cudaMemcpyAsync(ctx->d_RHOP, RHOP, size_3d, cudaMemcpyHostToDevice, ctx->stream_upload);
    cudaMemcpyAsync(ctx->d_MU, MU, size_3d, cudaMemcpyHostToDevice, ctx->stream_upload);
    cudaMemcpyAsync(ctx->d_DP, DP, size_3d, cudaMemcpyHostToDevice, ctx->stream_upload);
    if (TMP != NULL) {
        cudaMemcpyAsync(ctx->d_TMP, TMP, size_3d, cudaMemcpyHostToDevice, ctx->stream_upload);
    }
    if (KP != NULL) {
        cudaMemcpyAsync(ctx->d_KP, KP, size_3d, cudaMemcpyHostToDevice, ctx->stream_upload);
    }

    cudaStreamSynchronize(ctx->stream_upload);

    ctx->thermo_uploaded = 1;
    *ierr = GPU_SUCCESS;
}

/* Invalidate persistent field caches - call at start of new timestep */
void gpu_kernel_invalidate_cache_(int* mesh_id, int* ierr) {
    int slot = find_context_slot(*mesh_id);
    if (slot < 0) {
        *ierr = GPU_ERROR;
        return;
    }

    GPUKernelContext* ctx = &kernel_contexts[slot];
    ctx->velocity_uploaded = 0;
    ctx->thermo_uploaded = 0;
    *ierr = GPU_SUCCESS;
}

/* ============================================================================
 * Diffusion Kernel Wrappers
 * ============================================================================ */

/* Compute species diffusion flux on GPU */
void gpu_compute_species_diffusion_(int* mesh_id,
                                     double* ZZP, double* RHO_D,
                                     double* RHO_D_DZDX, double* RHO_D_DZDY, double* RHO_D_DZDZ,
                                     int* n_species, int* ierr) {
    int slot = find_context_slot(*mesh_id);
    if (slot < 0) {
        *ierr = GPU_ERROR;
        return;
    }

    GPUKernelContext* ctx = &kernel_contexts[slot];
    size_t n3d = calc_3d_size(ctx->ibar, ctx->jbar, ctx->kbar);
    size_t size_3d = n3d * sizeof(double);
    size_t size_4d = n3d * (*n_species) * sizeof(double);

    /* Allocate temporary device arrays */
    double *d_ZZP, *d_RHO_D;
    double *d_RHO_D_DZDX, *d_RHO_D_DZDY, *d_RHO_D_DZDZ;

    cudaError_t err;
    err = cudaMalloc(&d_ZZP, size_4d);
    if (err != cudaSuccess) { *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_RHO_D, size_3d);
    if (err != cudaSuccess) { cudaFree(d_ZZP); *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_RHO_D_DZDX, size_4d);
    if (err != cudaSuccess) { cudaFree(d_ZZP); cudaFree(d_RHO_D); *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_RHO_D_DZDY, size_4d);
    if (err != cudaSuccess) { cudaFree(d_ZZP); cudaFree(d_RHO_D); cudaFree(d_RHO_D_DZDX); *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_RHO_D_DZDZ, size_4d);
    if (err != cudaSuccess) { cudaFree(d_ZZP); cudaFree(d_RHO_D); cudaFree(d_RHO_D_DZDX); cudaFree(d_RHO_D_DZDY); *ierr = GPU_ERROR; return; }

    /* Upload data */
    cudaMemcpy(d_ZZP, ZZP, size_4d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_RHO_D, RHO_D, size_3d, cudaMemcpyHostToDevice);

    /* Launch kernel for each species */
    for (int n = 0; n < *n_species; n++) {
        int result = gpu_species_diffusion_flux(
            d_ZZP, d_RHO_D,
            ctx->d_RDXN, ctx->d_RDYN, ctx->d_RDZN,
            d_RHO_D_DZDX, d_RHO_D_DZDY, d_RHO_D_DZDZ,
            ctx->ibar, ctx->jbar, ctx->kbar,
            *n_species, n,
            ctx->stream
        );

        if (result != 0) {
            fprintf(stderr, "ERROR: Species diffusion kernel failed for species %d\n", n);
            *ierr = GPU_ERROR;
            goto cleanup;
        }
    }

    /* Synchronize and download results */
    cudaStreamSynchronize(ctx->stream);
    cudaMemcpy(RHO_D_DZDX, d_RHO_D_DZDX, size_4d, cudaMemcpyDeviceToHost);
    cudaMemcpy(RHO_D_DZDY, d_RHO_D_DZDY, size_4d, cudaMemcpyDeviceToHost);
    cudaMemcpy(RHO_D_DZDZ, d_RHO_D_DZDZ, size_4d, cudaMemcpyDeviceToHost);

    *ierr = GPU_SUCCESS;

cleanup:
    cudaFree(d_ZZP);
    cudaFree(d_RHO_D);
    cudaFree(d_RHO_D_DZDX);
    cudaFree(d_RHO_D_DZDY);
    cudaFree(d_RHO_D_DZDZ);
}

/* Compute single-species diffusion flux on GPU (for per-species RHO_D) */
void gpu_compute_species_diffusion_single_(int* mesh_id,
                                            double* ZZ_N, double* RHO_D,
                                            double* RHO_D_DZDX_N, double* RHO_D_DZDY_N, double* RHO_D_DZDZ_N,
                                            int* ierr) {
    int slot = find_context_slot(*mesh_id);
    if (slot < 0) {
        *ierr = GPU_ERROR;
        return;
    }

    GPUKernelContext* ctx = &kernel_contexts[slot];
    size_t n3d = calc_3d_size(ctx->ibar, ctx->jbar, ctx->kbar);
    size_t size_3d = n3d * sizeof(double);

    /* Allocate temporary device arrays */
    double *d_ZZ_N, *d_RHO_D;
    double *d_RHO_D_DZDX_N, *d_RHO_D_DZDY_N, *d_RHO_D_DZDZ_N;

    cudaError_t err;
    err = cudaMalloc(&d_ZZ_N, size_3d);
    if (err != cudaSuccess) { *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_RHO_D, size_3d);
    if (err != cudaSuccess) { cudaFree(d_ZZ_N); *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_RHO_D_DZDX_N, size_3d);
    if (err != cudaSuccess) { cudaFree(d_ZZ_N); cudaFree(d_RHO_D); *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_RHO_D_DZDY_N, size_3d);
    if (err != cudaSuccess) { cudaFree(d_ZZ_N); cudaFree(d_RHO_D); cudaFree(d_RHO_D_DZDX_N); *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_RHO_D_DZDZ_N, size_3d);
    if (err != cudaSuccess) { cudaFree(d_ZZ_N); cudaFree(d_RHO_D); cudaFree(d_RHO_D_DZDX_N); cudaFree(d_RHO_D_DZDY_N); *ierr = GPU_ERROR; return; }

    /* Upload data */
    cudaMemcpy(d_ZZ_N, ZZ_N, size_3d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_RHO_D, RHO_D, size_3d, cudaMemcpyHostToDevice);

    /* Launch kernel for single species (n_species=1, species_idx=0) */
    int result = gpu_species_diffusion_flux(
        d_ZZ_N, d_RHO_D,
        ctx->d_RDXN, ctx->d_RDYN, ctx->d_RDZN,
        d_RHO_D_DZDX_N, d_RHO_D_DZDY_N, d_RHO_D_DZDZ_N,
        ctx->ibar, ctx->jbar, ctx->kbar,
        1, 0,  /* n_species=1, species_idx=0 */
        ctx->stream
    );

    if (result != 0) {
        fprintf(stderr, "ERROR: Single species diffusion kernel failed\n");
        *ierr = GPU_ERROR;
        goto cleanup;
    }

    /* Synchronize and download results */
    cudaStreamSynchronize(ctx->stream);
    cudaMemcpy(RHO_D_DZDX_N, d_RHO_D_DZDX_N, size_3d, cudaMemcpyDeviceToHost);
    cudaMemcpy(RHO_D_DZDY_N, d_RHO_D_DZDY_N, size_3d, cudaMemcpyDeviceToHost);
    cudaMemcpy(RHO_D_DZDZ_N, d_RHO_D_DZDZ_N, size_3d, cudaMemcpyDeviceToHost);

    *ierr = GPU_SUCCESS;

cleanup:
    cudaFree(d_ZZ_N);
    cudaFree(d_RHO_D);
    cudaFree(d_RHO_D_DZDX_N);
    cudaFree(d_RHO_D_DZDY_N);
    cudaFree(d_RHO_D_DZDZ_N);
}

/* Compute thermal diffusion flux on GPU */
void gpu_compute_thermal_diffusion_(int* mesh_id,
                                     double* TMP, double* KP,
                                     double* KDTDX, double* KDTDY, double* KDTDZ,
                                     int* ierr) {
    int slot = find_context_slot(*mesh_id);
    if (slot < 0) {
        *ierr = GPU_ERROR;
        return;
    }

    GPUKernelContext* ctx = &kernel_contexts[slot];
    size_t n3d = calc_3d_size(ctx->ibar, ctx->jbar, ctx->kbar);
    size_t size_3d = n3d * sizeof(double);

    /* Allocate temporary device arrays */
    double *d_TMP, *d_KP;
    double *d_KDTDX, *d_KDTDY, *d_KDTDZ;

    cudaError_t err;
    err = cudaMalloc(&d_TMP, size_3d);
    if (err != cudaSuccess) { *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_KP, size_3d);
    if (err != cudaSuccess) { cudaFree(d_TMP); *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_KDTDX, size_3d);
    if (err != cudaSuccess) { cudaFree(d_TMP); cudaFree(d_KP); *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_KDTDY, size_3d);
    if (err != cudaSuccess) { cudaFree(d_TMP); cudaFree(d_KP); cudaFree(d_KDTDX); *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_KDTDZ, size_3d);
    if (err != cudaSuccess) { cudaFree(d_TMP); cudaFree(d_KP); cudaFree(d_KDTDX); cudaFree(d_KDTDY); *ierr = GPU_ERROR; return; }

    /* Upload data */
    cudaMemcpy(d_TMP, TMP, size_3d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_KP, KP, size_3d, cudaMemcpyHostToDevice);

    /* Launch kernel */
    int result = gpu_thermal_diffusion_flux(
        d_TMP, d_KP,
        ctx->d_RDXN, ctx->d_RDYN, ctx->d_RDZN,
        d_KDTDX, d_KDTDY, d_KDTDZ,
        ctx->ibar, ctx->jbar, ctx->kbar,
        ctx->stream
    );

    if (result != 0) {
        fprintf(stderr, "ERROR: Thermal diffusion kernel failed\n");
        *ierr = GPU_ERROR;
        goto cleanup;
    }

    /* Synchronize and download results */
    cudaStreamSynchronize(ctx->stream);
    cudaMemcpy(KDTDX, d_KDTDX, size_3d, cudaMemcpyDeviceToHost);
    cudaMemcpy(KDTDY, d_KDTDY, size_3d, cudaMemcpyDeviceToHost);
    cudaMemcpy(KDTDZ, d_KDTDZ, size_3d, cudaMemcpyDeviceToHost);

    *ierr = GPU_SUCCESS;

cleanup:
    cudaFree(d_TMP);
    cudaFree(d_KP);
    cudaFree(d_KDTDX);
    cudaFree(d_KDTDY);
    cudaFree(d_KDTDZ);
}

/* ============================================================================
 * Velocity Flux Kernel Wrappers
 * ============================================================================ */

/* Compute velocity flux (FVX, FVY, FVZ) on GPU */
/* Compute vorticity and stress tensor only (no EDGE boundary handling needed) */
void gpu_compute_vorticity_stress_(int* mesh_id,
                                    double* UU, double* VV, double* WW, double* MU,
                                    double* OMX, double* OMY, double* OMZ,
                                    double* TXY, double* TXZ, double* TYZ,
                                    int* ierr) {
    int slot = find_context_slot(*mesh_id);
    if (slot < 0) {
        *ierr = GPU_ERROR;
        return;
    }

    GPUKernelContext* ctx = &kernel_contexts[slot];
    size_t n3d = calc_3d_size(ctx->ibar, ctx->jbar, ctx->kbar);
    size_t size_3d = n3d * sizeof(double);

    /* Allocate temporary device arrays */
    double *d_UU, *d_VV, *d_WW, *d_MU;
    double *d_OMX, *d_OMY, *d_OMZ, *d_TXY, *d_TXZ, *d_TYZ;

    cudaError_t err;
    err = cudaMalloc(&d_UU, size_3d);
    if (err != cudaSuccess) { *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_VV, size_3d);
    if (err != cudaSuccess) { cudaFree(d_UU); *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_WW, size_3d);
    if (err != cudaSuccess) { cudaFree(d_UU); cudaFree(d_VV); *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_MU, size_3d);
    if (err != cudaSuccess) { cudaFree(d_UU); cudaFree(d_VV); cudaFree(d_WW); *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_OMX, size_3d);
    if (err != cudaSuccess) { cudaFree(d_UU); cudaFree(d_VV); cudaFree(d_WW); cudaFree(d_MU); *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_OMY, size_3d);
    if (err != cudaSuccess) { cudaFree(d_UU); cudaFree(d_VV); cudaFree(d_WW); cudaFree(d_MU); cudaFree(d_OMX); *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_OMZ, size_3d);
    if (err != cudaSuccess) { cudaFree(d_UU); cudaFree(d_VV); cudaFree(d_WW); cudaFree(d_MU); cudaFree(d_OMX); cudaFree(d_OMY); *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_TXY, size_3d);
    if (err != cudaSuccess) { cudaFree(d_UU); cudaFree(d_VV); cudaFree(d_WW); cudaFree(d_MU); cudaFree(d_OMX); cudaFree(d_OMY); cudaFree(d_OMZ); *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_TXZ, size_3d);
    if (err != cudaSuccess) { cudaFree(d_UU); cudaFree(d_VV); cudaFree(d_WW); cudaFree(d_MU); cudaFree(d_OMX); cudaFree(d_OMY); cudaFree(d_OMZ); cudaFree(d_TXY); *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_TYZ, size_3d);
    if (err != cudaSuccess) { cudaFree(d_UU); cudaFree(d_VV); cudaFree(d_WW); cudaFree(d_MU); cudaFree(d_OMX); cudaFree(d_OMY); cudaFree(d_OMZ); cudaFree(d_TXY); cudaFree(d_TXZ); *ierr = GPU_ERROR; return; }

    /* Upload data */
    cudaMemcpy(d_UU, UU, size_3d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_VV, VV, size_3d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_WW, WW, size_3d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MU, MU, size_3d, cudaMemcpyHostToDevice);

    /* Compute vorticity and stress tensor */
    int result = gpu_vorticity_stress(
        d_UU, d_VV, d_WW, d_MU,
        ctx->d_RDXN, ctx->d_RDYN, ctx->d_RDZN,
        d_OMX, d_OMY, d_OMZ,
        d_TXY, d_TXZ, d_TYZ,
        ctx->ibar, ctx->jbar, ctx->kbar,
        ctx->stream
    );

    if (result != 0) {
        fprintf(stderr, "ERROR: Vorticity/stress kernel failed\n");
        *ierr = GPU_ERROR;
        goto cleanup;
    }

    /* Synchronize and download results */
    cudaStreamSynchronize(ctx->stream);
    cudaMemcpy(OMX, d_OMX, size_3d, cudaMemcpyDeviceToHost);
    cudaMemcpy(OMY, d_OMY, size_3d, cudaMemcpyDeviceToHost);
    cudaMemcpy(OMZ, d_OMZ, size_3d, cudaMemcpyDeviceToHost);
    cudaMemcpy(TXY, d_TXY, size_3d, cudaMemcpyDeviceToHost);
    cudaMemcpy(TXZ, d_TXZ, size_3d, cudaMemcpyDeviceToHost);
    cudaMemcpy(TYZ, d_TYZ, size_3d, cudaMemcpyDeviceToHost);

    *ierr = GPU_SUCCESS;

cleanup:
    cudaFree(d_UU);
    cudaFree(d_VV);
    cudaFree(d_WW);
    cudaFree(d_MU);
    cudaFree(d_OMX);
    cudaFree(d_OMY);
    cudaFree(d_OMZ);
    cudaFree(d_TXY);
    cudaFree(d_TXZ);
    cudaFree(d_TYZ);
}

/* Compute full velocity flux (vorticity/stress + FVX/FVY/FVZ)
 * OPTIMIZED: Uses persistent GPU memory for velocity/thermo fields.
 * If persistent data not uploaded, falls back to per-call upload.
 */
void gpu_compute_velocity_flux_(int* mesh_id,
                                 double* UU, double* VV, double* WW,
                                 double* RHOP, double* MU, double* DP,
                                 double* FVX, double* FVY, double* FVZ,
                                 int* ierr) {
    int slot = find_context_slot(*mesh_id);
    if (slot < 0) {
        *ierr = GPU_ERROR;
        return;
    }

    GPUKernelContext* ctx = &kernel_contexts[slot];
    size_t n3d = calc_3d_size(ctx->ibar, ctx->jbar, ctx->kbar);
    size_t size_3d = n3d * sizeof(double);

    /* Use persistent arrays if available, otherwise upload fresh */
    double *d_UU_use, *d_VV_use, *d_WW_use;
    double *d_RHOP_use, *d_MU_use, *d_DP_use;
    int need_velocity_upload = !ctx->velocity_uploaded;
    int need_thermo_upload = !ctx->thermo_uploaded;

    /* Point to persistent storage */
    d_UU_use = ctx->d_UU;
    d_VV_use = ctx->d_VV;
    d_WW_use = ctx->d_WW;
    d_RHOP_use = ctx->d_RHOP;
    d_MU_use = ctx->d_MU;
    d_DP_use = ctx->d_DP;

    /* Upload velocity if not cached */
    if (need_velocity_upload) {
        cudaMemcpyAsync(d_UU_use, UU, size_3d, cudaMemcpyHostToDevice, ctx->stream_upload);
        cudaMemcpyAsync(d_VV_use, VV, size_3d, cudaMemcpyHostToDevice, ctx->stream_upload);
        cudaMemcpyAsync(d_WW_use, WW, size_3d, cudaMemcpyHostToDevice, ctx->stream_upload);
    }

    /* Upload thermodynamic properties if not cached */
    if (need_thermo_upload) {
        cudaMemcpyAsync(d_RHOP_use, RHOP, size_3d, cudaMemcpyHostToDevice, ctx->stream_upload);
        cudaMemcpyAsync(d_MU_use, MU, size_3d, cudaMemcpyHostToDevice, ctx->stream_upload);
        cudaMemcpyAsync(d_DP_use, DP, size_3d, cudaMemcpyHostToDevice, ctx->stream_upload);
    }

    /* Wait for uploads to complete before computing */
    if (need_velocity_upload || need_thermo_upload) {
        cudaStreamSynchronize(ctx->stream_upload);
    }

    /* Allocate ONLY output arrays (these must be downloaded every call) */
    double *d_FVX, *d_FVY, *d_FVZ;
    cudaError_t err;
    err = cudaMalloc(&d_FVX, size_3d);
    if (err != cudaSuccess) { *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_FVY, size_3d);
    if (err != cudaSuccess) { cudaFree(d_FVX); *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_FVZ, size_3d);
    if (err != cudaSuccess) { cudaFree(d_FVX); cudaFree(d_FVY); *ierr = GPU_ERROR; return; }

    /* Step 1: Compute vorticity and stress tensor */
    int result = gpu_vorticity_stress(
        d_UU_use, d_VV_use, d_WW_use, d_MU_use,
        ctx->d_RDXN, ctx->d_RDYN, ctx->d_RDZN,
        ctx->d_OMX, ctx->d_OMY, ctx->d_OMZ,
        ctx->d_TXY, ctx->d_TXZ, ctx->d_TYZ,
        ctx->ibar, ctx->jbar, ctx->kbar,
        ctx->stream
    );

    if (result != 0) {
        fprintf(stderr, "ERROR: Vorticity/stress kernel failed\n");
        *ierr = GPU_ERROR;
        goto cleanup;
    }

    /* Step 2: Compute velocity flux */
    result = gpu_velocity_flux(
        d_UU_use, d_VV_use, d_WW_use, d_RHOP_use, d_MU_use, d_DP_use,
        ctx->d_OMX, ctx->d_OMY, ctx->d_OMZ,
        ctx->d_TXY, ctx->d_TXZ, ctx->d_TYZ,
        ctx->d_RDX, ctx->d_RDXN, ctx->d_RDY, ctx->d_RDYN, ctx->d_RDZ, ctx->d_RDZN,
        ctx->d_RHO_0,
        ctx->d_GX, ctx->d_GY, ctx->d_GZ,
        d_FVX, d_FVY, d_FVZ,
        ctx->ibar, ctx->jbar, ctx->kbar,
        ctx->stream
    );

    if (result != 0) {
        fprintf(stderr, "ERROR: Velocity flux kernel failed\n");
        *ierr = GPU_ERROR;
        goto cleanup;
    }

    /* Synchronize and download results */
    cudaStreamSynchronize(ctx->stream);
    cudaMemcpy(FVX, d_FVX, size_3d, cudaMemcpyDeviceToHost);
    cudaMemcpy(FVY, d_FVY, size_3d, cudaMemcpyDeviceToHost);
    cudaMemcpy(FVZ, d_FVZ, size_3d, cudaMemcpyDeviceToHost);

    *ierr = GPU_SUCCESS;

cleanup:
    /* Only free output arrays - inputs are persistent */
    cudaFree(d_FVX);
    cudaFree(d_FVY);
    cudaFree(d_FVZ);
}

/* ============================================================================
 * Advection Kernel Wrappers
 * ============================================================================ */

/* Compute scalar advection on GPU
 * OPTIMIZED: Uses persistent GPU memory for velocity fields.
 */
void gpu_compute_advection_(int* mesh_id,
                             double* RHO_SCALAR,
                             double* UU, double* VV, double* WW,
                             double* U_DOT_DEL,
                             int* flux_limiter_type,
                             int* ierr) {
    int slot = find_context_slot(*mesh_id);
    if (slot < 0) {
        *ierr = GPU_ERROR;
        return;
    }

    GPUKernelContext* ctx = &kernel_contexts[slot];
    size_t n3d = calc_3d_size(ctx->ibar, ctx->jbar, ctx->kbar);
    size_t size_3d = n3d * sizeof(double);

    /* Use persistent velocity arrays */
    int need_velocity_upload = !ctx->velocity_uploaded;
    if (need_velocity_upload) {
        cudaMemcpyAsync(ctx->d_UU, UU, size_3d, cudaMemcpyHostToDevice, ctx->stream_upload);
        cudaMemcpyAsync(ctx->d_VV, VV, size_3d, cudaMemcpyHostToDevice, ctx->stream_upload);
        cudaMemcpyAsync(ctx->d_WW, WW, size_3d, cudaMemcpyHostToDevice, ctx->stream_upload);
        cudaStreamSynchronize(ctx->stream_upload);
    }

    /* Allocate only scalar and output arrays */
    double *d_RHO_SCALAR, *d_U_DOT_DEL;
    cudaError_t err;
    err = cudaMalloc(&d_RHO_SCALAR, size_3d);
    if (err != cudaSuccess) { *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_U_DOT_DEL, size_3d);
    if (err != cudaSuccess) { cudaFree(d_RHO_SCALAR); *ierr = GPU_ERROR; return; }

    /* Upload scalar data */
    cudaMemcpy(d_RHO_SCALAR, RHO_SCALAR, size_3d, cudaMemcpyHostToDevice);

    /* Step 1: Compute advection fluxes */
    int result = gpu_advection_flux(
        d_RHO_SCALAR, ctx->d_UU, ctx->d_VV, ctx->d_WW,
        ctx->d_FX, ctx->d_FY, ctx->d_FZ,
        *flux_limiter_type,
        ctx->ibar, ctx->jbar, ctx->kbar,
        ctx->stream
    );

    if (result != 0) {
        fprintf(stderr, "ERROR: Advection flux kernel failed\n");
        *ierr = GPU_ERROR;
        goto cleanup;
    }

    /* Step 2: Compute flux divergence */
    result = gpu_flux_divergence(
        ctx->d_FX, ctx->d_FY, ctx->d_FZ,
        ctx->d_RDX, ctx->d_RDY, ctx->d_RDZ,
        d_U_DOT_DEL,
        ctx->ibar, ctx->jbar, ctx->kbar,
        ctx->stream
    );

    if (result != 0) {
        fprintf(stderr, "ERROR: Flux divergence kernel failed\n");
        *ierr = GPU_ERROR;
        goto cleanup;
    }

    /* Synchronize and download results */
    cudaStreamSynchronize(ctx->stream);
    cudaMemcpy(U_DOT_DEL, d_U_DOT_DEL, size_3d, cudaMemcpyDeviceToHost);

    *ierr = GPU_SUCCESS;

cleanup:
    cudaFree(d_RHO_SCALAR);
    cudaFree(d_U_DOT_DEL);
}

/* Compute FDS-style enthalpy advection on GPU
 * OPTIMIZED: Uses persistent GPU memory for velocity fields.
 * This matches ENTHALPY_ADVECTION_NEW formulation (FDS Tech Guide B.12-B.14)
 */
void gpu_compute_enthalpy_advection_(int* mesh_id,
                                      double* RHO_H_S_P,
                                      double* UU, double* VV, double* WW,
                                      double* U_DOT_DEL_RHO_H_S,
                                      int* flux_limiter_type,
                                      int* ierr) {
    int slot = find_context_slot(*mesh_id);
    if (slot < 0) {
        *ierr = GPU_ERROR;
        return;
    }

    GPUKernelContext* ctx = &kernel_contexts[slot];
    size_t n3d = calc_3d_size(ctx->ibar, ctx->jbar, ctx->kbar);
    size_t size_3d = n3d * sizeof(double);

    /* Use persistent velocity arrays */
    int need_velocity_upload = !ctx->velocity_uploaded;
    if (need_velocity_upload) {
        cudaMemcpyAsync(ctx->d_UU, UU, size_3d, cudaMemcpyHostToDevice, ctx->stream_upload);
        cudaMemcpyAsync(ctx->d_VV, VV, size_3d, cudaMemcpyHostToDevice, ctx->stream_upload);
        cudaMemcpyAsync(ctx->d_WW, WW, size_3d, cudaMemcpyHostToDevice, ctx->stream_upload);
        cudaStreamSynchronize(ctx->stream_upload);
    }

    /* Allocate only scalar and output arrays */
    double *d_RHO_H_S_P, *d_U_DOT_DEL;
    cudaError_t err;
    err = cudaMalloc(&d_RHO_H_S_P, size_3d);
    if (err != cudaSuccess) { *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_U_DOT_DEL, size_3d);
    if (err != cudaSuccess) { cudaFree(d_RHO_H_S_P); *ierr = GPU_ERROR; return; }

    /* Upload scalar data */
    cudaMemcpy(d_RHO_H_S_P, RHO_H_S_P, size_3d, cudaMemcpyHostToDevice);

    /* Step 1: Compute face values with flux limiter - use context flux arrays */
    int result = gpu_advection_flux(
        d_RHO_H_S_P, ctx->d_UU, ctx->d_VV, ctx->d_WW,
        ctx->d_FX, ctx->d_FY, ctx->d_FZ,
        *flux_limiter_type,
        ctx->ibar, ctx->jbar, ctx->kbar,
        ctx->stream
    );

    if (result != 0) {
        fprintf(stderr, "ERROR: Advection flux kernel failed\n");
        *ierr = GPU_ERROR;
        goto cleanup;
    }

    /* Step 2: Compute FDS-style advection divergence */
    result = gpu_fds_advection_divergence(
        ctx->d_FX, ctx->d_FY, ctx->d_FZ, d_RHO_H_S_P, ctx->d_UU, ctx->d_VV, ctx->d_WW,
        ctx->d_RDX, ctx->d_RDY, ctx->d_RDZ,
        d_U_DOT_DEL,
        ctx->ibar, ctx->jbar, ctx->kbar,
        ctx->stream
    );

    if (result != 0) {
        fprintf(stderr, "ERROR: FDS advection divergence kernel failed\n");
        *ierr = GPU_ERROR;
        goto cleanup;
    }

    /* Synchronize and download results */
    cudaStreamSynchronize(ctx->stream);
    cudaMemcpy(U_DOT_DEL_RHO_H_S, d_U_DOT_DEL, size_3d, cudaMemcpyDeviceToHost);

    *ierr = GPU_SUCCESS;

cleanup:
    cudaFree(d_RHO_H_S_P);
    cudaFree(d_U_DOT_DEL);
}

/* Compute species advection on GPU
 * OPTIMIZED: Uses persistent GPU memory for velocity fields.
 * Same algorithm as enthalpy advection but for species mass fraction
 */
void gpu_compute_species_advection_(int* mesh_id,
                                     double* RHO_Z_P,
                                     double* UU, double* VV, double* WW,
                                     double* U_DOT_DEL_RHO_Z,
                                     int* flux_limiter_type,
                                     int* ierr) {
    int slot = find_context_slot(*mesh_id);
    if (slot < 0) {
        *ierr = GPU_ERROR;
        return;
    }

    GPUKernelContext* ctx = &kernel_contexts[slot];
    size_t n3d = calc_3d_size(ctx->ibar, ctx->jbar, ctx->kbar);
    size_t size_3d = n3d * sizeof(double);

    /* Use persistent velocity arrays */
    int need_velocity_upload = !ctx->velocity_uploaded;
    if (need_velocity_upload) {
        cudaMemcpyAsync(ctx->d_UU, UU, size_3d, cudaMemcpyHostToDevice, ctx->stream_upload);
        cudaMemcpyAsync(ctx->d_VV, VV, size_3d, cudaMemcpyHostToDevice, ctx->stream_upload);
        cudaMemcpyAsync(ctx->d_WW, WW, size_3d, cudaMemcpyHostToDevice, ctx->stream_upload);
        cudaStreamSynchronize(ctx->stream_upload);
    }

    /* Allocate only scalar and output arrays */
    double *d_RHO_Z_P, *d_U_DOT_DEL;
    cudaError_t err;
    err = cudaMalloc(&d_RHO_Z_P, size_3d);
    if (err != cudaSuccess) { *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_U_DOT_DEL, size_3d);
    if (err != cudaSuccess) { cudaFree(d_RHO_Z_P); *ierr = GPU_ERROR; return; }

    /* Upload scalar data */
    cudaMemcpy(d_RHO_Z_P, RHO_Z_P, size_3d, cudaMemcpyHostToDevice);

    /* Step 1: Compute face values with flux limiter - use context flux arrays */
    int result = gpu_advection_flux(
        d_RHO_Z_P, ctx->d_UU, ctx->d_VV, ctx->d_WW,
        ctx->d_FX, ctx->d_FY, ctx->d_FZ,
        *flux_limiter_type,
        ctx->ibar, ctx->jbar, ctx->kbar,
        ctx->stream
    );

    if (result != 0) {
        fprintf(stderr, "ERROR: Species advection flux kernel failed\n");
        *ierr = GPU_ERROR;
        goto cleanup_species;
    }

    /* Step 2: Compute FDS-style advection divergence */
    result = gpu_fds_advection_divergence(
        ctx->d_FX, ctx->d_FY, ctx->d_FZ, d_RHO_Z_P, ctx->d_UU, ctx->d_VV, ctx->d_WW,
        ctx->d_RDX, ctx->d_RDY, ctx->d_RDZ,
        d_U_DOT_DEL,
        ctx->ibar, ctx->jbar, ctx->kbar,
        ctx->stream
    );

    if (result != 0) {
        fprintf(stderr, "ERROR: Species FDS advection divergence kernel failed\n");
        *ierr = GPU_ERROR;
        goto cleanup_species;
    }

    /* Synchronize and download results */
    cudaStreamSynchronize(ctx->stream);
    cudaMemcpy(U_DOT_DEL_RHO_Z, d_U_DOT_DEL, size_3d, cudaMemcpyDeviceToHost);

    *ierr = GPU_SUCCESS;

cleanup_species:
    cudaFree(d_RHO_Z_P);
    cudaFree(d_U_DOT_DEL);
}

/* Compute density update for a single species on GPU
 * OPTIMIZED: Uses persistent GPU memory for velocity fields.
 * ZZS = RHO * ZZ - DT * RHS
 * where RHS = -DEL_RHO_D_DEL_Z + div(FX*U, FY*V, FZ*W)
 * For Cartesian meshes only (R=RRN=1)
 */
void gpu_compute_density_update_(int* mesh_id,
                                  double* RHO,
                                  double* ZZ,
                                  double* DEL_RHO_D_DEL_Z,
                                  double* FX, double* FY, double* FZ,
                                  double* UU, double* VV, double* WW,
                                  double* DT,
                                  double* ZZS,
                                  int* ierr) {
    int slot = find_context_slot(*mesh_id);
    if (slot < 0) {
        *ierr = GPU_ERROR;
        return;
    }

    GPUKernelContext* ctx = &kernel_contexts[slot];
    size_t n3d = calc_3d_size(ctx->ibar, ctx->jbar, ctx->kbar);
    size_t size_3d = n3d * sizeof(double);

    /* Use persistent velocity arrays */
    int need_velocity_upload = !ctx->velocity_uploaded;
    if (need_velocity_upload) {
        cudaMemcpyAsync(ctx->d_UU, UU, size_3d, cudaMemcpyHostToDevice, ctx->stream_upload);
        cudaMemcpyAsync(ctx->d_VV, VV, size_3d, cudaMemcpyHostToDevice, ctx->stream_upload);
        cudaMemcpyAsync(ctx->d_WW, WW, size_3d, cudaMemcpyHostToDevice, ctx->stream_upload);
        cudaStreamSynchronize(ctx->stream_upload);
    }

    /* Allocate only species-specific arrays */
    double *d_RHO, *d_ZZ, *d_DEL_RHO_D_DEL_Z;
    double *d_FX, *d_FY, *d_FZ;
    double *d_ZZS;

    cudaError_t err;
    err = cudaMalloc(&d_RHO, size_3d);
    if (err != cudaSuccess) { *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_ZZ, size_3d);
    if (err != cudaSuccess) { cudaFree(d_RHO); *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_DEL_RHO_D_DEL_Z, size_3d);
    if (err != cudaSuccess) { cudaFree(d_RHO); cudaFree(d_ZZ); *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_FX, size_3d);
    if (err != cudaSuccess) { cudaFree(d_RHO); cudaFree(d_ZZ); cudaFree(d_DEL_RHO_D_DEL_Z); *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_FY, size_3d);
    if (err != cudaSuccess) { cudaFree(d_RHO); cudaFree(d_ZZ); cudaFree(d_DEL_RHO_D_DEL_Z); cudaFree(d_FX); *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_FZ, size_3d);
    if (err != cudaSuccess) { cudaFree(d_RHO); cudaFree(d_ZZ); cudaFree(d_DEL_RHO_D_DEL_Z); cudaFree(d_FX); cudaFree(d_FY); *ierr = GPU_ERROR; return; }
    err = cudaMalloc(&d_ZZS, size_3d);
    if (err != cudaSuccess) { cudaFree(d_RHO); cudaFree(d_ZZ); cudaFree(d_DEL_RHO_D_DEL_Z); cudaFree(d_FX); cudaFree(d_FY); cudaFree(d_FZ); *ierr = GPU_ERROR; return; }

    /* Upload species-specific data */
    cudaMemcpy(d_RHO, RHO, size_3d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ZZ, ZZ, size_3d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_DEL_RHO_D_DEL_Z, DEL_RHO_D_DEL_Z, size_3d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_FX, FX, size_3d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_FY, FY, size_3d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_FZ, FZ, size_3d, cudaMemcpyHostToDevice);

    /* Launch kernel - use persistent velocity arrays */
    int result = gpu_density_update(
        d_RHO, d_ZZ, d_DEL_RHO_D_DEL_Z,
        d_FX, d_FY, d_FZ,
        ctx->d_UU, ctx->d_VV, ctx->d_WW,
        ctx->d_RDX, ctx->d_RDY, ctx->d_RDZ,
        *DT, d_ZZS,
        ctx->ibar, ctx->jbar, ctx->kbar,
        ctx->stream
    );

    if (result != 0) {
        fprintf(stderr, "ERROR: Density update kernel failed\n");
        *ierr = GPU_ERROR;
        goto cleanup_density;
    }

    /* Synchronize and download results */
    cudaStreamSynchronize(ctx->stream);
    cudaMemcpy(ZZS, d_ZZS, size_3d, cudaMemcpyDeviceToHost);

    *ierr = GPU_SUCCESS;

cleanup_density:
    cudaFree(d_RHO);
    cudaFree(d_ZZ);
    cudaFree(d_DEL_RHO_D_DEL_Z);
    cudaFree(d_FX);
    cudaFree(d_FY);
    cudaFree(d_FZ);
    cudaFree(d_ZZS);
}

/* ============================================================================
 * Query Functions
 * ============================================================================ */

/* Check if GPU kernels are available */
void gpu_kernel_available_(int* available) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    *available = (err == cudaSuccess && device_count > 0) ? 1 : 0;
}

/* Get mesh cell count */
void gpu_get_mesh_cells_(int* mesh_id, int* n_cells) {
    int slot = find_context_slot(*mesh_id);
    if (slot < 0) {
        *n_cells = 0;
        return;
    }

    GPUKernelContext* ctx = &kernel_contexts[slot];
    *n_cells = ctx->ibar * ctx->jbar * ctx->kbar;
}

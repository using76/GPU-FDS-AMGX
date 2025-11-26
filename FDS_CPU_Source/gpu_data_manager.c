/**
 * @file gpu_data_manager.c
 * @brief GPU memory manager for FDS field data (Advection/Diffusion modules)
 *
 * This module manages GPU-resident field data for advection and diffusion
 * computations. It follows the same zone-based pattern as amgx_c_wrapper.c.
 *
 * Features:
 * - Zone-based memory management (one per FDS mesh)
 * - GPU-resident field arrays (U, V, W, RHO, TMP, ZZ, etc.)
 * - Pinned memory for fast CPU-GPU transfers
 * - CUDA streams for async operations
 *
 * Build: nvcc -c gpu_data_manager.c -I${CUDA_HOME}/include
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

/* Maximum number of mesh zones */
#define MAX_GPU_MESHES 256

/* Error codes */
#define GPU_SUCCESS           0
#define GPU_ERR_NOT_INIT     -1
#define GPU_ERR_INVALID_MESH -2
#define GPU_ERR_ALLOC_FAIL   -3
#define GPU_ERR_TRANSFER     -4
#define GPU_ERR_INVALID_SIZE -5

/**
 * @brief GPU Mesh Data Structure
 *
 * Contains all GPU-resident field arrays for a single FDS mesh.
 * Mirrors the structure used in FDS Fortran code.
 */
typedef struct {
    int mesh_id;           /* FDS mesh number (1-based) */
    int initialized;       /* Flag: structure initialized */

    /* Mesh dimensions */
    int ibar, jbar, kbar;  /* Cell counts in each direction */
    int n_species;         /* Number of species tracked */

    /* Total array sizes */
    size_t n3d;            /* (IBAR+2) * (JBAR+2) * (KBAR+2) for 3D arrays */
    size_t n3d_face_x;     /* (IBAR+1) * (JBAR+2) * (KBAR+2) for X-face arrays */
    size_t n3d_face_y;     /* (IBAR+2) * (JBAR+1) * (KBAR+2) for Y-face arrays */
    size_t n3d_face_z;     /* (IBAR+2) * (JBAR+2) * (KBAR+1) for Z-face arrays */

    /* =====================================================
     * GPU-RESIDENT FIELD ARRAYS (Device Memory)
     * These stay on GPU permanently to minimize transfers
     * ===================================================== */

    /* Velocity fields (face-centered) */
    double *d_U;           /* X-velocity at X-faces */
    double *d_V;           /* Y-velocity at Y-faces */
    double *d_W;           /* Z-velocity at Z-faces */
    double *d_US;          /* Estimated X-velocity */
    double *d_VS;          /* Estimated Y-velocity */
    double *d_WS;          /* Estimated Z-velocity */

    /* Scalar fields (cell-centered) */
    double *d_RHO;         /* Density */
    double *d_RHOS;        /* Estimated density */
    double *d_TMP;         /* Temperature */
    double *d_MU;          /* Dynamic viscosity */
    double *d_D;           /* Divergence */
    double *d_DS;          /* Estimated divergence */

    /* Velocity flux arrays (cell-centered) */
    double *d_FVX;         /* X-momentum flux */
    double *d_FVY;         /* Y-momentum flux */
    double *d_FVZ;         /* Z-momentum flux */

    /* Diffusion work arrays (face-centered) */
    double *d_KDTDX;       /* Thermal flux X: k * dT/dx */
    double *d_KDTDY;       /* Thermal flux Y: k * dT/dy */
    double *d_KDTDZ;       /* Thermal flux Z: k * dT/dz */

    /* Species arrays (4D: n3d * n_species) */
    double *d_ZZ;          /* Species mass fractions */
    double *d_ZZS;         /* Estimated species mass fractions */
    double *d_RHO_D_DZDX;  /* Species diffusion flux X */
    double *d_RHO_D_DZDY;  /* Species diffusion flux Y */
    double *d_RHO_D_DZDZ;  /* Species diffusion flux Z */
    double *d_DEL_RHO_D_DEL_Z; /* Divergence of species diffusion */

    /* Thermal conductivity (cell-centered) */
    double *d_KP;          /* Thermal conductivity k(T) */

    /* Species diffusivity (cell-centered) */
    double *d_RHO_D;       /* rho * D diffusivity */

    /* Grid geometry (1D arrays) */
    double *d_DX;          /* Cell widths in X */
    double *d_DY;          /* Cell widths in Y */
    double *d_DZ;          /* Cell widths in Z */
    double *d_RDX;         /* 1/DX (cell) */
    double *d_RDY;         /* 1/DY (cell) */
    double *d_RDZ;         /* 1/DZ (cell) */
    double *d_RDXN;        /* 1/DX (face) */
    double *d_RDYN;        /* 1/DY (face) */
    double *d_RDZN;        /* 1/DZ (face) */

    /* Background density profile (1D) */
    double *d_RHO_0;       /* Reference density profile */

    /* =====================================================
     * PINNED MEMORY BUFFERS (Host Memory, Page-Locked)
     * For fast CPU-GPU DMA transfers
     * ===================================================== */
    double *pinned_buffer;     /* General-purpose pinned buffer */
    size_t pinned_buffer_size; /* Size of pinned buffer in bytes */

    /* =====================================================
     * CUDA RESOURCES
     * ===================================================== */
    cudaStream_t compute_stream;   /* Stream for compute operations */
    cudaStream_t transfer_stream;  /* Stream for data transfers */

    /* Status flags */
    int gpu_data_valid;    /* Flag: GPU data is current */
    int fields_allocated;  /* Flag: field arrays allocated */
    int geometry_uploaded; /* Flag: grid geometry uploaded */

} GPUMeshData;

/* Global array of mesh data structures */
static GPUMeshData gpu_meshes[MAX_GPU_MESHES];
static int gpu_manager_initialized = 0;

/* =========================================================================
 * INTERNAL HELPER FUNCTIONS
 * ========================================================================= */

/**
 * @brief Find mesh slot by mesh_id
 * @param mesh_id FDS mesh number (1-based)
 * @param allocate If true, allocate new slot if not found
 * @return Slot index (0 to MAX_GPU_MESHES-1), or -1 if not found/full
 */
static int find_mesh_slot(int mesh_id, int allocate)
{
    int i;
    int free_slot = -1;

    /* Look for existing mesh with this ID */
    for (i = 0; i < MAX_GPU_MESHES; i++) {
        if (gpu_meshes[i].initialized && gpu_meshes[i].mesh_id == mesh_id) {
            return i;
        }
        if (!gpu_meshes[i].initialized && free_slot < 0) {
            free_slot = i;
        }
    }

    /* Not found - return free slot if allocating */
    if (allocate) {
        return free_slot;
    }
    return -1;
}

/**
 * @brief Allocate GPU memory with error checking
 */
static int gpu_alloc(double **ptr, size_t count, const char *name)
{
    cudaError_t err = cudaMalloc((void**)ptr, count * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "[GPU Data] Failed to allocate %s (%zu elements): %s\n",
                name, count, cudaGetErrorString(err));
        *ptr = NULL;
        return GPU_ERR_ALLOC_FAIL;
    }
    /* Initialize to zero */
    cudaMemset(*ptr, 0, count * sizeof(double));
    return GPU_SUCCESS;
}

/**
 * @brief Free GPU memory safely
 */
static void gpu_free(double **ptr)
{
    if (*ptr != NULL) {
        cudaFree(*ptr);
        *ptr = NULL;
    }
}

/* =========================================================================
 * PUBLIC API FUNCTIONS (Called from Fortran via ISO_C_BINDING)
 * ========================================================================= */

/**
 * @brief Initialize GPU data manager
 * @param[out] ierr Error code (0 = success)
 */
void gpu_data_manager_init_(int *ierr)
{
    int i;

    if (gpu_manager_initialized) {
        *ierr = GPU_SUCCESS;
        return;
    }

    /* Initialize CUDA */
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GPU Data] Failed to set CUDA device: %s\n",
                cudaGetErrorString(err));
        *ierr = GPU_ERR_NOT_INIT;
        return;
    }

    /* Initialize all mesh structures */
    for (i = 0; i < MAX_GPU_MESHES; i++) {
        memset(&gpu_meshes[i], 0, sizeof(GPUMeshData));
        gpu_meshes[i].initialized = 0;
        gpu_meshes[i].mesh_id = -1;
    }

    gpu_manager_initialized = 1;
    *ierr = GPU_SUCCESS;

    printf("[GPU Data] Manager initialized successfully\n");
}

/**
 * @brief Finalize GPU data manager and free all resources
 * @param[out] ierr Error code
 */
void gpu_data_manager_finalize_(int *ierr)
{
    int i;
    int dummy_err;

    if (!gpu_manager_initialized) {
        *ierr = GPU_SUCCESS;
        return;
    }

    /* Free all mesh resources */
    for (i = 0; i < MAX_GPU_MESHES; i++) {
        if (gpu_meshes[i].initialized) {
            int mesh_id = gpu_meshes[i].mesh_id;
            gpu_deallocate_mesh_(&mesh_id, &dummy_err);
        }
    }

    gpu_manager_initialized = 0;
    *ierr = GPU_SUCCESS;

    printf("[GPU Data] Manager finalized\n");
}

/**
 * @brief Allocate GPU memory for a mesh
 * @param[in]  mesh_id   FDS mesh number (1-based)
 * @param[in]  ibar      Number of cells in X
 * @param[in]  jbar      Number of cells in Y
 * @param[in]  kbar      Number of cells in Z
 * @param[in]  n_species Number of tracked species
 * @param[out] ierr      Error code
 */
void gpu_allocate_mesh_(int *mesh_id, int *ibar, int *jbar, int *kbar,
                        int *n_species, int *ierr)
{
    int mid;
    GPUMeshData *mesh;
    cudaError_t cuda_err;

    if (!gpu_manager_initialized) {
        fprintf(stderr, "[GPU Data] Manager not initialized\n");
        *ierr = GPU_ERR_NOT_INIT;
        return;
    }

    /* Find or allocate slot */
    mid = find_mesh_slot(*mesh_id, 1);
    if (mid < 0) {
        fprintf(stderr, "[GPU Data] No available slots for mesh %d\n", *mesh_id);
        *ierr = GPU_ERR_INVALID_MESH;
        return;
    }

    mesh = &gpu_meshes[mid];

    /* Clean up if already allocated */
    if (mesh->initialized) {
        gpu_deallocate_mesh_(mesh_id, ierr);
    }

    /* Store mesh info */
    mesh->mesh_id = *mesh_id;
    mesh->ibar = *ibar;
    mesh->jbar = *jbar;
    mesh->kbar = *kbar;
    mesh->n_species = *n_species;

    /* Calculate array sizes */
    /* FDS uses ghost cells: arrays are (0:IBAR+1, 0:JBAR+1, 0:KBAR+1) */
    int ibp2 = *ibar + 2;  /* IBAR+2 for ghost cells */
    int jbp2 = *jbar + 2;
    int kbp2 = *kbar + 2;

    mesh->n3d = (size_t)ibp2 * jbp2 * kbp2;
    mesh->n3d_face_x = (size_t)(*ibar + 1) * jbp2 * kbp2;
    mesh->n3d_face_y = (size_t)ibp2 * (*jbar + 1) * kbp2;
    mesh->n3d_face_z = (size_t)ibp2 * jbp2 * (*kbar + 1);

    printf("[GPU Data] Mesh %d: Allocating GPU memory (IBAR=%d, JBAR=%d, KBAR=%d, n3d=%zu)\n",
           *mesh_id, *ibar, *jbar, *kbar, mesh->n3d);

    /* Allocate velocity fields (face-centered) */
    if (gpu_alloc(&mesh->d_U, mesh->n3d_face_x, "U") != GPU_SUCCESS ||
        gpu_alloc(&mesh->d_V, mesh->n3d_face_y, "V") != GPU_SUCCESS ||
        gpu_alloc(&mesh->d_W, mesh->n3d_face_z, "W") != GPU_SUCCESS ||
        gpu_alloc(&mesh->d_US, mesh->n3d_face_x, "US") != GPU_SUCCESS ||
        gpu_alloc(&mesh->d_VS, mesh->n3d_face_y, "VS") != GPU_SUCCESS ||
        gpu_alloc(&mesh->d_WS, mesh->n3d_face_z, "WS") != GPU_SUCCESS) {
        *ierr = GPU_ERR_ALLOC_FAIL;
        return;
    }

    /* Allocate scalar fields (cell-centered) */
    if (gpu_alloc(&mesh->d_RHO, mesh->n3d, "RHO") != GPU_SUCCESS ||
        gpu_alloc(&mesh->d_RHOS, mesh->n3d, "RHOS") != GPU_SUCCESS ||
        gpu_alloc(&mesh->d_TMP, mesh->n3d, "TMP") != GPU_SUCCESS ||
        gpu_alloc(&mesh->d_MU, mesh->n3d, "MU") != GPU_SUCCESS ||
        gpu_alloc(&mesh->d_D, mesh->n3d, "D") != GPU_SUCCESS ||
        gpu_alloc(&mesh->d_DS, mesh->n3d, "DS") != GPU_SUCCESS) {
        *ierr = GPU_ERR_ALLOC_FAIL;
        return;
    }

    /* Allocate velocity flux arrays */
    if (gpu_alloc(&mesh->d_FVX, mesh->n3d_face_x, "FVX") != GPU_SUCCESS ||
        gpu_alloc(&mesh->d_FVY, mesh->n3d_face_y, "FVY") != GPU_SUCCESS ||
        gpu_alloc(&mesh->d_FVZ, mesh->n3d_face_z, "FVZ") != GPU_SUCCESS) {
        *ierr = GPU_ERR_ALLOC_FAIL;
        return;
    }

    /* Allocate thermal diffusion work arrays */
    if (gpu_alloc(&mesh->d_KDTDX, mesh->n3d_face_x, "KDTDX") != GPU_SUCCESS ||
        gpu_alloc(&mesh->d_KDTDY, mesh->n3d_face_y, "KDTDY") != GPU_SUCCESS ||
        gpu_alloc(&mesh->d_KDTDZ, mesh->n3d_face_z, "KDTDZ") != GPU_SUCCESS ||
        gpu_alloc(&mesh->d_KP, mesh->n3d, "KP") != GPU_SUCCESS) {
        *ierr = GPU_ERR_ALLOC_FAIL;
        return;
    }

    /* Allocate species arrays (if species tracked) */
    if (*n_species > 0) {
        size_t n4d = mesh->n3d * (*n_species);

        if (gpu_alloc(&mesh->d_ZZ, n4d, "ZZ") != GPU_SUCCESS ||
            gpu_alloc(&mesh->d_ZZS, n4d, "ZZS") != GPU_SUCCESS ||
            gpu_alloc(&mesh->d_RHO_D_DZDX, n4d, "RHO_D_DZDX") != GPU_SUCCESS ||
            gpu_alloc(&mesh->d_RHO_D_DZDY, n4d, "RHO_D_DZDY") != GPU_SUCCESS ||
            gpu_alloc(&mesh->d_RHO_D_DZDZ, n4d, "RHO_D_DZDZ") != GPU_SUCCESS ||
            gpu_alloc(&mesh->d_DEL_RHO_D_DEL_Z, n4d, "DEL_RHO_D_DEL_Z") != GPU_SUCCESS ||
            gpu_alloc(&mesh->d_RHO_D, mesh->n3d, "RHO_D") != GPU_SUCCESS) {
            *ierr = GPU_ERR_ALLOC_FAIL;
            return;
        }
    }

    /* Allocate grid geometry arrays (1D) */
    if (gpu_alloc(&mesh->d_DX, ibp2, "DX") != GPU_SUCCESS ||
        gpu_alloc(&mesh->d_DY, jbp2, "DY") != GPU_SUCCESS ||
        gpu_alloc(&mesh->d_DZ, kbp2, "DZ") != GPU_SUCCESS ||
        gpu_alloc(&mesh->d_RDX, ibp2, "RDX") != GPU_SUCCESS ||
        gpu_alloc(&mesh->d_RDY, jbp2, "RDY") != GPU_SUCCESS ||
        gpu_alloc(&mesh->d_RDZ, kbp2, "RDZ") != GPU_SUCCESS ||
        gpu_alloc(&mesh->d_RDXN, *ibar + 1, "RDXN") != GPU_SUCCESS ||
        gpu_alloc(&mesh->d_RDYN, *jbar + 1, "RDYN") != GPU_SUCCESS ||
        gpu_alloc(&mesh->d_RDZN, *kbar + 1, "RDZN") != GPU_SUCCESS ||
        gpu_alloc(&mesh->d_RHO_0, kbp2, "RHO_0") != GPU_SUCCESS) {
        *ierr = GPU_ERR_ALLOC_FAIL;
        return;
    }

    /* Allocate pinned memory buffer for fast transfers */
    /* Size: enough for largest 3D array */
    mesh->pinned_buffer_size = mesh->n3d * sizeof(double);
    cuda_err = cudaMallocHost((void**)&mesh->pinned_buffer, mesh->pinned_buffer_size);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "[GPU Data] Warning: Pinned buffer allocation failed\n");
        mesh->pinned_buffer = NULL;
        mesh->pinned_buffer_size = 0;
    }

    /* Create CUDA streams */
    cuda_err = cudaStreamCreate(&mesh->compute_stream);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "[GPU Data] Warning: Compute stream creation failed\n");
        mesh->compute_stream = NULL;
    }

    cuda_err = cudaStreamCreate(&mesh->transfer_stream);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "[GPU Data] Warning: Transfer stream creation failed\n");
        mesh->transfer_stream = NULL;
    }

    /* Mark as initialized */
    mesh->initialized = 1;
    mesh->fields_allocated = 1;
    mesh->gpu_data_valid = 0;
    mesh->geometry_uploaded = 0;

    /* Calculate memory usage */
    size_t total_mem = 0;
    total_mem += (mesh->n3d_face_x * 6) * sizeof(double);  /* U, US, FVX, KDTDX, etc. */
    total_mem += (mesh->n3d_face_y * 4) * sizeof(double);
    total_mem += (mesh->n3d_face_z * 4) * sizeof(double);
    total_mem += (mesh->n3d * 10) * sizeof(double);        /* RHO, TMP, MU, etc. */
    if (*n_species > 0) {
        total_mem += (mesh->n3d * (*n_species) * 6) * sizeof(double);
    }

    printf("[GPU Data] Mesh %d: Allocated %.1f MB GPU memory\n",
           *mesh_id, (double)total_mem / (1024.0 * 1024.0));

    *ierr = GPU_SUCCESS;
}

/**
 * @brief Deallocate GPU memory for a mesh
 * @param[in]  mesh_id FDS mesh number
 * @param[out] ierr    Error code
 */
void gpu_deallocate_mesh_(int *mesh_id, int *ierr)
{
    int mid = find_mesh_slot(*mesh_id, 0);
    GPUMeshData *mesh;

    if (mid < 0) {
        *ierr = GPU_SUCCESS;  /* Not found, nothing to deallocate */
        return;
    }

    mesh = &gpu_meshes[mid];

    /* Free velocity arrays */
    gpu_free(&mesh->d_U);
    gpu_free(&mesh->d_V);
    gpu_free(&mesh->d_W);
    gpu_free(&mesh->d_US);
    gpu_free(&mesh->d_VS);
    gpu_free(&mesh->d_WS);

    /* Free scalar arrays */
    gpu_free(&mesh->d_RHO);
    gpu_free(&mesh->d_RHOS);
    gpu_free(&mesh->d_TMP);
    gpu_free(&mesh->d_MU);
    gpu_free(&mesh->d_D);
    gpu_free(&mesh->d_DS);

    /* Free flux arrays */
    gpu_free(&mesh->d_FVX);
    gpu_free(&mesh->d_FVY);
    gpu_free(&mesh->d_FVZ);

    /* Free diffusion work arrays */
    gpu_free(&mesh->d_KDTDX);
    gpu_free(&mesh->d_KDTDY);
    gpu_free(&mesh->d_KDTDZ);
    gpu_free(&mesh->d_KP);

    /* Free species arrays */
    gpu_free(&mesh->d_ZZ);
    gpu_free(&mesh->d_ZZS);
    gpu_free(&mesh->d_RHO_D_DZDX);
    gpu_free(&mesh->d_RHO_D_DZDY);
    gpu_free(&mesh->d_RHO_D_DZDZ);
    gpu_free(&mesh->d_DEL_RHO_D_DEL_Z);
    gpu_free(&mesh->d_RHO_D);

    /* Free geometry arrays */
    gpu_free(&mesh->d_DX);
    gpu_free(&mesh->d_DY);
    gpu_free(&mesh->d_DZ);
    gpu_free(&mesh->d_RDX);
    gpu_free(&mesh->d_RDY);
    gpu_free(&mesh->d_RDZ);
    gpu_free(&mesh->d_RDXN);
    gpu_free(&mesh->d_RDYN);
    gpu_free(&mesh->d_RDZN);
    gpu_free(&mesh->d_RHO_0);

    /* Free pinned memory */
    if (mesh->pinned_buffer) {
        cudaFreeHost(mesh->pinned_buffer);
        mesh->pinned_buffer = NULL;
    }

    /* Destroy streams */
    if (mesh->compute_stream) {
        cudaStreamDestroy(mesh->compute_stream);
        mesh->compute_stream = NULL;
    }
    if (mesh->transfer_stream) {
        cudaStreamDestroy(mesh->transfer_stream);
        mesh->transfer_stream = NULL;
    }

    /* Reset structure */
    mesh->initialized = 0;
    mesh->mesh_id = -1;
    mesh->fields_allocated = 0;
    mesh->gpu_data_valid = 0;

    printf("[GPU Data] Mesh %d: Deallocated\n", *mesh_id);

    *ierr = GPU_SUCCESS;
}

/**
 * @brief Upload grid geometry to GPU (called once during setup)
 * @param[in]  mesh_id FDS mesh number
 * @param[in]  dx, dy, dz     Cell widths
 * @param[in]  rdx, rdy, rdz  1/cell widths
 * @param[in]  rdxn, rdyn, rdzn  1/face widths
 * @param[in]  rho_0  Background density profile
 * @param[out] ierr   Error code
 */
void gpu_upload_geometry_(int *mesh_id,
                          double *dx, double *dy, double *dz,
                          double *rdx, double *rdy, double *rdz,
                          double *rdxn, double *rdyn, double *rdzn,
                          double *rho_0, int *ierr)
{
    int mid = find_mesh_slot(*mesh_id, 0);
    GPUMeshData *mesh;

    if (mid < 0 || !gpu_meshes[mid].initialized) {
        fprintf(stderr, "[GPU Data] Mesh %d not initialized\n", *mesh_id);
        *ierr = GPU_ERR_INVALID_MESH;
        return;
    }

    mesh = &gpu_meshes[mid];

    int ibp2 = mesh->ibar + 2;
    int jbp2 = mesh->jbar + 2;
    int kbp2 = mesh->kbar + 2;

    /* Upload geometry arrays */
    cudaMemcpy(mesh->d_DX, dx, ibp2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(mesh->d_DY, dy, jbp2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(mesh->d_DZ, dz, kbp2 * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(mesh->d_RDX, rdx, ibp2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(mesh->d_RDY, rdy, jbp2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(mesh->d_RDZ, rdz, kbp2 * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(mesh->d_RDXN, rdxn, (mesh->ibar + 1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(mesh->d_RDYN, rdyn, (mesh->jbar + 1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(mesh->d_RDZN, rdzn, (mesh->kbar + 1) * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(mesh->d_RHO_0, rho_0, kbp2 * sizeof(double), cudaMemcpyHostToDevice);

    mesh->geometry_uploaded = 1;
    *ierr = GPU_SUCCESS;

    printf("[GPU Data] Mesh %d: Geometry uploaded\n", *mesh_id);
}

/**
 * @brief Upload velocity fields to GPU
 * @param[in]  mesh_id FDS mesh number
 * @param[in]  u, v, w Velocity components from Fortran
 * @param[out] ierr    Error code
 */
void gpu_upload_velocity_(int *mesh_id, double *u, double *v, double *w, int *ierr)
{
    int mid = find_mesh_slot(*mesh_id, 0);
    GPUMeshData *mesh;

    if (mid < 0 || !gpu_meshes[mid].initialized) {
        *ierr = GPU_ERR_INVALID_MESH;
        return;
    }

    mesh = &gpu_meshes[mid];

    /* Use async copy with transfer stream if available */
    if (mesh->transfer_stream) {
        cudaMemcpyAsync(mesh->d_U, u, mesh->n3d_face_x * sizeof(double),
                        cudaMemcpyHostToDevice, mesh->transfer_stream);
        cudaMemcpyAsync(mesh->d_V, v, mesh->n3d_face_y * sizeof(double),
                        cudaMemcpyHostToDevice, mesh->transfer_stream);
        cudaMemcpyAsync(mesh->d_W, w, mesh->n3d_face_z * sizeof(double),
                        cudaMemcpyHostToDevice, mesh->transfer_stream);
        cudaStreamSynchronize(mesh->transfer_stream);
    } else {
        cudaMemcpy(mesh->d_U, u, mesh->n3d_face_x * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(mesh->d_V, v, mesh->n3d_face_y * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(mesh->d_W, w, mesh->n3d_face_z * sizeof(double), cudaMemcpyHostToDevice);
    }

    *ierr = GPU_SUCCESS;
}

/**
 * @brief Upload scalar fields (RHO, TMP, MU) to GPU
 * @param[in]  mesh_id FDS mesh number
 * @param[in]  rho, tmp, mu Scalar fields from Fortran
 * @param[out] ierr    Error code
 */
void gpu_upload_scalars_(int *mesh_id, double *rho, double *tmp, double *mu, int *ierr)
{
    int mid = find_mesh_slot(*mesh_id, 0);
    GPUMeshData *mesh;

    if (mid < 0 || !gpu_meshes[mid].initialized) {
        *ierr = GPU_ERR_INVALID_MESH;
        return;
    }

    mesh = &gpu_meshes[mid];

    if (mesh->transfer_stream) {
        cudaMemcpyAsync(mesh->d_RHO, rho, mesh->n3d * sizeof(double),
                        cudaMemcpyHostToDevice, mesh->transfer_stream);
        cudaMemcpyAsync(mesh->d_TMP, tmp, mesh->n3d * sizeof(double),
                        cudaMemcpyHostToDevice, mesh->transfer_stream);
        cudaMemcpyAsync(mesh->d_MU, mu, mesh->n3d * sizeof(double),
                        cudaMemcpyHostToDevice, mesh->transfer_stream);
        cudaStreamSynchronize(mesh->transfer_stream);
    } else {
        cudaMemcpy(mesh->d_RHO, rho, mesh->n3d * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(mesh->d_TMP, tmp, mesh->n3d * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(mesh->d_MU, mu, mesh->n3d * sizeof(double), cudaMemcpyHostToDevice);
    }

    *ierr = GPU_SUCCESS;
}

/**
 * @brief Upload species mass fractions to GPU
 * @param[in]  mesh_id   FDS mesh number
 * @param[in]  zz        Species mass fraction array (n3d * n_species)
 * @param[in]  n_species Number of species
 * @param[out] ierr      Error code
 */
void gpu_upload_species_(int *mesh_id, double *zz, int *n_species, int *ierr)
{
    int mid = find_mesh_slot(*mesh_id, 0);
    GPUMeshData *mesh;

    if (mid < 0 || !gpu_meshes[mid].initialized) {
        *ierr = GPU_ERR_INVALID_MESH;
        return;
    }

    mesh = &gpu_meshes[mid];
    size_t n4d = mesh->n3d * (*n_species);

    if (mesh->transfer_stream) {
        cudaMemcpyAsync(mesh->d_ZZ, zz, n4d * sizeof(double),
                        cudaMemcpyHostToDevice, mesh->transfer_stream);
        cudaStreamSynchronize(mesh->transfer_stream);
    } else {
        cudaMemcpy(mesh->d_ZZ, zz, n4d * sizeof(double), cudaMemcpyHostToDevice);
    }

    *ierr = GPU_SUCCESS;
}

/**
 * @brief Download velocity flux results from GPU
 * @param[in]  mesh_id FDS mesh number
 * @param[out] fvx, fvy, fvz Velocity flux arrays
 * @param[out] ierr    Error code
 */
void gpu_download_velocity_flux_(int *mesh_id, double *fvx, double *fvy, double *fvz, int *ierr)
{
    int mid = find_mesh_slot(*mesh_id, 0);
    GPUMeshData *mesh;

    if (mid < 0 || !gpu_meshes[mid].initialized) {
        *ierr = GPU_ERR_INVALID_MESH;
        return;
    }

    mesh = &gpu_meshes[mid];

    if (mesh->transfer_stream) {
        cudaMemcpyAsync(fvx, mesh->d_FVX, mesh->n3d_face_x * sizeof(double),
                        cudaMemcpyDeviceToHost, mesh->transfer_stream);
        cudaMemcpyAsync(fvy, mesh->d_FVY, mesh->n3d_face_y * sizeof(double),
                        cudaMemcpyDeviceToHost, mesh->transfer_stream);
        cudaMemcpyAsync(fvz, mesh->d_FVZ, mesh->n3d_face_z * sizeof(double),
                        cudaMemcpyDeviceToHost, mesh->transfer_stream);
        cudaStreamSynchronize(mesh->transfer_stream);
    } else {
        cudaMemcpy(fvx, mesh->d_FVX, mesh->n3d_face_x * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(fvy, mesh->d_FVY, mesh->n3d_face_y * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(fvz, mesh->d_FVZ, mesh->n3d_face_z * sizeof(double), cudaMemcpyDeviceToHost);
    }

    *ierr = GPU_SUCCESS;
}

/**
 * @brief Download diffusion flux results from GPU
 * @param[in]  mesh_id FDS mesh number
 * @param[out] kdtdx, kdtdy, kdtdz Thermal diffusion flux arrays
 * @param[out] ierr    Error code
 */
void gpu_download_diffusion_flux_(int *mesh_id, double *kdtdx, double *kdtdy, double *kdtdz, int *ierr)
{
    int mid = find_mesh_slot(*mesh_id, 0);
    GPUMeshData *mesh;

    if (mid < 0 || !gpu_meshes[mid].initialized) {
        *ierr = GPU_ERR_INVALID_MESH;
        return;
    }

    mesh = &gpu_meshes[mid];

    cudaMemcpy(kdtdx, mesh->d_KDTDX, mesh->n3d_face_x * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(kdtdy, mesh->d_KDTDY, mesh->n3d_face_y * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(kdtdz, mesh->d_KDTDZ, mesh->n3d_face_z * sizeof(double), cudaMemcpyDeviceToHost);

    *ierr = GPU_SUCCESS;
}

/**
 * @brief Check if mesh GPU data is allocated and valid
 * @param[in]  mesh_id   FDS mesh number
 * @param[out] is_valid  1 if valid, 0 otherwise
 * @param[out] ierr      Error code
 */
void gpu_is_mesh_ready_(int *mesh_id, int *is_valid, int *ierr)
{
    int mid = find_mesh_slot(*mesh_id, 0);

    if (mid < 0 || !gpu_meshes[mid].initialized || !gpu_meshes[mid].fields_allocated) {
        *is_valid = 0;
    } else {
        *is_valid = 1;
    }

    *ierr = GPU_SUCCESS;
}

/**
 * @brief Synchronize GPU compute stream
 * @param[in]  mesh_id FDS mesh number
 * @param[out] ierr    Error code
 */
void gpu_sync_compute_(int *mesh_id, int *ierr)
{
    int mid = find_mesh_slot(*mesh_id, 0);

    if (mid < 0 || !gpu_meshes[mid].initialized) {
        *ierr = GPU_ERR_INVALID_MESH;
        return;
    }

    if (gpu_meshes[mid].compute_stream) {
        cudaStreamSynchronize(gpu_meshes[mid].compute_stream);
    } else {
        cudaDeviceSynchronize();
    }

    *ierr = GPU_SUCCESS;
}

/**
 * @brief Get GPU memory usage statistics
 * @param[out] used_mb  Used GPU memory in MB
 * @param[out] total_mb Total GPU memory in MB
 * @param[out] ierr     Error code
 */
void gpu_get_memory_usage_(double *used_mb, double *total_mb, int *ierr)
{
    size_t free_bytes, total_bytes;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);

    if (err != cudaSuccess) {
        *used_mb = -1.0;
        *total_mb = -1.0;
        *ierr = GPU_ERR_NOT_INIT;
        return;
    }

    *total_mb = (double)total_bytes / (1024.0 * 1024.0);
    *used_mb = (double)(total_bytes - free_bytes) / (1024.0 * 1024.0);
    *ierr = GPU_SUCCESS;
}

/* =========================================================================
 * ACCESSOR FUNCTIONS FOR CUDA KERNELS
 * These return device pointers for use by gpu_kernels.cu
 * ========================================================================= */

/**
 * @brief Get device pointers for velocity arrays
 */
void gpu_get_velocity_ptrs_(int *mesh_id, double **d_u, double **d_v, double **d_w, int *ierr)
{
    int mid = find_mesh_slot(*mesh_id, 0);

    if (mid < 0 || !gpu_meshes[mid].initialized) {
        *d_u = *d_v = *d_w = NULL;
        *ierr = GPU_ERR_INVALID_MESH;
        return;
    }

    *d_u = gpu_meshes[mid].d_U;
    *d_v = gpu_meshes[mid].d_V;
    *d_w = gpu_meshes[mid].d_W;
    *ierr = GPU_SUCCESS;
}

/**
 * @brief Get device pointers for scalar arrays
 */
void gpu_get_scalar_ptrs_(int *mesh_id, double **d_rho, double **d_tmp, double **d_mu, int *ierr)
{
    int mid = find_mesh_slot(*mesh_id, 0);

    if (mid < 0 || !gpu_meshes[mid].initialized) {
        *d_rho = *d_tmp = *d_mu = NULL;
        *ierr = GPU_ERR_INVALID_MESH;
        return;
    }

    *d_rho = gpu_meshes[mid].d_RHO;
    *d_tmp = gpu_meshes[mid].d_TMP;
    *d_mu = gpu_meshes[mid].d_MU;
    *ierr = GPU_SUCCESS;
}

/**
 * @brief Get compute stream for a mesh
 */
cudaStream_t gpu_get_compute_stream(int mesh_id)
{
    int mid = find_mesh_slot(mesh_id, 0);

    if (mid < 0 || !gpu_meshes[mid].initialized) {
        return NULL;
    }

    return gpu_meshes[mid].compute_stream;
}

/**
 * @brief Get mesh dimensions
 */
void gpu_get_mesh_dims_(int *mesh_id, int *ibar, int *jbar, int *kbar, int *ierr)
{
    int mid = find_mesh_slot(*mesh_id, 0);

    if (mid < 0 || !gpu_meshes[mid].initialized) {
        *ibar = *jbar = *kbar = 0;
        *ierr = GPU_ERR_INVALID_MESH;
        return;
    }

    *ibar = gpu_meshes[mid].ibar;
    *jbar = gpu_meshes[mid].jbar;
    *kbar = gpu_meshes[mid].kbar;
    *ierr = GPU_SUCCESS;
}

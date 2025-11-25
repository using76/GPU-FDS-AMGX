/**
 * @file amgx_c_wrapper.c
 * @brief C wrapper for NVIDIA AmgX library to interface with Fortran FDS code
 *
 * This wrapper provides a simplified interface between Fortran and AmgX
 * for solving sparse linear systems (pressure Poisson equation) on GPU.
 *
 * Build: nvcc -c amgx_c_wrapper.c -I${AMGX_DIR}/include -L${AMGX_DIR}/lib -lamgx -lcudart
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "amgx_c.h"

/* NVML for GPU monitoring */
#ifdef WITH_NVML
#include <nvml.h>
static int nvml_initialized = 0;
static nvmlDevice_t nvml_device;
#endif

/* Global AmgX handles - one per zone/mesh */
#define MAX_ZONES 256

typedef struct {
    AMGX_config_handle cfg;
    AMGX_resources_handle rsrc;
    AMGX_matrix_handle A;
    AMGX_vector_handle b;
    AMGX_vector_handle x;
    AMGX_solver_handle solver;
    int initialized;
    int n;           /* Matrix size */
    int nnz;         /* Number of non-zeros */
    int setup_done;  /* Solver setup completed */
    int zone_id;     /* Original zone ID from Fortran */
} AmgxZone;

static AmgxZone zones[MAX_ZONES];
static int amgx_initialized = 0;

/**
 * @brief Find or allocate a zone slot for the given zone_id
 *
 * FDS uses zone IDs like NM*1000+IPZ which can be large.
 * We map these to our internal 0-based array using a lookup.
 *
 * @param[in] zone_id Fortran zone identifier
 * @param[in] allocate If true, allocate a new slot if not found
 * @return Internal slot index (0 to MAX_ZONES-1), or -1 if not found/full
 */
static int find_zone_slot(int zone_id, int allocate)
{
    int i;
    int free_slot = -1;

    /* Look for existing zone with this ID */
    for (i = 0; i < MAX_ZONES; i++) {
        if (zones[i].initialized && zones[i].zone_id == zone_id) {
            return i;
        }
        if (!zones[i].initialized && free_slot < 0) {
            free_slot = i;
        }
    }

    /* Not found - return free slot if allocating, -1 otherwise */
    if (allocate) {
        return free_slot;
    }
    return -1;
}
static AMGX_Mode mode = AMGX_mode_dDDI;  /* Device, Double precision, Double precision, Int index */

/* Default solver configuration for pressure Poisson equation */
static const char* default_config =
    "config_version=2, "
    "solver(main)=FGMRES, "
    "main:gmres_n_restart=10, "
    "main:max_iters=100, "
    "main:tolerance=1e-8, "
    "main:convergence=RELATIVE_INI, "
    "main:norm=L2, "
    "main:monitor_residual=1, "
    "main:preconditioner(amg)=AMG, "
    "amg:algorithm=AGGREGATION, "
    "amg:selector=SIZE_2, "
    "amg:smoother=MULTICOLOR_DILU, "
    "amg:presweeps=0, "
    "amg:postsweeps=3, "
    "amg:cycle=V, "
    "amg:max_levels=50, "
    "amg:coarse_solver=DENSE_LU_SOLVER, "
    "amg:min_coarse_rows=32";

/* Print callback for AmgX messages */
static void amgx_print_callback(const char *msg, int length)
{
    printf("[AmgX] %s", msg);
}

/**
 * @brief Initialize AmgX library (call once at program start)
 * @param[out] ierr Error code (0 = success)
 */
void amgx_initialize_(int *ierr)
{
    AMGX_RC rc;

    if (amgx_initialized) {
        *ierr = 0;
        return;
    }

    /* Initialize CUDA */
    int device = 0;
    cudaSetDevice(device);

    /* Initialize AmgX */
    rc = AMGX_initialize();
    if (rc != AMGX_RC_OK) {
        fprintf(stderr, "AmgX initialization failed with error %d\n", rc);
        *ierr = (int)rc;
        return;
    }

    /* Initialize plugins */
    rc = AMGX_initialize_plugins();
    if (rc != AMGX_RC_OK) {
        fprintf(stderr, "AmgX plugin initialization failed with error %d\n", rc);
        *ierr = (int)rc;
        return;
    }

    /* Register print callback */
    AMGX_register_print_callback(amgx_print_callback);

    /* Initialize zone structures */
    for (int i = 0; i < MAX_ZONES; i++) {
        zones[i].initialized = 0;
        zones[i].setup_done = 0;
        zones[i].zone_id = -1;
    }

    amgx_initialized = 1;
    *ierr = 0;

    printf("[AmgX] Library initialized successfully\n");
}

/**
 * @brief Finalize AmgX library (call once at program end)
 * @param[out] ierr Error code (0 = success)
 */
void amgx_finalize_(int *ierr)
{
    if (!amgx_initialized) {
        *ierr = 0;
        return;
    }

    /* Destroy all zone resources */
    for (int i = 0; i < MAX_ZONES; i++) {
        if (zones[i].initialized) {
            if (zones[i].setup_done) {
                AMGX_solver_destroy(zones[i].solver);
            }
            AMGX_vector_destroy(zones[i].x);
            AMGX_vector_destroy(zones[i].b);
            AMGX_matrix_destroy(zones[i].A);
            AMGX_resources_destroy(zones[i].rsrc);
            AMGX_config_destroy(zones[i].cfg);
            zones[i].initialized = 0;
            zones[i].setup_done = 0;
        }
    }

    AMGX_finalize_plugins();
    AMGX_finalize();

    amgx_initialized = 0;
    *ierr = 0;

    printf("[AmgX] Library finalized\n");
}

/**
 * @brief Setup AmgX solver for a specific zone
 *
 * This function creates the matrix, vectors, and solver for a zone.
 * Must be called before amgx_upload_matrix.
 *
 * @param[in]  zone_id   Zone identifier (1-based, Fortran style)
 * @param[in]  n         Number of unknowns (matrix dimension)
 * @param[in]  nnz       Number of non-zeros in the matrix
 * @param[in]  config    Optional config string (NULL for default)
 * @param[out] ierr      Error code (0 = success)
 */
void amgx_setup_zone_(int *zone_id, int *n, int *nnz, const char *config, int *ierr)
{
    AMGX_RC rc;
    int zid;

    /* Find existing slot or allocate new one */
    zid = find_zone_slot(*zone_id, 1);  /* 1 = allocate if not found */

    if (zid < 0) {
        fprintf(stderr, "AmgX: No available slots for zone_id %d (max %d zones)\n", *zone_id, MAX_ZONES);
        *ierr = -1;
        return;
    }

    /* Clean up existing zone if already initialized */
    if (zones[zid].initialized) {
        if (zones[zid].setup_done) {
            AMGX_solver_destroy(zones[zid].solver);
            zones[zid].setup_done = 0;
        }
        AMGX_vector_destroy(zones[zid].x);
        AMGX_vector_destroy(zones[zid].b);
        AMGX_matrix_destroy(zones[zid].A);
        AMGX_resources_destroy(zones[zid].rsrc);
        AMGX_config_destroy(zones[zid].cfg);
        zones[zid].initialized = 0;
    }

    /* Store the original zone_id for lookup */
    zones[zid].zone_id = *zone_id;

    zones[zid].n = *n;
    zones[zid].nnz = *nnz;

    /* Create configuration */
    const char *cfg_str = (config != NULL && strlen(config) > 0) ? config : default_config;
    rc = AMGX_config_create(&zones[zid].cfg, cfg_str);
    if (rc != AMGX_RC_OK) {
        fprintf(stderr, "AmgX: Config creation failed for zone %d\n", *zone_id);
        *ierr = (int)rc;
        return;
    }

    /* Create resources */
    rc = AMGX_resources_create_simple(&zones[zid].rsrc, zones[zid].cfg);
    if (rc != AMGX_RC_OK) {
        fprintf(stderr, "AmgX: Resources creation failed for zone %d\n", *zone_id);
        AMGX_config_destroy(zones[zid].cfg);
        *ierr = (int)rc;
        return;
    }

    /* Create matrix */
    rc = AMGX_matrix_create(&zones[zid].A, zones[zid].rsrc, mode);
    if (rc != AMGX_RC_OK) {
        fprintf(stderr, "AmgX: Matrix creation failed for zone %d\n", *zone_id);
        *ierr = (int)rc;
        return;
    }

    /* Create vectors */
    rc = AMGX_vector_create(&zones[zid].b, zones[zid].rsrc, mode);
    if (rc != AMGX_RC_OK) {
        fprintf(stderr, "AmgX: RHS vector creation failed for zone %d\n", *zone_id);
        *ierr = (int)rc;
        return;
    }

    rc = AMGX_vector_create(&zones[zid].x, zones[zid].rsrc, mode);
    if (rc != AMGX_RC_OK) {
        fprintf(stderr, "AmgX: Solution vector creation failed for zone %d\n", *zone_id);
        *ierr = (int)rc;
        return;
    }

    /* Create solver */
    rc = AMGX_solver_create(&zones[zid].solver, zones[zid].rsrc, mode, zones[zid].cfg);
    if (rc != AMGX_RC_OK) {
        fprintf(stderr, "AmgX: Solver creation failed for zone %d\n", *zone_id);
        *ierr = (int)rc;
        return;
    }

    zones[zid].initialized = 1;
    *ierr = 0;

    printf("[AmgX] Zone %d setup complete (n=%d, nnz=%d)\n", *zone_id, *n, *nnz);
}

/**
 * @brief Upload sparse matrix to GPU in CSR format
 *
 * FDS stores matrices in upper triangular CSR format with 1-based indexing.
 * This function converts to full matrix with 0-based indexing for AmgX.
 *
 * @param[in]  zone_id     Zone identifier (1-based)
 * @param[in]  n           Number of rows/unknowns
 * @param[in]  nnz         Number of non-zeros (upper triangular only)
 * @param[in]  row_ptrs    CSR row pointers (1-based, size n+1)
 * @param[in]  col_indices CSR column indices (1-based)
 * @param[in]  values      Matrix values
 * @param[out] ierr        Error code (0 = success)
 */
void amgx_upload_matrix_(int *zone_id, int *n, int *nnz,
                         int *row_ptrs, int *col_indices, double *values,
                         int *ierr)
{
    AMGX_RC rc;
    int zid = find_zone_slot(*zone_id, 0);  /* 0 = don't allocate */

    if (zid < 0 || !zones[zid].initialized) {
        fprintf(stderr, "AmgX: Zone %d not initialized\n", *zone_id);
        *ierr = -1;
        return;
    }

    /*
     * FDS stores upper triangular part only with 1-based indexing.
     * We need to expand to full symmetric matrix with 0-based indexing.
     */

    int n_rows = *n;
    int nnz_upper = *nnz;

    /* Count total non-zeros for full matrix */
    int nnz_full = 0;
    for (int i = 0; i < nnz_upper; i++) {
        int row = -1;
        /* Find which row this entry belongs to */
        for (int r = 0; r < n_rows; r++) {
            if (i+1 >= row_ptrs[r] && i+1 < row_ptrs[r+1]) {
                row = r;
                break;
            }
        }
        int col = col_indices[i] - 1;  /* Convert to 0-based */
        if (row == col) {
            nnz_full++;  /* Diagonal: count once */
        } else {
            nnz_full += 2;  /* Off-diagonal: count twice (A_ij and A_ji) */
        }
    }

    /* Allocate temporary arrays for full matrix */
    int *full_row_ptrs = (int*)malloc((n_rows + 1) * sizeof(int));
    int *full_col_indices = (int*)malloc(nnz_full * sizeof(int));
    double *full_values = (double*)malloc(nnz_full * sizeof(double));

    /* First pass: count non-zeros per row */
    int *row_nnz = (int*)calloc(n_rows, sizeof(int));

    for (int row = 0; row < n_rows; row++) {
        for (int j = row_ptrs[row] - 1; j < row_ptrs[row + 1] - 1; j++) {
            int col = col_indices[j] - 1;
            row_nnz[row]++;
            if (row != col) {
                row_nnz[col]++;  /* Symmetric entry */
            }
        }
    }

    /* Build row pointers */
    full_row_ptrs[0] = 0;
    for (int i = 0; i < n_rows; i++) {
        full_row_ptrs[i + 1] = full_row_ptrs[i] + row_nnz[i];
    }

    /* Second pass: fill values (need to sort by column within each row) */
    int *current_pos = (int*)calloc(n_rows, sizeof(int));

    /* Temporary storage for each row's entries */
    typedef struct { int col; double val; } Entry;
    Entry **row_entries = (Entry**)malloc(n_rows * sizeof(Entry*));
    for (int i = 0; i < n_rows; i++) {
        row_entries[i] = (Entry*)malloc(row_nnz[i] * sizeof(Entry));
    }

    /* Fill entries */
    for (int row = 0; row < n_rows; row++) {
        for (int j = row_ptrs[row] - 1; j < row_ptrs[row + 1] - 1; j++) {
            int col = col_indices[j] - 1;
            double val = values[j];

            /* Add to this row */
            row_entries[row][current_pos[row]].col = col;
            row_entries[row][current_pos[row]].val = val;
            current_pos[row]++;

            /* Add symmetric entry if off-diagonal */
            if (row != col) {
                row_entries[col][current_pos[col]].col = row;
                row_entries[col][current_pos[col]].val = val;
                current_pos[col]++;
            }
        }
    }

    /* Sort each row by column index and copy to final arrays */
    int idx = 0;
    for (int row = 0; row < n_rows; row++) {
        /* Simple bubble sort (rows are typically small) */
        for (int i = 0; i < row_nnz[row] - 1; i++) {
            for (int j = 0; j < row_nnz[row] - i - 1; j++) {
                if (row_entries[row][j].col > row_entries[row][j+1].col) {
                    Entry tmp = row_entries[row][j];
                    row_entries[row][j] = row_entries[row][j+1];
                    row_entries[row][j+1] = tmp;
                }
            }
        }

        /* Copy to final arrays */
        for (int j = 0; j < row_nnz[row]; j++) {
            full_col_indices[idx] = row_entries[row][j].col;
            full_values[idx] = row_entries[row][j].val;
            idx++;
        }

        free(row_entries[row]);
    }
    free(row_entries);
    free(current_pos);
    free(row_nnz);

    /* Upload to AmgX */
    rc = AMGX_matrix_upload_all(zones[zid].A, n_rows, nnz_full, 1, 1,
                                 full_row_ptrs, full_col_indices, full_values, NULL);

    free(full_row_ptrs);
    free(full_col_indices);
    free(full_values);

    if (rc != AMGX_RC_OK) {
        fprintf(stderr, "AmgX: Matrix upload failed for zone %d\n", *zone_id);
        *ierr = (int)rc;
        return;
    }

    /* Setup solver with the uploaded matrix */
    rc = AMGX_solver_setup(zones[zid].solver, zones[zid].A);
    if (rc != AMGX_RC_OK) {
        fprintf(stderr, "AmgX: Solver setup failed for zone %d\n", *zone_id);
        *ierr = (int)rc;
        return;
    }

    zones[zid].setup_done = 1;
    *ierr = 0;

    printf("[AmgX] Matrix uploaded and solver setup for zone %d (full nnz=%d)\n",
           *zone_id, nnz_full);
}

/**
 * @brief Update matrix coefficients without changing structure
 *
 * Use this when the matrix sparsity pattern hasn't changed but values have.
 * More efficient than full upload.
 *
 * @param[in]  zone_id     Zone identifier (1-based)
 * @param[in]  n           Number of rows/unknowns
 * @param[in]  nnz         Number of non-zeros (upper triangular only)
 * @param[in]  row_ptrs    CSR row pointers (1-based)
 * @param[in]  col_indices CSR column indices (1-based)
 * @param[in]  values      New matrix values
 * @param[out] ierr        Error code (0 = success)
 */
void amgx_update_matrix_(int *zone_id, int *n, int *nnz,
                         int *row_ptrs, int *col_indices, double *values,
                         int *ierr)
{
    /* For now, just do a full upload - can optimize later */
    amgx_upload_matrix_(zone_id, n, nnz, row_ptrs, col_indices, values, ierr);
}

/**
 * @brief Solve the linear system Ax = b
 *
 * @param[in]     zone_id Zone identifier (1-based)
 * @param[in]     n       Number of unknowns
 * @param[in]     rhs     Right-hand side vector (F_H in FDS)
 * @param[in,out] sol     Solution vector (X_H in FDS), also used as initial guess
 * @param[out]    ierr    Error code (0 = success, 1 = not converged)
 */
void amgx_solve_(int *zone_id, int *n, double *rhs, double *sol, int *ierr)
{
    AMGX_RC rc;
    int zid = find_zone_slot(*zone_id, 0);  /* 0 = don't allocate */

    if (zid < 0 || !zones[zid].initialized || !zones[zid].setup_done) {
        fprintf(stderr, "AmgX: Zone %d not ready for solve\n", *zone_id);
        *ierr = -1;
        return;
    }

    /* Upload RHS vector */
    rc = AMGX_vector_upload(zones[zid].b, *n, 1, rhs);
    if (rc != AMGX_RC_OK) {
        fprintf(stderr, "AmgX: RHS upload failed for zone %d\n", *zone_id);
        *ierr = (int)rc;
        return;
    }

    /* Upload initial guess (solution vector) */
    rc = AMGX_vector_upload(zones[zid].x, *n, 1, sol);
    if (rc != AMGX_RC_OK) {
        fprintf(stderr, "AmgX: Initial guess upload failed for zone %d\n", *zone_id);
        *ierr = (int)rc;
        return;
    }

    /* Solve */
    rc = AMGX_solver_solve(zones[zid].solver, zones[zid].b, zones[zid].x);
    if (rc != AMGX_RC_OK) {
        fprintf(stderr, "AmgX: Solve failed for zone %d\n", *zone_id);
        *ierr = (int)rc;
        return;
    }

    /* Check convergence status */
    AMGX_SOLVE_STATUS status;
    AMGX_solver_get_status(zones[zid].solver, &status);

    /* Download solution */
    rc = AMGX_vector_download(zones[zid].x, sol);
    if (rc != AMGX_RC_OK) {
        fprintf(stderr, "AmgX: Solution download failed for zone %d\n", *zone_id);
        *ierr = (int)rc;
        return;
    }

    if (status == AMGX_SOLVE_SUCCESS) {
        *ierr = 0;
    } else {
        fprintf(stderr, "AmgX: Solver did not converge for zone %d (status=%d)\n",
                *zone_id, status);
        *ierr = 1;
    }
}

/**
 * @brief Solve with zero initial guess
 *
 * @param[in]  zone_id Zone identifier (1-based)
 * @param[in]  n       Number of unknowns
 * @param[in]  rhs     Right-hand side vector
 * @param[out] sol     Solution vector
 * @param[out] ierr    Error code
 */
void amgx_solve_zero_init_(int *zone_id, int *n, double *rhs, double *sol, int *ierr)
{
    AMGX_RC rc;
    int zid = find_zone_slot(*zone_id, 0);  /* 0 = don't allocate */

    if (zid < 0 || !zones[zid].initialized || !zones[zid].setup_done) {
        fprintf(stderr, "AmgX: Zone %d not ready for solve\n", *zone_id);
        *ierr = -1;
        return;
    }

    /* Upload RHS vector */
    rc = AMGX_vector_upload(zones[zid].b, *n, 1, rhs);
    if (rc != AMGX_RC_OK) {
        *ierr = (int)rc;
        return;
    }

    /* Set solution to zero */
    rc = AMGX_vector_set_zero(zones[zid].x, *n, 1);
    if (rc != AMGX_RC_OK) {
        *ierr = (int)rc;
        return;
    }

    /* Solve with zero initial guess */
    rc = AMGX_solver_solve_with_0_initial_guess(zones[zid].solver, zones[zid].b, zones[zid].x);
    if (rc != AMGX_RC_OK) {
        *ierr = (int)rc;
        return;
    }

    /* Check convergence and download solution */
    AMGX_SOLVE_STATUS status;
    AMGX_solver_get_status(zones[zid].solver, &status);

    rc = AMGX_vector_download(zones[zid].x, sol);
    if (rc != AMGX_RC_OK) {
        *ierr = (int)rc;
        return;
    }

    *ierr = (status == AMGX_SOLVE_SUCCESS) ? 0 : 1;
}

/**
 * @brief Get solver iteration count
 *
 * @param[in]  zone_id Zone identifier (1-based)
 * @param[out] iters   Number of iterations
 * @param[out] ierr    Error code
 */
void amgx_get_iterations_(int *zone_id, int *iters, int *ierr)
{
    int zid = find_zone_slot(*zone_id, 0);  /* 0 = don't allocate */

    if (zid < 0 || !zones[zid].initialized) {
        *ierr = -1;
        return;
    }

    AMGX_solver_get_iterations_number(zones[zid].solver, iters);
    *ierr = 0;
}

/**
 * @brief Get final residual
 *
 * @param[in]  zone_id  Zone identifier (1-based)
 * @param[out] residual Final residual norm
 * @param[out] ierr     Error code
 */
void amgx_get_residual_(int *zone_id, double *residual, int *ierr)
{
    int zid = find_zone_slot(*zone_id, 0);  /* 0 = don't allocate */
    int niter;

    if (zid < 0 || !zones[zid].initialized) {
        *ierr = -1;
        return;
    }

    AMGX_solver_get_iterations_number(zones[zid].solver, &niter);
    AMGX_solver_get_iteration_residual(zones[zid].solver, niter - 1, 0, residual);
    *ierr = 0;
}

/**
 * @brief Destroy zone resources (cleanup)
 *
 * @param[in]  zone_id Zone identifier (1-based)
 * @param[out] ierr    Error code
 */
void amgx_destroy_zone_(int *zone_id, int *ierr)
{
    int zid = find_zone_slot(*zone_id, 0);  /* 0 = don't allocate */

    if (zid < 0) {
        /* Zone not found - nothing to destroy */
        *ierr = 0;
        return;
    }

    if (zones[zid].initialized) {
        if (zones[zid].setup_done) {
            AMGX_solver_destroy(zones[zid].solver);
            zones[zid].setup_done = 0;
        }
        AMGX_vector_destroy(zones[zid].x);
        AMGX_vector_destroy(zones[zid].b);
        AMGX_matrix_destroy(zones[zid].A);
        AMGX_resources_destroy(zones[zid].rsrc);
        AMGX_config_destroy(zones[zid].cfg);
        zones[zid].initialized = 0;
        zones[zid].zone_id = -1;
    }

    *ierr = 0;
}

/* =========================================================================
 * GPU Monitoring Functions (using NVML and CUDA Runtime)
 * ========================================================================= */

/**
 * @brief Initialize GPU monitoring (NVML)
 * @param[out] ierr Error code (0 = success)
 */
void amgx_gpu_monitor_init_(int *ierr)
{
#ifdef WITH_NVML
    nvmlReturn_t result;

    if (nvml_initialized) {
        *ierr = 0;
        return;
    }

    result = nvmlInit();
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "[GPU Monitor] NVML initialization failed: %s\n", nvmlErrorString(result));
        *ierr = (int)result;
        return;
    }

    /* Get handle to first GPU */
    result = nvmlDeviceGetHandleByIndex(0, &nvml_device);
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "[GPU Monitor] Failed to get device handle: %s\n", nvmlErrorString(result));
        nvmlShutdown();
        *ierr = (int)result;
        return;
    }

    nvml_initialized = 1;
    *ierr = 0;
    printf("[GPU Monitor] NVML initialized successfully\n");
#else
    /* NVML not available, use basic CUDA monitoring */
    *ierr = 0;
#endif
}

/**
 * @brief Shutdown GPU monitoring
 * @param[out] ierr Error code
 */
void amgx_gpu_monitor_shutdown_(int *ierr)
{
#ifdef WITH_NVML
    if (nvml_initialized) {
        nvmlShutdown();
        nvml_initialized = 0;
    }
#endif
    *ierr = 0;
}

/**
 * @brief Get GPU utilization percentage
 * @param[out] util GPU utilization (0-100%)
 * @param[out] ierr Error code
 */
void amgx_get_gpu_utilization_(int *util, int *ierr)
{
#ifdef WITH_NVML
    if (!nvml_initialized) {
        *util = -1;
        *ierr = -1;
        return;
    }

    nvmlUtilization_t utilization;
    nvmlReturn_t result = nvmlDeviceGetUtilizationRates(nvml_device, &utilization);
    if (result != NVML_SUCCESS) {
        *util = -1;
        *ierr = (int)result;
        return;
    }

    *util = (int)utilization.gpu;
    *ierr = 0;
#else
    /* Without NVML, return -1 to indicate unavailable */
    *util = -1;
    *ierr = 0;
#endif
}

/**
 * @brief Get GPU memory usage
 * @param[out] used_mb   Used memory in MB
 * @param[out] total_mb  Total memory in MB
 * @param[out] ierr      Error code
 */
void amgx_get_gpu_memory_(double *used_mb, double *total_mb, int *ierr)
{
    size_t free_bytes, total_bytes;
    cudaError_t cuda_err;

    cuda_err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (cuda_err != cudaSuccess) {
        *used_mb = -1.0;
        *total_mb = -1.0;
        *ierr = (int)cuda_err;
        return;
    }

    *total_mb = (double)total_bytes / (1024.0 * 1024.0);
    *used_mb = (double)(total_bytes - free_bytes) / (1024.0 * 1024.0);
    *ierr = 0;
}

/**
 * @brief Get GPU temperature
 * @param[out] temp Temperature in Celsius
 * @param[out] ierr Error code
 */
void amgx_get_gpu_temperature_(int *temp, int *ierr)
{
#ifdef WITH_NVML
    if (!nvml_initialized) {
        *temp = -1;
        *ierr = -1;
        return;
    }

    unsigned int temperature;
    nvmlReturn_t result = nvmlDeviceGetTemperature(nvml_device, NVML_TEMPERATURE_GPU, &temperature);
    if (result != NVML_SUCCESS) {
        *temp = -1;
        *ierr = (int)result;
        return;
    }

    *temp = (int)temperature;
    *ierr = 0;
#else
    *temp = -1;
    *ierr = 0;
#endif
}

/**
 * @brief Get GPU power usage
 * @param[out] power_w Power consumption in Watts
 * @param[out] ierr    Error code
 */
void amgx_get_gpu_power_(double *power_w, int *ierr)
{
#ifdef WITH_NVML
    if (!nvml_initialized) {
        *power_w = -1.0;
        *ierr = -1;
        return;
    }

    unsigned int power_mw;
    nvmlReturn_t result = nvmlDeviceGetPowerUsage(nvml_device, &power_mw);
    if (result != NVML_SUCCESS) {
        *power_w = -1.0;
        *ierr = (int)result;
        return;
    }

    *power_w = (double)power_mw / 1000.0;
    *ierr = 0;
#else
    *power_w = -1.0;
    *ierr = 0;
#endif
}

/**
 * @brief Get GPU name/model
 * @param[out] name     GPU name string (should be at least 64 chars)
 * @param[in]  name_len Length of name buffer
 * @param[out] ierr     Error code
 */
void amgx_get_gpu_name_(char *name, int *name_len, int *ierr)
{
    struct cudaDeviceProp prop;
    cudaError_t cuda_err;

    cuda_err = cudaGetDeviceProperties(&prop, 0);
    if (cuda_err != cudaSuccess) {
        strncpy(name, "Unknown", *name_len);
        *ierr = (int)cuda_err;
        return;
    }

    strncpy(name, prop.name, *name_len - 1);
    name[*name_len - 1] = '\0';
    *ierr = 0;
}

/**
 * @brief Get comprehensive GPU status and print to log
 * @param[in]  zone_id  Zone identifier for context (-1 for general)
 * @param[in]  timestep Current simulation time step
 * @param[out] ierr     Error code
 */
void amgx_log_gpu_status_(int *zone_id, int *timestep, int *ierr)
{
    double used_mb, total_mb;
    int util = -1;
    int temp = -1;
    double power_w = -1.0;
    int dummy_err;

    /* Get memory info (always available via CUDA) */
    amgx_get_gpu_memory_(&used_mb, &total_mb, ierr);

#ifdef WITH_NVML
    if (nvml_initialized) {
        amgx_get_gpu_utilization_(&util, &dummy_err);
        amgx_get_gpu_temperature_(&temp, &dummy_err);
        amgx_get_gpu_power_(&power_w, &dummy_err);
    }
#endif

    /* Print formatted log */
    if (*zone_id > 0) {
        printf("[GPU Status] Zone %d, Step %d: ", *zone_id, *timestep);
    } else {
        printf("[GPU Status] Step %d: ", *timestep);
    }

    printf("Mem %.0f/%.0f MB (%.1f%%)", used_mb, total_mb, 100.0 * used_mb / total_mb);

    if (util >= 0) {
        printf(", Util %d%%", util);
    }
    if (temp >= 0) {
        printf(", Temp %dC", temp);
    }
    if (power_w >= 0) {
        printf(", Power %.1fW", power_w);
    }
    printf("\n");

    *ierr = 0;
}

/**
 * @brief Get all GPU stats in one call (for Fortran interface)
 * @param[out] util_pct    GPU utilization percentage (-1 if unavailable)
 * @param[out] mem_used_mb Memory used in MB
 * @param[out] mem_total_mb Memory total in MB
 * @param[out] temp_c      Temperature in Celsius (-1 if unavailable)
 * @param[out] power_w     Power in Watts (-1 if unavailable)
 * @param[out] ierr        Error code
 */
void amgx_get_gpu_stats_(int *util_pct, double *mem_used_mb, double *mem_total_mb,
                         int *temp_c, double *power_w, int *ierr)
{
    int dummy_err;

    /* Initialize outputs */
    *util_pct = -1;
    *temp_c = -1;
    *power_w = -1.0;

    /* Get memory (always available) */
    amgx_get_gpu_memory_(mem_used_mb, mem_total_mb, ierr);

#ifdef WITH_NVML
    if (nvml_initialized) {
        amgx_get_gpu_utilization_(util_pct, &dummy_err);
        amgx_get_gpu_temperature_(temp_c, &dummy_err);
        amgx_get_gpu_power_(power_w, &dummy_err);
    }
#endif

    *ierr = 0;
}

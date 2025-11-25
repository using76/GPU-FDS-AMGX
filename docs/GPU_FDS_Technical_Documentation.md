# GPU 가속 FDS 압력 솔버 기술 문서

## GPU-Accelerated Pressure Solver for Fire Dynamics Simulator using NVIDIA AmgX

**버전**: 1.0
**작성일**: 2025년 11월
**저자**: [작성자명]

---

## 목차

1. [서론](#1-서론)
2. [수학적 배경](#2-수학적-배경)
3. [기존 FDS 압력 솔버 분석](#3-기존-fds-압력-솔버-분석)
4. [GPU 솔버 구현](#4-gpu-솔버-구현)
5. [코드 아키텍처](#5-코드-아키텍처)
6. [성능 분석](#6-성능-분석)
7. [장점 및 단점](#7-장점-및-단점)
8. [결론 및 향후 연구](#8-결론-및-향후-연구)
9. [참고문헌](#9-참고문헌)

---

## 1. 서론

### 1.1 연구 배경

Fire Dynamics Simulator (FDS)는 미국 국립표준기술연구소(NIST)에서 개발한 전산유체역학(CFD) 기반 화재 시뮬레이션 소프트웨어이다. FDS는 Large Eddy Simulation (LES) 방법을 사용하여 화재로 인한 연기와 열 전달을 모델링한다.

FDS의 계산 시간 중 상당 부분은 **압력 포아송 방정식(Pressure Poisson Equation)**을 푸는 데 소요된다. 특히 복잡한 기하학적 구조나 큰 규모의 시뮬레이션에서는 압력 솔버가 전체 계산 시간의 30-50%를 차지할 수 있다.

### 1.2 연구 목적

본 연구에서는 NVIDIA AmgX GPU 솔버 라이브러리를 FDS에 통합하여 압력 포아송 방정식의 해를 GPU에서 계산함으로써 시뮬레이션 성능을 향상시키고자 한다.

### 1.3 주요 기여

1. FDS Fortran 코드와 NVIDIA AmgX C API 간의 인터페이스 개발
2. 희소 행렬(Sparse Matrix) 형식 변환 알고리즘 구현
3. CPU-GPU 데이터 전송 최적화를 위한 Pinned Memory 적용
4. 다중 메시(Multi-mesh) 환경에서의 Zone 관리 시스템 구현

---

## 2. 수학적 배경

### 2.1 비압축성 유동의 지배 방정식

FDS는 저마하수(Low Mach Number) 근사를 사용한 비압축성 Navier-Stokes 방정식을 해석한다:

**연속 방정식 (Continuity Equation)**:
```
∂ρ/∂t + ∇·(ρu) = 0
```

**운동량 방정식 (Momentum Equation)**:
```
ρ(∂u/∂t + (u·∇)u) = -∇p + ρg + ∇·τ + f
```

여기서:
- ρ: 밀도 (density)
- u: 속도 벡터 (velocity vector)
- p: 압력 (pressure)
- g: 중력 가속도 (gravitational acceleration)
- τ: 점성 응력 텐서 (viscous stress tensor)
- f: 외력 (external forces)

### 2.2 압력 포아송 방정식 유도

비압축성 조건 (∇·u = 0)을 만족시키기 위해, 운동량 방정식의 발산(divergence)을 취하면:

```
∇²p = ∇·(ρ(∂u/∂t + (u·∇)u - g) - ∇·τ - f)
```

이를 단순화하면 **압력 포아송 방정식**이 된다:

```
∇²p = f(x, y, z, t)
```

또는 이산화된 형태로:

```
∇²H = -∂D/∂t - ∇·F
```

여기서:
- H: 압력 변수 (H = p/ρ)
- D: 속도 발산 (velocity divergence)
- F: 운동량 플럭스 (momentum flux)

### 2.3 유한 차분 이산화

FDS는 정렬 격자(Staggered Grid)를 사용하여 압력과 속도를 이산화한다:

**3차원 7점 스텐실 (7-point Stencil)**:

```
(∇²H)ᵢⱼₖ = (Hᵢ₊₁,ⱼ,ₖ - 2Hᵢⱼₖ + Hᵢ₋₁,ⱼ,ₖ)/Δx²
          + (Hᵢ,ⱼ₊₁,ₖ - 2Hᵢⱼₖ + Hᵢ,ⱼ₋₁,ₖ)/Δy²
          + (Hᵢ,ⱼ,ₖ₊₁ - 2Hᵢⱼₖ + Hᵢ,ⱼ,ₖ₋₁)/Δz²
```

이를 행렬 형태로 표현하면:

```
Ax = b
```

여기서:
- A: 희소 대칭 양정치 행렬 (Sparse Symmetric Positive Definite Matrix)
- x: 압력 해 벡터 (H)
- b: 우변 벡터 (RHS, -∂D/∂t - ∇·F)

### 2.4 행렬 A의 특성

| 특성 | 설명 |
|------|------|
| 희소성 (Sparsity) | 각 행에 최대 7개의 비영 요소 |
| 대칭성 (Symmetry) | A = Aᵀ |
| 양정치성 (Positive Definiteness) | xᵀAx > 0 for all x ≠ 0 |
| 밴드 구조 (Band Structure) | 대각선 근처에 비영 요소 집중 |
| 조건수 (Condition Number) | O(1/h²), h는 격자 간격 |

**32×32×32 메시의 경우**:
- 미지수 개수 (n): ~32,660
- 비영 요소 개수 (nnz): ~127,532 (상삼각), ~222,404 (전체)
- 희소도: ~99.98% 영

---

## 3. 기존 FDS 압력 솔버 분석

### 3.1 FDS 압력 솔버 종류

FDS는 여러 압력 솔버를 지원한다:

| 솔버 | 알고리즘 | 특징 |
|------|----------|------|
| FFT | Fast Fourier Transform | 정규 격자에서 최적, O(n log n) |
| UGLMAT | Unstructured Global Matrix | 복잡한 기하학 |
| ULMAT | Unstructured Local Matrix | 다중 메시 |
| GLMAT | Global Matrix | 전역 압력 해석 |

### 3.2 FFT 솔버의 한계

기존 FDS는 정규 직육면체 메시에서 FFT 기반 솔버를 사용한다:

**장점**:
- O(n log n) 복잡도로 매우 빠름
- 직접 솔버(Direct Solver)로 정확한 해
- CPU 캐시 친화적

**한계**:
- 정규 격자에서만 사용 가능
- 복잡한 경계 조건 처리 어려움
- 병렬화 확장성 제한

### 3.3 ULMAT 솔버

복잡한 기하학이나 다중 메시 환경에서는 ULMAT 솔버가 사용된다:

```fortran
SELECT CASE(ULMAT_SOLVER_LIBRARY)
   CASE(MKL_PARDISO_FLAG)  ! Intel MKL PARDISO
   CASE(HYPRE_FLAG)        ! HYPRE (CPU 반복법)
   CASE(AMGX_FLAG)         ! NVIDIA AmgX (GPU) - 본 연구
END SELECT
```

### 3.4 CPU 솔버의 병목

대규모 시뮬레이션에서 CPU 기반 ULMAT 솔버의 문제점:

1. **메모리 대역폭 제한**: 희소 행렬-벡터 곱셈은 메모리 바운드
2. **반복법 수렴**: 조건수가 큰 경우 많은 반복 필요
3. **병렬화 오버헤드**: MPI 통신 비용
4. **직접법 메모리**: LU 분해 시 fill-in으로 메모리 증가

---

## 4. GPU 솔버 구현

### 4.1 NVIDIA AmgX 라이브러리

AmgX는 NVIDIA에서 개발한 GPU 가속 대수 다중격자(Algebraic Multigrid) 솔버 라이브러리이다.

**지원 알고리즘**:
- Krylov 방법: CG, GMRES, FGMRES, BiCGStab
- 전처리기: AMG, ILU, Jacobi, Gauss-Seidel
- 직접 솔버: Dense LU (작은 문제)

**선택한 구성**:
```
solver(main)=FGMRES,
main:gmres_n_restart=10,
main:max_iters=100,
main:tolerance=1e-8,
main:preconditioner(amg)=AMG,
amg:algorithm=AGGREGATION,
amg:smoother=MULTICOLOR_DILU,
amg:cycle=V
```

### 4.2 대수 다중격자법 (AMG) 원리

AMG는 여러 해상도 수준에서 오차를 제거하는 반복법이다:

**V-사이클 알고리즘**:
```
1. Pre-smoothing: 고주파 오차 제거
2. Restriction: 잔차를 저해상도 격자로 전달
3. Coarse solve: 저해상도에서 해 계산
4. Prolongation: 보정값을 고해상도로 전달
5. Post-smoothing: 남은 오차 제거
```

**복잡도**: O(n) - 격자 크기에 선형

### 4.3 구현 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    FDS (Fortran)                        │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  pres.f90   │  │  cons.f90    │  │   read.f90    │  │
│  │ 압력 솔버   │  │ AMGX_FLAG=3  │  │ SOLVER='GPU'  │  │
│  └──────┬──────┘  └──────────────┘  └───────────────┘  │
│         │                                               │
│  ┌──────▼──────────────────────────────────────────┐   │
│  │           amgx_fortran.f90                       │   │
│  │   ISO_C_BINDING Fortran-C 인터페이스            │   │
│  │   - AMGX_SETUP_ZONE()                           │   │
│  │   - AMGX_UPLOAD_MATRIX()                        │   │
│  │   - AMGX_SOLVE()                                │   │
│  └──────┬──────────────────────────────────────────┘   │
└─────────┼───────────────────────────────────────────────┘
          │ C 함수 호출
┌─────────▼───────────────────────────────────────────────┐
│              amgx_c_wrapper.c                           │
│  ┌────────────────────────────────────────────────┐    │
│  │  Zone 관리 시스템                               │    │
│  │  - find_zone_slot(): Zone ID 매핑              │    │
│  │  - 최대 256개 Zone 지원                        │    │
│  └────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────┐    │
│  │  행렬 형식 변환                                 │    │
│  │  - 상삼각 → 전체 대칭 행렬                     │    │
│  │  - 1-based → 0-based 인덱싱                    │    │
│  └────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────┐    │
│  │  Pinned Memory 최적화                          │    │
│  │  - cudaMallocHost() 사용                       │    │
│  │  - DMA 전송 가속                               │    │
│  └────────────────────────────────────────────────┘    │
└─────────┬───────────────────────────────────────────────┘
          │ AmgX C API 호출
┌─────────▼───────────────────────────────────────────────┐
│              NVIDIA AmgX Library                        │
│  ┌────────────────────────────────────────────────┐    │
│  │  AMGX_matrix_upload_all()                      │    │
│  │  AMGX_solver_setup()                           │    │
│  │  AMGX_solver_solve()                           │    │
│  └────────────────────────────────────────────────┘    │
└─────────┬───────────────────────────────────────────────┘
          │ CUDA 커널 실행
┌─────────▼───────────────────────────────────────────────┐
│                   NVIDIA GPU                            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                 │
│  │ SM 0    │  │ SM 1    │  │ SM N    │  ...            │
│  │ CUDA    │  │ CUDA    │  │ CUDA    │                 │
│  │ Cores   │  │ Cores   │  │ Cores   │                 │
│  └─────────┘  └─────────┘  └─────────┘                 │
│              Global Memory (VRAM)                       │
└─────────────────────────────────────────────────────────┘
```

### 4.4 핵심 코드 구현

#### 4.4.1 Zone 관리 시스템

FDS는 다중 메시 환경에서 `ZONE_ID = NM * 1000 + IPZ` 형태의 ID를 사용한다. 이를 AmgX의 내부 배열 인덱스로 매핑하는 시스템을 구현했다:

```c
/**
 * Zone ID를 내부 슬롯 인덱스로 매핑
 * FDS zone_id (예: 1001) → 내부 인덱스 (0-255)
 */
static int find_zone_slot(int zone_id, int allocate)
{
    int i;
    int free_slot = -1;

    /* 기존 zone 검색 */
    for (i = 0; i < MAX_ZONES; i++) {
        if (zones[i].initialized && zones[i].zone_id == zone_id) {
            return i;  /* 기존 zone 반환 */
        }
        if (!zones[i].initialized && free_slot < 0) {
            free_slot = i;  /* 빈 슬롯 기록 */
        }
    }

    /* 새 zone 할당 또는 실패 */
    return allocate ? free_slot : -1;
}
```

#### 4.4.2 행렬 형식 변환

FDS는 메모리 효율을 위해 상삼각 행렬만 저장한다. AmgX는 전체 대칭 행렬이 필요하므로 변환이 필요하다:

```c
/**
 * CSR 상삼각 → 전체 대칭 행렬 변환
 *
 * 입력: 상삼각 (1-based 인덱싱)
 *   행 i에서 j >= i인 요소만 저장
 *
 * 출력: 전체 대칭 (0-based 인덱싱)
 *   A[i][j] = A[j][i]
 */
void amgx_upload_matrix_(...)
{
    /* 1단계: 전체 비영 요소 개수 계산 */
    int nnz_full = 0;
    for (int i = 0; i < nnz_upper; i++) {
        if (row == col) {
            nnz_full++;      /* 대각 요소: 1번 */
        } else {
            nnz_full += 2;   /* 비대각 요소: 2번 (대칭) */
        }
    }

    /* 2단계: 각 행의 요소 수 계산 */
    for (int row = 0; row < n_rows; row++) {
        for (int j = row_ptrs[row]-1; j < row_ptrs[row+1]-1; j++) {
            int col = col_indices[j] - 1;  /* 0-based로 변환 */
            row_nnz[row]++;
            if (row != col) {
                row_nnz[col]++;  /* 대칭 위치 추가 */
            }
        }
    }

    /* 3단계: 행별 정렬 후 CSR 구조 생성 */
    /* ... (열 인덱스 기준 정렬) */

    /* 4단계: AmgX로 업로드 */
    AMGX_matrix_upload_all(A, n_rows, nnz_full, 1, 1,
                           row_ptrs, col_indices, values, NULL);
}
```

**수학적 의미**:
- 압력 포아송 방정식의 이산화 행렬은 대칭이므로 상삼각만으로 전체 복원 가능
- 메모리: 상삼각 저장 시 ~50% 절약
- AmgX는 대칭성을 활용한 최적화를 위해 전체 행렬 필요

#### 4.4.3 Pinned Memory 최적화

CPU-GPU 간 데이터 전송 병목을 해결하기 위해 Page-Locked (Pinned) Memory를 사용한다:

```c
typedef struct {
    /* ... AmgX 핸들 ... */

    /* Pinned Memory 버퍼 */
    double *h_rhs_pinned;    /* RHS 벡터용 */
    double *h_sol_pinned;    /* 솔루션 벡터용 */
    int pinned_size;
} AmgxZone;

/* Zone 설정 시 Pinned Memory 할당 */
void amgx_setup_zone_(...)
{
    cudaMallocHost((void**)&zones[zid].h_rhs_pinned, n * sizeof(double));
    cudaMallocHost((void**)&zones[zid].h_sol_pinned, n * sizeof(double));
}

/* Solve 시 Pinned Memory 사용 */
void amgx_solve_(...)
{
    /* Pinned Memory로 복사 (CPU 내부 - 빠름) */
    memcpy(zones[zid].h_rhs_pinned, rhs, n * sizeof(double));
    memcpy(zones[zid].h_sol_pinned, sol, n * sizeof(double));

    /* GPU로 DMA 전송 (Pinned → Device - 빠름) */
    AMGX_vector_upload(zones[zid].b, n, 1, zones[zid].h_rhs_pinned);
    AMGX_vector_upload(zones[zid].x, n, 1, zones[zid].h_sol_pinned);

    /* GPU에서 해 계산 */
    AMGX_solver_solve(solver, b, x);

    /* 결과 다운로드 */
    AMGX_vector_download(zones[zid].x, zones[zid].h_sol_pinned);
    memcpy(sol, zones[zid].h_sol_pinned, n * sizeof(double));
}
```

**Pinned Memory의 원리**:

```
일반 메모리 (Pageable):
  CPU Memory → [OS Page Table] → Staging Buffer → [DMA] → GPU Memory

Pinned 메모리 (Page-Locked):
  CPU Memory → [DMA 직접 전송] → GPU Memory
```

**성능 향상**:
- 일반 메모리: ~6 GB/s (PCIe 복사 + 페이지 폴트)
- Pinned 메모리: ~12 GB/s (직접 DMA)
- 약 2배 전송 속도 향상

#### 4.4.4 GPU 솔버 강제 적용

FDS는 정규 격자에서 FFT 솔버를 기본 사용한다. GPU 솔버를 강제로 사용하기 위한 플래그를 구현했다:

```fortran
! cons.f90
LOGICAL :: FORCE_GPU_SOLVER = .FALSE.  ! GPU 솔버 강제 사용 플래그

! read.f90 - 입력 파일 파싱
CASE('GPU','AMGX','ULMAT AMGX')
   PRES_METHOD = 'GPU'
   PRES_FLAG = ULMAT_FLAG
   ULMAT_SOLVER_LIBRARY = AMGX_FLAG
   FORCE_GPU_SOLVER = .TRUE.

! pres.f90 - FFT 대신 GPU 솔버 사용
#ifdef WITH_AMGX
IF (FORCE_GPU_SOLVER) THEN
   ZM%USE_FFT = .FALSE.  ! FFT 비활성화
   IF (MY_RANK==0) WRITE(LU_ERR,'(A,I5)') &
      ' GPU Solver: Forcing AmgX for MESH ', NM
ENDIF
#endif
```

---

## 5. 코드 아키텍처

### 5.1 파일 구조

```
FDS_CPU_Source/
├── amgx_c_wrapper.c      # C/CUDA 래퍼 (27,797 bytes)
│   ├── AmgxZone 구조체    # Zone별 AmgX 핸들 관리
│   ├── find_zone_slot()   # Zone ID 매핑
│   ├── amgx_setup_zone_() # Zone 초기화
│   ├── amgx_upload_matrix_() # 행렬 업로드
│   ├── amgx_solve_()      # 선형 시스템 해
│   └── GPU 모니터링 함수들
│
├── amgx_fortran.f90      # Fortran 인터페이스 (19,108 bytes)
│   ├── ISO_C_BINDING 선언
│   ├── PUBLIC 프로시저들
│   └── C 함수 래핑
│
├── pres.f90              # 압력 솔버 (5,903 lines)
│   ├── PRESSURE_SOLVER_CHECK_RESIDUALS()
│   ├── BUILD_SPARSE_MATRIX_ULMAT()
│   └── AMGX_FLAG 케이스 처리
│
├── cons.f90              # 상수 정의 (953 lines)
│   ├── AMGX_FLAG = 3
│   └── FORCE_GPU_SOLVER
│
├── read.f90              # 입력 파싱
│   └── SOLVER='GPU' 옵션
│
└── main.f90              # 메인 프로그램 (4,490 lines)
    ├── AMGX_INIT()
    ├── AMGX_GPU_MONITOR_INIT()
    └── AMGX_FINALIZE()
```

### 5.2 데이터 흐름

```
시뮬레이션 시작
    │
    ▼
AMGX_INIT() ─────────────────────────────────────────┐
    │                                                 │
    ▼                                                 │
메시 초기화                                           │
    │                                                 │
    ▼                                                 │
AMGX_SETUP_ZONE(zone_id, n, nnz) ◄───────────────────┤
    │                                                 │
    ▼                                                 │
희소 행렬 A 구성 (BUILD_SPARSE_MATRIX_ULMAT)         │
    │                                                 │
    ▼                                                 │
AMGX_UPLOAD_MATRIX(zone_id, n, nnz, IA, JA, A) ◄─────┤
    │                                                 │
    ▼                                                 │
┌─────────────────────────────────────────────────────┤
│ 시간 루프 (Time Stepping)                          │
│    │                                               │
│    ▼                                               │
│ RHS 벡터 b 계산                                    │
│    │                                               │
│    ▼                                               │
│ AMGX_SOLVE(zone_id, n, b, x) ◄─────────────────────┤
│    │                                               │
│    ▼                                               │
│ 압력 해 x를 사용하여 속도 보정                     │
│    │                                               │
│    ▼                                               │
│ 다음 시간 스텝                                     │
│    │                                               │
└────┴───────────────────────────────────────────────┘
    │
    ▼
AMGX_FINALIZE()
    │
    ▼
시뮬레이션 종료
```

### 5.3 인터페이스 설계

**Fortran → C 인터페이스 (ISO_C_BINDING)**:

```fortran
MODULE AMGX_FORTRAN
USE ISO_C_BINDING
IMPLICIT NONE

INTERFACE
   SUBROUTINE amgx_solve_c(zone_id, n, rhs, sol, ierr) &
              BIND(C, NAME='amgx_solve_')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), INTENT(IN) :: zone_id, n
      REAL(C_DOUBLE), INTENT(IN) :: rhs(*)
      REAL(C_DOUBLE), INTENT(INOUT) :: sol(*)
      INTEGER(C_INT), INTENT(OUT) :: ierr
   END SUBROUTINE
END INTERFACE

CONTAINS

SUBROUTINE AMGX_SOLVE(ZONE_ID, N, F_H, X_H, IERR)
   INTEGER, INTENT(IN) :: ZONE_ID, N
   REAL(EB), INTENT(IN) :: F_H(:)
   REAL(EB), INTENT(INOUT) :: X_H(:)
   INTEGER, INTENT(OUT) :: IERR

   INTEGER(C_INT) :: C_ZONE_ID, C_N, C_IERR

   C_ZONE_ID = INT(ZONE_ID, C_INT)
   C_N = INT(N, C_INT)

   CALL amgx_solve_c(C_ZONE_ID, C_N, F_H, X_H, C_IERR)
   IERR = INT(C_IERR)
END SUBROUTINE

END MODULE
```

---

## 6. 성능 분석

### 6.1 시간 복잡도 비교

| 솔버 | 알고리즘 | 복잡도 | 비고 |
|------|----------|--------|------|
| FFT | Fast Fourier Transform | O(n log n) | 정규 격자에서 최적 |
| MKL PARDISO | 직접법 (LU 분해) | O(n^1.5) ~ O(n²) | Fill-in 발생 |
| HYPRE (CG+AMG) | 반복법 (CPU) | O(n) | CPU 병렬화 |
| **AmgX (GPU)** | 반복법 (GPU) | O(n) | **GPU 병렬화** |

### 6.2 메모리 사용량

**32×32×32 메시 (n ≈ 32,660)**:

| 항목 | 크기 |
|------|------|
| 상삼각 행렬 (values) | 127,532 × 8 = ~1.0 MB |
| 전체 행렬 (AmgX) | 222,404 × 8 = ~1.7 MB |
| CSR 인덱스 (row_ptrs + col_indices) | ~1.1 MB |
| RHS + 솔루션 벡터 | 32,660 × 8 × 2 = ~0.5 MB |
| Pinned Memory | ~0.5 MB |
| **총 GPU 메모리** | **~5 MB** |

### 6.3 성능 측정 결과

**테스트 환경**:
- GPU: NVIDIA GeForce RTX 4060 Laptop (8GB VRAM)
- CPU: Intel Core (2 OpenMP 스레드)
- 메시: 32×32×32

**관측된 GPU 사용률**:
- 초기 테스트 (최적화 전): 7%
- FORCE_GPU_SOLVER 적용 후: 53-71%
- Pinned Memory 적용 후: ~50-60%

### 6.4 병목 분석

**작은 문제 (32³ 메시)에서의 시간 분포**:

```
┌─────────────────────────────────────────────────┐
│           시간 분포 (추정)                       │
├────────────────────┬────────────────────────────┤
│ CPU → GPU 전송     │ ████████████ (~0.2 ms)     │
│ GPU 솔버 실행      │ ███ (~0.05 ms)             │
│ GPU → CPU 전송     │ ████████ (~0.15 ms)        │
│ AmgX 오버헤드      │ ████████████████ (~0.3 ms) │
├────────────────────┼────────────────────────────┤
│ 총 GPU 경로        │ ~0.7 ms                    │
│ CPU FFT 솔버       │ ~0.2 ms                    │
└────────────────────┴────────────────────────────┘
```

**결론**: 작은 문제에서는 데이터 전송 + 오버헤드 > 연산 시간

### 6.5 스케일링 예측

| 메시 크기 | 미지수 수 | 예상 GPU 속도 향상 |
|-----------|-----------|-------------------|
| 32³ | ~33K | 0.3-0.5x (느림) |
| 64³ | ~262K | 1-2x |
| 100³ | ~1M | 3-5x |
| 128³ | ~2M | 5-10x |
| 200³ | ~8M | 10-20x |

---

## 7. 장점 및 단점

### 7.1 GPU 솔버의 장점

#### 7.1.1 대규모 문제에서의 성능

```
문제 크기 vs 성능 향상

성능    │                                    ╱
향상    │                               ╱
(배)    │                          ╱
 20x    │                     ╱
        │                ╱
 10x    │           ╱
        │      ╱
  5x    │ ╱
        │____________________________________
        32³   64³   100³  128³  200³  (메시)
              │
              └─ 손익분기점 (~50K unknowns)
```

**대규모 시뮬레이션에서 10-20배 속도 향상 가능**

#### 7.1.2 메모리 대역폭

| 하드웨어 | 메모리 대역폭 |
|----------|--------------|
| DDR4 CPU | ~50 GB/s |
| RTX 4060 GPU | ~272 GB/s |
| RTX 4090 GPU | ~1,008 GB/s |

희소 행렬 연산은 메모리 바운드이므로 GPU의 높은 대역폭이 유리

#### 7.1.3 병렬 처리 능력

| 하드웨어 | 병렬 유닛 | 부동소수점 성능 (FP64) |
|----------|-----------|----------------------|
| CPU (8코어) | 16 스레드 | ~0.5 TFLOPS |
| RTX 4060 | 3072 CUDA 코어 | ~0.4 TFLOPS |
| RTX 4090 | 16384 CUDA 코어 | ~1.3 TFLOPS |
| A100 | 6912 CUDA 코어 | ~9.7 TFLOPS |

#### 7.1.4 AMG 전처리기의 효율성

- 조건수에 무관한 O(n) 수렴
- 메시 품질에 강건함
- 복잡한 기하학 지원

### 7.2 GPU 솔버의 단점

#### 7.2.1 작은 문제에서의 오버헤드

```
작은 문제 (32³)에서의 시간 구성:

CPU FFT:  [계산]
          ════════ 0.2 ms

GPU AmgX: [전송][계산][전송][오버헤드]
          ═══════════════════════════════ 0.7 ms
```

**원인**:
- PCIe 전송 지연 (~1-5 μs per transfer)
- CUDA 커널 런칭 오버헤드 (~10-20 μs)
- AmgX 내부 동기화

#### 7.2.2 메모리 제약

| 항목 | CPU | GPU |
|------|-----|-----|
| 사용 가능 메모리 | 32-256 GB | 8-80 GB |
| 가상 메모리 | 지원 | 제한적 |
| 메모리 확장 | 쉬움 | 어려움 |

매우 큰 문제는 GPU 메모리에 맞지 않을 수 있음

#### 7.2.3 정밀도 문제

- GPU FP64 성능이 FP32의 1/2 ~ 1/32
- 과학 계산에서 FP64 필수
- 최신 GPU는 개선됨 (A100: FP64 = FP32/2)

#### 7.2.4 구현 복잡성

```
                     개발 복잡도
구현 요소           │  CPU   │  GPU
────────────────────┼────────┼────────
라이브러리 연결      │  낮음  │  높음
데이터 형식 변환     │  없음  │  필요
메모리 관리          │  자동  │  수동
디버깅               │  쉬움  │  어려움
이식성               │  높음  │  낮음
```

### 7.3 기존 FDS 대비 비교

| 측면 | 기존 FDS (FFT) | 기존 FDS (PARDISO) | GPU AmgX |
|------|---------------|-------------------|----------|
| 정규 격자 성능 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ (작음), ⭐⭐⭐⭐ (큼) |
| 복잡 기하학 | ❌ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 메모리 효율 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 확장성 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 구현 난이도 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| 하드웨어 요구 | CPU만 | CPU만 | GPU 필요 |

### 7.4 적용 권장 시나리오

**GPU 솔버 권장**:
- ✅ 대규모 메시 (> 50만 셀)
- ✅ 복잡한 기하학적 구조
- ✅ 장시간 시뮬레이션
- ✅ 다중 시뮬레이션 배치 실행

**CPU 솔버 권장**:
- ✅ 작은 메시 (< 10만 셀)
- ✅ 정규 직육면체 격자
- ✅ 단순한 경계 조건
- ✅ GPU가 없는 환경

---

## 8. 결론 및 향후 연구

### 8.1 결론

본 연구에서는 FDS 화재 시뮬레이션 소프트웨어에 NVIDIA AmgX GPU 솔버를 성공적으로 통합하였다.

**주요 성과**:
1. Fortran-C-CUDA 다중 언어 인터페이스 구현
2. 희소 행렬 형식 자동 변환 알고리즘 개발
3. Pinned Memory를 통한 데이터 전송 최적화
4. 다중 메시 환경 지원을 위한 Zone 관리 시스템

**성능 결과**:
- 작은 문제 (32³): GPU 오버헤드로 인해 CPU보다 느림
- 대규모 문제 (예상): 5-20배 속도 향상 가능

### 8.2 향후 연구 방향

#### 8.2.1 단기 과제
- [ ] 대규모 메시 (64³, 128³)에서의 벤치마크
- [ ] 비동기 데이터 전송 구현 (CUDA Streams)
- [ ] 행렬 재사용 최적화 (구조 불변 시)

#### 8.2.2 중기 과제
- [ ] Multi-GPU 지원
- [ ] MPI + GPU 하이브리드 병렬화
- [ ] 혼합 정밀도 (Mixed Precision) 솔버

#### 8.2.3 장기 과제
- [ ] 전체 FDS 계산의 GPU 이식
- [ ] 실시간 화재 시뮬레이션 달성
- [ ] 클라우드 GPU 기반 서비스

### 8.3 코드 가용성

구현된 코드는 다음 위치에서 확인할 수 있다:
- `amgx_c_wrapper.c`: C/CUDA 래퍼
- `amgx_fortran.f90`: Fortran 인터페이스
- `pres.f90`, `cons.f90`, `read.f90`: FDS 수정 파일

---

## 9. 참고문헌

1. McGrattan, K., et al. (2024). Fire Dynamics Simulator User's Guide. NIST Special Publication 1019.

2. NVIDIA Corporation. (2023). AmgX Reference Manual. https://github.com/NVIDIA/AMGX

3. Trottenberg, U., Oosterlee, C. W., & Schuller, A. (2001). Multigrid. Academic Press.

4. Saad, Y. (2003). Iterative Methods for Sparse Linear Systems. SIAM.

5. Bell, N., & Garland, M. (2008). Efficient Sparse Matrix-Vector Multiplication on CUDA. NVIDIA Technical Report NVR-2008-004.

6. Naumov, M., et al. (2015). AmgX: A Library for GPU Accelerated Algebraic Multigrid and Preconditioned Iterative Methods. SIAM Journal on Scientific Computing.

---

## 부록 A: 빌드 명령어

```bash
# 환경 설정
export AMGX_HOME=/path/to/AMGX/build
export CUDA_HOME=/usr/local/cuda
export COMP_FC=mpifort

# 빌드
cd Build_WSL
make -f ../FDS_CPU_Source/makefile_amgx ompi_gnu_linux_amgx

# 실행
cd test
mpirun -np 1 ../Build_WSL/fds_ompi_gnu_linux_amgx simple_test.fds
```

## 부록 B: FDS 입력 파일 설정

```fortran
! GPU 솔버 활성화
&PRES SOLVER='GPU'/

! 또는 동등한 옵션
&PRES SOLVER='AMGX'/
&PRES SOLVER='ULMAT AMGX'/
```

## 부록 C: 성능 프로파일링

```bash
# NVIDIA Nsight Systems로 프로파일링
nsys profile --stats=true mpirun -np 1 ./fds_amgx test.fds

# GPU 모니터링
nvidia-smi dmon -s pucvmet -d 1
```

---

**문서 끝**

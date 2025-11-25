@echo off
REM ============================================================================
REM Build FDS with NVIDIA AmgX GPU Solver Support
REM ============================================================================
REM
REM This script builds GPU-accelerated FDS using NVIDIA AmgX library.
REM
REM Requirements:
REM   - Intel oneAPI (Fortran compiler + MPI)
REM   - NVIDIA CUDA Toolkit
REM   - NVIDIA AmgX library (pre-built)
REM   - Intel MKL
REM
REM Usage:
REM   1. Open "Intel oneAPI command prompt for Intel 64"
REM   2. Navigate to this directory
REM   3. Run: build_fds_amgx.bat
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo  FDS + AmgX GPU Solver Build Script
echo ============================================================
echo.

REM Set paths
set "SOURCE_DIR=%~dp0..\Source"
set "AMGX_DIR=%~dp0..\AMGX\build"
set "CUDA_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"

REM Check Intel compiler environment
IF NOT X%SETVARS_COMPLETED% == X1 (
    echo Setting up Intel compiler environment...
    set "ONEAPIDIR=C:\Program Files (x86)\Intel\oneAPI"
    IF DEFINED ONEAPI_ROOT set "ONEAPIDIR=%ONEAPI_ROOT%"
    IF NOT EXIST "!ONEAPIDIR!\setvars.bat" (
        echo ERROR: Intel oneAPI not found at !ONEAPIDIR!
        echo Please install Intel oneAPI or run this from oneAPI command prompt
        pause
        exit /b 1
    )
    call "!ONEAPIDIR!\setvars" intel64
)

REM Verify compilers
echo.
echo Checking compilers...
where ifort >nul 2>&1
if errorlevel 1 (
    echo ERROR: Intel Fortran compiler ^(ifort^) not found
    pause
    exit /b 1
)
echo   Intel Fortran: OK

where cl >nul 2>&1
if errorlevel 1 (
    echo WARNING: Microsoft C compiler ^(cl^) not found
    echo   Will try to use gcc if available
)

REM Check CUDA
if not exist "%CUDA_DIR%\bin\nvcc.exe" (
    echo ERROR: CUDA not found at %CUDA_DIR%
    pause
    exit /b 1
)
echo   CUDA: OK ^(%CUDA_DIR%^)

REM Check AmgX
if not exist "%AMGX_DIR%\libamgx.a" (
    echo ERROR: AmgX library not found at %AMGX_DIR%\libamgx.a
    pause
    exit /b 1
)
echo   AmgX: OK ^(%AMGX_DIR%^)

REM Create build directory
set "BUILD_DIR=%~dp0obj"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
cd /d "%BUILD_DIR%"

echo.
echo ============================================================
echo  Step 1: Compile C Wrapper (amgx_c_wrapper.c)
echo ============================================================
echo.

REM Compile C wrapper with CUDA
set CUDA_INC=-I"%CUDA_DIR%\include" -I"%AMGX_DIR%\..\base\include" -I"%AMGX_DIR%\..\include"
set CUDA_NVML_FLAG=-DWITH_NVML

"%CUDA_DIR%\bin\nvcc" -c -O2 %CUDA_INC% %CUDA_NVML_FLAG% "%SOURCE_DIR%\amgx_c_wrapper.c" -o amgx_c_wrapper.obj
if errorlevel 1 (
    echo ERROR: Failed to compile amgx_c_wrapper.c
    pause
    exit /b 1
)
echo   amgx_c_wrapper.obj: OK

echo.
echo ============================================================
echo  Step 2: Compile Fortran Sources
echo ============================================================
echo.

REM Fortran compiler flags
set FFLAGS=-O2 -DWITH_AMGX -DWITH_MKL
set FFLAGS=%FFLAGS% -I"%MKLROOT%\include"

REM Compile Fortran modules in order
set FORTRAN_FILES=prec cons chem prop devc type data mesh func gsmv smvv rcal turb soot pois geom ccib radi part vege ctrl hvac mass imkl wall fire velo amgx_fortran amgx_validation pres init dump read divg main

for %%f in (%FORTRAN_FILES%) do (
    echo Compiling %%f.f90...
    ifort -c %FFLAGS% "%SOURCE_DIR%\%%f.f90"
    if errorlevel 1 (
        echo ERROR: Failed to compile %%f.f90
        pause
        exit /b 1
    )
)

echo.
echo ============================================================
echo  Step 3: Link Executable
echo ============================================================
echo.

REM Set library paths
set MKL_LIBS="%MKLROOT%\lib\mkl_intel_lp64.lib" "%MKLROOT%\lib\mkl_core.lib"
set MKL_LIBS=%MKL_LIBS% "%MKLROOT%\lib\mkl_intel_thread.lib" "%MKLROOT%\lib\mkl_blacs_intelmpi_lp64.lib"
set CUDA_LIBS="%CUDA_DIR%\lib\x64\cudart.lib" "%CUDA_DIR%\lib\x64\cublas.lib"
set CUDA_LIBS=%CUDA_LIBS% "%CUDA_DIR%\lib\x64\cusparse.lib" "%CUDA_DIR%\lib\x64\cusolver.lib"
set CUDA_LIBS=%CUDA_LIBS% "%CUDA_DIR%\lib\x64\nvml.lib"
set AMGX_LIBS="%AMGX_DIR%\libamgx.a"
set MPI_LIBS="%I_MPI_ROOT%\lib\impi.lib"

REM Link
set OBJ_FILES=*.obj amgx_c_wrapper.obj
set OUTPUT=fds_amgx_win.exe

echo Linking %OUTPUT%...
ifort -o %OUTPUT% %OBJ_FILES% %MKL_LIBS% %CUDA_LIBS% %AMGX_LIBS% %MPI_LIBS% -Qopenmp

if errorlevel 1 (
    echo ERROR: Linking failed
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Build Complete!
echo ============================================================
echo.
echo Executable: %BUILD_DIR%\%OUTPUT%
echo.
echo To run a simulation:
echo   mpiexec -n 1 %OUTPUT% your_input.fds
echo.

pause

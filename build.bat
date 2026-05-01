@echo off
REM ============================================================
REM  build.bat — one-shot build for Windows
REM  Requires: Python 3.11 at the path below + MSVC Build Tools
REM  Install MSVC: https://visualstudio.microsoft.com/visual-cpp-build-tools/
REM ============================================================

SET PY=C:\Users\zhdanovandrei\AppData\Local\Programs\Python\Python311\python.exe

echo [1/2] Building Cython extensions...
%PY% setup.py build_ext --inplace
IF ERRORLEVEL 1 (
    echo.
    echo ERROR: Cython build failed.
    echo Install MSVC Build Tools from:
    echo   https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo Then re-run this script.
    pause
    exit /b 1
)

echo.
echo [2/2] Building C shared library (requires gcc in PATH)...
where gcc >nul 2>&1
IF ERRORLEVEL 1 (
    echo WARNING: gcc not found. Skipping C shared library build.
    echo Install MinGW-w64 and add to PATH, then run:
    echo   gcc -O2 -shared -o src\c_impl\matrix_ops.dll src\c_impl\matrix_ops.c -lm
) ELSE (
    gcc -O2 -shared -o src\c_impl\matrix_ops.dll src\c_impl\matrix_ops.c -lm
    IF ERRORLEVEL 1 (
        echo ERROR: C library build failed.
    ) ELSE (
        echo C library built: src\c_impl\matrix_ops.dll
    )
)

echo.
echo Build complete. Run benchmarks with:
echo   %PY% benchmarks\bench_runner.py --sizes 64 128 256 --repeats 5
pause

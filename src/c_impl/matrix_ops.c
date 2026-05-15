/*
 * matrix_ops.c — pure C matrix operations + Monte Carlo pi estimation.
 *
 * Compile standalone benchmark (outputs JSON):
 *   Windows MSVC (from VS Developer Prompt):
 *     cl /O2 /Fe:src\c_impl\matrix_bench.exe src\c_impl\matrix_ops.c
 *
 *   Linux / macOS:
 *     gcc -O2 -o src/c_impl/matrix_bench src/c_impl/matrix_ops.c -lm
 *
 * Run:
 *   src\c_impl\matrix_bench.exe
 *   -> prints JSON to stdout, parsed by bench_runner.py
 */

#include "matrix_ops.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

/* -------------------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------------- */

double *alloc_matrix(int n, int m) {
    double *p = (double *)calloc((size_t)n * m, sizeof(double));
    if (!p) { fprintf(stderr, "OOM\n"); exit(1); }
    return p;
}

void free_matrix(double *A) { free(A); }

/* Row-major index macro */
#define IDX(A, cols, i, j)  ((A)[(i)*(cols)+(j)])

/* -------------------------------------------------------------------------
 * matmul  C[n x m] = A[n x k] x B[k x m]
 * ---------------------------------------------------------------------- */
void matmul(const double * restrict A,
            const double * restrict B,
            double       * restrict C,
            int n, int k, int m) {
    memset(C, 0, (size_t)n * m * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int p = 0; p < k; p++) {
            double a_ip = IDX(A, k, i, p);
            for (int j = 0; j < m; j++) {
                IDX(C, m, i, j) += a_ip * IDX(B, m, p, j);
            }
        }
    }
}

/* -------------------------------------------------------------------------
 * matrix_add  C = A + B
 * ---------------------------------------------------------------------- */
void matrix_add(const double * restrict A,
                const double * restrict B,
                double       * restrict C,
                int n, int m) {
    int total = n * m;
    for (int i = 0; i < total; i++) C[i] = A[i] + B[i];
}

/* -------------------------------------------------------------------------
 * monte_carlo_pi — estimate pi by throwing random darts
 *
 * Throw n_samples points uniformly into [0,1)^2.
 * Count how many land inside the unit circle (x^2 + y^2 < 1).
 * pi ~ 4 * inside / n_samples
 *
 * Uses rand() — same approach as Cython typed version.
 * All variables are C types on the stack: no malloc, no boxing.
 * ---------------------------------------------------------------------- */
double monte_carlo_pi(int n_samples) {
    int inside = 0;
    double inv = 1.0 / RAND_MAX;
    for (int i = 0; i < n_samples; i++) {
        double x = rand() * inv;
        double y = rand() * inv;
        if (x * x + y * y < 1.0) inside++;
    }
    return 4.0 * inside / n_samples;
}

/* -------------------------------------------------------------------------
 * Standalone benchmark — outputs JSON to stdout
 * Compiled with: cl /O2 /Fe:matrix_bench.exe matrix_ops.c
 *            or: gcc -O2 -o matrix_bench matrix_ops.c -lm
 * ---------------------------------------------------------------------- */

/* High-resolution wall-clock timer */
static double wall_time(void) {
#ifdef _WIN32
    /* Windows: QueryPerformanceCounter gives ~100ns resolution */
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

static void fill_random(double *A, int total) {
    for (int i = 0; i < total; i++)
        A[i] = (double)rand() / RAND_MAX;
}

int main(void) {
    srand(42);
    int sizes[]  = {64, 128};
    int n_sizes  = 2;
    int nruns    = 5;
    int mc_n     = 1000000;

    printf("[\n");
    int first = 1;

    /* ── Matrix operations ─────────────────────────────────────────────── */
    for (int si = 0; si < n_sizes; si++) {
        int N = sizes[si];
        double *A = alloc_matrix(N, N);
        double *B = alloc_matrix(N, N);
        double *C = alloc_matrix(N, N);
        fill_random(A, N * N);
        fill_random(B, N * N);

        /* matmul — best of nruns */
        double best_mm = 1e18;
        for (int r = 0; r < nruns; r++) {
            double t0 = wall_time();
            matmul(A, B, C, N, N, N);
            double dt = wall_time() - t0;
            if (dt < best_mm) best_mm = dt;
        }

        /* matrix_add — best of nruns */
        double best_add = 1e18;
        for (int r = 0; r < nruns; r++) {
            double t0 = wall_time();
            matrix_add(A, B, C, N, N);
            double dt = wall_time() - t0;
            if (dt < best_add) best_add = dt;
        }

        if (!first) printf(",\n"); first = 0;
        printf("  {\"impl\":\"c\",\"N\":%d,\"op\":\"matmul\",\"time_s\":%.9f,\"mem_bytes\":0}",
               N, best_mm);
        printf(",\n  {\"impl\":\"c\",\"N\":%d,\"op\":\"matrix_add\",\"time_s\":%.9f,\"mem_bytes\":0}",
               N, best_add);

        free_matrix(A); free_matrix(B); free_matrix(C);
    }

    /* ── Monte Carlo ───────────────────────────────────────────────────── */
    double best_mc = 1e18;
    for (int r = 0; r < nruns; r++) {
        srand(42);
        double t0 = wall_time();
        monte_carlo_pi(mc_n);
        double dt = wall_time() - t0;
        if (dt < best_mc) best_mc = dt;
    }
    printf(",\n  {\"impl\":\"c\",\"N\":%d,\"op\":\"monte_carlo\",\"time_s\":%.9f,\"mem_bytes\":0}",
           mc_n, best_mc);

    printf("\n]\n");
    return 0;
}

/*
 * matrix_ops.c — pure C matrix operations for performance comparison.
 *
 * Compile standalone:
 *   gcc -O2 -march=native -o matrix_bench matrix_ops.c -lm
 *
 * Or as a shared library (called from Python via ctypes):
 *   gcc -O2 -march=native -shared -fPIC -o matrix_ops.so matrix_ops.c -lm
 */

#include "matrix_ops.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

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
 * matmul  C[n×m] = A[n×k] × B[k×m]
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
 * dot_product
 * ---------------------------------------------------------------------- */
double dot_product(const double * restrict a,
                   const double * restrict b,
                   int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
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
 * transpose  T[m×n] = A[n×m]^T
 * ---------------------------------------------------------------------- */
void transpose(const double * restrict A,
               double       * restrict T,
               int n, int m) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            IDX(T, n, j, i) = IDX(A, m, i, j);
}

/* -------------------------------------------------------------------------
 * frobenius_norm
 * ---------------------------------------------------------------------- */
double frobenius_norm(const double *A, int n, int m) {
    double s = 0.0;
    int total = n * m;
    for (int i = 0; i < total; i++) s += A[i] * A[i];
    return sqrt(s);
}

/* -------------------------------------------------------------------------
 * Standalone benchmark (compiled with -DSTANDALONE)
 * ---------------------------------------------------------------------- */
#ifdef STANDALONE

static double wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void fill_random(double *A, int total) {
    for (int i = 0; i < total; i++)
        A[i] = (double)rand() / RAND_MAX;
}

int main(void) {
    srand(42);
    int sizes[] = {64, 128, 256, 512};
    int nruns = 5;

    printf("%-6s  %-12s  %-12s  %-12s\n",
           "N", "matmul(s)", "add(s)", "norm");

    for (int si = 0; si < 4; si++) {
        int N = sizes[si];
        double *A = alloc_matrix(N, N);
        double *B = alloc_matrix(N, N);
        double *C = alloc_matrix(N, N);
        fill_random(A, N * N);
        fill_random(B, N * N);

        /* matmul */
        double t0 = wall_time();
        for (int r = 0; r < nruns; r++) matmul(A, B, C, N, N, N);
        double t_mm = (wall_time() - t0) / nruns;

        /* add */
        t0 = wall_time();
        for (int r = 0; r < nruns; r++) matrix_add(A, B, C, N, N);
        double t_add = (wall_time() - t0) / nruns;

        double norm = frobenius_norm(A, N, N);

        printf("%-6d  %-12.6f  %-12.6f  %-12.4f\n",
               N, t_mm, t_add, norm);

        free_matrix(A); free_matrix(B); free_matrix(C);
    }
    return 0;
}
#endif /* STANDALONE */

#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

/* Export symbols from DLL on Windows (MSVC) */
#ifdef _WIN32
#  ifdef BUILDING_MATRIX_OPS_DLL
#    define MATRIX_API __declspec(dllexport)
#  else
#    define MATRIX_API __declspec(dllimport)
#  endif
#else
#  define MATRIX_API
#endif

MATRIX_API double *alloc_matrix(int n, int m);
MATRIX_API void    free_matrix(double *A);

MATRIX_API void   matmul(const double *A, const double *B, double *C, int n, int k, int m);
MATRIX_API double dot_product(const double *a, const double *b, int n);
MATRIX_API void   matrix_add(const double *A, const double *B, double *C, int n, int m);
MATRIX_API void   transpose(const double *A, double *T, int n, int m);
MATRIX_API double frobenius_norm(const double *A, int n, int m);

#endif /* MATRIX_OPS_H */

#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

double *alloc_matrix(int n, int m);
void    free_matrix(double *A);

void   matmul(const double *A, const double *B, double *C, int n, int k, int m);
void   matrix_add(const double *A, const double *B, double *C, int n, int m);
double monte_carlo_pi(int n_samples);

#endif /* MATRIX_OPS_H */

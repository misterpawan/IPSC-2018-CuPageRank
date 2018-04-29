#ifndef ARNOLDI_H
#define ARNOLDI_H
#include "matrix.h"
void plainArnoldi(Matrix *QMat, Matrix *HMat,Matrix *A, double *q, int k);
void parallelArnoldi(double* d_A, double* d_q, int k, double* d_Q, double* d_H, int n, double* z, double* temp);
#endif

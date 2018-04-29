/*
    This file calls the filereader function to read the a file into a matrix and then transfers data to GPU memory and runs arnoldi and svd in pipeline until convergence.
    More details about the control flow is described in the powerpoint presentation.
*/

#include <stdio.h>
#include <stdlib.h>
#include "arnoldi.h"
#include "matrix.h"
#include <math.h>
#include <string.h>
#include "filereader.h"
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <pthread.h>

struct ArnoldiArgs
{
    double *d_q1;
    double *d_H1;
    double *d_Q;
    double *temp, *d_z;
    double *d_A, *h_A, *h_q;
    double *d_v;
    int k, n;
};

struct SVDArgs
{
    cusolverDnHandle_t cusolverH;
    int rows;
    int cols;
    int lda;

    double *d_H1;
    double *d_S;
    double *d_U;
    double *d_VT;

    int *devInfo;
    double *d_work;
    double *d_rwork;
    int lwork;
    int n;

    double *d_q1, *d_Q, *d_v;
};


__global__ void check(double* d_norm, int n, double* val)
{
    double temp = 0;
    double f ;
    for(int i=0; i<n; i++)
    {
        f = d_norm[i];
        temp += f*f;
    }
    *val = sqrt(temp);
}


__global__ void calc_norm(double* d_A, double* d_q, int n, double* d_norm)
{
    int id = (blockIdx.x*blockDim.x) + threadIdx.x;
    if(id<n)
    {
        double temp = 0;
        int c = id*n;
        for(int i=0; i<n; i++, c++) temp += d_A[c]*d_q[i];
        d_norm[id] = temp - d_q[id];
    }
}


__global__ void saxpy(double *Q, double* v ,double* q, int n, int k)
{
    int id = (blockIdx.x*blockDim.x) + threadIdx.x;
    if(id < n){
        double temp = 0;
        int i = id*(k+1), j = i+k , count=-1;

        for(; i<j; i++) temp += Q[i]*v[++count];
        q[id] = temp;
        }
}


void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++)
        {
            for(int col = 0 ; col < n ; col++)
                {
                    double Areg = A[row + col*lda];
                    printf(" %f ", Areg);
                }
            printf("\n");
        }
}


__global__ void get_v(double* d_VT, double* d_v, int k)
{
    int id = threadIdx.x;
    d_v[id] = d_VT[id*k +k-1+id];
}


__global__ void initCudaMat(double* A, int row, int col)
{
    for(int i=0; i<row*col; i++) A[i] = 0;
}


__global__ void printMatt(double* A, int row, int col)
{
    int z;
    for(z=0; z< (row*col); z++)
    {
        printf("%f  ", A[z]);
        if((z+1)%col == 0) printf("\n");
    }
}


void *threadArnoldi(void *vargp)
{
    struct ArnoldiArgs *args = (struct ArnoldiArgs *)vargp;
    parallelArnoldi(args->d_A, args->d_q1, args->k, args->d_Q, args->d_H1, args->n, args->d_z, args->temp);
    return NULL;
}


void *threadSVD(void *vargp)
{
        struct SVDArgs *args = (struct SVDArgs *)vargp;
        signed char jobu = 'A';
        signed char jobvt = 'A';
        cusolverDnDgesvd (
        args->cusolverH,
        jobu,
        jobvt,
        args->rows,
        args->cols,
        args->d_H1,
        args->lda,
        args->d_S,
        args->d_U,
        args->lda,
        args->d_VT,
        args->lda,
        args->d_work,
        args->lwork,
        args->d_rwork,
        args->devInfo);

        cudaDeviceSynchronize();

        get_v<<<1, args->cols>>>(args->d_VT, args->d_v, args->cols);
        saxpy<<<(args->n)/1024 + 1, 1024>>>(args->d_Q, args->d_v, args->d_q1, args->n, args->cols);

    return NULL;
}


int main()
{
    cudaError_t err = cudaSuccess;
    int k = 16;
    int n;
    double tempNorm;

    Matrix A;

    double *d_norm;
    double *d_q1;
    double *d_H1;
    double *d_Q;
    double *temp, *d_z;
    double *d_A, *h_A, *h_q;
    double *d_v;
    double *d_H2, *d_q2, *d_Q2;
    n = 50000;
    printf("Starting malloc\n");
    cudaMalloc((void**) &d_A, sizeof(double)*n*n);
    if(d_A == NULL){
        printf("Cuda mai memero nahi mila :'(");
    }
    else{
        printf("mila\n");
        exit(0);
    }
    printf("Ending malloc");
    readInput(&A,"ip3.txt",0.85);
    n = A.A_rows;

    matrixTranspose(&A);
    double **AMat = A.A;
    h_A = (double*)malloc(sizeof(double)*n*n);
    h_q = (double*)malloc(sizeof(double)*n);

    double val = (1.0/sqrt(n));
    for(int i=0, c=0; i<n; i++)
    {
        for(int j=0; j<n; j++, c++) h_A[c] = AMat[i][j];
        h_q[i] = val;
    }

    cusolverDnHandle_t cusolverH = NULL;

    const int rows = k+1;
    const int cols = k;
    const int lda = rows;

    double *d_S = NULL;
    double *d_U = NULL;
    double *d_VT = NULL;
    double *d_work = NULL;
    double *d_rwork = NULL;
    int *devInfo = NULL;
    int lwork = 0;

    cudaMalloc ((void**)&d_S , sizeof(double)*cols);
    cudaMalloc ((void**)&d_U , sizeof(double)*lda*rows);
    cudaMalloc ((void**)&d_VT , sizeof(double)*lda*cols);
    cudaMalloc ((void**)&devInfo, sizeof(int));

    cusolverDnCreate(&cusolverH);
    cusolverDnDgesvd_bufferSize( cusolverH, rows, cols, &lwork );
    cudaMalloc((void**)&d_work , sizeof(double)*lwork);

    cudaMalloc((void**) &d_norm, sizeof(double)*n);
    cudaMalloc((void**) &d_A, sizeof(double)*n*n);
    cudaMalloc((void**) &d_q1, sizeof(double)*n);
    cudaMalloc((void**) &d_q2, sizeof(double)*n);
    cudaMalloc((void**) &d_H1, sizeof(double)*k*(k+1));
    cudaMalloc((void**) &d_H2, sizeof(double)*k*(k+1));
    cudaMalloc((void**) &d_Q, sizeof(double)*n*(k+1));
    cudaMalloc((void**) &d_Q2, sizeof(double)*n*(k+1));
    cudaMalloc((void**) &temp, sizeof(double));
    cudaMalloc((void**) &d_z, sizeof(double)*n);
    cudaMalloc((void**) &d_v, sizeof(double)*k);

    cudaMemcpy(d_A, h_A, n*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q1, h_q, n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q2, h_q, n*sizeof(double), cudaMemcpyHostToDevice);

    initCudaMat<<<1, 1>>>(d_H1, k+1, k);
    initCudaMat<<<1, 1>>>(d_Q, n, k+1);
    initCudaMat<<<1, 1>>>(d_z, n, 1);

    //thread
    pthread_t tid1, tid2;
    //thread


    ArnoldiArgs *arnoldiArgs = (struct ArnoldiArgs *)malloc(sizeof(struct ArnoldiArgs));
    SVDArgs *svdArgs = (struct SVDArgs *)malloc(sizeof(struct SVDArgs));

	arnoldiArgs->d_A = d_A;
	arnoldiArgs->d_q1 = d_q1;
	arnoldiArgs->k = k;
	arnoldiArgs->d_Q = d_Q;
	arnoldiArgs->d_H1 = d_H1;
	arnoldiArgs->n = n;
	arnoldiArgs->d_z = d_z;
	arnoldiArgs->temp = temp;


	svdArgs->cusolverH = cusolverH;
	svdArgs->rows = rows;
	svdArgs->cols = cols;
	svdArgs->lda = lda;
	svdArgs->d_H1 = d_H2;
	svdArgs->d_S = d_S;
	svdArgs->d_U = d_U;
	svdArgs->d_VT = d_VT;
	svdArgs->devInfo = devInfo;
	svdArgs->d_work = d_work;
	svdArgs->d_rwork = d_rwork;
	svdArgs->lwork = lwork;
	svdArgs->n = n;
	svdArgs->d_q1 = d_q2;
	svdArgs->d_Q = d_Q2;
	svdArgs->d_v = d_v;

    for(int fold = 0; fold < 100; fold++)
    {
        //printf("Fold = %d\n\n", fold);

        clock_t begin = clock();
        //#############  Calling Arnoldi to get Q and H  ##################

	    pthread_create(&tid1, NULL, threadArnoldi, (void *)arnoldiArgs);

        if(fold > 0) pthread_create(&tid2, NULL, threadSVD, (void *)svdArgs);


	    pthread_join(tid1, NULL);
	    pthread_join(tid2, NULL);

        // update q

    	double *tempH1 = arnoldiArgs->d_H1;
    	arnoldiArgs->d_H1 = svdArgs->d_H1;
    	svdArgs->d_H1 = tempH1;

    	double *tempq1 = arnoldiArgs->d_q1;
    	arnoldiArgs->d_q1 = svdArgs->d_q1;
    	svdArgs->d_q1 = tempq1;

    	double *tempQ = arnoldiArgs->d_Q;
    	arnoldiArgs->d_Q = svdArgs->d_Q;
    	svdArgs->d_Q = tempQ;

        calc_norm<<<n/1024 + 1, 1024>>>(d_A, arnoldiArgs->d_q1, n, d_norm);
        check<<<1, 1>>>(d_norm, n, temp);
        cudaMemcpy(&tempNorm, temp, sizeof(double), cudaMemcpyDeviceToHost);

	    printf("Aq - q norm for iteration %d is %.15lf\n", fold, tempNorm);

        clock_t end = clock();

        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        printf("Time for this iteration = %lf\n", elapsed_secs);

        cudaMemcpy(h_q , d_q1 , sizeof(double)*n, cudaMemcpyDeviceToHost);

        if (err != cudaSuccess) printf("Error above: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_norm);
    cudaFree(d_A);
    cudaFree(d_H1);
    cudaFree(d_H2);
    cudaFree(d_Q);
    cudaFree(d_Q2);
    cudaFree(d_q1);
    cudaFree(d_q2);
    cudaFree(d_z);
    cudaFree(d_v);
    cudaFree(d_rwork);
    cudaFree(d_S);
    cudaFree(d_U);
    cudaFree(d_VT);
    cudaFree(d_work);

    // end fold
    int x = 1;
    if(h_q[0] < 0) x = -1;
    //printf("Eigenvector : \n");
    for(int i=0;i < n;i++){
           //printf("%lf\n",x*h_q[i]);
        }

    free(A.A[0]);
    free(A.A);
    free(h_A);
    free(h_q);
    return 0;
}

/*
* The names of the kernels are fairly self explanatory
* about the task that they are performing. You
* may refer to the Arnoldi algorithm on page 4 of the
* attached paper to understand the flow.
*/


#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>


__global__ void assign_Q(double* d_Q, double* d_q, int j, int k, int n)
{
    int id = (blockIdx.x*blockDim.x) + threadIdx.x;
    if(id<n)
    {
        d_Q[j+(id*k)] = d_q[id];
    }
}


__global__ void update_q(double* d_q, double* d_Q, int i, int col, int n)
{
    int id = (blockIdx.x*blockDim.x) + threadIdx.x;
    if(id<n)
    {
        d_q[id] = d_Q[id*col + i];
    }
}


__global__ void update_z(double* d_z, double*d_q, double* temp, int n)
{
        __shared__ double t;
    if(threadIdx.x == 0) t = *temp;
    __syncthreads();

    int id = (blockIdx.x*blockDim.x) + threadIdx.x;
    if(id<n)
    {
        d_z[id] -= t * d_q[id];
    }
}


__global__ void dot_prod_assign_H(double* a, double*b, int n, double* temp, double* d_H, int i, int j, int k)
{
    double t=0;
    for(int i=0; i<n; i++) t += a[i]*b[i];
    if(i==j) d_H[i + j*(k+1)] = t-1;
    else d_H[i + j*(k+1)] = t;
    *temp = t;
}


__global__ void saxpy(double *A, int n, double* z, double* q)
{
    int id = (blockIdx.x*blockDim.x) + threadIdx.x;
    if(id < n){
        double temp = 0;
        int i = id*n, j = i+n , count=-1;

        for(; i<j; i++) temp += A[i]*q[++count];
        z[id] = temp;
        }
}


__global__ void normalize_1(double* q, double* temp, int n, double* d_H, int loc)
{
    double sum = 0;
    for(int i=0; i<n; i++)
    {
        double f = q[i];
        sum += f*f;
    }
*temp = sqrt(sum);
d_H[loc] = *temp;
}


__global__ void normalize_2(double* q, double* temp, int n)
{
    int id = (blockIdx.x*blockDim.x) + threadIdx.x;
    if(id<n){
    q[id] /= *temp ;
    }
}


void normalize_assign_H(double *q, double* temp, int n, double* d_H, int loc)
{
    normalize_1<<<1, 1>>>(q, temp, n, d_H, loc);

    normalize_2<<<n/1024 + 1, 1024>>>(q, temp, n);
}


__global__ void printMat(double* A, int row, int col)
{
    printf("Printing matrix\n");
    int z;
    for(z=0; z< (row*col); z++)
    {
        printf("%f  ", A[z]);
        if((z+1)%col == 0) printf("\n");
    }
}


void parallelArnoldi(double* d_A, double* d_q, int k, double* d_Q, double* d_H, int n, double* d_z, double* temp)
{
    cudaError_t err = cudaSuccess;

    assign_Q<<<n/1024 + 1, 1024>>>(d_Q, d_q, 0, k+1, n);

    for(int j=0; j<k; j++)
    {
        saxpy<<<n/1024 + 1, 1024>>>(d_A, n, d_z, d_q);

        for(int i=0; i<=j; i++)
        {
            update_q<<<n/1024 + 1, 1024>>>(d_q, d_Q, i, k+1, n);
            dot_prod_assign_H<<< 1, 1>>>(d_q, d_z, n, temp, d_H, i, j, k);
            update_z<<<n/1024 + 1, 1024>>>(d_z, d_q, temp, n);
        }

        normalize_assign_H(d_z, temp, n, d_H, j+1 + j*(k+1));

        cudaMemcpy(d_q, d_z, sizeof(double)*n, cudaMemcpyDeviceToDevice);
        assign_Q<<<n/1024 + 1, 1024>>>(d_Q, d_q, j+1, k+1, n);
    }
    if (err != cudaSuccess) printf("Error above: %s\n", cudaGetErrorString(err));
     /*
        printf("H\n");
        printMat<<<1, 1>>>(d_H, k+1, k);
        cudaDeviceSynchronize();

        printf("Q\n");
        printMat<<<1, 1>>>(d_Q, n, k+1);
        cudaDeviceSynchronize();
    */
}

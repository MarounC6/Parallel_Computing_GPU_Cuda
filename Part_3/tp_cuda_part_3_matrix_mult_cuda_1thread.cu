/*
**  PROGRAM: Matrix Multiply - CUDA Version (1 thread per block)
**
**  PURPOSE: This is a simple matrix multiply program using CUDA
**           It will compute the product
**
**                C  = A * B
**
**           A and B are set to constant matrices so we
**           can make a quick test of the multiplication.
**
**  Q3.1: Port the code to CUDA (1 thread per block)
**
**  HISTORY: Written by Tim Mattson, Nov 1999.
*            Modified and extended by Jonathan Rouzaud-Cornabas, Oct 2022
*/

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define AVAL 3.14
#define BVAL 5.42
#define TOL  0.001

// CUDA Kernel: Each block computes one element of C
// 1 thread per block (blockDim.x = 1)
__global__
void matrixMultKernel(double* A, double* B, double* C, int Ndim, int Pdim, int Mdim)
{
    // With 1 thread per block, we use blockIdx to identify which element to compute
    int row = blockIdx.y;
    int col = blockIdx.x;
    
    if (row < Ndim && col < Mdim) {
        double tmp = 0.0;
        for (int k = 0; k < Pdim; k++) {
            // C(row,col) = sum(over k) A(row,k) * B(k,col)
            tmp += A[row * Ndim + k] * B[k * Pdim + col];
        }
        C[row * Ndim + col] = tmp;
    }
}

void matrixMultCuda(double* A, double* B, double* C, int Ndim, int Pdim, int Mdim)
{
    // Allocate device memory
    double *A_gpu, *B_gpu, *C_gpu;
    
    cudaError_t err = cudaMalloc((void**)&A_gpu, Ndim * Pdim * sizeof(double));
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(1);
    }
    
    err = cudaMalloc((void**)&B_gpu, Pdim * Mdim * sizeof(double));
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(1);
    }
    
    err = cudaMalloc((void**)&C_gpu, Ndim * Mdim * sizeof(double));
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(1);
    }
    
    // Copy data from host to device
    cudaMemcpy(A_gpu, A, Ndim * Pdim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, Pdim * Mdim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(C_gpu, C, Ndim * Mdim * sizeof(double), cudaMemcpyHostToDevice);
    
    // Launch kernel with 1 thread per block
    // Grid dimensions: Mdim x Ndim blocks (one block per output element)
    // Block dimensions: 1 x 1 threads
    dim3 gridDim(Mdim, Ndim);
    dim3 blockDim(1, 1);
    
    matrixMultKernel<<<gridDim, blockDim>>>(A_gpu, B_gpu, C_gpu, Ndim, Pdim, Mdim);
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(C, C_gpu, Ndim * Mdim * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
}

int main(int argc, char **argv)
{
    int Ndim = 1000, Pdim = 1000, Mdim = 1000;   /* A[N][P], B[P][M], C[N][M] */
    int i, j, k;
    double *A, *B, *C, cval, tmp, err, errsq;
    double dN, dM, dP, mflops;

    // Read command line arguments.
    for (int i = 0; i < argc; i++) {
        if ((strcmp(argv[i], "-N") == 0)) {
            Ndim = atoi(argv[++i]);
            printf("  User N is %d\n", Ndim);
        } else if ((strcmp(argv[i], "-M") == 0)) {
            Mdim = atoi(argv[++i]);
            printf("  User M is %d\n", Mdim);
        } else if ((strcmp(argv[i], "-P") == 0)) {
            Pdim = atoi(argv[++i]);
            printf("  User P is %d\n", Pdim);
        } else if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "-help") == 0)) {
            printf("  Matrix multiplication Options:\n");
            printf("  -N <int>:              Size of the dimension N (by default 1000)\n");
            printf("  -M <int>:              Size of the dimension M (by default 1000)\n");
            printf("  -P <int>:              Size of the dimension P (by default 1000)\n");
            printf("  -help (-h):            print this message\n\n");
            exit(1);
        }
    }

    A = (double *)malloc(Ndim * Pdim * sizeof(double));
    B = (double *)malloc(Pdim * Mdim * sizeof(double));
    C = (double *)malloc(Ndim * Mdim * sizeof(double));

    /* Initialize matrices */
    for (i = 0; i < Ndim; i++)
        for (j = 0; j < Pdim; j++)
            A[i * Ndim + j] = AVAL;

    for (i = 0; i < Pdim; i++)
        for (j = 0; j < Mdim; j++)
            B[i * Pdim + j] = BVAL;

    for (i = 0; i < Ndim; i++)
        for (j = 0; j < Mdim; j++)
            C[i * Ndim + j] = 0.0;

    /* Do the matrix product using CUDA */
    struct timeval begin, end;

    gettimeofday(&begin, NULL);

    matrixMultCuda(A, B, C, Ndim, Pdim, Mdim);

    gettimeofday(&end, NULL);

    // Calculate time.
    double time = 1.0 * (end.tv_sec - begin.tv_sec) +
                  1.0e-6 * (end.tv_usec - begin.tv_usec);

    printf(" N %d M %d P %d multiplication in %f seconds \n", Ndim, Mdim, Pdim, time);

    dN = (double)Ndim;
    dM = (double)Mdim;
    dP = (double)Pdim;
    mflops = 2.0 * dN * dM * dP / (1000000.0 * time);

    printf(" N %d M %d P %d multiplication at %f mflops\n", Ndim, Mdim, Pdim, mflops);

    /* Check the answer */
    cval = Pdim * AVAL * BVAL;
    errsq = 0.0;
    for (i = 0; i < Ndim; i++) {
        for (j = 0; j < Mdim; j++) {
            err = C[i * Ndim + j] - cval;
            errsq += err * err;
        }
    }

    if (errsq > TOL)
        printf("\n Errors in multiplication: %f", errsq);
    else
        printf("\n Hey, it worked");

    printf("\n all done \n");
    printf(" N %d M %d P %d multiplication in %f seconds \n", Ndim, Mdim, Pdim, time);

    free(A);
    free(B);
    free(C);

    return 0;
}

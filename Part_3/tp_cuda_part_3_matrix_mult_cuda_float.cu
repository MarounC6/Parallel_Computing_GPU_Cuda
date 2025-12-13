/*
**  PROGRAM: Matrix Multiply - CUDA Shared Memory Version (FLOAT precision)
**
**  PURPOSE: This is a matrix multiply program using CUDA with:
**           - Multiple threads per block
**           - Shared memory for optimization
**           - FLOAT precision (instead of double)
**
**                C  = A * B
**
**  Q3.9: Modify the code to change the type of the matrix from double to float
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

#define AVAL 3.14f
#define BVAL 5.42f
#define TOL  0.001f
#define TILE_SIZE 16  // Tile size for shared memory

// CUDA Kernel: Multiple threads per block with shared memory (FLOAT)
__global__
void matrixMultSharedKernel(float* A, float* B, float* C, int Ndim, int Pdim, int Mdim)
{
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Calculate row and column index of C element
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    // Shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    float tmp = 0.0f;
    
    // Loop over tiles of A and B required to compute C element
    int numTiles = (Pdim + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        // Load tile of A into shared memory
        int a_col = t * TILE_SIZE + tx;
        if (row < Ndim && a_col < Pdim) {
            As[ty][tx] = A[row * Ndim + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile of B into shared memory
        int b_row = t * TILE_SIZE + ty;
        if (b_row < Pdim && col < Mdim) {
            Bs[ty][tx] = B[b_row * Pdim + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        // Synchronize to ensure tiles are loaded
        __syncthreads();
        
        // Compute partial dot product for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            tmp += As[ty][k] * Bs[k][tx];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result to C
    if (row < Ndim && col < Mdim) {
        C[row * Ndim + col] = tmp;
    }
}

void matrixMultCuda(float* A, float* B, float* C, int Ndim, int Pdim, int Mdim)
{
    // Allocate device memory
    float *A_gpu, *B_gpu, *C_gpu;
    
    cudaError_t err = cudaMalloc((void**)&A_gpu, Ndim * Pdim * sizeof(float));
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(1);
    }
    
    err = cudaMalloc((void**)&B_gpu, Pdim * Mdim * sizeof(float));
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(1);
    }
    
    err = cudaMalloc((void**)&C_gpu, Ndim * Mdim * sizeof(float));
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(1);
    }
    
    // Copy data from host to device
    cudaMemcpy(A_gpu, A, Ndim * Pdim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, Pdim * Mdim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_gpu, C, Ndim * Mdim * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel with multiple threads per block
    dim3 gridDim((Mdim + TILE_SIZE - 1) / TILE_SIZE, (Ndim + TILE_SIZE - 1) / TILE_SIZE);
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    
    matrixMultSharedKernel<<<gridDim, blockDim>>>(A_gpu, B_gpu, C_gpu, Ndim, Pdim, Mdim);
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(C, C_gpu, Ndim * Mdim * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
}

int main(int argc, char **argv)
{
    int Ndim = 1000, Pdim = 1000, Mdim = 1000;   /* A[N][P], B[P][M], C[N][M] */
    int i, j, k;
    float *A, *B, *C, cval, tmp, err, errsq;
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

    A = (float *)malloc(Ndim * Pdim * sizeof(float));
    B = (float *)malloc(Pdim * Mdim * sizeof(float));
    C = (float *)malloc(Ndim * Mdim * sizeof(float));

    /* Initialize matrices */
    for (i = 0; i < Ndim; i++)
        for (j = 0; j < Pdim; j++)
            A[i * Ndim + j] = AVAL;

    for (i = 0; i < Pdim; i++)
        for (j = 0; j < Mdim; j++)
            B[i * Pdim + j] = BVAL;

    for (i = 0; i < Ndim; i++)
        for (j = 0; j < Mdim; j++)
            C[i * Ndim + j] = 0.0f;

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
    errsq = 0.0f;
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

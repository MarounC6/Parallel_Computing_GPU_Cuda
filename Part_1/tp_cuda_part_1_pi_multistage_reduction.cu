/*

This program will numerically compute the integral of

                  4/(1+x*x) 
				  
from 0 to 1.  The value of this integral is pi -- which 
is great since it gives us an easy way to check the answer.

History: Written by Tim Mattson, 11/1999.
         Modified/extended by Jonathan Rouzaud-Cornabas, 10/2022
*/

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


static long num_steps = 1000000;
static int threadsPerBlock = 256;
double step;

__global__
void computeFirstTableKernel(float* block_sums, float step, int num_steps)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < num_steps) {
        float val = (i + 1 - 0.5) * step;
        block_sums[i] = 4.0 / (1.0 + val * val);  // Calcul de la valeur de pi pour ce thread
    }
}

__global__
void reduceKernel(int* size, float *tableToReduce)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float shared_tab[];

    if(i<*size) {
        shared_tab[i] = tableToReduce[i];
    }
    __syncthreads();

    if(i<*size && i%2==0) {
        if(i+1 != *size) {
            tableToReduce[i/2] = shared_tab[i] + shared_tab[i+1];
        } else {
            tableToReduce[i/2] = shared_tab[i];
        }
    }
    __syncthreads();
    if(threadIdx.x == 0) {
        *size = (*size + 1) / 2;
    }
}



void computePi(float* sum, int num_steps, float step, int threadsPerBlock)
{
    int size = num_steps * sizeof(float);
    
    // Allouer de la mémoire sur le GPU pour stocker les résultats de chaque thread
    float* block_sums_gpu;
    cudaError_t err1 = cudaMalloc((void**)&block_sums_gpu, size);
    if (err1 != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err1), __FILE__, __LINE__);
        exit(1);
    }

    // Lancer le kernel pour calculer les sommes des blocs
    int blocks = (num_steps + threadsPerBlock - 1) / threadsPerBlock;  // Calcul du nombre de blocs nécessaires
    
    computeFirstTableKernel<<<blocks, threadsPerBlock>>>(block_sums_gpu, step, num_steps);
    
    // Attendre que le calcul soit terminé
    cudaDeviceSynchronize();

    // Récupérer les résultats dans le tableau 'sum' sur le GPU
    float* block_sums_host = (float*)malloc(size);
    cudaMemcpy(block_sums_host, block_sums_gpu, size, cudaMemcpyDeviceToHost);
    

















    int *size_current_table = (int*)malloc(sizeof(int));
    *size_current_table = num_steps;

    while (*size_current_table > 1) {
        blocks = (*size_current_table + threadsPerBlock - 1) / (2 * threadsPerBlock);
        int shared_mem_size = *size_current_table * sizeof(float) * 2;
        reduceKernel<<<blocks, threadsPerBlock, shared_mem_size>>>(size_current_table, block_sums_gpu);
    }

    cudaMemcpy ( block_sums_host , block_sums_gpu , size , cudaMemcpyDeviceToHost );
    cudaFree ( block_sums_gpu ) ;
    *sum = block_sums_host[0];
    free(block_sums_host);
    free(size_current_table);
}


int main (int argc, char** argv)
{
    
      // Read command line arguments.
      for ( int i = 0; i < argc; i++ ) {
        if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-num_steps" ) == 0 ) ) {
            num_steps = atol( argv[ ++i ] );
            printf( "  User num_steps is %ld\n", num_steps );
        } else if( ( strcmp( argv[ i ], "-T" ) == 0 ) || ( strcmp( argv[ i ], "-threadsPerBlock" ) == 0 ) ) {
            threadsPerBlock = atoi(argv[++i]);
            printf( "  User threadsPerBlock is %d\n", threadsPerBlock );
        } else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
            printf( "  Pi Options:\n" );
            printf( "  -num_steps (-N) <int>:      Number of steps to compute Pi (by default 100000000)\n" );
            printf( "  -help (-h):            print this message\n\n" );
            exit( 1 );
        }
      }
      
	  float pi = 0.0;

      float *sum = (float*)malloc(sizeof(float));
      *sum = 0.0;
	  
      step = 1.0/(float) num_steps;

      // Timer products.
      struct timeval begin, end;

      gettimeofday( &begin, NULL );
      
	  computePi(sum, num_steps, step, threadsPerBlock);

	  pi = step * (*sum);

      
      gettimeofday( &end, NULL );

      // Calculate time.
      double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
                1.0e-6 * ( end.tv_usec - begin.tv_usec );
                
      printf("\n pi with %ld steps is %lf in %lf seconds\n ",num_steps,pi,time);

    free(sum);
    return 0;
}

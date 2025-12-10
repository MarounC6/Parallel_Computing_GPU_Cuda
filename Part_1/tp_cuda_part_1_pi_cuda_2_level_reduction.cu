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
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


static long num_steps = 1000000;
static int threadsPerBlock = 12;
double step;

__global__
void computePiKernel(float* sum, float step, int num_steps)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Vérifie si le thread est dans les limites
    if (i < num_steps) {
        float val = (i + 1 - 0.5) * step;
        float y = 4.0/(1.0+val*val);
        atomicAdd(sum, y);
    }
}


void computePi ( float *sum, int num_steps, float step, int threadsPerBlock )
{
    int size = 1 * sizeof ( float );
    float * sum_gpu;
    cudaError_t err = cudaMalloc( ( void ** ) &sum_gpu , size );
    if ( err != cudaSuccess ) {
        printf ( "%s in %s at line %d\n" ,
        cudaGetErrorString ( err ) , __FILE__ , __LINE__ ) ;
        exit ( 1 ) ;
    }
    cudaMemcpy ( sum_gpu , sum , size , cudaMemcpyHostToDevice );

    int blocks = (num_steps + threadsPerBlock - 1) / threadsPerBlock;  // Calcul du nombre de blocs nécessaires

    computePiKernel<<<blocks, threadsPerBlock>>>( sum_gpu, step, num_steps);

    cudaMemcpy ( sum , sum_gpu , size , cudaMemcpyDeviceToHost );
    cudaFree ( sum_gpu ) ;
}



int main (int argc, char** argv)
{
    
      // Read command line arguments.
      for ( int i = 0; i < argc; i++ ) {
        if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-num_steps" ) == 0 ) ) {
            num_steps = atol( argv[ ++i ] );
            printf( "  User num_steps is %ld\n", num_steps );
        } else if ( ( strcmp( argv[ i ], "-T" ) == 0 ) || ( strcmp( argv[ i ], "-threadsPerBlock" ) == 0 ) ) {
            threadsPerBlock = atoi( argv[ ++i ] );
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

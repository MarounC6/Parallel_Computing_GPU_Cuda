#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

static long num_steps = 1000000;
static int threadsPerBlock = 256;
double step;

__global__
void computePiKernel(float* block_sums, float step, int num_steps)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Vérifie si le thread est dans les limites
    if (i < num_steps) {
        float val = (i + 1 - 0.5) * step;
        block_sums[i] = 4.0 / (1.0 + val * val);  // Calcul de la valeur de pi pour ce thread
    }
}

void computePi(float* sum, int num_steps, float step, int threadsPerBlock)
{
    int size = num_steps * sizeof(float);
    
    // Allouer de la mémoire sur le GPU pour stocker les résultats de chaque thread
    float* block_sums_gpu;
    cudaError_t err = cudaMalloc((void**)&block_sums_gpu, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(1);
    }

    // Lancer le kernel pour calculer les sommes des blocs
    int blocks = (num_steps + threadsPerBlock - 1) / threadsPerBlock;  // Calcul du nombre de blocs nécessaires
    
    computePiKernel<<<blocks, threadsPerBlock>>>(block_sums_gpu, step, num_steps);
    
    // Attendre que le calcul soit terminé
    cudaDeviceSynchronize();

    // Récupérer les résultats dans le tableau 'sum' sur le GPU
    float* block_sums_host = (float*)malloc(size);
    cudaMemcpy(block_sums_host, block_sums_gpu, size, cudaMemcpyDeviceToHost);
    
    // Somme des résultats sur le CPU
    *sum = 0.0;
    for (int i = 0; i < num_steps; i++) {
        *sum += block_sums_host[i];
    }

    // Libérer la mémoire GPU
    cudaFree(block_sums_gpu);
    free(block_sums_host);
}

int main(int argc, char** argv)
{
    // Lire les arguments de la ligne de commande
    for (int i = 0; i < argc; i++) {
        if ((strcmp(argv[i], "-N") == 0) || (strcmp(argv[i], "-num_steps") == 0)) {
            num_steps = atol(argv[++i]);
            printf("  User num_steps is %ld\n", num_steps);
        } else if( (strcmp(argv[i], "-T") == 0) || (strcmp(argv[i], "-threadsPerBlock") == 0)) {
            threadsPerBlock = atoi(argv[++i]);
            printf("  User threadsPerBlock is %d\n", threadsPerBlock);
        } else if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "-help") == 0)) {
            printf("  Pi Options:\n");
            printf("  -num_steps (-N) <int>:      Number of steps to compute Pi (by default 100000000)\n");
            printf("  -help (-h):            print this message\n\n");
            exit(1);
        }
    }

    float pi = 0.0;
    float* sum = (float*)malloc(sizeof(float));
    *sum = 0.0;

    step = 1.0 / (float)num_steps;

    // Timer products
    struct timeval begin, end;

    gettimeofday(&begin, NULL);

    computePi(sum, num_steps, step, threadsPerBlock);

    pi = step * (*sum);

    gettimeofday(&end, NULL);

    // Calcul du temps
    double time = 1.0 * (end.tv_sec - begin.tv_sec) + 1.0e-6 * (end.tv_usec - begin.tv_usec);
    printf("\n Pi avec %ld étapes est %lf en %lf secondes\n ", num_steps, pi, time);

    free(sum);
    return 0;
}

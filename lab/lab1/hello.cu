#include <cstdio>
#include "cuda.h"

__global__ void cudaHello(){
  // Afficher le message Hello World ainsi que blockidx et threadidx depuis chaque thread
  // A FAIRE ...
  printf("Hello World from block %d, thread %d\n", blockIdx.x, threadIdx.x);
  printf("Total number of blocks: %d, Total number of threads/block: %d\n", numBlocks, blockSize);
}

int main() {
  int totalThreads = 64;
  int numBlocks;
  int blockSize;
  
  // Experimenter avec de differents blockSize (nombre de threads par block) pour les puissances de 2
  // tout en gardant le nombre total de threads egale a 64
  // A FAIRE ...
  for (blockSize = 64; blockSize >= 1; blockSize /= 2) {
    numBlocks = totalThreads / blockSize;
    printf("\nLaunching kernel with %d blocks and %d threads per block\n", numBlocks, blockSize);
    cudaHello<<<numBlocks, blockSize>>>(numBlocks, blockSize); 

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
      printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    }

    return 0;
}

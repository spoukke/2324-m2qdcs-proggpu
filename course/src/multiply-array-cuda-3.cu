#include <cstdio>    
#include "cuda.h"    

#define N 1024                     
float A[N];    
float c = 2.0;    

__device__ float dA[N];    

__global__ void multiplyArray(int n, float c)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i <  n)
  dA[i] *= c;
}

int main(int argc, char **argv)
{
  // Initialisation
  for (int i = 0; i < N; i++) { A[i] = i; }
  // Copier le tableau vers le GPU
  cudaMemcpyToSymbol(dA, A, N * sizeof(float), 0,
      cudaMemcpyHostToDevice);
  int blockSize = 128;
  int numBlocks = N / blockSize;
  if (N % blockSize) numBlocks++;
  multiplyArray<<<(numBlocks, blockSize>>>(n, c); 
  // Recopier le tableau multiplie vers le CPU
  cudaMemcpyFromSymbol(A, dA, N * sizeof(float), 0,
      cudaMemcpyDeviceToHost);
  printf("%lf\n", A[2]);
  return 0;
}

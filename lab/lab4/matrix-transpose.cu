/**
  * Kernel 1: Implement a first kernel where each block (using BSXY x BSXY threads) transposes a BSXY x BSXY tile of A, and writes it into the corresponding location in At. Do without using shared memory.
  *
  * Kernel 2: In the second kernel, do the same, but using the shared memory. Each block should load a tile of BSXY x BSXY of A into the shared memory, then perform the transposition using this tile in the shared memory into At. Test the difference in speedup. Test the performance using shared memory without padding and with padding (to avoid shared memory bank conflicts).
  *
  * Kernel 3: In this kernel, perform the transpose in-place on the matrix A (do not use At). A block should be transpose two tiles simultenously to be able to do this.
  *
  */
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <ctime>
#include "cuda.h"
#include <cfloat>

#define BSXY 32

// Kernel 1: Transpose without using shared memory
__global__ void transposeNoShared(float *A, float *At, int N) {
  int x = blockIdx.x * BSXY + threadIdx.x;
  int y = blockIdx.y * BSXY + threadIdx.y;

  if (x < N && y < N) {
    At[y * N + x] = A[x * N + y];
  }
}

// Kernel 2: Transpose using shared memory
__global__ void transposeWithShared(float *A, float *At, int N) {
  __shared__ float tile[BSXY][BSXY];

  int x = blockIdx.x * BSXY + threadIdx.x;
  int y = blockIdx.y * BSXY + threadIdx.y;

  if (x < N && y < N) {
    tile[threadIdx.y][threadIdx.x] = A[y * N + x];
  }

  __syncthreads();

  x = blockIdx.y * BSXY + threadIdx.x; // Transpose block offset
  y = blockIdx.x * BSXY + threadIdx.y;

  if (x < N && y < N) {
    At[y * N + x] = tile[threadIdx.x][threadIdx.y];
  }
}

// Kernel 3: In-place transpose
__global__ void transposeInPlace(float *A, int N) {
  __shared__ float tile[BSXY][BSXY + 1]; // +1 for avoiding bank conflicts

  int blockIdx_x, blockIdx_y;

  if (blockIdx.x > blockIdx.y) {
    blockIdx_x = blockIdx.y;
    blockIdx_y = blockIdx.x;
  } else {
    blockIdx_x = blockIdx.x;
    blockIdx_y = blockIdx.y;
  }

  int x = blockIdx_x * BSXY + threadIdx.x;
  int y = blockIdx_y * BSXY + threadIdx.y;

  if (x < N && y < N) {
    tile[threadIdx.y][threadIdx.x] = A[y * N + x];
  }

  __syncthreads();

  x = blockIdx_y * BSXY + threadIdx.x;
  y = blockIdx_x * BSXY + threadIdx.y;

  if (x < N && y < N) {
    A[y * N + x] = tile[threadIdx.x][threadIdx.y];
  }
}


void transposeCPU(float *A, float *At, int N)
{
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      At[i * N + j] = A[j * N + i];
    }
  }
}

void verifyResult(float *A, float *At, int N) {
  float *At_ref = new float[N * N];

  transposeCPU(A, At_ref, N);

  // Compare each element of the CPU-transposed matrix with the GPU-transposed matrix
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (fabs(At_ref[i * N + j] - At[i * N + j]) > 1e-5) { // Use an epsilon value for floating point comparison
        delete[] At_ref;
        std::cout << "Transpose is incorrect At[" << i << "][" << j << "]" << std::endl;
        return; // Return false if any element does not match
      }
    }
  }

  delete[] At_ref;
  std::cout << "Transpose is correct!" << std::endl;
}

int main()
{
  // Allocate A and At
  // A is an N * N matrix stored by rows, i.e. A(i, j) = A[i * N + j]
  // At is also stored by rows and is the transpose of A, i.e., At(i, j) = A(j, i)
  int N = 1024;
  float *A = (float *) malloc(N * N * sizeof(A[0]));
  float *At = (float *) malloc(N * N * sizeof(At[0]));
  float *A_copy = (float *) malloc(N * N * sizeof(A[0])); // used to verify in-place transpose
  
  // Allocate dA and dAt, and call the corresponding matrix transpose kernel
  // TODO / A FAIRE ...
  float *dA, *dAt;
  cudaMalloc(&dA, N * N * sizeof(float));
  cudaMalloc(&dAt, N * N * sizeof(float));

  for (int j = 0; j < N; j++) { 
    for (int i = 0; i < N; i++) { 
      A[i + j * N] = i + j;
      A_copy[i + j * N] = i + j;
    }
  }
  cudaMemcpy(dA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);

  {
    dim3 dimGrid((N + BSXY - 1) / BSXY, (N + BSXY - 1) / BSXY);
    dim3 dimBlock(BSXY, BSXY);
    transposeNoShared<<<dimGrid, dimBlock>>>(dA, dAt, N);
    cudaMemcpy(At, dAt, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    verifyResult(A, At, N);
    cudaMemset(At, 0, N * N * sizeof(float));
  }
  {
    dim3 dimGrid((N + BSXY - 1) / BSXY, (N + BSXY - 1) / BSXY);
    dim3 dimBlock(BSXY, BSXY);
    transposeWithShared<<<dimGrid, dimBlock>>>(dA, dAt, N);
    cudaMemcpy(At, dAt, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    verifyResult(A, At, N);
    cudaMemset(At, 0, N * N * sizeof(float));
  }
  {
    dim3 dimGrid((N + BSXY - 1) / BSXY, (N + BSXY - 1) / BSXY);
    dim3 dimBlock(BSXY, BSXY);
    transposeInPlace<<<dimGrid, dimBlock>>>(dA, N);
    cudaMemcpy(A, dA, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    verifyResult(A_copy, A, N);
  }

  // Deallocate dA and dAt
  // TODO / A FAIRE ...
  cudaFree(dA);
  cudaFree(dAt);

  // Deallocate A and At
  free(A);
  free(At);
  return 0;
}

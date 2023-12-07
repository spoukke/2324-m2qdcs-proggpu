/**
  * In this exercise, we will implement GPU kernels for computing the average of 9 points on a 2D array.
  * Dans cet exercice, nous implantons un kernel GPU pour un calcul de moyenne de 9 points sur un tableau 2D.
  *
  * Kernel 1: Use 1D grid of blocks (only blockIdx.x), no additional threads (1 thread per block)
  * Kernel 1: Utiliser grille 1D de blocs (seulement blockIdx.x), pas de threads (1 thread par bloc)
  *
  * Kernel 2: Use 2D grid of blocks (blockIdx.x/.y), no additional threads (1 thread per block)
  * Kernel 2: Utiliser grille 2D de blocs (blockIdx.x/.y), pas de threads (1 thread par bloc)
  *
  * Kernel 3: Use 2D grid of blocks and 2D threads (BSXY x BSXY), each thread computing 1 element of Aavg
  * Kernel 3: Utiliser grille 2D de blocs, threads de 2D (BSXY x BSXY), chaque thread calcule 1 element de Aavg
  *
  * Kernel 4: Use 2D grid of blocks and 2D threads, each thread computing 1 element of Aavg, use shared memory. Each block should load BSXY x BSXY elements of A, then compute (BSXY - 2) x (BSXY - 2) elements of Aavg. Borders of tiles loaded by different blocks must overlap to be able to compute all elements of Aavg.
  * Kernel 4: Utiliser grille 2D de blocs, threads de 2D, chaque thread calcule 1 element de Aavg, avec shared memory. Chaque bloc doit lire BSXY x BSXY elements de A, puis calculer avec ceci (BSXY - 2) x (BSXY - 2) elements de Aavg. Les bords des tuiles chargees par de differents blocs doivent chevaucher afin de pouvoir calculer tous les elements de Aavg.
  *
  * Kernel 5: Use 2D grid of blocks and 2D threads, use shared memory, each thread computes KxK elements of Aavg
  * Kernel 5: Utiliser grille 2D de blocs, threads de 2D, avec shared memory et KxK ops par thread
  *
  * For all kernels: Make necessary memory allocations/deallocations and memcpy in the main.
  * Pour tous les kernels: Effectuer les allocations/desallocations et memcpy necessaires dans le main.
  */

#include <iostream>
#include <cstdio>
#include "cuda.h"
#include "omp.h"

#define N 1024
#define K 4
#define BSXY 32

// The matrix is stored by rows, that is A(i, j) = A[i + j * N]. The average should be computed on Aavg array.
// La matrice A est stockee par lignes, a savoir A(i, j) = A[i + j * N]
float *A;
float *Aavg;

float *dA, *dAavg;

__global__ void ninePointAverageKernel1D(const float *A, float *Aavg, int n) {
  int idx = blockIdx.x;
  int row = idx / n;
  int col = idx % n;
    
  if (row > 0 && row < n - 1 && col > 0 && col < n - 1) {
    Aavg[row * n + col] = (
        A[(row - 1) * n + (col - 1)] + A[(row - 1) * n + col] + A[(row - 1) * n + (col + 1)] +
        A[row * n + (col - 1)] + A[row * n + col] + A[row * n + (col + 1)] +
        A[(row + 1) * n + (col - 1)] + A[(row + 1) * n + col] + A[(row + 1) * n + (col + 1)]
    ) / 9.0f;
  }
}

__global__ void ninePointAverageKernel2D(const float *A, float *Aavg, int n) {
  int row = blockIdx.x;
  int col = blockIdx.y;
    
  if (row > 0 && row < n - 1 && col > 0 && col < n - 1) {
    Aavg[row * n + col] = (
      A[(row - 1) * n + (col - 1)] + A[(row - 1) * n + col] + A[(row - 1) * n + (col + 1)] +
      A[row * n + (col - 1)] + A[row * n + col] + A[row * n + (col + 1)] +
      A[(row + 1) * n + (col - 1)] + A[(row + 1) * n + col] + A[(row + 1) * n + (col + 1)]
    ) / 9.0f;
  }
}

__global__ void ninePointAverageKernel2DThreads(const float *A, float *Aavg, int n) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
    
  if (row > 0 && row < n - 1 && col > 0 && col < n - 1) {
    Aavg[row * n + col] = (
      A[(row - 1) * n + (col - 1)] + A[(row - 1) * n + col] + A[(row - 1) * n + (col + 1)] +
      A[row * n + (col - 1)] + A[row * n + col] + A[row * n + (col + 1)] +
      A[(row + 1) * n + (col - 1)] + A[(row + 1) * n + col] + A[(row + 1) * n + (col + 1)]
    ) / 9.0f;
  }
}

__global__ void ninePointAverageKernelSharedMemory(const float *A, float *Aavg, int n) {
  __shared__ float shA[BSXY + 2][BSXY + 2];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.x * (BSXY - 2) + tx;
  int col = blockIdx.y * (BSXY - 2) + ty;

  // Load data into shared memory
  if (row < n && col < n) {
    shA[tx][ty] = A[row * n + col];
  }

  __syncthreads();

  // Compute the 9-point average for the inner (BSXY - 2) x (BSXY - 2) elements
  if (tx > 0 && tx < BSXY - 1 && ty > 0 && ty < BSXY - 1 && row < n - 1 && col < n - 1) {
    Aavg[(row - 1) * n + (col - 1)] = (
      shA[tx - 1][ty - 1] + shA[tx - 1][ty] + shA[tx - 1][ty + 1] +
      shA[tx][ty - 1] + shA[tx][ty] + shA[tx][ty + 1] +
      shA[tx + 1][ty - 1] + shA[tx + 1][ty] + shA[tx + 1][ty + 1]
    ) / 9.0f;
  }
}

__global__ void ninePointAverageKernelKxK(const float *A, float *Aavg, int n) {
  __shared__ float shA[BSXY + 2 * K][BSXY + 2 * K];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.x * BSXY * K + tx * K;
  int col = blockIdx.y * BSXY * K + ty * K;

  // Load KxK elements into shared memory per thread
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      if (row + i < n && col + j < n) {
        shA[tx * K + i][ty * K + j] = A[(row + i) * n + col + j];
      }
    }
  }

  __syncthreads();

  // Compute the 9-point average for KxK elements per thread
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      if (tx * K + i > 0 && tx * K + i < BSXY + K - 1 && ty * K + j > 0 && ty * K + j < BSXY + K - 1 && row + i < n - 1 && col + j < n - 1) {
        Aavg[(row + i - 1) * n + col + j - 1] = (
          shA[tx * K + i - 1][ty * K + j - 1] + shA[tx * K + i - 1][ty * K + j] + shA[tx * K + i - 1][ty * K + j + 1] +
          shA[tx * K + i][ty * K + j - 1] + shA[tx * K + i][ty * K + j] + shA[tx * K + i][ty * K + j + 1] +
          shA[tx * K + i + 1][ty * K + j - 1] + shA[tx * K + i + 1][ty * K + j] + shA[tx * K + i + 1][ty * K + j + 1]
        ) / 9.0f;
      }
    }
  }
}




// Reference CPU implementation
// Code de reference pour le CPU
void ninePointAverageCPU(const float *A, float *Aavg)
{
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
      Aavg[i + j * N] = (A[i - 1 + (j - 1) * N] + A[i - 1 + (j) * N] + A[i - 1 + (j + 1) * N] +
          A[i + (j - 1) * N] + A[i + (j) * N] + A[i + (j + 1) * N] +
<<<<<<< HEAD
          A[i + 1 + (j - 1) * N] + A[i + 1 + (j) * N] + A[i + 1 + (j + 1) * N]) * (1.0 / 9.0);
=======
          A[i + 1 + (j - 1) * N] + A[i + 1 + (j) * N] + A[i + 1 + (j + 1) * N]) / 9.0f;
>>>>>>> f6d9519 (feat: implement all kernels)
    }
  }
}

void verifyResults(const float *AavgGPU, float *AavgCPU, int n) {
  ninePointAverageCPU(A, AavgCPU);
  const float tolerance = 1e-5; // Tolerance for floating-point comparison

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      float diff = fabs(AavgGPU[i * n + j] - AavgCPU[i * n + j]);
      if (diff > tolerance) {
        std::cout << "Mismatch at [" << i << "][" << j << "]: GPU = " 
                  << AavgGPU[i * n + j] << ", CPU = " << AavgCPU[i * n + j] << std::endl;
        return;
      }
    }
  }
  std::cout << "Results verified: All elements match!" << std::endl;
}



int main()
{
  A = (float *) malloc (N * N * sizeof(float));
  Aavg = (float *) malloc (N * N * sizeof(float));

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i + j * N] = (float)i * (float)j;
    }
  }

  float* CPUresult;
  CPUresult = (float *) malloc(N * N * sizeof(float));
  ninePointAverageCPU(A, CPUresult);

  cudaMalloc(&dA, N * N * sizeof(float));
  cudaMalloc(&dAavg, N * N * sizeof(float));

  cudaMemcpy(dA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);

  {
    dim3 dimGrid1D(N * N, 1, 1);
    dim3 dimBlock1D(1, 1, 1);
    ninePointAverageKernel1D<<<dimGrid1D, dimBlock1D>>>(dA, dAavg, N);
    cudaMemcpy(Aavg, dAavg, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "First kernel:" << std::endl;
    verifyResults(CPUresult, Aavg, N);
    std::cout << "\n" << std::endl;

    cudaMemset(dAavg, 0, N * N * sizeof(float));
  }
  {
    dim3 dimGrid2D(N , N, 1);
    dim3 dimBlock2D(1, 1, 1);
    ninePointAverageKernel2D<<<dimGrid2D, dimBlock2D>>>(dA, dAavg, N);
    cudaMemcpy(Aavg, dAavg, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Second kernel:" << std::endl;
    verifyResults(CPUresult, Aavg, N);
    std::cout << "\n" << std::endl;

    cudaMemset(dAavg, 0, N * N * sizeof(float));
  }
  {
    dim3 dimGrid3((N + BSXY - 1) / BSXY, (N + BSXY - 1) / BSXY, 1);
    dim3 dimBlock3(BSXY, BSXY, 1);
    ninePointAverageKernel2DThreads<<<dimGrid3, dimBlock3>>>(dA, dAavg, N);
    cudaMemcpy(Aavg, dAavg, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Third kernel:" << std::endl;
    verifyResults(CPUresult, Aavg, N);
    std::cout << "\n" << std::endl;

    cudaMemset(dAavg, 0, N * N * sizeof(float));
  }
  {
    dim3 dimGrid4((N + BSXY - 3) / (BSXY - 2), (N + BSXY - 3) / (BSXY - 2));
    dim3 dimBlock4(BSXY + 2, BSXY + 2);
    ninePointAverageKernelSharedMemory<<<dimGrid4, dimBlock4>>>(dA, dAavg, N);
    cudaMemcpy(Aavg, dAavg, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Fourth kernel:" << std::endl;
    verifyResults(CPUresult, Aavg, N);
    std::cout << "\n" << std::endl;

    cudaMemset(dAavg, 0, N * N * sizeof(float));
  }
  {
    dim3 dimGrid5((N + BSXY * K - 3) / (BSXY * K), (N + BSXY * K - 3) / (BSXY * K));
    dim3 dimBlock5(BSXY, BSXY);
    ninePointAverageKernelKxK<<<dimGrid5, dimBlock5>>>(dA, dAavg, N);
    cudaMemcpy(Aavg, dAavg, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Fifth kernel:" << std::endl;
    verifyResults(CPUresult, Aavg, N);
  }

  free(A);
  free(Aavg);
  free(CPUresult);

  cudaFree(dA);
  cudaFree(dAavg);

  return 0;
}

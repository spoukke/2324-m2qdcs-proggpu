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


// Reference CPU implementation
// Code de reference pour le CPU
void ninePointAverageCPU(const float *A, float *Aavg)
{
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
      Aavg[i + j * N] = (A[i - 1 + (j - 1) * N] + A[i - 1 + (j) * N] + A[i - 1 + (j + 1) * N] +
          A[i + (j - 1) * N] + A[i + (j) * N] + A[i + (j + 1) * N] +
          A[i + 1 + (j - 1) * N] + A[i + 1 + (j) * N] + A[i + 1 + (j + 1) * N]) * (1.0 / 9.0);
    }
  }
}

void verifyResults(const float *AavgGPU, const float *AavgCPU, int n) {
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
  ninePointAverageCPU(A, CPUresult);

  cudaMalloc(&dA, N * N * sizeof(float));
  cudaMalloc(&dAavg, N * N * sizeof(float));

  cudaMemcpy(dA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);

  {
    dim3 dimGrid1D(N * N);
    ninePointAverageKernel1D<<<dimGrid1D, 1>>>(dA, dAavg, N);

    cudaMemcpy(Aavg, dAavg, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    verifyResults(CPUresult, Aavg, N);
    cudaMemset(dAavg, 0, N * N * sizeof(float));
  }
  {

    cudaMemcpy(Aavg, dAavg, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    verifyResults(CPUresult, Aavg, N);
    cudaMemset(dAavg, 0, N * N * sizeof(float));
  }
  {

    cudaMemcpy(Aavg, dAavg, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    verifyResults(CPUresult, Aavg, N);
    cudaMemset(dAavg, 0, N * N * sizeof(float));
  }
  {

    cudaMemcpy(Aavg, dAavg, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    verifyResults(CPUresult, Aavg, N);
    cudaMemset(dAavg, 0, N * N * sizeof(float));
  }
  {

    cudaMemcpy(Aavg, dAavg, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    verifyResults(CPUresult, Aavg, N);
    cudaMemset(dAavg, 0, N * N * sizeof(float));
  }

  free(A);
  free(Am);

  cudaFree(dA);
  cudaFree(dAavg);

  return 0;
}

/**
  * Compilation: nvcc sgemm-tile-stream.cu -o sgemm-tile-stream -lcublas
  * Execution: ./sgemm-tile-stream
  * 
  * Task 1: Basic CuBLAS execution and benchmarking.
  *   -Allocate and initialize three N * N column-major float matrices A, B, C on the CPU.
  *   -Allocate dA, dB, dC on the GPU.
  *   -Copy contents of A, B to dA, dB
  *   -Execute cublasSgemm(...)
  *   -Copy dC back to C
  *   -Measure and print the total execution time including host-to-device copy, sgemm, and device-to-host copy and flops/s (sgemm performs 2*N*N*(N-1) flops)
  *
  *
  * Task 2: Implementing tiled cublasSgemm with pipelining
  *   -Create one transfer stream for host-to-device transfers and P x P streams for computing each tile of C(pi, pj) for 0 <= pi, pj < P
  *   -Transfer all tiles A(pi, pj) and B(pi, pj) to dA(pi, pj) and dB(pi, pj) in the transfer stream for 0 <= pi, pj < P, and launch an event ea(pi, pj) and eb(pi, pj) for each tile transfer
  *   -Schedule all tile sgemms required to compute dC(pi, pj) into stream(pi, pj), add data dependencies for each operation with event wait. Use cublasSetStream(handle, stream) each time to make sure that sgemm is placed onto the stream(pi, pj).
  *   -Once all sgemms for a tile dC(pi, pj) are completed, copy dC(pi, pj) into the tile C(pi, pj) in the stream (pi, pj).
  *   -Measure and print the total execution time including tile data transfers and sgemm calls, and print flops/s (sgemm performs 2*N*N*(N-1) flops)
  *   -Tune the value of P by experimentation for N=4096.
  */

/** cublasSgemm signature:
  *
  * cublasStatus_t cublasSgemm(
  * cublasHandle_t handle,
  * cublasOperation_t transa,
  * cublasOperation_t transb,
  * int m, int n, int k,
  * const float *alpha,
  * const float *A, int lda,
  * const float *B, int ldb,
  * const float *beta,
  * float *C, int ldc)
  *
  * See https://docs.nvidia.com/cuda/cublas/index.html for details of usage.
  */

/** cudaMemcpy2DAsync signature:
  *
  * cudaError_t cudaMemcpy2DAsync(
  * void* dst,
  * size_t dpitch,
  * const void* src,
  * size_t spitch,
  * size_t width,
  * size_t height,
  * cudaMemcpyKind kind,
  * cudaStream_t stream = 0)
  *
  * See https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ge529b926e8fb574c2666a9a1d58b0dc1 for details of usage.
  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define N 4096
#define P 2

static void simple_sgemm(int n, float alpha, const float *A, const float *B,
    float beta, float *C)
{
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      float prod = 0;
      for (int k = 0; k < n; ++k) { prod += A[k * n + i] * B[j * n + k]; }
      C[j * n + i] = alpha * prod + beta * C[j * n + i];
    }
  }
}

int main(int argc, char **argv) {
  cublasStatus_t status;
  float *A;
  float *B;
  float *C;
  float *C_ref;
  float *d_A = 0;
  float *d_B = 0;
  float *d_C = 0;
  float alpha = 1.0f;
  float beta = 0.0f;
  int n2 = N * N;
  cublasHandle_t handle;

  // CUDA Event creation for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // CUBLAS init
  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUBLAS initialization error!\n");
    return 1;
  }

  // TODO / A FAIRE ...
  A = (float *)malloc(n2 * sizeof(float));
  B = (float *)malloc(n2 * sizeof(float));
  C = (float *)malloc(n2 * sizeof(float));
  C_ref = (float *)malloc(n2 * sizeof(float)); // For validation


  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = i + j; // A(i, j) = i + j
      B[i + j * N] = i - j; // B(i, j) = 1
      C[i + j * N] = 0; // C(i, j) = 0
      C_ref[i + j * N] = 0; // C(i, j) = 0
    }
  }
  // uncomment for verification
  // simple_sgemm(N, alpha, A, B, beta, C_ref);

  // Start timing
  cudaEventRecord(start);

  cudaMalloc((void **)&d_A, n2 * sizeof(float));
  cudaMalloc((void **)&d_B, n2 * sizeof(float));
  cudaMalloc((void **)&d_C, n2 * sizeof(float));

  cudaMemcpy(d_A, A, n2 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, n2 * sizeof(float), cudaMemcpyHostToDevice);


  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
  cudaMemcpy(C, d_C, n2 * sizeof(float), cudaMemcpyDeviceToHost);

  // Stop timing
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Calculate and print execution time and flops/s
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  float flops = 2.0 * N * N * N;
  float flopsPerSecond = flops / (milliseconds / 1000.0);
  printf("Elapsed time task 1: %f ms\n", milliseconds);
  printf("Performance task 1: %f GFlops/s\n", flopsPerSecond / 1e9);

  // uncomment for verification
  // for (int i = 0; i < N; i++) {
  //   for (int j = 0; j < N; j++) {
  //     if (fabs(C[i * N + j] - C_ref[i * N + j]) > 1e-5) {
  //       printf("Mismatch at row %d, column %d: GPU %f, CPU %f\n", i, j, C[i * N + j], C_ref[i * N + j]);
  //       break;
  //     }
  //   }
  // }

  cudaStream_t transferStream;
  cudaStreamCreate(&transferStream);
  cudaStream_t computeStreams[P][P];
  for (int i = 0; i < P; i++) {
    for (int j = 0; j < P; j++) {
      cudaStreamCreate(&computeStreams[i][j]);
    }
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      C[i + j * N] = 0; // C(i, j) = 0
    }
  }

  // Start timing
  cudaEventRecord(start);

  // Allocate memory for tiles and perform host-to-device transfers
  float *d_A_tiles[P][P], *d_B_tiles[P][P], *d_C_tiles[P][P];
  int tileSize = N / P;
  int tileBytes = tileSize * tileSize * sizeof(float);
  
  size_t originalPitch = N * sizeof(float); // Pitch of the original full matrix
  size_t width = tileSize * sizeof(float);  // Width of the tile in bytes
  for (int i = 0; i < P; i++) {
    for (int j = 0; j < P; j++) {
      cudaMalloc((void **)&d_A_tiles[i][j], tileBytes);
      cudaMalloc((void **)&d_B_tiles[i][j], tileBytes);
      cudaMalloc((void **)&d_C_tiles[i][j], tileBytes);

      float* srcA = A + (i * tileSize * N) + (j * tileSize);
      float* srcB = B + (i * tileSize * N) + (j * tileSize);

      cudaMemcpy2DAsync(d_A_tiles[i][j], width, srcA, originalPitch, width, tileSize, cudaMemcpyHostToDevice, transferStream);
      cudaMemcpy2DAsync(d_B_tiles[i][j], width, srcB, originalPitch, width, tileSize, cudaMemcpyHostToDevice, transferStream);
    }
  }

  // Perform tiled matrix multiplication
  for (int pi = 0; pi < P; pi++) {
    for (int pj = 0; pj < P; pj++) {
      // Set the stream for cublas operations
      cublasSetStream(handle, computeStreams[pi][pj]);

      // Call cublasSgemm for each tile
      beta = 0.0f;
      for (int k = 0; k < P; k++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, tileSize, tileSize, tileSize, 
                        &alpha, d_A_tiles[k][pi], tileSize, 
                        d_B_tiles[pj][k], tileSize, &beta, 
                        d_C_tiles[pi][pj], tileSize);
        beta = 1.0f;
      }

      // Copy each tile of d_C back to C asynchronously
      float* dstC = C + (pj * tileSize * N) + (pi * tileSize);
      cudaMemcpy2DAsync(dstC, originalPitch, d_C_tiles[pi][pj], width, width, tileSize, cudaMemcpyDeviceToHost, computeStreams[pi][pj]);
    }
  }

  for (int pi = 0; pi < P; pi++) {
    for (int pj = 0; pj < P; pj++) {
      cudaStreamSynchronize(computeStreams[pi][pj]);
    }
  }


  // Stop timing and synchronize
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

   // Calculate and print execution time and flops/s
  cudaEventElapsedTime(&milliseconds, start, stop);
  flopsPerSecond = flops / (milliseconds / 1000.0);
  printf("Elapsed time task 2: %f ms\n", milliseconds);
  printf("Performance task 2: %f GFlops/s\n", flopsPerSecond / 1e9);

  // uncomment for verification
//   for (int i = 0; i < N; i++) {
//     for (int j = 0; j < N; j++) {
//       if (fabs(C[i * N + j] / C_ref[i * N + j]) < 0.99) {
//         printf("Mismatch at row %d, column %d: GPU %f, CPU %f\n", i, j, C[i * N + j], C_ref[i * N + j]);
//         break;
//       }
//     }
//   }

  // Clean-up: Destroy streams
  cudaStreamDestroy(transferStream);
  for (int i = 0; i < P; i++) {
    for (int j = 0; j < P; j++) {
      cudaStreamDestroy(computeStreams[i][j]);
    }
  }


  // CUBLAS destroy
  status = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUBLAS shutdown error!\n");
    return 1;
  }

  return 0;
}

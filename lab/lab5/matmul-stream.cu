#include <cstdio>
#include <cuda.h>

#define THREADS_PER_BLOCK 256

int N = 1023;
const int nStreams = 4;
float *A, *B, *C;
float *dA, *dB, *dC;
cudaStream_t streams[nStreams];

// Kernel that performs the matrix vector multiplication b(i) = sum_j(A(i, j), x(j))
// A is row-major (stored row-by-row in memory)
// TODO: should be __device__
__global__ void matvec(float *A, float *x, float *b, int n)
{
  // TODO / A FAIRE ...
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    float sum = 0;
    for (int j = 0; j < n; j++) {
      sum += A[i * n + j] * x[j];
    }
    b[i] = sum;
  }
}

void matvecCPU(float *A, float *x, float *b, int n) {
  for (int i = 0; i < n; i++) {
    b[i] = 0;
    for (int j = 0; j < n; j++) {
      b[i] += A[i * n + j] * x[j];
    }
  }
}

bool verifyResult(float *A, float *B, float *C, int n) {
  bool isCorrect = true;
  float *C_ref = (float *)malloc(n * sizeof(float));

  for (int j = 0; j < n; j++) {
    matvecCPU(A, &B[j * n], C_ref, n);

    for (int i = 0; i < n; i++) {
      if (fabs(C_ref[i] - C[i + j * n]) > 1e-5) {
        isCorrect = false;
        break;
      }
    }

    if (!isCorrect) {
      break;
    }
  }

  free(C_ref);
  return isCorrect;
}

int main()
{
  // A is stored by rows, A(i, j) = A[i * N + j]
  A = (float *) malloc (N * N * sizeof(float));
  // B and C are stored by columns, B(i, j) = B[i + j * N]
  B = (float *) malloc (N * N * sizeof(float));
  C = (float *) malloc (N * N * sizeof(float));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = i + j; // A(i, j) = i + j
      B[i + j * N] = i - j; // B(i, j) = i - j
      C[i + j * N] = 0; // C(i, j) = 0
    }
  }
  cudaMalloc(&dA, N * N * sizeof(float));
  cudaMalloc(&dB, N * nStreams * sizeof(float));
  cudaMalloc(&dC, N * nStreams * sizeof(float));

  // Only copy the entire matrix A. For B and C, they need to be copied and computed one column vector at a time in a streaming manner
  cudaMemcpy(dA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);

  // Create streams
  for (int i = 0; i < nStreams; i++) {
    cudaStreamCreate(&streams[i]);
  }

  // Compute the matrix-vector multiplication C(:, j) = A * B(:, j) column-by-column using nStreams streams
  for (int j = 0; j < N; j++) {
    // Copy the column j of B into one of slots in dB using the stream no (j % nStreams) and cudaMemcpyAsync
    // TODO / A FAIRE ...
    cudaMemcpyAsync(&dB[N * (j % nStreams)], &B[j * N], N * sizeof(float), cudaMemcpyHostToDevice, streams[j % nStreams]);

    // Perform the matrix-vector multiplication on A and the column vector in dB(:, j % nStreams), compute on dC(:, j % nStreams), using stream no (j % nStreams)
    // TODO / A FAIRE ...
    matvec<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, streams[j % nStreams]>>>(dA, &dB[N * (j % nStreams)], &dC[N * (j % nStreams)], N);

    // Copy back the computed vector dC(:, j % nStreams) into the column C(:, j) using the same stream no (j % nStreams) and cudaMemcpyAsync
    cudaMemcpyAsync(&C[j * N], &dC[N * (j % nStreams)], N * sizeof(float), cudaMemcpyDeviceToHost, streams[j % nStreams]);
  }
  
  cudaDeviceSynchronize();

  if (verifyResult(A, B, C, N)) {
    printf("GPU computation is correct! \n");
  } else {
    printf("GPU computation is incorrect! \n");
  }

  for (int i = 0; i < nStreams; i++) {
    cudaStreamDestroy(streams[i]);
  }


  free(A); free(B); free(C);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  return 0;
}

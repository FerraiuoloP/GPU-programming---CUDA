// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Include associated header file.
#include "../include/cuda_kernel.cuh"



//Define the kernel function right here
__global__ void VectorAdd(int *A, int *B, int *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) { //check if the thread is within the range of the vector size (because the last block may have more threads than the remaining elements)
        C[idx] = A[idx] + B[idx];
    }
}

/**
 * Wrapper function for the CUDA kernel function.
 */
void vecAddKernelWrap(int *h_A, int *h_B, int *h_C, int N) {
    //create the device pointers
    int *d_A, *d_B, *d_C;

    //allocate device memory. Hint: used cudaMalloc
    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_B, N * sizeof(int));
    cudaMalloc((void**)&d_C, N * sizeof(int));

    //Copy data from host to device. Hint: use cudaMemcpy
    cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice);

    //define the thread dimentions 
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Issue the kernel on the GPU 
    VectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    
    //Copy the computed results from device to host
    cudaMemcpy(h_C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    //free the device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

//kernel function for matrix multiplication
__global__ void MatrixMul(int *A, int *B, int *C, int M, int K, int N) {
    //from MxK and KxN, we get MxN matrix
    //row of C
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    //column of C
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (row < M && col < N) {
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }

}




void matrixMulKernelWrap(int *h_A, int *h_B, int *h_C, int N, int M, int K) {
    //matrix A is MxK, matrix B is KxN, and matrix C is MxN

    //create the device pointers
    int *d_A, *d_B, *d_C;
   
    //allocate device memory. Hint: used cudaMalloc
    cudaMalloc((void**)&d_A, M * K * sizeof(int));
    cudaMalloc((void**)&d_B, K * N * sizeof(int));
    cudaMalloc((void**)&d_C, M * N * sizeof(int));

    //Copy data from host to device. Hint: use cudaMemcpy
    cudaMemcpy(d_A, h_A, M * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(int), cudaMemcpyHostToDevice);

    //define the thread dimentions
    dim3 blockSize(16, 16); //each block will be 16x16=256 threads. We are arranging the threads in the blox in a 2D grid
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    printf("Grid size: %d, %d\n", gridSize.x, gridSize.y);


    // Issue the kernel on the GPU
    MatrixMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);

    //Copy the computed results from device to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    //free the device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}












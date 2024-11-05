// Include C++ header files.
#include <iostream>
#include <chrono>
// Include local CUDA header files.
#include "include/cuda_kernel.cuh"

#define N 3 //array/matrix size
#define M 4 //matrix size
#define K 5 //matrix size




int CPU_vecAdd(int *A, int *B, int *C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
    return 0;
}

int vecAdd() {

    //Create empty arrays on the host (CPU)
    int *h_A = (int *)malloc(N * sizeof(int));
    int *h_B = (int *)malloc(N * sizeof(int));
    int *h_C = (int *)malloc(N * sizeof(int));

    //initialize the arrays with random data (e.g., use rand function)
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_A[i] = rand() % 100;
        h_B[i] = rand() % 100;
    }
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    //Peform the computation on the CPU 
    CPU_vecAdd(h_A, h_B, h_C, N);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> time_taken = end - start;
    std::cout << "Time taken by CPU computation: " << time_taken.count() << " ms" << std::endl;

    free(h_C);
    h_C = (int *)malloc(N * sizeof(int));
    //call a function passing the pointer of the arrays as arguments to compute on the GPU
    std::chrono::steady_clock::time_point start2 = std::chrono::steady_clock::now();
    vecAddKernelWrap(h_A, h_B, h_C, N);
    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> time_taken2 = end2 - start2;
    std::cout << "Time taken by GPU computation: " << time_taken2.count() << " ms" << std::endl;


 
    
    printf("\nProgramm Finished!\n");
    return 0;
}


int CPU_MulMatrix(int *A, int *B, int *C, int n,int m,int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i*n+j] = 0;
            for (int l = 0; l < k; l++) {
                C[i*n+j] += A[i*k+l] * B[l*n+j];
            }
        }
    }
    return 0;

}

int matrixMul() {
    //Create empty matrix on the host (CPU)
    int *h_A= (int*)malloc(M * K * sizeof(int));
    int *h_B= (int*)malloc(K * N * sizeof(int));
    int *h_C= (int*)malloc(M * N * sizeof(int));


    //initialize the arrays with random data (e.g., use rand function)
    srand(time(NULL));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            h_A[i*K+j] = rand() % 100;
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            h_B[i*N+j] = rand() % 100;
        }
    }
    

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    CPU_MulMatrix(h_A, h_B, h_C, N,M,K);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> time_taken = end - start;
    std::cout << "Time taken by CPU computation: " << time_taken.count() << " ms" << std::endl;
    std::cout << "CPU Result: " << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_C[i*N+j] << " ";
        }
        std::cout << std::endl;
    }
    free(h_C);
    h_C = (int*)malloc(M * N * sizeof(int));


    //call a function passing the pointer of the arrays as arguments to compute on the GPU
    std::chrono::steady_clock::time_point start2 = std::chrono::steady_clock::now();
    matrixMulKernelWrap(h_A, h_B, h_C, N,M,K);
    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> time_taken2 = end2 - start2;
    std::cout << "Time taken by GPU computation: " << time_taken2.count() << " ms" << std::endl;
    std::cout << "GPU Result: " << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_C[i*N+j] << " ";
        }
        std::cout << std::endl;
    }

    printf("\nProgramm Finished!\n");
    return 0;
}

int main(void){
matrixMul();

}
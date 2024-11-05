
// List wrapper function callable by .cpp file.

// TODO: define the wrapper funtions to be used wherever it is required by other CPP files
void vecAddKernelWrap(int *h_A, int *h_B, int *h_C, int N);
void matrixMulKernelWrap(int *h_A, int *h_B, int *h_C, int N,int M,int K);





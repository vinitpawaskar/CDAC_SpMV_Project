void mat_vec_allocation(int N, float **A, float **B, float **C_cuda, float **C_csr, float **C_cublas, float **C_spmv, float **val, int **row, int **col);

void sparse_mat_vec_initializer(int N, float *A, float *B);

int mat_to_csr(int N, float *A, float *val, int *row, int *col);

void print_mat(int N, float *A);

void print_vec(int N, float *B);

void print_csr(int N, float *val, int *row, int * col, int nnz);

void error_routine(int N, float *C_cuda, float *C_csr, float *C_cublas, float *C_spmv);

__global__ void cuda_multiplication(int N, float *A_d, float *B_d, float *C_cuda_d);

__global__ void csr_multiplication(int N, float *B_d, float *C_csr_d, int *row_d, int *col_d, float *val_d);
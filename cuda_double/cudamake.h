void mat_vec_allocation(int N, double **A, double **B, double **C_cuda, double **C_csr, double **C_cublas, double **C_spmv, double **val, int **row, int **col);

void sparse_mat_vec_initializer(int N, double *A, double *B);

int mat_to_csr(int N, double *A, double *val, int *row, int *col);

void print_mat(int N, double *A);

void print_vec(int N, double *B);

void print_csr(int N, double *val, int *row, int * col, int nnz);

void error_routine(int N, double *C_cuda, double *C_csr, double *C_cublas, double *C_spmv);

__global__ void cuda_multiplication(int N, double *A_d, double *B_d, double *C_cuda_d);

__global__ void csr_multiplication(int N, double *B_d, double *C_csr_d, int *row_d, int *col_d, double *val_d);

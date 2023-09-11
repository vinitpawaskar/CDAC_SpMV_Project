void mat_vec_allocation(int N, float **A, float **B, float **C_seq, float **C_csr, float **C_cblas, float **C_spmv, float **val, int **row, int **col);

void sparse_mat_vec_initializer(int N, float *A, float *B);

void print_mat(int N, float *A);

void print_vec(int N, float *B);

int mat_to_csr(int N, float *A, float *val, int *row, int *col);

void print_csr(int N, float *val, int *row, int * col, int nnz);

void seq_mult(int N, float *A, float *B, float *C_seq);

void csr_mult(int N, float *A, float *B, float *C_csr, int *row, int *col, float *val);

void cblas_Dgemv(int N, float *A, float *B, float *C_cblas);

void sparse_d_mv(int N, float *A, float *B, float *C_spmv, int *row, int *col, float *val);

void error_routine(int N, float *C_seq, float *C_csr, float *C_blas, float *C_spmv);

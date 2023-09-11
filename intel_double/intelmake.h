void mat_vec_allocation(int N, double **A, double **B, double **C_seq, double **C_csr, double **C_cblas, double **C_spmv, double **val, int **row, int **col);

void sparse_mat_vec_initializer(int N, double *A, double *B);

void print_mat(int N, double *A);

void print_vec(int N, double *B);

int mat_to_csr(int N, double *A, double *val, int *row, int *col);

void print_csr(int N, double *val, int *row, int * col, int nnz);

void seq_mult(int N, double *A, double *B, double *C_seq);

void csr_mult(int N, double *A, double *B, double *C_csr, int *row, int *col, double *val);

void cblas_Dgemv(int N, double *A, double *B, double *C_cblas);

void sparse_d_mv(int N, double *A, double *B, double *C_spmv, int *row, int *col, double *val);

void error_routine(int N, double *C_seq, double *C_csr, double *C_blas, double *C_spmv);

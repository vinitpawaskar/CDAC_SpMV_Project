#include<stdio.h>
#include<stdlib.h>
#include"intelmake.h"

int main(int argc, char *argv[])
{
        float *A, *B, *C_seq, *C_csr, *C_cblas, *C_spmv;
	float *val;
	int *row, *col;
        int nnz;
        int N = atoi(argv[1]);
              
        mat_vec_allocation(N, &A, &B, &C_seq, &C_csr, &C_cblas, &C_spmv, &val, &row, &col);
        
        sparse_mat_vec_initializer(N, A, B);

        nnz = mat_to_csr(N, A, val, row, col);

//	print_mat(N, A);

//	print_vec(N, B);
	
//	print_csr(N, val, row, col, nnz);
	
	seq_mult(N, A, B, C_seq);
//	print_vec(N, C_seq);

	csr_mult(N, A, B, C_csr, row, col, val);
//	print_vec(N, C_csr);
	
	cblas_Dgemv(N, A, B, C_cblas);
//	print_vec(N, C_cblas);
	
	sparse_d_mv(N, A, B, C_spmv, row, col, val);
//	print_vec(N, C_spmv);
	
	error_routine(N, C_seq, C_csr, C_cblas, C_spmv);

        free(A);
        free(B);
        free(C_seq);
        free(C_csr);
        free(C_cblas);
        free(C_spmv);
        free(val);
        free(row);
        free(col);
        return 0;
}

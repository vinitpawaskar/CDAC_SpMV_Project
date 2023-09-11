#include<stdio.h>
#include<stdlib.h>
#include"intelmake.h"

int main(int argc, char *argv[])
{
        float *A, *B, *C_csr, *C_spmv;
	float *val;
	int *row, *col;
        int N = atoi(argv[1]);
        int nnz = N  * 5 / 100;
   	printf("%d\n", nnz);
        
              
        mat_vec_allocation(N, nnz, &A, &B, &C_csr, &C_spmv, &val, &row, &col); 
        
        sparse_mat_vec_initializer(N, nnz, A, B, val, col, row);

//	print_vec(N, B);
	
//	print_csr(N, val, row, col, nnz);

	csr_mult(N, A, B, C_csr, row, col, val);
//	print_vec(N, C_csr);

	sparse_d_mv(N, A, B, C_spmv, row, col, val);
//	print_vec(N, C_spmv);
	
	error_routine(N, C_csr, C_spmv);

        free(A);
        free(B);

        free(C_csr);

        free(C_spmv);
        free(val);
        free(row);
        free(col);
        return 0;
}

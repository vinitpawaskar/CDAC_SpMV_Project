#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include<cublas_v2.h>
#include<sys/time.h>
#include"cudamake.h"

int main(int argc, char *argv[])
{
        double *A, *B, *C_csr, *C_spmv, *val;
        int *row, *col;
       	double *B_d, *C_csr_d, *C_spmv_d;
        int N = atoi(argv[1]);
        int nnz = N  * 5 / 100;
   	printf("%d\n", nnz);
	double exe_time;
	struct timeval stop_time, start_time;
	
        mat_vec_allocation(N, nnz, &A, &B, &C_csr, &C_spmv, &val, &row, &col); 
        
        sparse_mat_vec_initializer(N, nnz, A, B, val, col, row);
                
//        print_mat(N, A);
//        print_vec(N, B);

        double *val_d; 
        int *col_d, *row_d;
        
        cudaMalloc(&B_d,N*sizeof(double));

        cudaMalloc(&C_csr_d,N*sizeof(double));

        cudaMalloc(&C_spmv_d,N*sizeof(double));

        cudaMemcpy(B_d,B,N*sizeof(double),cudaMemcpyHostToDevice);

	int Total_num_Threads = N;
        int num_threads_per_block = 256;
        int numblocks = Total_num_Threads/num_threads_per_block + 1;
        int  loop;
	//////////////////////////////////////////////////////////////////////

	///////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////
	

//	print_csr(N, val, row, col, nnz);
	
	cudaMalloc(&val_d,N*nnz*sizeof(double));
        cudaMalloc(&col_d,N*nnz*sizeof(int));
        cudaMalloc(&row_d,(N+1)*sizeof(int));

        cudaMemcpy(val_d,val,N*nnz*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(col_d,col,N*nnz*sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(row_d,row,(N+1)*sizeof(int),cudaMemcpyHostToDevice);
        
        exe_time = 0.0;
	for(loop = 0; loop < 5; loop++)
	{
        gettimeofday(&start_time, NULL);
	csr_multiplication<<<numblocks,num_threads_per_block>>>(N, B_d, C_csr_d, row_d, col_d, val_d);
	gettimeofday(&stop_time, NULL);
	exe_time += (stop_time.tv_sec+(stop_time.tv_usec/1000000.0)) - (start_time.tv_sec+(start_time.tv_usec/1000000.0));
	}
	printf("\nMatrix Size : %d X %d : Executed 5 times : Average execution time for csr mult kernel is = %lf seconds.\n", N,N, exe_time/5);
	cudaMemcpy(C_csr, C_csr_d, N*sizeof(double), cudaMemcpyDeviceToHost);
//	print_vec(N, C_csr);
	//////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////
	const double alpha = 1.0;
        const double beta = 0.0;
        
        cusparseHandle_t handle1;
        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t Bd,Cspmvd;
        void* dBuffer = NULL;
        size_t bufferSize = 0;
        cusparseCreate(&handle1);

        cusparseCreateCsr(&matA, N, N, N*nnz, row_d, col_d, val_d, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

        cusparseCreateDnVec(&Bd, N, B_d, CUDA_R_64F);

        cusparseCreateDnVec(&Cspmvd, N, C_spmv_d, CUDA_R_64F);

        cusparseSpMV_bufferSize(handle1, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, Bd, &beta, Cspmvd, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);

        cudaMalloc(&dBuffer, bufferSize);

	exe_time = 0.0;
	for(loop = 0; loop < 5; loop++)
	{
	gettimeofday(&start_time, NULL);
        cusparseSpMV(handle1, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, Bd, &beta, Cspmvd, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
	gettimeofday(&stop_time, NULL);
	exe_time += (stop_time.tv_sec+(stop_time.tv_usec/1000000.0)) - (start_time.tv_sec+(start_time.tv_usec/1000000.0));
	}
        printf("\nMatrix Size : %d X %d : Executed 5 times : Average execution time for cusparseSpMV kernel is = %lf seconds.\n", N,N, exe_time/5);

        cusparseDestroySpMat(matA);
        cusparseDestroyDnVec(Bd);
        cusparseDestroyDnVec(Cspmvd);
        cusparseDestroy(handle1);

        cudaMemcpy(C_spmv, C_spmv_d, N*sizeof(double), cudaMemcpyDeviceToHost);
//        print_vec(N, C_spmv);
        ////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////
        
        error_routine(N, C_csr, C_spmv);
	
        cudaFree(B_d);

        cudaFree(C_csr_d);

        cudaFree(C_spmv_d);
        cudaFree(val_d);
        cudaFree(row_d);
        cudaFree(col_d);
        cudaFree(dBuffer);

	free(A);
        free(B);

        free(C_csr);

        free(C_spmv);
        free(val);
        free(row);
        free(col);
        
        return 0;
}

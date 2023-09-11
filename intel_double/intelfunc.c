#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<mkl.h>
#include"intelmake.h"


void mat_vec_allocation(int N, double **A, double **B, double **C_seq, double **C_csr, double **C_cblas, double **C_spmv, double **val, int **row, int **col)
{
	*A = (double *)calloc(N*N ,sizeof(double));
	if ( A == NULL ) 
	{
		fprintf(stderr, "\nOut of memory!... Matrix cannot be Initialized.\n");
		exit(1);
	}
	
        *B = (double *)calloc(N, sizeof(double));
        *C_seq = (double *)calloc(N, sizeof(double));
        *C_csr = (double *)calloc(N, sizeof(double));
        *C_cblas = (double *)calloc(N, sizeof(double));
        *C_spmv = (double *)calloc(N, sizeof(double));
        if (B==NULL || C_seq==NULL || C_csr==NULL || C_cblas==NULL || C_spmv==NULL) 
	{
		fprintf(stderr, "\nOut of memory!... Vectors cannot be Initialized.\n");
		exit(1);
	}
	      
        *val = (double *)calloc((3*N+1) ,sizeof(double));
        *row = (int *)calloc((N+1), sizeof(int));
        *col = (int *)calloc((3*N+1), sizeof(int));
        if (val==NULL || row==NULL || col==NULL) 
	{
		fprintf(stderr, "\nOut of memory!\n");
		exit(1);
	}
}


void sparse_mat_vec_initializer(int N, double *A, double *B)
{
	int i, j;
        for(i=0;i<N;i++)
        {
                for(j=0;j<N;j++)
                {
                        if((i*N+j) % N == i-1 || (i*N+j) % N == i || (i*N+j) % N == i+1 )
                        {
                                A[i*N+j] = (rand()%100);
                        }

                }
                B[i] = (rand()%100);

        }	
}

void print_mat(int N, double *A)
{
	int i, j, k;

	printf("\n A matrix:\n");
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
		{
			printf("\t%f ", A[i*N+j]);	
		}
		printf("\n");
	}
}

void print_vec(int N, double *B)
{
	int i;
	
	printf("\n B vector:\n");
        for(i=N-5; i<N; i++)
        {
                printf("\t%lf", B[i]);
        }
        printf("\n");
}

int mat_to_csr(int N, double *A, double *val, int *row, int *col)
{
	row[0] = 0;
        int nnz = 0;
        
        int i, j;
        for (i=0; i<N; i++)
        {
                for(j=0; j<N; j++)
                {
                        if(A[i*N+j] != 0.0)
                        {
                                val[nnz]=A[i*N+j];
                                col[nnz] = j;
                                nnz++;
                        }
                }
                row[i+1]=nnz;
        }
        
        printf("%d\n",nnz);


        val = (double *)realloc(val, nnz*sizeof(double));
        col = (int *)realloc(col, nnz*sizeof(int));
        
        return nnz;
}

void print_csr(int N, double *val, int *row, int * col, int nnz)
{
	int i;
	
	printf("\nValues: \n");
	for(i=0; i<nnz; i++)
        {
                printf("%lf  ",val[i]);
        }
        printf("\nRow Indices: \n");
        for(i=0; i<N+1; i++)
        {
                printf("%d  ",row[i]);
        }
        printf("\nColumn Indices: \n");
        for(i=0; i<nnz; i++)
        {
                printf("%d  ",col[i]);
        }
        printf("\n");
}

void seq_mult(int N, double *A, double *B, double *C_seq)
{
	double exec_time = 0.0;
        struct timeval stop_time, start_time;

	int i, j, loop;
	for(loop = 0; loop < 5; loop++)
	{
	gettimeofday(&start_time, NULL);
	for(i=0;i<N;i++)
        {
		C_seq[i] = 0.0;        
                for(j=0;j<N;j++)
                {
                        C_seq[i] += A[i*N+j]*B[j];
                }
        }
        gettimeofday(&stop_time, NULL);
        exec_time += (stop_time.tv_sec+(stop_time.tv_usec/1000000.0)) - (start_time.tv_sec+(start_time.tv_usec/1000000.0));
	}
	printf("\nMatrix Size : %d X %d : Executed 5 times : Average execution time for sequential multiplication is = %lf seconds.\n", N,N, exec_time/5);
}

void csr_mult(int N, double *A, double *B, double *C_csr, int *row, int *col, double *val)
{
	double exec_time = 0.0;
        struct timeval stop_time, start_time;

	int i, j, loop;
	for(loop = 0; loop < 5; loop++)
	{
	gettimeofday(&start_time, NULL);
	for(i =0; i<N; i++)
        {
        	C_csr[i] = 0.0;
                for(j=row[i]; j<row[i+1]; j++)
                {
                        C_csr[i] += val[j] * B[col[j]];
                }
        }
        gettimeofday(&stop_time, NULL);
        exec_time += (stop_time.tv_sec+(stop_time.tv_usec/1000000.0)) - (start_time.tv_sec+(start_time.tv_usec/1000000.0));
	}
	printf("\nMatrix Size : %d X %d : Executed 5 times : Average execution time for csr multiplication is = %lf seconds.\n", N,N, exec_time/5);
}


void cblas_Dgemv(int N, double *A, double *B, double *C_cblas)
{
	double exec_time = 0.0;
        struct timeval stop_time, start_time;
	
	int loop;
	for(loop = 0; loop<5; loop++)
	{
	gettimeofday(&start_time, NULL);
	cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0, A, N, B, 1, 0.0, C_cblas, 1);
	gettimeofday(&stop_time, NULL);
        exec_time += (stop_time.tv_sec+(stop_time.tv_usec/1000000.0)) - (start_time.tv_sec+(start_time.tv_usec/1000000.0));
	}
	printf("\nMatrix Size : %d X %d : Executed 5 times : Average execution time for cbla_Dgemv multiplication is = %lf seconds.\n", N,N, exec_time/5);
}


void sparse_d_mv(int N, double *A, double *B, double *C_spmv, int *row, int *col, double *val)
{
	double exec_time;
        struct timeval stop_time, start_time;

	struct matrix_descr descrA;
        sparse_matrix_t csrA;
        sparse_status_t status;

        mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO, N, N, row, row+1, col, val);

        descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
        
        int loop;
        for(loop = 0; loop < 5; loop++)
        {
	gettimeofday(&start_time, NULL);
        status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrA, descrA, B, 0.0, C_spmv);
 	gettimeofday(&stop_time, NULL);
        exec_time += (stop_time.tv_sec+(stop_time.tv_usec/1000000.0)) - (start_time.tv_sec+(start_time.tv_usec/1000000.0));
	}
	printf("\nMatrix Size : %d X %d : Executed 5 times : Average execution time for sparse_d_mv multiplication is = %lf seconds.\n", N,N, exec_time/5);
}

void error_routine(int N, double *C_seq, double *C_csr, double *C_blas, double *C_spmv)
{
	int i;
	int flag = 0;
	for(i=0; i<N; i++)
	{
		if((C_seq[i] != C_csr[i]) || (C_seq[i] != C_blas[i]) || (C_seq[i] != C_spmv[i]))
		{
			flag = 1;
			printf("\nOutput Vectors are not matching...!!!\n");
			break;
		}
	}
	if(flag == 0)
	{
		printf("\nOutput Vectors are correct... :)\n");
	}
	
}

#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<mkl.h>
#include"intelmake.h"




void mat_vec_allocation(int N, int nnz, double **A, double **B, double **C_csr, double **C_spmv, double **val, int **row, int **col)
{
	*A = (double *)calloc(N,sizeof(double));
	if ( A == NULL ) 
	{
		fprintf(stderr, "\nOut of memory!... Matrix cannot be Initialized.\n");
		exit(1);
	}
	
        *B = (double *)calloc(N, sizeof(double));

        *C_csr = (double *)calloc(N, sizeof(double));

        *C_spmv = (double *)calloc(N, sizeof(double));
        if (B==NULL ||  C_csr==NULL || C_spmv==NULL) 
	{
		fprintf(stderr, "\nOut of memory!... Vectors cannot be Initialized.\n");
		exit(1);
	}
	      
        *val = (double *)calloc((N*nnz) ,sizeof(double));
        *row = (int *)calloc((N+1), sizeof(int));
        *col = (int *)calloc((N*nnz), sizeof(int));
        if (val==NULL || row==NULL || col==NULL) 
	{
		fprintf(stderr, "\nOut of memory!\n");
		exit(1);
	}
}


void sparse_mat_vec_initializer(int N, int nnz, double *A, double *B, double *val, int *col, int *row)
{
	int i, j;
	row[0] = 0;
	for( i=0; i<N; i++)
	{	
		A = (double *)calloc(N, sizeof(double));
		
		for( j=0; j<nnz; j++)
		{
			int ind = rand() % N;	
			while (A[ind] !=0.0)
			{
				ind = rand()%100;
			}
			A[ind] = rand()%100;
	//		printf("%lf ", A[ind]);
			val[i*nnz+j] = rand()%100;
			col[i*nnz+j] = ind;
		}
	//	printf("\n");
		row[i+1]=row[i]+nnz;
		free(A);	
		B[i] = rand()%100;
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
        for(i=0; i<N; i++)
        {
                printf("\t%lf", B[i]);
        }
        printf("\n");
}


void print_csr(int N, double *val, int *row, int * col, int nnz)
{
	int i;
	
	printf("\nValues: \n");
	for(i=0; i<N*nnz; i++)
        {
                printf("%lf  ",val[i]);
        }
        printf("\nRow Indices: \n");
        for(i=0; i<N+1; i++)
        {
                printf("%d  ",row[i]);
        }
        printf("\nColumn Indices: \n");
        for(i=0; i<N*nnz; i++)
        {
                printf("%d  ",col[i]);
        }
        printf("\n");
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

void error_routine(int N, double *C_csr, double *C_spmv)
{
	int i;
	int flag = 0;
	for(i=0; i<N; i++)
	{
		if((C_csr[i] != C_spmv[i]))
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

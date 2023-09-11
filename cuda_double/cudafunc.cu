#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include<cublas_v2.h>
#include<sys/time.h>
#include"cudamake.h"

void mat_vec_allocation(int N, double **A, double **B, double **C_cuda, double **C_csr, double **C_cublas, double **C_spmv, double **val, int **row, int **col)
{
	*A = (double *)calloc(N*N ,sizeof(double));
	if ( A == NULL ) 
	{
		fprintf(stderr, "\nOut of memory!... Matrix cannot be Initialized.\n");
		exit(1);
	}
	
        *B = (double *)calloc(N, sizeof(double));
        *C_cuda = (double *)calloc(N, sizeof(double));
        *C_csr = (double *)calloc(N, sizeof(double));
        *C_cublas = (double *)calloc(N, sizeof(double));
        *C_spmv = (double *)calloc(N, sizeof(double));
        if (B==NULL || C_cuda==NULL || C_csr==NULL || C_cublas==NULL || C_spmv==NULL) 
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
        
//        printf("%d\n",nnz);


        val = (double *)realloc(val, nnz*sizeof(double));
        col = (int *)realloc(col, nnz*sizeof(int));
        
        return nnz;
}

void print_mat(int N, double *A)
{
	int i, j;

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

void error_routine(int N, double *C_cuda, double *C_csr, double *C_cublas, double *C_spmv)
{
	int i;
	int flag = 0;
	for(i=0; i<N; i++)
	{
		if((C_cuda[i] != C_csr[i]) || (C_cuda[i] != C_cublas[i]) || (C_cuda[i] != C_spmv[i]))
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

__global__ void cuda_multiplication(int N, double *A_d, double *B_d, double *C_cuda_d)
{

        int i = blockIdx.x*blockDim.x + threadIdx.x;
        int j;
        int sum = 0;
        if(i<N)
        {
        	C_cuda_d[i] = 0.0;
                for(j=0;j<N;j++)
                {
                        sum = sum + A_d[i*N+j]*B_d[j];
                }
                C_cuda_d[i] =  sum;
        }
}


__global__ void csr_multiplication(int N, double *B_d, double *C_csr_d, int *row_d, int *col_d, double *val_d)
{
        int my_row = blockDim.x* blockIdx.x + threadIdx.x;

        if(my_row < N)
        {
                double sum = 0;

                int row_start = row_d[my_row];
                int row_end = row_d[my_row + 1];
		
		C_csr_d[my_row] = 0.0;
		
                for ( int i = row_start; i < row_end; i++)
                {
                        sum += val_d[i] * B_d[col_d[i]];
                }
                C_csr_d[my_row] += sum;
        }
 
}

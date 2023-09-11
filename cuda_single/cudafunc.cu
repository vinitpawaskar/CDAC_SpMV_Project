#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include<cublas_v2.h>
#include<sys/time.h>
#include"cudamake.h"

void mat_vec_allocation(int N, float **A, float **B, float **C_cuda, float **C_csr, float **C_cublas, float **C_spmv, float **val, int **row, int **col)
{
	*A = (float *)calloc(N*N ,sizeof(float));
	if ( A == NULL )
	{
	fprintf(stderr, "\nOut of memory!... Matrix cannot be Initialized.\n");
	exit(1);
	}

        *B = (float *)calloc(N, sizeof(float));
        *C_cuda = (float *)calloc(N, sizeof(float));
        *C_csr = (float *)calloc(N, sizeof(float));
        *C_cublas = (float *)calloc(N, sizeof(float));
        *C_spmv = (float *)calloc(N, sizeof(float));
        if (B==NULL || C_cuda==NULL || C_csr==NULL || C_cublas==NULL || C_spmv==NULL)
	{
	fprintf(stderr, "\nOut of memory!... Vectors cannot be Initialized.\n");
	exit(1);
	}
     
        *val = (float *)calloc((3*N+1) ,sizeof(float));
        *row = (int *)calloc((N+1), sizeof(int));
        *col = (int *)calloc((3*N+1), sizeof(int));
        if (val==NULL || row==NULL || col==NULL)
	{
	fprintf(stderr, "\nOut of memory!\n");
	exit(1);
	}
	}

	void sparse_mat_vec_initializer(int N, float *A, float *B)
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

	int mat_to_csr(int N, float *A, float *val, int *row, int *col)
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


        val = (float *)realloc(val, nnz*sizeof(float));
        col = (int *)realloc(col, nnz*sizeof(int));
       
        return nnz;
	}

void print_mat(int N, float *A)
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

void print_vec(int N, float *B)
{
	int i;

	printf("\n B vector:\n");
        for(i=0; i<N; i++)
        {
                printf("\t%lf", B[i]);
        }
        printf("\n");
	}

void print_csr(int N, float *val, int *row, int * col, int nnz)
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

void error_routine(int N, float *C_cuda, float *C_csr, float *C_cublas, float *C_spmv)
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

__global__ void cuda_multiplication(int N, float *A_d, float *B_d, float *C_cuda_d)
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


__global__ void csr_multiplication(int N, float *B_d, float *C_csr_d, int *row_d, int *col_d, float *val_d)
{
        int my_row = blockDim.x* blockIdx.x + threadIdx.x;

        if(my_row < N)
        {
                float sum = 0;

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

#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include<cublas_v2.h>
#include<sys/time.h>
#include"cudamake.h"


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

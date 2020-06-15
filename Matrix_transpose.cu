#define TILE_WIDTH 100
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace std;
//cuda kernel
__global__ void sharedMem_transpose_pad(float* M, float* R, int dim1, int dim2) 
{

	// fill data into shared memory
	__shared__ float M_Shared[TILE_WIDTH][TILE_WIDTH + 1];
	
	int ix, iy, index_in;
	int i_row, i_col, _id_index, out_ix, out_iy, index_out;

	ix = blockDim.x * blockIdx.x + threadIdx.x;
	iy = blockDim.y * blockIdx.y + threadIdx.y;

	index_in = iy * dim1 + ix;
	_id_index = threadIdx.y * blockDim.x + threadIdx.x;

	i_row = _id_index / blockDim.y;
	i_col = _id_index % blockDim.y;

	out_ix = blockIdx.y * blockDim.y + i_col;
	out_iy = blockIdx.x * blockDim.x + i_row;

	index_out = out_iy * dim2 + out_ix;

	if (ix < dim1 && iy < dim2) 
	{
		M_Shared[threadIdx.y][threadIdx.x] = M[index_in];

		cudaDeviceSynchronize(); // wait all other threads to go further.

		R[index_out] = M_Shared[i_col][i_row];
	}
	
}


//host code
int main()
{
	int const tile_size = 100;
	int const dim1 = 3000;
	int const dim2 = 3000;

	float* M_h;
	float* R_h;
	float* M_d;
	float* R_d;

	size_t size = dim1 * dim2 * sizeof(float);

	cudaMallocHost((float**)& M_h, size); //page locked host mem allocation
	R_h = (float*)malloc(size);
	cudaMalloc((float**)& M_d, size);


	// init matrix
	for (int i = 0; i < dim1 * dim2; ++i) 
	{
		M_h[i] = i;
	}

	cudaMemcpyAsync(M_d, M_h, size, cudaMemcpyHostToDevice);
	cudaMalloc((float**)& R_d, size);
	cudaMemset(R_d, 0, size);
	
	int threadNumX = tile_size;
	int threadNumY = tile_size;
	int blockNumX = dim1 / tile_size + (dim1 % tile_size == 0 ? 0 : 1);
	int blockNumY = dim2 / tile_size + (dim2 % tile_size == 0 ? 0 : 1);

	dim3 blockSize(threadNumX, threadNumY);
	dim3 gridSize(blockNumX, blockNumY);


	sharedMem_transpose_pad<<<gridSize, blockSize>>>(M_d, R_d, dim1, dim2);

	cudaMemcpy(R_h, R_d, size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < dim1; ++i) 
	{
		for (int j = 0; j < dim2; ++j) 
		{
			float num = R_h[i*dim2 + j];
			cout << num;
		}
		cout << endl;
	}


	free(M_h);
	free(R_h);
	cudaFree(R_d);
	cudaFree(M_d);
	return 0;
}

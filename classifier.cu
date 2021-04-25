/**
 * classifier.cu
 * 
 * A CUDA kernel for accelerating a fully-connected neural network layer.
 */

#include <iostream>
#include <string>

using namespace std;

#ifndef Ni
  #define Ni 4096  
#endif
#ifndef Nn
  #define Nn 1024  
#endif
#ifndef Nb
  #define Nb 1
#endif

#define DEBUG (false)

/* The weights of the layer*/
__device__ float d_weights[Nn][Ni];
__device__ float d_inputs[Ni][Nb];
__device__ float d_outputs[Nn][Nb];

#define BLK_SZ (16)

/* Performs a matrix multiply. Assumes kernel block size is (BLK_SZ, BLK_SZ, 1)*/
__device__ void matmul_blk(float a[BLK_SZ][BLK_SZ], float b[BLK_SZ][BLK_SZ], float c[BLK_SZ][BLK_SZ])
{
    const int row = threadIdx.x;
    const int col = threadIdx.y;

	for(int i = 0; i < BLK_SZ; i++)
	{
		c[row][col] += a[row][i] * b[i][col];
	}
}

/* Performs C = A + B. Assumes kernel block size is (BLK_SZ, BLK_SZ, 1)*/
__device__ void matadd_blk(float a[BLK_SZ][BLK_SZ], float b[BLK_SZ][BLK_SZ], float c[BLK_SZ][BLK_SZ])
{
    const int row = threadIdx.x;
    const int col = threadIdx.y;

    c[row][col] = a[row][col] + b[row][col];
}

__device__ void matadd_blk_global(float *dst, int2 dstDim, float mat[BLK_SZ][BLK_SZ], int blkX, int blkY)
{
    const int row = threadIdx.x;
    const int col = threadIdx.y;

    atomicAdd(&dst[dstDim.y*(BLK_SZ*blkX + row) + (BLK_SZ*blkY + col)], mat[row][col]);
}

__device__ void matread_blk(float *src, int2 srcDim, float mat[BLK_SZ][BLK_SZ], int blkX, int blkY)
{
    const int row = threadIdx.x;
    const int col = threadIdx.y;

    const int idx = srcDim.y*(BLK_SZ*blkX + row) + (BLK_SZ*blkY + col);
    if(idx >= (srcDim.x * srcDim.y))
    {
    	printf("bounds error srcDim=(%d,%d) (%d,%d) block (%d,%d): %d\n", srcDim.x, srcDim.y, row, col, blkX, blkY, idx);
    }

    mat[row][col] = src[idx];
}

__device__ void matwrite_blk(float *dst, int2 srcDim, float mat[BLK_SZ][BLK_SZ], int blkX, int blkY)
{
    const int row = threadIdx.x;
    const int col = threadIdx.y;

    dst[srcDim.y*(BLK_SZ*blkX + row) + (BLK_SZ*blkY + col)] = mat[row][col];
}

__device__ void blk_zero(float mat[BLK_SZ][BLK_SZ])
{
    const int row = threadIdx.x;
    const int col = threadIdx.y;

    mat[row][col] = 0;
}

__device__ void printBlock(float mat[BLK_SZ][BLK_SZ], int blkX, int blkY)
{
	if(blockIdx.x == blkX && blockIdx.y == blkY)
	{
		printf("mat[%d][%d] = %f\n", threadIdx.x, threadIdx.y, mat[threadIdx.x][threadIdx.y]);
	}
}

__global__ void classify()
{
    const int2 weightDim = make_int2(Nn, Ni);
    const int2 inDim = make_int2(Ni, Nb);
    const int2 outDim = make_int2(Nn, Nb);

    const int blkX = blockIdx.x;
    const int blkY = blockIdx.y;

    __shared__ float weightBlk[BLK_SZ][BLK_SZ];
    __shared__ float inBlk[BLK_SZ][BLK_SZ];
    __shared__ float outBlk[BLK_SZ][BLK_SZ];

    matread_blk((float*)d_inputs, inDim, inBlk, blkX, blkY);

    for(int i = 0; i < (weightDim.x / BLK_SZ); i++)
    {
        matread_blk((float*)d_weights, weightDim, weightBlk, i, blkX);
        blk_zero(outBlk);


        //__threadfence();

        matmul_blk(weightBlk, inBlk, outBlk);

        //__threadfence();

        matadd_blk_global((float*)d_outputs, outDim, outBlk, i, blkY);

    }
}

void randomizeArray(float *data, int len)
{
	for(int i = 0; i < len; i++)
	{
		data[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 16.0f - 8.0f;
	}
}

int main(int argc, char **argv) 
{
     const int2 in_dim = make_int2(Ni, Nb);    					// The dimensions of the input matrix
     const int in_size = sizeof(float) * in_dim.x * in_dim.y;   // Total size of input buffer in bytes

     const int2 out_dim = make_int2(Nn, Nb);               // The dimensions of the output matrix
     const int out_size = sizeof(float) * out_dim.x * out_dim.y; // Total size of output buffer in bytes

     const dim3 grid_size(in_dim.x/16,in_dim.y/16,1);
     const dim3 block_size(16,16,1);

     float *h_in_data = new float[in_dim.x * in_dim.y]; // Input data on host
     float *h_out_data = new float[out_dim.x * out_dim.y];    // Output on device
     float *h_random_weights = new float[Ni*Nn];

     // Make some random data.

     randomizeArray(h_random_weights, Ni*Nn);
     randomizeArray(h_in_data, in_dim.x * in_dim.y);

     cudaMemcpyToSymbol(d_weights, h_random_weights, Nn*Ni*sizeof(float));

     cudaMemcpyToSymbol(d_inputs, h_in_data, in_size); // Give the GPU our input data.

     classify<<<grid_size, block_size>>>();
     cudaDeviceSynchronize();

     cudaMemcpyFromSymbol(h_out_data, d_outputs, out_size); // Retrieve the neuron outputs.


     if(DEBUG)
     {
		 for(int i = 0; i < Nn; i++)
		 {
			 for(int j = 0; j < Ni; j++)
			 {
				 printf("%f ", h_random_weights[i*Ni + j]);
			 }
			 printf("\n");
		 }
		 printf("\n");

		 for(int i = 0; i < in_dim.x; i++)
		 {
			 for(int j = 0; j < in_dim.y; j++)
			 {
				 printf("%f ", h_in_data[i*in_dim.y + j]);
			 }
			 printf("\n");
		 }
		 printf("\n");

		 for(int i = 0; i < out_dim.x; i++)
		 {
			 for(int j = 0; j < out_dim.y; j++)
			 {
				 printf("%f ", h_out_data[i*out_dim.y + j]);
			 }
			 printf("\n");
		 }
     }

    return 0;
}


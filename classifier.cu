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

/* The weights of the layer*/
__device__ float d_weights[Ni*Nn];

/*
 * Matrix multiply. Performs the operation c = ab', where b' is the transpose of b.
 */
__device__ void mmult(float *a, float *b, float *c, int2 adim, int2 bdim)
{
    int row = threadIdx.x + blockIdx.x*blockDim.x;
    int col = threadIdx.y + blockIdx.y*blockDim.y;

    if(row < bdim.x && col < adim.y)
    {
		for(int i = 0; i < adim.x; i++)
		{
			c[row*bdim.x + col] += a[i*adim.y + col] * b[row*adim.y + i];
		}
    }
}

__device__ void relu(float *mtx, int2 dim)
{
    int row = threadIdx.x + blockIdx.x*blockDim.x;
    int col = threadIdx.y + blockIdx.y*blockDim.y;

    if(row < dim.x && col < dim.y)
    {
			mtx[col*dim.y + row] *= (mtx[col*dim.y + row] > 0);
    }
}

__global__ void classify(float *in, float *out)
{
    const int2 in_dim = make_int2(Ni, 1);
    const int2 weight_dim = make_int2(Ni, Nn);

    mmult(d_weights, in, out, weight_dim, in_dim);
    //relu(out, make_int2(weight_dim.y, in_dim.x));
}

void randomizeArray(float *data, int len)
{
	for(int i = 0; i < len; i++)
	{
		data[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 16.0f;
	}
}

int main(int argc, char **argv) 
{
     const int in_size = sizeof(float) * Ni;  // Total size of input buffer in bytes
     const int out_size = sizeof(float) * Nn; // Total size of output buffer in bytes

     const dim3 grid_size(Nn/16,16,1);
     const dim3 block_size(16,16,1);

     float h_in_data[Ni]; // Input data on host
     float *d_in_data;    // Input data on device

     float h_out_data[Nn]; // Output on device
     float *d_out_data;    // Output on host

     float *h_random_weights = new float[Ni*Nn];

     // Make some random data.

     randomizeArray(h_random_weights, Ni*Nn);
     randomizeArray(h_in_data, Ni);

     cudaMemcpyToSymbol(d_weights, h_random_weights, Nn*Ni*sizeof(float));

     cudaMalloc(&d_in_data, in_size);
     cudaMemcpy(d_in_data, h_in_data, in_size, cudaMemcpyHostToDevice); // Give the GPU our input data.

     cudaMalloc(&d_out_data, out_size);

     classify<<<grid_size, block_size>>>((float*)d_in_data, (float*)d_out_data);

     cudaDeviceSynchronize();

     cudaMemcpy(h_out_data, d_out_data, out_size, cudaMemcpyDeviceToHost); // Retrieve the neuron outputs.


     for(int i = 0; i < Nn*Ni; i++)
		 printf("%f ", h_random_weights[i]);
     printf("\n\n");

     for(int i = 0; i < Nn; i++)
    	 printf("%f ", h_in_data[i]);
     printf("\n\n");

     for(int i = 0; i < Nn; i++)
    	 printf("%f ", h_out_data[i]);
     printf("\n\n");

    return 0;
}


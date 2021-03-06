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

#define DEBUG false

/* The weights of the layer*/
__device__ float d_weights[Ni*Nn];

/*
 * Matrix multiply. Performs the operation c = b'a, where b' is the transpose of b.
 */
__device__ void mmult(float *a, float *b, float *c, int2 adim, int2 bdim)
{
    int row = threadIdx.x + blockIdx.x*blockDim.x;
    int col = threadIdx.y + blockIdx.y*blockDim.y;

    if(adim.y != bdim.y)
    {
        if(row == 0 & col == 0)
        {
            printf("Error: Incompatible matrix dimensions: [%dx%d] * [%dx%d]\n", adim.y, adim.x, bdim.x, bdim.y);
        }
    }

    if(row < adim.x && col < bdim.x)
    {
        for(int i = 0; i < adim.y; i++)
        {
            c[row*bdim.x + col] += a[row*adim.y + i] * b[col*adim.y + i];
        }
    }
}

__device__ void relu(float *mtx, int2 dim)
{
    int row = threadIdx.x + blockIdx.x*blockDim.x;
    int col = threadIdx.y + blockIdx.y*blockDim.y;

    if(row < dim.x && col < dim.y)
    {
        mtx[col*dim.x + row] *= (mtx[col*dim.x + row] > 0);
    }
}

__global__ void classify(float *in, float *out, int2 in_dim)
{
    const int2 weight_dim = make_int2(Nn, Ni);

    mmult(d_weights, in, out, weight_dim, in_dim);
    //relu(out, make_int2(weight_dim.y, in_dim.x));
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
    const int2 in_dim = make_int2(Nb, Ni);    					// The dimensions of the input matrix
    const int in_size = sizeof(float) * in_dim.x * in_dim.y;   // Total size of input buffer in bytes

    const int2 out_dim = make_int2(Nn, in_dim.x);               // The dimensions of the output matrix
    const int out_size = sizeof(float) * out_dim.x * out_dim.y; // Total size of output buffer in bytes

    const dim3 grid_size((out_dim.x * out_dim.y)/16,16,1);
    const dim3 block_size(16,16,1);

    float *h_in_data = new float[in_dim.x * in_dim.y]; // Input data on host
    float *d_in_data;    // Input data on device

    float *h_out_data = new float[out_dim.x * out_dim.y];    // Output on device
    float *d_out_data;    // Output on host

    float *h_random_weights = new float[Ni*Nn];

    // Make some random data.

    randomizeArray(h_random_weights, Ni*Nn);
    randomizeArray(h_in_data, in_dim.x * in_dim.y);

    cudaMemcpyToSymbol(d_weights, h_random_weights, Nn*Ni*sizeof(float));

    cudaMalloc(&d_in_data, in_size);
    cudaMemcpy(d_in_data, h_in_data, in_size, cudaMemcpyHostToDevice); // Give the GPU our input data.

    cudaMalloc(&d_out_data, out_size);

    classify<<<grid_size, block_size>>>((float*)d_in_data, (float*)d_out_data, in_dim);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out_data, d_out_data, out_size, cudaMemcpyDeviceToHost); // Retrieve the neuron outputs.


    if(DEBUG)
    {
        for(int i = 0; i < Nn*Ni; i++)
            printf("%f ", h_random_weights[i]);
        printf("\n\n");

        for(int i = 0; i < in_dim.y; i++)
        {
            for(int j = 0; j < in_dim.x; j++)
            {
                printf("%f ", h_in_data[i*in_dim.x + j]);
            }
            printf("\n");
        }

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


/**
 * classifier.cu
 * 
 * A CUDA kernel for accelerating a fully-connected neural network layer.
 */

#include <iostream>
#include <string>
#include <fstream>

using namespace std;

/* The weights of the current layer*/
__device__ float d_weights[785*16];

/*
 * Matrix multiply. Performs the operation c = ab', where b' is the transpose of b.
 */
__device__ void mmult(float *a, float *b, float *c, int2 adim, int2 bdim)
{
    int row = threadIdx.x + blockIdx.x*blockDim.x;
    int col = threadIdx.y + blockIdx.y*blockDim.y;

    if(adim.x != bdim.x)
    {
    	if(row == 0 & col == 0)
    	{
    		printf("Error: Incompatible matrix dimensions: [%dx%d] * [%dx%d]\n", adim.y, adim.x, bdim.x, bdim.y);
    	}
    }

    if(row < bdim.y && col < adim.x)
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
			mtx[col*dim.x + row] *= (mtx[col*dim.x + row] > 0);
    }
}

__global__ void classify(float *in, float *out, int2 in_dim, int2 weight_dim)
{
    mmult(d_weights, in, out, weight_dim, in_dim);
    relu(out, make_int2(weight_dim.y, in_dim.x));
}


int main(int argc, char **argv) 
{
     const int2 in_dim = make_int2(785, 1);    					// The dimensions of the input matrix
     const int in_size = sizeof(float) * in_dim.x * in_dim.y;   // Total size of input buffer in bytes

     const int2 out_dim = make_int2(1, 10);               // The dimensions of the output matrix
     const int out_size = sizeof(float) * out_dim.x * out_dim.y; // Total size of output buffer in bytes

     const dim3 grid_size(50,16,1);
     const dim3 block_size(16,16,1);

     float h_in_data[785]; // Input data on host
     float *d_in_data;    // Input data on device

     float h_hidden[17];
     float *d_hidden; // Hidden layer activations

     float h_out_data[10]; // Output on device
     float *d_out_data;    // Output on host

     float *weights_1 = new float[785*16]();
     float *weights_2 = new float[17*10]();

     ifstream weightfile;

     weightfile.open("example/weights_1");

     int arrayIdx = 0;
     while(!weightfile.eof())
    	 weightfile >> weights_1[arrayIdx++];

     weightfile.close();
     weightfile.open("example/weights_2");

     arrayIdx = 0;
     while(!weightfile.eof())
    	 weightfile >> weights_2[arrayIdx++];

     weightfile.close();
     weightfile.open("example/input_1");

     arrayIdx = 0;
     h_in_data[arrayIdx++] = 1.0f;

     while(!weightfile.eof())
    	 weightfile >> h_in_data[arrayIdx++];


     //////////////////////////
     // First Layer
     //////////////////////////

     cudaMemcpyToSymbol(d_weights, weights_1, 785*16*sizeof(float));

     cudaMalloc(&d_in_data, in_size);
     cudaMemcpy(d_in_data, h_in_data, in_size, cudaMemcpyHostToDevice); // Give the GPU our input data.

     cudaMalloc(&d_hidden, 17);

     classify<<<grid_size, block_size>>>(d_in_data, &d_hidden[1], make_int2(785,1), make_int2(785,16));
     cudaDeviceSynchronize();

     //////////////////////////
     // Second Layer
     //////////////////////////

     cudaMemcpyToSymbol(d_weights, weights_2, 17*10*sizeof(float));

     h_hidden[0] = 1.0f;
     cudaMemcpy(d_hidden, h_hidden, 1, cudaMemcpyHostToDevice);

     cudaMalloc(&d_out_data, out_size);

     classify<<<grid_size, block_size>>>(d_hidden, d_out_data, make_int2(17,1), make_int2(17,10));
     cudaDeviceSynchronize();

     cudaMemcpy(h_out_data, d_out_data, out_size, cudaMemcpyDeviceToHost); // Retrieve the neuron outputs.

     printf("Results: \n");
     for(int i = 0; i < out_dim.x; i++)
     {
         float max = -1;
         int argmax = 0;

    	 for(int j = 0; j < out_dim.y; j++)
    	 {
    		 float neuron = h_out_data[i*out_dim.y + j];

    		 if(neuron > max)
    		 {
    			 max = neuron;
    			 argmax = j;
    		 }
        	 printf("%f ", neuron);
    	 }
         printf("\n");
         printf("Predicted number: %d\n\n", argmax);
     }



    return 0;
}


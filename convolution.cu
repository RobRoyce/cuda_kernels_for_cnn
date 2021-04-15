#include <iostream>
#include <string>
#include <cuda.h>
#include <math.h>
#include "dnn.h"

using namespace std;

#define V100_CUDA_CAP 7.0
#define V100_GLOBAL_MEM_TOTAL 12621381632
#define V100_SM_COUNT 80
#define V100_CUDA_CORES_PER_SM 64
#define V100_CUDA_CORES_TOTAL 5120
#define V100_L2_SIZE 4718592
#define V100_SH_MEM_PER_BLOCK 49152
#define V100_REGS_PER_BLOCK 65536
#define V100_WARP_SIZE 32
#define V100_MAX_THREADS_PER_SM 2048
#define V100_MAX_THREADS_PER_BLOCK 1024


//Define the parameters if not defined externally
#ifndef Sy
#define Sy 1 // Stride in y (?)
#define Sx 1 // Stride in x (?)
#endif

#ifndef Pad
#define Pad 1
#endif

#define SYNAPSE_SIZE (1L*Ky*Kx*Nn*Ni)
#define Ox (Nx - Kx + 2*Pad)/(Sx + 1)
#define Oy Ox

#ifndef TYPE
#define TYPE float
#endif

__global__
void convolution(TYPE* X, TYPE* w, TYPE* y, int3 inDim, int4 wtsDim, int3 outDim) {
    y[blockIdx.x + threadIdx.x] = X[blockIdx.x + threadIdx.x] * w[blockIdx.x + threadIdx.x];
}

int main(int argc, char **argv) {
    const int3 INPUT_DIM = make_int3(Nx, Ny, Ni);
    const int INPUT_SIZE = Nx * Ny * Ni;
    const int INPUT_MEM = INPUT_SIZE * sizeof(TYPE);

    const int3 OUTPUT_DIM = make_int3(Ox, Oy, Nn);
    const int OUTPUT_SIZE = Ox * Oy * Nn;
    const int OUTPUT_MEM = OUTPUT_SIZE * sizeof(TYPE);

    const int4 WEIGHTS_DIM = make_int4(Kx, Ky, Ni, Nn);
    const int WEIGHTS_SIZE = Kx * Ky * Ni * Nn;
    const int WEIGHTS_MEM = WEIGHTS_SIZE * sizeof(TYPE);


    printf("Stride (Sx, Sy): (%d, %d)\n", Sx, Sy);
    printf("Padding: %d\n", Pad);
    printf("Input dimensions (Nx, Ny, Ni): (%d, %d, %d)\n", INPUT_DIM.x, INPUT_DIM.y, INPUT_DIM.z);
    printf("Input memory size: %d bytes\n", INPUT_MEM);
    printf("Output dimensions (Nx, Ny, Nn): (%d, %d, %d)\n", OUTPUT_DIM.x, OUTPUT_DIM.y, OUTPUT_DIM.z);
    printf("Output memory size: %d bytes\n", OUTPUT_MEM);
    printf("Weight dimensions (Kx, Ky, Ni, Nn): (%d, %d, %d, %d)\n", WEIGHTS_DIM.x, WEIGHTS_DIM.y, WEIGHTS_DIM.z, WEIGHTS_DIM.w);
    printf("Weight memory size: %d bytes\n", WEIGHTS_MEM);


    TYPE *h_input = new TYPE[INPUT_SIZE];
    TYPE *h_weights = new TYPE[WEIGHTS_SIZE];
    TYPE *h_output = new TYPE[INPUT_SIZE];
    TYPE *d_input, *d_weights, *d_output;

    cudaMalloc((void**)&d_input, INPUT_MEM);
    cudaMalloc((void**)&d_output, OUTPUT_MEM);
    cudaMalloc((void**)&d_weights, WEIGHTS_MEM);

    randomizeArray(h_input, INPUT_SIZE);
    randomizeArray(h_weights, WEIGHTS_SIZE);

    cudaMemcpy(d_weights, h_weights, WEIGHTS_MEM, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, INPUT_MEM, cudaMemcpyHostToDevice);

    dim3 blocksPerGrid(5, 1, 1);
    dim3 threadsPerBlock(224, 1, 1);

    begin_roi();
    convolution<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_weights, d_output, INPUT_DIM, WEIGHTS_DIM, OUTPUT_DIM);
    end_roi();


    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, OUTPUT_MEM, cudaMemcpyDeviceToHost); // Retrieve the neuron outputs.


    cout << "Sample output from device calculations:" << endl;
    int nonZero = 0;
    for(int i = 0; i < OUTPUT_SIZE; i++) {
        if(h_output[i] != 0)
            nonZero++;
    }
    cout << "Number of non-zero entries: " << nonZero << endl;


    free(h_input);
    free(h_weights);
    free(h_output);
    return 0;
}
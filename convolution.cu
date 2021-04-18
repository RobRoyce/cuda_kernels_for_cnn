#include <iostream>
#include <string>
#include <cuda.h>
#include <math.h>
#include <memory>
#include "dnn.h"

#define Batch 1
#define Pad 1
#define StrideX 1
#define StrideY 1

#define NxPad (Nx + (2*Pad))
#define NyPad (Ny + (2*Pad))
#define Ox (((Nx - Kx + 2*Pad) / StrideX) + 1)
#define Oy Ox

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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

using namespace std;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


__device__ float d_input[Ni][NyPad][NxPad];
__device__ float d_output[Nn][Oy][Ox];
__device__ float d_filters[Nn][Ni][Ky][Kx];

float h_input[Ni][NyPad][NxPad];
float h_output[Nn][Oy][Ox];
float h_filters[Nn][Ni][Ky][Kx];

bool DEBUG = true;

__global__ void convolution();
__host__ void randomizeFilters();
__host__ void randomizeInput();
__host__ void padInput();




int main(int argc, char **argv) {
    // Input, output, and filter size (N elements)
    const int I_SIZE = Ni * NyPad * NxPad;
    const int O_SIZE = Nn * Oy * Ox;
    const int F_SIZE = Nn * Ni * Ky * Kx;

    // Input, output, and filter size in memory
    const int I_MEM_SIZE = I_SIZE * sizeof(float);
    const int O_MEM_SIZE = O_SIZE * sizeof(float);
    const int F_MEM_SIZE = F_SIZE * sizeof(float);


    dim3 blocksPerGrid(Ox, Oy, Nn);
    dim3 threadsPerBlock(Kx, Ky, 64);

    // Randomize inputs/filters and set padded regions to 0
    randomizeFilters();
    randomizeInput();
    padInput();


    if (DEBUG) {
        printf("\n\n");
        printf("Padding: %d\n", Pad);
        printf("Stride (StrideX, StrideY): (%d, %d)\n", StrideX, StrideY);

        printf("\n\n");
        printf("Input dimensions (Nx, Ny, Ni): (%d, %d, %d)\n", Nx, Ny, Ni);
        printf("Input dimensions with Pad (Nx+%d, Ny+%d, Ni): (%d, %d, %d)\n", (2 * Pad), (2 * Pad), NxPad, NyPad,
               Ni);
        printf("Input number of elements: %dx%dx%d = %d\n", Nx, Ny, Ni, Nx * Ny * Ni);
        printf("Input memory size: %d bytes\n", I_MEM_SIZE);

        printf("\n\n");
        printf("Output dimensions (Ox, Oy, Nn): (%d, %d, %d)\n", Ox, Oy, Nn);
        printf("Output number of elements: %dx%dx%d = %d\n", Ox, Oy, Nn, Ox * Oy * Nn);
        printf("Output memory size: %d bytes\n", O_MEM_SIZE);

        printf("\n\n");
        printf("Weights dimensions (Kx, Ky, Ni, Nn): (%d, %d, %d, %d)\n", Kx, Ky, Ni, Nn);
        printf("Weights number of elements: %dx%dx%dx%d = %d\n", Kx, Ky, Ni, Nn, Kx * Ky * Ni * Nn);
        printf("Weights memory size: %d bytes\n", F_MEM_SIZE);

        printf("\n\n");
        printf("Blocks-Per-Grid: (%d, %d, %d)\n", blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z);
        printf("Threads-Per-Block: (%d, %d, %d)\n", threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);
        printf("\n\n");
    }

    int nonZero = 0;
    for (int nn = 0; nn < Nn; nn++)
        for (int oy = 0; oy < Oy; oy++)
            for (int ox = 0; ox < Ox; ox++)
                if (h_output[nn][oy][ox] != 0)
                    nonZero++;
    printf("Number of non-zero elements in h_output: %d\n", nonZero);


    // Begin GPU operations
    gpuErrchk(cudaMemcpyToSymbol(d_filters, h_filters, F_MEM_SIZE));
    gpuErrchk(cudaMemcpyToSymbol(d_input, h_input, I_MEM_SIZE));

    begin_roi();
    convolution<<<blocksPerGrid, threadsPerBlock>>>();
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    end_roi();


    gpuErrchk(cudaMemcpyFromSymbol(h_output, d_output, O_MEM_SIZE));

    nonZero = 0;
    for (int nn = 0; nn < Nn; nn++)
        for (int oy = 0; oy < Oy; oy++)
            for (int ox = 0; ox < Ox; ox++)
                if (h_output[nn][oy][ox] != 0)
                    nonZero++;
    printf("Number of non-zero elements in h_output: %d\n", nonZero);

    return 0;
}







__global__
void convolution() {
    int ox = blockIdx.x;
    int oy = blockIdx.y;
    int nn = blockIdx.z;
    int kx = threadIdx.x;
    int ky = threadIdx.y;
    int ni = threadIdx.z;


    // TODO Account for larger input (Ni)
    __shared__ float sum[Ni];
    float value;

    if (kx == 0 && ky == 0 && ni == 0)
        for (int i = 0; i < Ni; i++)
            sum[i] = 0;

    __syncthreads();

    for (int in_chunk = 0; in_chunk < Ni / 64; in_chunk++) {
        value = d_input[in_chunk * 64 + ni][oy + ky][ox + kx] * d_filters[nn][in_chunk * 64 + ni][ky][kx];
        atomicAdd(&sum[in_chunk * 64 + ni], value);
    }

    __syncthreads();

    if (kx == 0 && ky == 0)
        for (int in_chunk = 0; in_chunk < Ni / 64; in_chunk++) {
            atomicAdd(&d_output[nn][oy][ox], sum[in_chunk * 64 + ni]);
        }


//    for (int ny = ny_start; ny < ny_end; ny++)
//        for (int nx = nx_start; nx < nx_end; nx++)
//                d_output[nn][oy][ox] += d_input[ni][ny][nx] * d_filters[nn][ni][ky][kx];

//            d_output[nn][oy][ox] =     in[ni][ny][nx] * filt[nn][ni][ky][kx]
//            d_output[0][0][0] =     in[0][0][0] * filt[0][0][0][0]
//                                +   in[0][0][1] * filt[0][0][0][1]
//                                +   in[0][0][2] * filt[0][0][0][2]
//                                +   in[0][1][0] * filt[0][0][1][0]
//                                +   in[0][1][1] * filt[0][0][1][1]
//                                +   in[0][1][2] * filt[0][0][1][2]
//                                +   in[0][2][0] * filt[0][0][2][0]
//                                +   in[0][2][1] * filt[0][0][2][1]
//                                +   in[0][2][2] * filt[0][0][2][2]
//                                +   in[1][0][0] * filt[0][1][0][0]
//                                +   in[1][0][1] * filt[0][1][0][1]
//                                +   in[1][0][2] * filt[0][1][0][2]
//                                +   in[1][1][0] * filt[0][1][1][0]
//                                +   in[1][1][1] * filt[0][1][1][1]
//                                +   in[1][1][2] * filt[0][1][1][2]
//                                +   in[1][2][0] * filt[0][1][2][0]
//                                +   in[1][2][1] * filt[0][1][2][1]
//                                +   in[1][2][2] * filt[0][1][2][2]
//                                ...
//                                ...
//                                ...
//                                +   in[63][0][0] * filt[0][63][0][0]
//                                +   in[63][0][1] * filt[0][63][0][1]
//                                +   in[63][0][2] * filt[0][63][0][2]
//                                +   in[63][1][0] * filt[0][63][1][0]
//                                +   in[63][1][1] * filt[0][63][1][1]
//                                +   in[63][1][2] * filt[0][63][1][2]
//                                +   in[63][2][0] * filt[0][63][2][0]
//                                +   in[63][2][1] * filt[0][63][2][1]
//                                +   in[63][2][2] * filt[0][63][2][2]
//
//        d_output[0][0][1] =         in[0][0][1] * filt[0][0][0][0]
//                                +   in[0][0][2] * filt[0][0][0][1]
//                                +   in[0][0][3] * filt[0][0][0][2]
//                                +   in[0][1][1] * filt[0][0][1][0]
//                                +   in[0][1][2] * filt[0][0][1][1]
//                                +   in[0][1][3] * filt[0][0][1][2]
//                                +   in[0][2][1] * filt[0][0][2][0]
//                                +   in[0][2][2] * filt[0][0][2][1]
//                                +   in[0][2][3] * filt[0][0][2][2]
//                                +   in[1][0][1] * filt[0][1][0][0]
//                                +   in[1][0][2] * filt[0][1][0][1]
//                                +   in[1][0][3] * filt[0][1][0][2]
//                                +   in[1][1][1] * filt[0][1][1][0]
//                                +   in[1][1][2] * filt[0][1][1][1]
//                                +   in[1][1][3] * filt[0][1][1][2]
//                                +   in[1][2][1] * filt[0][1][2][0]
//                                +   in[1][2][2] * filt[0][1][2][1]
//                                +   in[1][2][3] * filt[0][1][2][2]
//                                ...
//                                ...
//                                ...
//                                +   in[63][0][1] * filt[0][63][0][0]
//                                +   in[63][0][2] * filt[0][63][0][1]
//                                +   in[63][0][3] * filt[0][63][0][2]
//                                +   in[63][1][1] * filt[0][63][1][0]
//                                +   in[63][1][2] * filt[0][63][1][1]
//                                +   in[63][1][3] * filt[0][63][1][2]
//                                +   in[63][2][1] * filt[0][63][2][0]
//                                +   in[63][2][2] * filt[0][63][2][1]
//                                +   in[63][2][3] * filt[0][63][2][2]
//
//        d_output[1][0][0] =         in[0][0][0] * filt[1][0][0][0]
//                                +   in[0][0][1] * filt[1][0][0][1]
//                                +   in[0][0][2] * filt[1][0][0][2]
//                                +   in[0][1][0] * filt[1][0][1][0]
//                                +   in[0][1][1] * filt[1][0][1][1]
//                                +   in[0][1][2] * filt[1][0][1][2]
//                                +   in[0][2][0] * filt[1][0][2][0]
//                                +   in[0][2][1] * filt[1][0][2][1]
//                                +   in[0][2][2] * filt[1][0][2][2]
//                                +   in[1][0][0] * filt[1][1][0][0]
//                                +   in[1][0][1] * filt[1][1][0][1]
//                                +   in[1][0][2] * filt[1][1][0][2]
//                                +   in[1][1][0] * filt[1][1][1][0]
//                                +   in[1][1][1] * filt[1][1][1][1]
//                                +   in[1][1][2] * filt[1][1][1][2]
//                                +   in[1][2][0] * filt[1][1][2][0]
//                                +   in[1][2][1] * filt[1][1][2][1]
//                                +   in[1][2][2] * filt[1][1][2][2]
//                                ...
//                                ...
//                                ...
//                                +   in[63][0][0] * filt[1][63][0][0]
//                                +   in[63][0][1] * filt[1][63][0][1]
//                                +   in[63][0][2] * filt[1][63][0][2]
//                                +   in[63][1][0] * filt[1][63][1][0]
//                                +   in[63][1][1] * filt[1][63][1][1]
//                                +   in[63][1][2] * filt[1][63][1][2]
//                                +   in[63][2][0] * filt[1][63][2][0]
//                                +   in[63][2][1] * filt[1][63][2][1]
//                                +   in[63][2][2] * filt[1][63][2][2]
//
//    d_output[nn][oy][ox] = d_input[ni][ny][nx] * d_filters[nn][ni][ky][kx];
}







__host__
void randomizeFilters() {
    for (int yy = 0; yy < Ky; ++yy)
        for (int xx = 0; xx < Kx; ++xx)
            for (int nn = 0; nn < Nn; ++nn)
                for (int ni = 0; ni < Ni; ++ni)
                    h_filters[nn][ni][yy][xx] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
}

__host__
void randomizeInput() {
    for (int ni = 0; ni < Ni; ++ni)
        for (int yy = 0; yy < NyPad; ++yy)
            for (int xx = 0; xx < NxPad; ++xx)
                h_input[ni][yy][xx] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
}

__host__
void padInput() {
    // Set padded regions to 0
    for (int z = 0; z < Ni; z++) {
        for (int x = 0; x < NxPad; x++) {
            h_input[z][0][x] = 0;
            h_input[z][NyPad - 1][x] = 0;
        }
        for (int y = 0; y < NyPad; y++) {
            h_input[z][y][0] = 0;
            h_input[z][y][NxPad - 1] = 0;
        }
    }
}
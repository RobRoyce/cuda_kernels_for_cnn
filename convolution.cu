/*
V100_CUDA_CAP 7.0
V100_GLOBAL_MEM_TOTAL 12621381632
V100_SM_COUNT 80
V100_CUDA_CORES_PER_SM 64
V100_CUDA_CORES_TOTAL 5120
V100_L2_SIZE 4718592
V100_SH_MEM_PER_BLOCK 49152
V100_REGS_PER_BLOCK 65536
V100_WARP_SIZE 32
V100_MAX_THREADS_PER_SM 2048
V100_MAX_THREADS_PER_BLOCK 1024
 */

#include <string>
#include <cmath>
#include "dnn.h"

#define Pad 1
#define StrideX 1
#define StrideY 1
#define NxPad (Nx + (2*Pad))
#define NyPad (Ny + (2*Pad))
#define Ox (((Nx - Kx + 2*Pad) / StrideX) + 1)
#define Oy Ox
#define I_SIZE (Ni * NyPad * NxPad)
#define O_SIZE (Nn * Oy * Ox)
#define F_SIZE (Nn * Ni * Ky * Kx)
#define I_MEM_SIZE (I_SIZE * sizeof(float))
#define O_MEM_SIZE (O_SIZE * sizeof(float))
#define F_MEM_SIZE (F_SIZE * sizeof(float))
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

using namespace std;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void convolution(int);

__host__ void randomizeFilters();
__host__ void randomizeInput();
__host__ void padInput();
__host__ void printParameters();

__device__ float d_input[Batch][Ni][NyPad][NxPad];
__device__ float d_output[Nn][Oy][Ox];
__device__ float d_filters[Nn][Ni][Ky][Kx];

float h_input[Batch][Ni][NyPad][NxPad];
float h_output[Nn][Oy][Ox];
float h_filters[Nn][Ni][Ky][Kx];

int main(int argc, char **argv) {
    bool DEBUG = ((argc > 1) && (std::string(argv[1]) == "--debug"));

    dim3 blocksPerGrid(Ox, Oy, Nn);
    dim3 threadsPerBlock(Kx, Ky, 64);

    // Randomize inputs/filters and set padded regions to 0
    randomizeFilters();
    randomizeInput();
    padInput();

    if (DEBUG) {
        printParameters();
        printf("\n\n");
        printf("Blocks-Per-Grid: (%d, %d, %d)\n", blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z);
        printf("Threads-Per-Block: (%d, %d, %d)\n\n\n", threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);

        int nonZero = 0;
        for (auto & nn : h_output)
            for (auto & oy : nn)
                for (float ox : oy)
                    if (ox != 0)
                        nonZero++;
        printf("Number of non-zero elements in h_output: %d\n", nonZero);
    }

    // Copy filters and input : host -> device
    gpuErrchk(cudaMemcpyToSymbol(d_input, h_input, I_MEM_SIZE * Batch));
    gpuErrchk(cudaMemcpyToSymbol(d_filters, h_filters, F_MEM_SIZE));


    // Start timer and execute kernel
    cudaStream_t streams[Batch];
    for (int i = 0; i < Batch; i++)
        cudaStreamCreate(&streams[i]);

    begin_roi();

    for (int batch = 0; batch < Batch; batch++)
        convolution<<<blocksPerGrid, threadsPerBlock, 0, streams[batch]>>>(batch);

    gpuErrchk(cudaDeviceSynchronize());
    end_roi();

    // Copy output : device -> host
    gpuErrchk(cudaMemcpyFromSymbol(h_output, d_output, O_MEM_SIZE));

    // Check output
    if (DEBUG) {
        int nonZero = 0;
        for (auto & nn : h_output)
            for (auto & oy : nn)
                for (float ox : oy)
                    if (ox != 0)
                        nonZero++;
        printf("Number of non-zero elements in h_output: %d\n", nonZero);
    }

    return 0;
}

__global__
void convolution(int batch) {
    unsigned int ox = blockIdx.x;
    unsigned int oy = blockIdx.y;
    unsigned int nn = blockIdx.z;
    unsigned int kx = threadIdx.x;
    unsigned int ky = threadIdx.y;
    unsigned int ni = threadIdx.z;

    __shared__ float sum[Ni];
    float value;

    // Use the first thread of each block to set accum. to 0
    if (kx == 0 && ky == 0 && ni == 0)
        for (int i = 0; i < Ni; i++)
            sum[i] = 0;

    // Wait until accum. is initialized
    __syncthreads();

    // Multiply-Accumulate
    for (int in_chunk = 0; in_chunk < Ni / 64; in_chunk++) {
        value = d_input[batch][in_chunk * 64 + ni][oy + ky][ox + kx] * d_filters[nn][in_chunk * 64 + ni][ky][kx];
        atomicAdd(&sum[in_chunk * 64 + ni], value);
    }

    __syncthreads();

    // Store results
    if (kx == 0 && ky == 0)
        for (int in_chunk = 0; in_chunk < Ni / 64; in_chunk++) {
            atomicAdd(&d_output[nn][oy][ox], sum[in_chunk * 64 + ni]);
        }
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
    for (int batch = 0; batch < Batch; batch++)
        for (int ni = 0; ni < Ni; ++ni)
            for (int yy = 0; yy < NyPad; ++yy)
                for (int xx = 0; xx < NxPad; ++xx)
                    h_input[batch][ni][yy][xx] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
}

__host__
void padInput() {
    // Set padded regions to 0
    for (int batch = 0; batch < Batch; batch++)
        for (int z = 0; z < Ni; z++) {
            for (int x = 0; x < NxPad; x++) {
                h_input[batch][z][0][x] = 0;
                h_input[batch][z][NyPad - 1][x] = 0;
            }
            for (int y = 0; y < NyPad; y++) {
                h_input[batch][z][y][0] = 0;
                h_input[batch][z][y][NxPad - 1] = 0;
            }
        }
}

__host__
void printParameters() {
    printf("\n\n");
    printf("Padding: %d\n", Pad);
    printf("Stride (StrideX, StrideY): (%d, %d)\n", StrideX, StrideY);

    printf("\n\n");
    printf("Input dimensions (Nx, Ny, Ni): (%d, %d, %d)\n", Nx, Ny, Ni);
    printf("Input dimensions with Pad (Nx+%d, Ny+%d, Ni): (%d, %d, %d)\n", (2 * Pad), (2 * Pad), NxPad, NyPad,
           Ni);
    printf("Input number of elements: %dx%dx%d = %d\n", Nx, Ny, Ni, Nx * Ny * Ni);
    printf("Input memory size: %lu bytes\n", I_MEM_SIZE);

    printf("\n\n");
    printf("Output dimensions (Ox, Oy, Nn): (%d, %d, %d)\n", Ox, Oy, Nn);
    printf("Output number of elements: %dx%dx%d = %d\n", Ox, Oy, Nn, Ox * Oy * Nn);
    printf("Output memory size: %lu bytes\n", O_MEM_SIZE);

    printf("\n\n");
    printf("Weights dimensions (Kx, Ky, Ni, Nn): (%d, %d, %d, %d)\n", Kx, Ky, Ni, Nn);
    printf("Weights number of elements: %dx%dx%dx%d = %d\n", Kx, Ky, Ni, Nn, Kx * Ky * Ni * Nn);
    printf("Weights memory size: %lu bytes\n", F_MEM_SIZE);
}
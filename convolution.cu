#include <iostream>
#include <string>
#include <cuda.h>
#include <math.h>
#include "dnn.h"

using namespace std;

#define CUDA_CAP_V100 7.0
#define SM_COUNT_2070 40
#define CUDA_CAP_2070 7.5

//Define the parameters if not defined externally
#ifndef Sy
#define Sy 1 // Stride in y (?)
#define Sx 1 // Stride in x (?)
#endif

#ifndef Tnn
//Tiling Sizes
#define Tnn 32
#define Tn  16
#define Ti  16

#define Ty  8
#define Tx  8
#endif

#define NYPAD (Ny+Ky)
#define NXPAD (Nx+Kx)

#define NYSCL (Ny/Sy)
#define NXSCL (Nx/Sx)

#define SYNAPSE_SIZE (1L*Ky*Kx*Nn*Ni)

VTYPE (*synapse)[Ky][Kx][Nn][Ni];
VTYPE  (*neuron_i)[NYPAD][NXPAD][Ni];
VTYPE  (*neuron_n)[NYSCL][NXSCL][Nn];
VTYPE (*neuron_n2)[NYSCL][NXSCL][Nn];


__global__
void add(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    y[index] = x[index] + y[index];
}


void fill_convolution_shared_simple(float (&synapse)[Ky][Kx][Nn][Ni],
                                    float (&neuron_i)[NYPAD][NXPAD][Ni]) {
    for (int yy = 0; yy < Ky; ++yy) {
        for (int xx = 0; xx < Kx; ++xx) {
            for (int nn = 0; nn < Nn; ++nn) {
                for (int ni = 0; ni < Ni; ++ni) {
                    synapse[yy][xx][nn][ni] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
                }
            }
        }
    }

    for (int yy = 0; yy < NYPAD; ++yy) {
        for (int xx = 0; xx < NXPAD; ++xx) {
            for (int ni = 0; ni < Ni; ++ni) {
                neuron_i[yy][xx][ni] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
            }
        }
    }
}


std::pair<int, int> convolution_layer_blocked(
        VTYPE (&synapse)[Ky][Kx][Nn][Ni],
        VTYPE (&neuron_i)[NYPAD][NXPAD][Ni],
        VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {
    int c1 = 0, c2 = 0;
    VTYPE sum[Nn] = {0};

    for (int yy = 0; yy < Ny; yy += Ty) {
        for (int xx = 0; xx < Nx; xx += Tx) {
            for (int nnn = 0; nnn < Nn; nnn += Tnn) {
                int yout = yy / Sy;
                for (int y = yy; y < yy + Ty; y += Sy) { // tiling for y;
                    int xout = xx / Sx;

                    for (int x = xx; x < xx + Tx; x += Sx) { // tiling for x;

                        for (int nn = nnn; nn < nnn + Tnn; nn += Tn) {
                            for (int n = nn; n < nn + Tn; n++) {
                                sum[n] = 0;
                            }

                            for (int ky = 0; ky < Ky; ky++) {  // sliding window;
                                for (int kx = 0; kx < Kx; kx++) {

                                    int ii = 0;
                                    VTYPE sum_sc;

                                    for (; ii < Ni - Ti + 1; ii += Ti) {
                                        for (int n = nn; n < nn + Tn; n++) {
                                            sum_sc = 0;
                                            for (int i = ii; i < ii + Ti; i++) {
                                                VTYPE sv = synapse[ky][kx][n][i];
                                                VTYPE nv = neuron_i[ky + y][kx + x][i];
                                                sum_sc += sv * nv;
                                            }
                                            sum[n] += sum_sc;
                                        }
                                    }
                                }
                            }

                            //transfer
                            for (int n = nn; n < nn + Tn; n++) {
                                neuron_n[yout][xout][n] = transfer(sum[n]);
                            }
                        }
                        xout++;
                    }
                    yout++;
                }
            }
        }
    }
}


//void convolution_layer(VTYPE (&synapse)[Ky][Kx][Nn][Ni],
//                       VTYPE (&neuron_i)[NYPAD][NXPAD][Ni],
//                       VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {
//    VTYPE sum[Nn] = {0};
//    int nPrintouts = 0;
//
//    // — Original code — (excluding nn, ii loops)
//    int yout = 0;
//    for (int y = 0; y < Ny; y += Sy) { // tiling for y;
//        int xout = 0;
//        for (int x = 0; x < Nx; x += Sx) { // tiling for x;
//            for (int nn = 0; nn < Nn; nn += Tn) {
//                for (int n = nn; n < nn + Tn; n++) {
//                    sum[n] = 0;
//                }
//                // sliding window;
//                for (int ky = 0; ky < Ky; ky++)
//                    for (int kx = 0; kx < Kx; kx++)
//                        for (int n = nn; n < nn + Tn; n++)
//                            for (int i = 0; i < Ni; i++) {
//                                VTYPE sv = synapse[ky][kx][n][i];
//                                VTYPE nv = neuron_i[ky + y][kx + x][i];
//                                sum[n] += sv * nv;
//
////                                if(nPrintouts++ <= 1024 * 256) {
////                                    printf("Input %d - Output %d - Filter x %d - Filter y %d - Image x %d - Image y %d - Tile %d\n", i, n, kx, ky, x, y, nn);
////                                }
//                            }
//                for (int n = nn; n < nn + Tn; n++) {
//                    neuron_n[yout][xout][n] = transfer(sum[n]);
//                }
//            }
//            xout++;
//        }
//        yout++;
//    }
//}


void convolution_layer(void** synapse_d, void** neuron_i_d, void** neuron_o_d) {
    return;
}




int main(int argc, char **argv) {
    int synapseMemSize = SYNAPSE_SIZE * sizeof(VTYPE);
    int neuronInputMemSize = NYPAD * NXPAD * Ni * sizeof(VTYPE);
    int neuronOutputMemSize = NYSCL * NXSCL * Nn * sizeof(VTYPE);

    // Allocate host memory
    synapse_h = (VTYPE (*)[Ky][Kx][Nn][Ni]) aligned_malloc(64, synapseMemSize);
    neuron_i_h = (VTYPE (*)[NYPAD][NXPAD][Ni]) aligned_malloc(64, neuronInputMemSize);
    neuron_n_h = (VTYPE (*)[NYSCL][NXSCL][Nn]) aligned_malloc(64, neuronOutputMemSize);

    // Allocate device memory
    VTYPE (*synapse_d)[Ky][Kx][Nn][Ni];
    VTYPE  (*neuron_i_d)[NYPAD][NXPAD][Ni];
    VTYPE  (*neuron_n_d)[NYSCL][NXSCL][Nn];
    cudaMalloc((void**)&synapse_d, synapseMemSize);
    cudaMalloc((void**)&neuron_i_d, neuronInputMemSize);
    cudaMalloc((void**)&neuron_n_d, neuronOutputMemSize);


    // Populate input and synapse with random values
    fill_convolution_shared_simple(*synapse, *neuron_i);

    // Copy input neuron and synapse from host to device
    cudaMemcpy(synapse_h, synapse_d, synapseMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(neuron_i_h, neuron_i_d, neuronInputMemSize, cudaMemcpyHostToDevice);

    //Simple Version
    begin_roi();
    convolution_layer(*synapse, *neuron_i, *neuron_n);
    end_roi();




//    compare((VTYPE*)*neuron_n,(VTYPE*)*neuron_n2,NYSCL*NXSCL*Nn);

    cout << "done\n";

    return 0;
}
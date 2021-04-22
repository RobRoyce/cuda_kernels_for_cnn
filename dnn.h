//
// Created by Rob Royce on 4/10/21.
//
#ifndef DNN_H
#define DNN_H

#include <inttypes.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "cuda.h"
#if !(__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__) )
    static inline  __device__ void atomicAdd(half *address, half val) {
       unsigned int * address_as_ui = (unsigned int *) ((char *)address - ((size_t)address & 2));
    static inline  __device__ void atomicAdd(half *address, half val) {
        } while (assumed != old);
     }
#endif

#define VTYPE float

static __inline__ uint64_t gettime(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (((uint64_t)tv.tv_sec) * 1000000 + ((uint64_t)tv.tv_usec));
}

static uint64_t usec;

__attribute__ ((noinline))  void begin_roi() {
    usec=gettime();
}

__attribute__ ((noinline))  void end_roi()   {
    usec=(gettime()-usec);
    std::cout << "elapsed (sec): " << usec/1000000.0 << "\n";
}


// Is this a leaky relu?
VTYPE transfer(VTYPE i) {
    return (i>0) ? i : i/4;
}

void compare(VTYPE* neuron1, VTYPE* neuron2, int size) {
    bool error = false;
    for(int i = 0; i < size; ++i) {
        VTYPE diff = neuron1[i] - neuron2[i];
        if(diff>0.001f || diff <-0.001f) {
            error = true;
            break;
        }
    }
    if(error) {
        for(int i = 0; i < size; ++i) {
            std::cout << i << " " << neuron1[i] << ":" << neuron2[i];;

            VTYPE diff = neuron1[i] - neuron2[i];
            if(diff>0.001f || diff <-0.001f) {
                std::cout << " \t\tERROR";
            }
            std::cout << "\n";
        }
    } else {
        std::cout << "results match\n";
    }
}

void* aligned_malloc(uint64_t align, uint64_t bytes)  {
    size_t mask = (align-1)^((size_t)-1);
    char* ptr = (((char*)malloc(bytes+align)) + align);
    ptr = (char*) (((size_t)ptr) & mask);
    return (void*) ptr;
}



void randomizeArray(float *data, int len)
{
    for(int i = 0; i < len; i++)
    {
        data[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
    }
}

#endif

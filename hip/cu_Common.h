#pragma once
#include "hip/hip_runtime.h"
#include "device_launch_parameters.h"
#include <hiprand.h>
#include <hiprand_kernel.h>
#include "hip/hip_runtime_api.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <vector_functions.h>

#ifndef __HIPCC__
#define __HIPCC__
#endif
#include "device_atomic_functions.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char* file, int line, bool abort = true)
{
    if (code != hipSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void cudaVerifyLaunch() {
    auto err = hipGetLastError();
    if (err != hipSuccess)
    {
        printf("%s", hipGetErrorString(err));
        exit(11);
    }
}

//cu_rand64, cu_rand32
__device__ uint64_t cu_rand64(hiprandStateXORWOW* rnd) {
    return (((uint64_t)hiprand(rnd)) << 32) | hiprand(rnd);
}
__device__ uint64_t cu_rand32(uint32_t& x, uint32_t& y, uint32_t& z) {
    uint32_t t;
    x ^= x << 16; x ^= x >> 5; x ^= x << 1;
    t = x; x = y; y = z; z = t ^ x ^ y;
    return z;
}
__device__ uint64_t cu_rand64(uint32_t& x, uint32_t& y, uint32_t& z) {
    return (((uint64_t)cu_rand32(x, y, z)) << 32) | cu_rand32(x, y, z);
}
__device__ uint64_t cu_rand32(hiprandStateXORWOW* rnd) {
    return hiprand(rnd);
}

//getIdx
__device__ uint32_t getIdx() { return blockIdx.x * blockDim.x + threadIdx.x; }





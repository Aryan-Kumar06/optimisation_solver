#pragma once

// Deterministic block reductions.
//
// Solver reductions are never accumulated with floating-point atomics: the
// order in which blocks reach an atomicAdd is a scheduling accident, so two
// runs of the same solve would diverge in the last bits and then, through the
// linesearch, in their iteration counts. Instead each block reduces its own
// values in a fixed tree order and writes one partial per block; a separate
// single-block pass sums the partials in index order. The result is
// reproducible run to run for a given launch configuration.

#include <cuda_runtime.h>

namespace cuda_support {

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;

// Sum across the 32 lanes of a warp. Every lane must call it.
__device__ inline double warpSum(double value) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullMask, value, offset);
    }
    return value;
}

// Sums K values across the block. `scratch` must hold K * (blockDim.x / 32)
// doubles and blockDim.x must be a multiple of 32. The totals are valid in
// thread 0 only. Every thread of the block must call it.
template <int K>
__device__ inline void blockSum(double (&values)[K], double* scratch) {
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const int warps = blockDim.x / kWarpSize;

#pragma unroll
    for (int k = 0; k < K; ++k) {
        const double warpTotal = warpSum(values[k]);
        if (lane == 0) {
            scratch[k * warps + warp] = warpTotal;
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int k = 0; k < K; ++k) {
            double v = lane < warps ? scratch[k * warps + lane] : 0.0;
            v = warpSum(v);
            values[k] = v;
        }
    }
    __syncthreads();
}

// Sums `count` partials laid out as partials[k * stride + i] for each of the K
// sums, in index order, into out[k]. Launched as a single block; count may be
// zero, in which case every output is exactly 0.0.
template <int K>
__global__ void sumPartialsKernel(
    const double* __restrict__ partials,
    int count,
    int stride,
    double* __restrict__ out
) {
    extern __shared__ double scratch[];
    double local[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        local[k] = 0.0;
    }
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
#pragma unroll
        for (int k = 0; k < K; ++k) {
            local[k] += partials[k * stride + i];
        }
    }
    blockSum<K>(local, scratch);
    if (threadIdx.x == 0) {
#pragma unroll
        for (int k = 0; k < K; ++k) {
            out[k] = local[k];
        }
    }
}

}  // namespace cuda_support

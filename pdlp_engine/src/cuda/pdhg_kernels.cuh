#pragma once

// Device kernels for the CUDA PDHG backend.
//
// The CPU kernel's structure is preserved: the primal half walks CSC columns
// and consumes each column's (A^T y)_j straight into its prox; the dual half
// walks CSR rows and consumes (A x')_i straight into its prox. No A^T y or A x'
// vector is ever materialised, and the halves are separated by exactly one
// global dependency -- the kernel boundary -- which is the one PDHG requires.
//
// Work mapping. A line (a row or a column) is reduced by a GROUP of G adjacent
// lanes, G a power of two chosen per matrix from its mean line length: one
// thread per line wastes bandwidth on long lines, one warp per line idles most
// lanes on the short lines typical of LPs. Lines longer than a threshold are
// excluded here and reduced by heavyLineKernel, one thread block per line, so
// a handful of dense rows (material balances, linking constraints) cannot
// serialise a warp while the rest of the grid waits.
//
// Reductions. Each block writes its partial sums to its own slot; nothing is
// accumulated with floating-point atomics, so results are reproducible run to
// run. See cuda_support/reduce.cuh.

#include "cuda_support/reduce.cuh"
#include "pdlp/pdhg_math.h"

#include <cstdint>

namespace pdlp {
namespace cuda {

using Offset = std::int64_t;

constexpr int kBlockSize = 256;

// Every kernel reports the same three sums so that all partial slots share one
// layout: [0] primal movement, [1] dual movement, [2] interaction.
constexpr int kSums = 3;

// A sparse matrix seen as a sequence of lines: CSR rows or CSC columns.
struct LineMatrix {
    const Offset* start;   // lines + 1 offsets (64-bit: nnz may exceed 2^31)
    const int* index;      // column (CSR) or row (CSC) of each entry
    const double* value;
    int lines;
    Offset heavyNonzeros;  // longer lines are left to heavyLineKernel
};

struct HeavyLines {
    const int* line;
    int count;
};

// sum_k value[k] * vector[index[k]] over one line, by the G lanes of a group.
// Every lane of the warp must call this, because the group reduction is a
// shuffle; lanes with no line pass active = false and contribute zero. The
// result is valid in the group's first lane.
template <int G>
__device__ inline double groupDot(
    const LineMatrix& matrix,
    const double* __restrict__ vector,
    int line,
    int laneInGroup,
    bool active
) {
    double sum = 0.0;
    if (active) {
        const Offset end = matrix.start[line + 1];
        for (Offset k = matrix.start[line] + laneInGroup; k < end; k += G) {
            sum += matrix.value[k] * vector[matrix.index[k]];
        }
    }
#pragma unroll
    for (int offset = G / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(cuda_support::kFullMask, sum, offset, G);
    }
    return sum;
}

// ---------------------------------------------------------------------------
// Epilogues: what happens to a line's dot product. Each mirrors one loop body
// of CpuPdhgKernel and delegates the arithmetic to pdhg_math.h.
// ---------------------------------------------------------------------------

// Primal half, one column: x'_j from (A^T y)_j. See CpuPdhgKernel::primalHalf.
struct PrimalStep {
    const double* __restrict__ objective;
    const double* __restrict__ lower;
    const double* __restrict__ upper;
    const double* __restrict__ scale;
    const double* __restrict__ scaleInverse;
    const double* __restrict__ primal;
    double* __restrict__ trial;
    double multiplier;  // eta / omega

    __device__ void apply(int column, double dot, double (&sums)[kSums]) const {
        const double gradient = objective[column] + dot;
        const double previous = primal[column];
        const double updated = math::primalUpdate(
            previous, multiplier, scale[column], gradient, lower[column], upper[column]);
        const double delta = updated - previous;
        trial[column] = updated;
        sums[0] += delta * delta * scaleInverse[column];
    }
};

// Dual half, one row: y'_i and (A x')_i from (A x')_i. See
// CpuPdhgKernel::dualHalf.
struct DualStep {
    const double* __restrict__ lower;
    const double* __restrict__ upper;
    const double* __restrict__ scale;
    const double* __restrict__ scaleInverse;
    const double* __restrict__ dual;
    const double* __restrict__ activity;
    double* __restrict__ trialDual;
    double* __restrict__ trialActivity;
    double multiplier;  // eta * omega

    __device__ void apply(int row, double nextActivity, double (&sums)[kSums]) const {
        const double previous = dual[row];
        const math::DualUpdate step = math::dualUpdate(
            previous, nextActivity, activity[row], multiplier, scale[row],
            lower[row], upper[row]);
        const double delta = step.value - previous;
        trialDual[row] = step.value;
        trialActivity[row] = nextActivity;
        sums[1] += delta * delta * scaleInverse[row];
        sums[2] += delta * step.deltaActivity;
    }
};

// Plain product, out = A x: refreshes the carried activity after a restart.
struct StoreProduct {
    double* __restrict__ out;

    __device__ void apply(int line, double dot, double (&)[kSums]) const {
        out[line] = dot;
    }
};

// Writes a block's sums to its slot, or nothing when partials is null (the
// plain product reduces nothing). `slot` is the block's first slot index.
__device__ inline void storeBlockSums(
    double (&sums)[kSums],
    double* scratch,
    double* __restrict__ partials,
    int stride,
    int slot
) {
    if (partials == nullptr) {
        return;
    }
    cuda_support::blockSum<kSums>(sums, scratch);
    if (threadIdx.x == 0) {
#pragma unroll
        for (int k = 0; k < kSums; ++k) {
            partials[k * stride + slot + blockIdx.x] = sums[k];
        }
    }
}

// Group-per-line kernel over every non-heavy line.
//
// The loop bound depends only on blockIdx, so all lanes of a warp run the same
// number of iterations and every shuffle has its full warp present. Grid-stride
// rather than one group per line: the grid is sized to the device, not to the
// matrix, so the partial-sum buffer has a fixed, small size.
template <int G, class Epilogue>
__global__ void __launch_bounds__(kBlockSize) lineKernel(
    LineMatrix matrix,
    const double* __restrict__ vector,
    Epilogue epilogue,
    double* __restrict__ partials,
    int stride,
    int slot
) {
    static_assert(G >= 1 && G <= cuda_support::kWarpSize && (G & (G - 1)) == 0,
                  "group size must be a power of two no larger than a warp");
    __shared__ double scratch[kSums * (kBlockSize / cuda_support::kWarpSize)];

    double sums[kSums] = {0.0, 0.0, 0.0};
    constexpr int kGroupsPerBlock = kBlockSize / G;
    const int laneInGroup = static_cast<int>(threadIdx.x) % G;
    const int groupInBlock = static_cast<int>(threadIdx.x) / G;
    const long long lines = matrix.lines;

    for (long long base = static_cast<long long>(blockIdx.x) * kGroupsPerBlock;
         base < lines;
         base += static_cast<long long>(gridDim.x) * kGroupsPerBlock) {
        const long long candidate = base + groupInBlock;
        bool active = candidate < lines;
        const int line = active ? static_cast<int>(candidate) : 0;
        if (active) {
            active = matrix.start[line + 1] - matrix.start[line] <= matrix.heavyNonzeros;
        }
        const double dot = groupDot<G>(matrix, vector, line, laneInGroup, active);
        if (active && laneInGroup == 0) {
            epilogue.apply(line, dot, sums);
        }
    }

    storeBlockSums(sums, scratch, partials, stride, slot);
}

// Block-per-line kernel for the heavy lines only.
template <class Epilogue>
__global__ void __launch_bounds__(kBlockSize) heavyLineKernel(
    LineMatrix matrix,
    HeavyLines heavy,
    const double* __restrict__ vector,
    Epilogue epilogue,
    double* __restrict__ partials,
    int stride,
    int slot
) {
    __shared__ double scratch[kSums * (kBlockSize / cuda_support::kWarpSize)];
    __shared__ double dotScratch[kBlockSize / cuda_support::kWarpSize];

    double sums[kSums] = {0.0, 0.0, 0.0};
    // Uniform per block, so every thread reaches every blockSum.
    for (int h = blockIdx.x; h < heavy.count; h += gridDim.x) {
        const int line = heavy.line[h];
        double dot[1] = {0.0};
        const Offset end = matrix.start[line + 1];
        for (Offset k = matrix.start[line] + threadIdx.x; k < end; k += blockDim.x) {
            dot[0] += matrix.value[k] * vector[matrix.index[k]];
        }
        cuda_support::blockSum<1>(dot, dotScratch);
        if (threadIdx.x == 0) {
            epilogue.apply(line, dot[0], sums);
        }
    }

    storeBlockSums(sums, scratch, partials, stride, slot);
}

// avg += w/W * (x - avg) over the primal and the dual in one launch; see
// IterateAverage::add.
__global__ void __launch_bounds__(kBlockSize) blendKernel(
    double* __restrict__ averagePrimal,
    const double* __restrict__ primal,
    int columns,
    double* __restrict__ averageDual,
    const double* __restrict__ dual,
    int rows,
    double fraction
) {
    const long long total = static_cast<long long>(columns) + rows;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        if (i < columns) {
            averagePrimal[i] = math::blended(averagePrimal[i], primal[i], fraction);
        } else {
            const long long row = i - columns;
            averageDual[row] = math::blended(averageDual[row], dual[row], fraction);
        }
    }
}

}  // namespace cuda
}  // namespace pdlp

// Hybrid CUDA implementation of AdmmBackend.
//
// Ownership, explicitly:
//   device  A, A^T, P (cuSPARSE CSR), q, l, u; z, y, A*x, zOld, yOld, bestY;
//           a device copy of x for the products.
//   host    the KKT factorisation and both triangular solves (KktSolver),
//           rho adaptation, the termination test, the certificates, and the
//           authoritative x, which the KKT solve produces.
//
// So each iteration moves n doubles down (the x-update right-hand side) and n
// doubles up (its solution), plus three scalars down for the residual norms and
// objective that rho adaptation and best-iterate tracking read every
// iteration. Everything else stays resident; y, z and A*x are downloaded only
// at termination-check boundaries and at the end. BackendProfile and
// AdmmResult::kktSolveSeconds record what that costs, so whether the hybrid
// pays for itself is a measurement, not a claim.
//
// Products. A^T is stored as its own CSR matrix (built from the host CSC
// arrays, which ARE the CSR of A^T) rather than applied as a transposed CSR
// product: cuSPARSE's transposed CSR SpMV accumulates with atomics and is not
// reproducible, while every product here is a plain, deterministic one
// (CUSPARSE_SPMV_CSR_ALG2). P is stored in full symmetric form exactly as on the
// host, so P*x is a plain CSR product and no off-diagonal is counted twice.

#include "qp/admm_backend.h"
#include "qp/compute_backend.h"
#include "qp/kkt_solver.h"

#include "cuda_support/cuda_check.h"
#include "cuda_support/device_buffer.h"
#include "cuda_support/reduce.cuh"

#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace qp {
namespace {

using cuda_support::DeviceBuffer;
using Clock = std::chrono::steady_clock;

constexpr int kBlockSize = 256;
constexpr int kSums = 3;

double secondsSince(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// std::min / std::max semantics (NaN in the first argument propagates), not
// fmin/fmax, which would silently repair a NaN; see pdlp's pdhg_math.h.
__device__ inline double minOf(double a, double b) { return (b < a) ? b : a; }
__device__ inline double maxOf(double a, double b) { return (a < b) ? b : a; }

// ---------------------------------------------------------------------------
// Kernels. Each is one pass over a vector, fusing what the CPU loop does in
// consecutive loops over the same index range.
// ---------------------------------------------------------------------------

// v = z - y/rho (input to A^T v).
__global__ void scaledDifferenceKernel(double* __restrict__ v, const double* __restrict__ z,
                                       const double* __restrict__ y, int m, double rho) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < m; i += static_cast<long long>(gridDim.x) * blockDim.x) {
        v[i] = z[i] - y[i] / rho;
    }
}

// rhs = (sigma*x - q) + rho*Atv, the CPU's two loops in their order.
__global__ void rhsKernel(double* __restrict__ rhs, const double* __restrict__ x,
                          const double* __restrict__ q, const double* __restrict__ Atv,
                          int n, double rho, bool withConstraints) {
    for (long long j = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; j < n; j += static_cast<long long>(gridDim.x) * blockDim.x) {
        double value = KktSolver::kSigma * x[j] - q[j];
        if (withConstraints) {
            value += rho * Atv[j];
        }
        rhs[j] = value;
    }
}

// z = min(max(Ax + y/rho, l), u); y += rho*(Ax - z). Infinite bounds are
// handled by the comparisons; no bound is replaced by a large finite value.
__global__ void projectAndAscendKernel(double* __restrict__ z, double* __restrict__ y,
                                       const double* __restrict__ Ax,
                                       const double* __restrict__ lower,
                                       const double* __restrict__ upper, int m, double rho) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < m; i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const double ax = Ax[i];
        const double v = ax + y[i] / rho;
        const double projected = minOf(maxOf(v, lower[i]), upper[i]);
        z[i] = projected;
        y[i] += rho * (ax - projected);
    }
}

// zDiff = z - zOld.
__global__ void differenceKernel(double* __restrict__ out, const double* __restrict__ a,
                                 const double* __restrict__ b, int count) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += static_cast<long long>(gridDim.x) * blockDim.x) {
        out[i] = a[i] - b[i];
    }
}

// Per-block partials of the three per-iteration scalars:
//   [0] sum (Ax - z)^2                          (m > 0)
//   [1] sum (rho * s_j)^2 with s = A^T zDiff    (m > 0), or
//       sum (Px + q)_j^2                        (m == 0)
//   [2] q'x + 0.5 x'Px                          (objective)
__global__ void __launch_bounds__(kBlockSize) metricsKernel(
    const double* __restrict__ Ax, const double* __restrict__ z, int m,
    const double* __restrict__ s, const double* __restrict__ x,
    const double* __restrict__ Px, const double* __restrict__ q, int n,
    double rho, bool withConstraints, double* __restrict__ partials, int stride) {
    __shared__ double scratch[kSums * (kBlockSize / cuda_support::kWarpSize)];
    double sums[kSums] = {0.0, 0.0, 0.0};
    const long long total = m > n ? m : n;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        if (i < m) {
            const double r = Ax[i] - z[i];
            sums[0] += r * r;
        }
        if (i < n) {
            if (withConstraints) {
                const double scaled = rho * s[i];
                sums[1] += scaled * scaled;
            } else {
                const double gradient = Px[i] + q[i];
                sums[1] += gradient * gradient;
            }
            sums[2] += q[i] * x[i] + 0.5 * x[i] * Px[i];
        }
    }
    cuda_support::blockSum<kSums>(sums, scratch);
    if (threadIdx.x == 0) {
        for (int k = 0; k < kSums; ++k) {
            partials[k * stride + blockIdx.x] = sums[k];
        }
    }
}

// ---------------------------------------------------------------------------
// cuSPARSE wrappers
// ---------------------------------------------------------------------------

class SparseHandle {
public:
    explicit SparseHandle(cudaStream_t stream) {
        CUDA_SUPPORT_CHECK(cusparseCreate(&handle_));
        CUDA_SUPPORT_CHECK(cusparseSetStream(handle_, stream));
    }
    ~SparseHandle() {
        if (handle_ != nullptr) {
            cusparseDestroy(handle_);
        }
    }
    SparseHandle(const SparseHandle&) = delete;
    SparseHandle& operator=(const SparseHandle&) = delete;
    [[nodiscard]] cusparseHandle_t get() const noexcept { return handle_; }

private:
    cusparseHandle_t handle_ = nullptr;
};

class DenseVector {
public:
    DenseVector(double* data, std::int64_t size) {
        CUDA_SUPPORT_CHECK(cusparseCreateDnVec(&descriptor_, size, data, CUDA_R_64F));
    }
    ~DenseVector() {
        if (descriptor_ != nullptr) {
            cusparseDestroyDnVec(descriptor_);
        }
    }
    DenseVector(const DenseVector&) = delete;
    DenseVector& operator=(const DenseVector&) = delete;
    [[nodiscard]] cusparseDnVecDescr_t get() const noexcept { return descriptor_; }

private:
    cusparseDnVecDescr_t descriptor_ = nullptr;
};

// A CSR matrix resident on the device, with its SpMV descriptor and workspace
// created once. Offsets and indices are stored 32-bit when nnz provably fits
// (the common case, and the index type every cuSPARSE SpMV algorithm
// supports); otherwise both are stored 64-bit. Offsets are never truncated.
class DeviceCsr {
public:
    DeviceCsr(int rows, int columns,
              const std::vector<Offset>& start, const std::vector<Index>& index,
              const std::vector<double>& value, cudaStream_t stream, BackendProfile& profile)
        : rows_(rows), columns_(columns), nonzeros_(static_cast<std::int64_t>(value.size())) {
        wide_ = nonzeros_ > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max());
        if (wide_) {
            startWide_ = cuda_support::uploadNew(start, stream);
            std::vector<std::int64_t> widened(index.begin(), index.end());
            indexWide_ = cuda_support::uploadNew(widened, stream);
            // uploadNew is asynchronous from pageable memory, which the driver
            // stages before returning, so `widened` may be released afterwards.
            profile.hostToDeviceBytes += static_cast<std::int64_t>(startWide_.bytes() + indexWide_.bytes());
        } else {
            std::vector<std::int32_t> narrowed(start.size());
            for (std::size_t k = 0; k < start.size(); ++k) {
                narrowed[k] = static_cast<std::int32_t>(start[k]);  // <= nnz, checked above
            }
            startNarrow_ = cuda_support::uploadNew(narrowed, stream);
            indexNarrow_ = cuda_support::uploadNew(index, stream);
            profile.hostToDeviceBytes += static_cast<std::int64_t>(startNarrow_.bytes() + indexNarrow_.bytes());
        }
        value_ = cuda_support::uploadNew(value, stream);
        profile.hostToDeviceBytes += static_cast<std::int64_t>(value_.bytes());

        // cuSPARSE rejects null pointers even for an empty matrix, so an empty
        // one is simply never multiplied; see multiply().
        if (nonzeros_ > 0 && rows_ > 0 && columns_ > 0) {
            const cusparseIndexType_t type = wide_ ? CUSPARSE_INDEX_64I : CUSPARSE_INDEX_32I;
            void* offsets = wide_ ? static_cast<void*>(startWide_.data()) : static_cast<void*>(startNarrow_.data());
            void* indices = wide_ ? static_cast<void*>(indexWide_.data()) : static_cast<void*>(indexNarrow_.data());
            CUDA_SUPPORT_CHECK(cusparseCreateCsr(&descriptor_, rows_, columns_, nonzeros_,
                                                 offsets, indices, value_.data(), type, type,
                                                 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        }
    }

    ~DeviceCsr() {
        if (descriptor_ != nullptr) {
            cusparseDestroySpMat(descriptor_);
        }
    }
    DeviceCsr(const DeviceCsr&) = delete;
    DeviceCsr& operator=(const DeviceCsr&) = delete;

    // Sizes the workspace for out = M * in with these vector descriptors; the
    // workspace is reused for every later product on the same pair.
    void prepare(cusparseHandle_t handle, const DenseVector& in, const DenseVector& out) {
        if (descriptor_ == nullptr) {
            return;
        }
        const double one = 1.0;
        const double zero = 0.0;
        std::size_t bytes = 0;
        CUDA_SUPPORT_CHECK(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   &one, descriptor_, in.get(), &zero, out.get(),
                                                   CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &bytes));
        if (bytes > workspace_.size()) {
            workspace_.allocate(bytes);
        }
    }

    // out = M * in. An empty matrix writes zeros, as the host product does.
    void multiply(cusparseHandle_t handle, const DenseVector& in, const DenseVector& out,
                  double* outData, cudaStream_t stream) {
        if (descriptor_ == nullptr) {
            if (rows_ > 0) {
                CUDA_SUPPORT_CHECK(cudaMemsetAsync(outData, 0, static_cast<std::size_t>(rows_) * sizeof(double), stream));
            }
            return;
        }
        const double one = 1.0;
        const double zero = 0.0;
        CUDA_SUPPORT_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &one, descriptor_, in.get(), &zero, out.get(),
                                        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, workspace_.data()));
    }

    [[nodiscard]] std::size_t deviceBytes() const noexcept {
        return startWide_.bytes() + indexWide_.bytes() + startNarrow_.bytes() +
               indexNarrow_.bytes() + value_.bytes() + workspace_.bytes();
    }

private:
    int rows_;
    int columns_;
    std::int64_t nonzeros_;
    bool wide_ = false;
    DeviceBuffer<std::int64_t> startWide_;
    DeviceBuffer<std::int64_t> indexWide_;
    DeviceBuffer<std::int32_t> startNarrow_;
    DeviceBuffer<std::int32_t> indexNarrow_;
    DeviceBuffer<double> value_;
    DeviceBuffer<unsigned char> workspace_;
    cusparseSpMatDescr_t descriptor_ = nullptr;
};

int gridFor(long long items, int maxGrid) {
    if (items <= 0) {
        return 0;
    }
    return static_cast<int>(std::min<long long>((items + kBlockSize - 1) / kBlockSize, maxGrid));
}

// A^T in CSR form is A's CSC arrays.
std::vector<Offset> transposeStart(const SparseMatrix& a) { return a.cscColumnStart(); }

class CudaAdmmBackend final : public AdmmBackend {
public:
    CudaAdmmBackend(const QpModel& model, int device)
        : guard_(device),
          n_(model.numVariables()),
          m_(model.numConstraints()),
          sparse_(stream_.get()) {
        const Clock::time_point setupStart = Clock::now();
        const cudaStream_t stream = stream_.get();

        // A^T's CSR is A's CSC. The host SparseMatrix always carries CSC when
        // built through fromTriplets (scaling.cpp does), but check: an empty
        // CSC on a matrix with entries would silently produce A^T = 0.
        if (model.A.cscColumnStart().size() != static_cast<std::size_t>(n_) + 1 ||
            model.A.cscValues().size() != model.A.csrValues().size()) {
            throw std::invalid_argument("hybrid CUDA backend requires A with its CSC form");
        }

        cudaDeviceProp properties{};
        CUDA_SUPPORT_CHECK(cudaGetDeviceProperties(&properties, device));
        maxGrid_ = std::max(properties.multiProcessorCount * 8, 1);
        metricsGrid_ = std::max(gridFor(std::max(n_, m_), maxGrid_), 1);

        // Up-front memory check (matrices twice for A and A^T, 64-bit worst case).
        cuda_support::ByteBudget budget;
        const auto nnzA = model.A.nonzeros();
        const auto nnzP = model.P.nonzeros();
        budget.add<std::int64_t>(2 * nnzA + nnzP + 2 * static_cast<std::size_t>(n_) + static_cast<std::size_t>(m_) + 3);
        budget.add<double>(2 * nnzA + nnzP);
        budget.add<double>(6 * static_cast<std::size_t>(n_) + 9 * static_cast<std::size_t>(m_));
        budget.add<double>(static_cast<std::size_t>(kSums) * static_cast<std::size_t>(metricsGrid_) + kSums);
        std::size_t freeBytes = 0;
        std::size_t totalBytes = 0;
        CUDA_SUPPORT_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
        if (budget.total() > freeBytes / 100 * 95) {
            throw cuda_support::CudaError(
                "insufficient device memory: need " + std::to_string(budget.total()) +
                " bytes, " + std::to_string(freeBytes) + " free on device " + std::to_string(device));
        }

        a_ = std::make_unique<DeviceCsr>(m_, n_, model.A.csrRowStart(), model.A.csrColumnIndex(),
                                         model.A.csrValues(), stream, profile_);
        at_ = std::make_unique<DeviceCsr>(n_, m_, transposeStart(model.A), model.A.cscRowIndex(),
                                          model.A.cscValues(), stream, profile_);
        p_ = std::make_unique<DeviceCsr>(n_, n_, model.P.csrRowStart(), model.P.csrColumnIndex(),
                                         model.P.csrValues(), stream, profile_);

        const auto uploadVector = [&](DeviceBuffer<double>& target, const std::vector<double>& source) {
            target = cuda_support::uploadNew(source, stream);
            profile_.hostToDeviceBytes += static_cast<std::int64_t>(target.bytes());
        };
        uploadVector(q_, model.q);
        uploadVector(lower_, model.l);
        uploadVector(upper_, model.u);

        const auto zeroed = [&](DeviceBuffer<double>& buffer, int count) {
            buffer.allocate(static_cast<std::size_t>(count));
            if (count > 0) {
                CUDA_SUPPORT_CHECK(cudaMemsetAsync(buffer.data(), 0, buffer.bytes(), stream));
            }
        };
        // Same initial state as the CPU backend: every vector starts at +0.0.
        zeroed(x_, n_);
        zeroed(rhs_, n_);
        zeroed(Atv_, n_);
        zeroed(Px_, n_);
        zeroed(z_, m_);
        zeroed(y_, m_);
        zeroed(Ax_, m_);
        zeroed(zOld_, m_);
        zeroed(yOld_, m_);
        zeroed(bestY_, m_);
        zeroed(scratchM_, m_);
        partials_.allocate(static_cast<std::size_t>(kSums) * static_cast<std::size_t>(metricsGrid_));
        reduced_.allocate(kSums);
        hostReduced_ = cuda_support::PinnedBuffer<double>(kSums);

        // Descriptors over fixed buffers, created once. cuSPARSE rejects a
        // zero-length dense vector, so each is created only when non-empty and
        // every product involving an empty side is skipped.
        if (n_ > 0) {
            xVec_ = std::make_unique<DenseVector>(x_.data(), n_);
            AtvVec_ = std::make_unique<DenseVector>(Atv_.data(), n_);
            PxVec_ = std::make_unique<DenseVector>(Px_.data(), n_);
        }
        if (m_ > 0) {
            AxVec_ = std::make_unique<DenseVector>(Ax_.data(), m_);
            scratchMVec_ = std::make_unique<DenseVector>(scratchM_.data(), m_);
        }
        if (n_ > 0 && m_ > 0) {
            a_->prepare(sparse_.get(), *xVec_, *AxVec_);
            at_->prepare(sparse_.get(), *scratchMVec_, *AtvVec_);
        }
        if (n_ > 0) {
            p_->prepare(sparse_.get(), *xVec_, *PxVec_);
        }

        hostX_.assign(static_cast<std::size_t>(n_), 0.0);
        hostXOld_.assign(static_cast<std::size_t>(n_), 0.0);

        stream_.synchronize();
        ++profile_.synchronisations;
        profile_.setupSeconds = secondsSince(setupStart);
    }

    void saveIterate() override {
        const cudaStream_t stream = stream_.get();
        if (m_ > 0) {
            CUDA_SUPPORT_CHECK(cudaMemcpyAsync(zOld_.data(), z_.data(), z_.bytes(), cudaMemcpyDeviceToDevice, stream));
            CUDA_SUPPORT_CHECK(cudaMemcpyAsync(yOld_.data(), y_.data(), y_.bytes(), cudaMemcpyDeviceToDevice, stream));
        }
        hostXOld_ = hostX_;
        invalidateSnapshots();
    }

    void buildRhs(double rho, std::vector<double>& rhs) override {
        const cudaStream_t stream = stream_.get();
        const bool constrained = m_ > 0;
        if (constrained && n_ > 0) {
            scaledDifferenceKernel<<<gridFor(m_, maxGrid_), kBlockSize, 0, stream>>>(
                scratchM_.data(), z_.data(), y_.data(), m_, rho);
            CUDA_SUPPORT_CHECK_LAUNCH();
            at_->multiply(sparse_.get(), *scratchMVec_, *AtvVec_, Atv_.data(), stream);
        }
        if (n_ > 0) {
            rhsKernel<<<gridFor(n_, maxGrid_), kBlockSize, 0, stream>>>(
                rhs_.data(), x_.data(), q_.data(), Atv_.data(), n_, rho, constrained);
            CUDA_SUPPORT_CHECK_LAUNCH();
        }
        // The KKT solve on the host needs the right-hand side: unavoidable in
        // the hybrid design.
        cuda_support::download(rhs, rhs_.data(), static_cast<std::size_t>(n_), stream);
        ++profile_.synchronisations;
        profile_.deviceToHostBytes += static_cast<std::int64_t>(rhs_.bytes());
    }

    void acceptPrimal(std::vector<double>&& solution, double rho) override {
        if (solution.size() != static_cast<std::size_t>(n_)) {
            throw std::invalid_argument("ADMM x-update has the wrong dimension");
        }
        hostX_ = std::move(solution);
        const cudaStream_t stream = stream_.get();
        cuda_support::uploadAsync(x_, hostX_, stream);
        profile_.hostToDeviceBytes += static_cast<std::int64_t>(x_.bytes());
        if (m_ > 0 && n_ > 0) {
            a_->multiply(sparse_.get(), *xVec_, *AxVec_, Ax_.data(), stream);
        } else if (m_ > 0) {
            CUDA_SUPPORT_CHECK(cudaMemsetAsync(Ax_.data(), 0, Ax_.bytes(), stream));
        }
        if (m_ > 0) {
            projectAndAscendKernel<<<gridFor(m_, maxGrid_), kBlockSize, 0, stream>>>(
                z_.data(), y_.data(), Ax_.data(), lower_.data(), upper_.data(), m_, rho);
            CUDA_SUPPORT_CHECK_LAUNCH();
        }
        invalidateSnapshots();
    }

    AdmmIterationMetrics metrics(double rho) override {
        const cudaStream_t stream = stream_.get();
        const bool constrained = m_ > 0;
        // A*x is already current: acceptPrimal computed it from this x, and the
        // CPU's second product in the residual pass reproduces the same vector.
        if (n_ > 0) {
            p_->multiply(sparse_.get(), *xVec_, *PxVec_, Px_.data(), stream);
        }
        if (constrained && n_ > 0) {
            differenceKernel<<<gridFor(m_, maxGrid_), kBlockSize, 0, stream>>>(
                scratchM_.data(), z_.data(), zOld_.data(), m_);
            CUDA_SUPPORT_CHECK_LAUNCH();
            at_->multiply(sparse_.get(), *scratchMVec_, *AtvVec_, Atv_.data(), stream);
        }
        metricsKernel<<<metricsGrid_, kBlockSize, 0, stream>>>(
            Ax_.data(), z_.data(), m_, Atv_.data(), x_.data(), Px_.data(), q_.data(), n_,
            rho, constrained, partials_.data(), metricsGrid_);
        CUDA_SUPPORT_CHECK_LAUNCH();
        constexpr int kFinalThreads = 256;
        cuda_support::sumPartialsKernel<kSums>
            <<<1, kFinalThreads, kSums * (kFinalThreads / cuda_support::kWarpSize) * sizeof(double), stream>>>(
                partials_.data(), metricsGrid_, metricsGrid_, reduced_.data());
        CUDA_SUPPORT_CHECK_LAUNCH();
        CUDA_SUPPORT_CHECK(cudaMemcpyAsync(hostReduced_.data(), reduced_.data(), kSums * sizeof(double),
                                           cudaMemcpyDeviceToHost, stream));
        // rho adaptation and best-iterate tracking read these every iteration.
        stream_.synchronize();
        ++profile_.synchronisations;
        profile_.deviceToHostBytes += kSums * static_cast<std::int64_t>(sizeof(double));

        AdmmIterationMetrics result;
        result.primalResidualNorm = constrained ? std::sqrt(hostReduced_[0]) : 0.0;
        result.dualResidualNorm = std::sqrt(hostReduced_[1]);
        result.objective = hostReduced_[2];
        return result;
    }

    void recordBest() override {
        hostBestX_ = hostX_;
        if (m_ > 0) {
            CUDA_SUPPORT_CHECK(cudaMemcpyAsync(bestY_.data(), y_.data(), y_.bytes(),
                                               cudaMemcpyDeviceToDevice, stream_.get()));
        }
        hasBest_ = true;
    }

    void restoreBest() override {
        const cudaStream_t stream = stream_.get();
        // Mirrors the CPU's `x = bestX; y = bestY`: before any recordBest()
        // those were empty vectors, and so are the views afterwards.
        if (!hasBest_) {
            hostX_.clear();
            emptied_ = true;
            invalidateSnapshots();
            return;
        }
        hostX_ = hostBestX_;
        cuda_support::uploadAsync(x_, hostX_, stream);
        profile_.hostToDeviceBytes += static_cast<std::int64_t>(x_.bytes());
        if (m_ > 0) {
            CUDA_SUPPORT_CHECK(cudaMemcpyAsync(y_.data(), bestY_.data(), y_.bytes(),
                                               cudaMemcpyDeviceToDevice, stream));
        }
        invalidateSnapshots();
    }

    AdmmCheckView checkView() override {
        refreshSnapshot(hostY_, y_, hostYValid_);
        refreshSnapshot(hostYOld_, yOld_, hostYOldValid_);
        refreshSnapshot(hostZ_, z_, hostZValid_);
        refreshSnapshot(hostAx_, Ax_, hostAxValid_);
        return AdmmCheckView{hostX_, hostXOld_, hostY_, hostYOld_, hostZ_, hostAx_};
    }

    const std::vector<double>& primal() override { return hostX_; }

    const std::vector<double>& dual() override {
        if (emptied_) {
            hostY_.clear();
            return hostY_;
        }
        refreshSnapshot(hostY_, y_, hostYValid_);
        return hostY_;
    }

    ComputeBackend kind() const noexcept override { return ComputeBackend::Cuda; }
    BackendProfile profile() const noexcept override { return profile_; }

private:
    void invalidateSnapshots() {
        hostYValid_ = false;
        hostYOldValid_ = false;
        hostZValid_ = false;
        hostAxValid_ = false;
    }

    void refreshSnapshot(std::vector<double>& target, const DeviceBuffer<double>& source, bool& valid) {
        if (valid) {
            return;
        }
        const Clock::time_point start = Clock::now();
        cuda_support::download(target, source.data(), source.size(), stream_.get());
        ++profile_.synchronisations;
        profile_.deviceToHostBytes += static_cast<std::int64_t>(source.bytes());
        profile_.snapshotSeconds += secondsSince(start);
        valid = true;
    }

    // Destroyed last, so every resource below is released on its own device.
    cuda_support::DeviceGuard guard_;
    cuda_support::Stream stream_;
    int n_;
    int m_;
    SparseHandle sparse_;
    int maxGrid_ = 1;
    int metricsGrid_ = 1;

    std::unique_ptr<DeviceCsr> a_;
    std::unique_ptr<DeviceCsr> at_;
    std::unique_ptr<DeviceCsr> p_;

    DeviceBuffer<double> q_, lower_, upper_;
    DeviceBuffer<double> x_, rhs_, Atv_, Px_;
    DeviceBuffer<double> z_, y_, Ax_, zOld_, yOld_, bestY_, scratchM_;
    DeviceBuffer<double> partials_, reduced_;
    cuda_support::PinnedBuffer<double> hostReduced_;

    // Declared after the buffers they point into, so they are destroyed first.
    std::unique_ptr<DenseVector> xVec_, AtvVec_, PxVec_, AxVec_, scratchMVec_;

    std::vector<double> hostX_, hostXOld_, hostBestX_;
    std::vector<double> hostY_, hostYOld_, hostZ_, hostAx_;
    bool hostYValid_ = false;
    bool hostYOldValid_ = false;
    bool hostZValid_ = false;
    bool hostAxValid_ = false;
    bool hasBest_ = false;
    bool emptied_ = false;

    BackendProfile profile_;
};

}  // namespace

CudaAvailability cudaAvailability(int device) {
    CudaAvailability availability;
    availability.compiled = true;
    if (cudaRuntimeGetVersion(&availability.runtimeVersion) != cudaSuccess) {
        availability.runtimeVersion = 0;
    }
    if (cudaDriverGetVersion(&availability.driverVersion) != cudaSuccess) {
        availability.driverVersion = 0;
    }
    cudaGetLastError();
    availability.reason = cuda_support::probeDevice(device, &availability.deviceName);
    availability.usable = availability.reason.empty();
    return availability;
}

std::unique_ptr<AdmmBackend> makeCudaAdmmBackend(const QpModel& model, int device, std::string& error) {
    const std::string unusable = cuda_support::probeDevice(device);
    if (!unusable.empty()) {
        error = unusable;
        return nullptr;
    }
    try {
        return std::make_unique<CudaAdmmBackend>(model, device);
    } catch (const std::exception& failure) {
        error = failure.what();
        return nullptr;
    }
}

}  // namespace qp

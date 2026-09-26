// CUDA implementation of IterationBackend.
//
// Residency. The scaled problem (CSR and CSC of A, bounds, objective,
// preconditioner) is uploaded once, when the backend is created. The iterate,
// its trial buffers, the carried activity A*x and the running average live on
// the device for the whole solve. Per iteration the host receives exactly three
// doubles -- the linesearch reductions -- which it needs to accept or reject
// the step; that is the only synchronisation in the hot loop. Full vectors
// cross the bus only when the solver asks for a host view, which it does at
// termination-check boundaries (every terminationCheckFrequency iterations),
// and at the end.
//
// Both CSR and CSC are kept on the device, deliberately: the fused primal half
// is a gather down columns and the fused dual half a gather along rows, and
// storing both is what lets each run without atomics. It doubles the matrix
// footprint, which the up-front memory check accounts for.

#include "pdlp/compute_backend.h"
#include "pdlp/iteration_backend.h"

#include "cuda_support/cuda_check.h"
#include "cuda_support/device_buffer.h"
#include "cuda_support/reduce.cuh"
#include "pdhg_kernels.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace pdlp {
namespace {

using cuda::kBlockSize;
using cuda::kSums;
using cuda::Offset;
using cuda_support::DeviceBuffer;

using Clock = std::chrono::steady_clock;

double secondsSince(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// Group size from the mean length of the lines the group kernel will see:
// the largest power of two not above it, so no lane of a group is idle on a
// typical line. Heuristic and PROVISIONAL -- see docs/cuda.md; override with
// CudaBackendConfig::forcedGroupSize.
int groupSizeFor(double meanLineNonzeros) {
    int group = 1;
    while (group < cuda_support::kWarpSize && 2.0 * group <= meanLineNonzeros) {
        group *= 2;
    }
    return group;
}

struct LinePlan {
    std::vector<int> heavy;   // lines reduced by the block-per-line kernel
    int group = 1;
};

LinePlan planLines(const std::vector<Offset>& start, std::int64_t heavyNonzeros, int forcedGroup) {
    LinePlan plan;
    const std::size_t lines = start.empty() ? 0 : start.size() - 1;
    Offset lightNonzeros = 0;
    std::size_t lightLines = 0;
    for (std::size_t line = 0; line < lines; ++line) {
        const Offset length = start[line + 1] - start[line];
        if (length > heavyNonzeros) {
            plan.heavy.push_back(static_cast<int>(line));
        } else {
            lightNonzeros += length;
            ++lightLines;
        }
    }
    if (forcedGroup > 0) {
        plan.group = forcedGroup;
    } else {
        const double mean = lightLines > 0
            ? static_cast<double>(lightNonzeros) / static_cast<double>(lightLines)
            : 1.0;
        plan.group = groupSizeFor(mean);
    }
    return plan;
}

// Blocks for a grid-stride kernel over `items` work units of `perBlock` each,
// capped at what the device keeps resident. The cap bounds the partial-sum
// buffer independently of problem size.
int gridFor(long long items, int perBlock, int maxGrid) {
    if (items <= 0) {
        return 0;
    }
    const long long needed = (items + perBlock - 1) / perBlock;
    return static_cast<int>(std::min<long long>(needed, maxGrid));
}

class CudaIterationBackend final : public IterationBackend {
public:
    CudaIterationBackend(
        const CompiledLp& problem,
        const DiagonalPreconditioner& preconditioner,
        const CudaBackendConfig& config
    )
        : guard_(config.device),
          rows_(problem.numRows()),
          columns_(problem.numColumns()) {
        const Clock::time_point setupStart = Clock::now();
        const SparseMatrix& matrix = problem.matrix;

        if (config.forcedGroupSize != 0 &&
            (config.forcedGroupSize < 1 || config.forcedGroupSize > 32 ||
             (config.forcedGroupSize & (config.forcedGroupSize - 1)) != 0)) {
            throw std::invalid_argument("CudaBackendConfig::forcedGroupSize must be 0 or a power of two <= 32");
        }
        heavyNonzeros_ = std::max<std::int64_t>(config.heavyLineNonzeros, 1);

        const LinePlan rowPlan = planLines(matrix.csrRowStart(), heavyNonzeros_, config.forcedGroupSize);
        const LinePlan columnPlan = planLines(matrix.cscColumnStart(), heavyNonzeros_, config.forcedGroupSize);
        rowGroup_ = rowPlan.group;
        columnGroup_ = columnPlan.group;
        heavyRowCount_ = static_cast<int>(rowPlan.heavy.size());
        heavyColumnCount_ = static_cast<int>(columnPlan.heavy.size());

        cudaDeviceProp properties{};
        CUDA_SUPPORT_CHECK(cudaGetDeviceProperties(&properties, config.device));
        // Enough resident blocks to cover the device several times over.
        const int maxGrid = std::max(properties.multiProcessorCount * 8, 1);
        const int groupsPerBlockColumns = kBlockSize / columnGroup_;
        const int groupsPerBlockRows = kBlockSize / rowGroup_;
        columnGrid_ = gridFor(columns_, groupsPerBlockColumns, maxGrid);
        rowGrid_ = gridFor(rows_, groupsPerBlockRows, maxGrid);
        heavyColumnGrid_ = std::min(heavyColumnCount_, maxGrid);
        heavyRowGrid_ = std::min(heavyRowCount_, maxGrid);
        vectorGrid_ = gridFor(static_cast<long long>(columns_) + rows_, kBlockSize, maxGrid);
        slots_ = columnGrid_ + heavyColumnGrid_ + rowGrid_ + heavyRowGrid_;

        // Refuse up front rather than fail on the tenth allocation.
        const auto nnz = static_cast<std::size_t>(matrix.nonzeros());
        const auto n = static_cast<std::size_t>(columns_);
        const auto m = static_cast<std::size_t>(rows_);
        cuda_support::ByteBudget budget;
        budget.add<Offset>(m + 1);          // CSR starts
        budget.add<Offset>(n + 1);          // CSC starts
        budget.add<int>(2 * nnz);           // CSR + CSC indices
        budget.add<double>(2 * nnz);        // CSR + CSC values
        budget.add<int>(rowPlan.heavy.size() + columnPlan.heavy.size());
        budget.add<double>(8 * n);          // c, l, u, T, 1/T, x, x', avg x
        budget.add<double>(10 * m);         // l, u, S, 1/S, y, y', Ax, Ax', avg y, spare
        budget.add<double>(static_cast<std::size_t>(kSums) * static_cast<std::size_t>(slots_) + kSums);
        std::size_t freeBytes = 0;
        std::size_t totalBytes = 0;
        CUDA_SUPPORT_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
        if (budget.total() > freeBytes / 100 * 95) {
            throw cuda_support::CudaError(
                "insufficient device memory: the problem needs " +
                std::to_string(budget.total()) + " bytes, " + std::to_string(freeBytes) +
                " are free on device " + std::to_string(config.device));
        }

        const cudaStream_t stream = stream_.get();
        const auto upload = [&](auto& target, const auto& source) {
            target = cuda_support::uploadNew(source, stream);
            profile_.hostToDeviceBytes += static_cast<std::int64_t>(target.bytes());
        };
        upload(csrStart_, matrix.csrRowStart());
        upload(csrIndex_, matrix.csrColumnIndex());
        upload(csrValue_, matrix.csrValues());
        upload(cscStart_, matrix.cscColumnStart());
        upload(cscIndex_, matrix.cscRowIndex());
        upload(cscValue_, matrix.cscValues());
        upload(heavyRows_, rowPlan.heavy);
        upload(heavyColumns_, columnPlan.heavy);

        upload(objective_, problem.objective);
        upload(variableLower_, problem.variableLower);
        upload(variableUpper_, problem.variableUpper);
        upload(rowLower_, problem.rowLower);
        upload(rowUpper_, problem.rowUpper);
        upload(primalScale_, preconditioner.primalScale);
        upload(primalScaleInverse_, preconditioner.primalScaleInverse);
        upload(dualScale_, preconditioner.dualScale);
        upload(dualScaleInverse_, preconditioner.dualScaleInverse);

        primal_.allocate(n);
        primalTrial_.allocate(n);
        averagePrimal_.allocate(n);
        dual_.allocate(m);
        dualTrial_.allocate(m);
        activity_.allocate(m);
        activityTrial_.allocate(m);
        averageDual_.allocate(m);
        partials_.allocate(static_cast<std::size_t>(kSums) * static_cast<std::size_t>(std::max(slots_, 1)));
        reduced_.allocate(kSums);
        hostReduced_ = cuda_support::PinnedBuffer<double>(kSums);

        stream_.synchronize();
        ++profile_.synchronisations;
        profile_.setupSeconds = secondsSince(setupStart);
    }

    void reset(const std::vector<double>& primal, const std::vector<double>& dual) override {
        if (primal.size() != static_cast<std::size_t>(columns_) ||
            dual.size() != static_cast<std::size_t>(rows_)) {
            throw std::invalid_argument("PDHG state dimensions do not match the LP");
        }
        cuda_support::uploadAsync(primal_, primal, stream_.get());
        cuda_support::uploadAsync(dual_, dual, stream_.get());
        profile_.hostToDeviceBytes += static_cast<std::int64_t>(primal_.bytes() + dual_.bytes());
        refreshActivity();
        iteration_ = 0;
        resetAverage();

        // The device now holds exactly these values, so the host views are
        // already valid without a download.
        hostPrimal_ = primal;
        hostDual_ = dual;
        hostPrimalValid_ = true;
        hostDualValid_ = true;
    }

    KernelTrialResult trial(const StepParameters& steps) override {
        const cudaStream_t stream = stream_.get();
        int slot = 0;

        // Primal half over CSC columns: (A^T y)_j -> x'_j.
        const cuda::PrimalStep primalStep{
            objective_.data(), variableLower_.data(), variableUpper_.data(),
            primalScale_.data(), primalScaleInverse_.data(),
            primal_.data(), primalTrial_.data(),
            steps.globalStep / steps.primalWeight};
        launchLines(columnMatrix(), columnGroup_, columnGrid_, dual_.data(), primalStep, partials_.data(), slot);
        slot += columnGrid_;
        launchHeavy(columnMatrix(), heavyColumns_.data(), heavyColumnCount_, heavyColumnGrid_,
                    dual_.data(), primalStep, partials_.data(), slot);
        slot += heavyColumnGrid_;

        // The kernel boundary above is the one global dependency PDHG has: every
        // row's A x' reads the complete trial primal.

        // Dual half over CSR rows: (A x')_i -> y'_i and the carried activity.
        const cuda::DualStep dualStep{
            rowLower_.data(), rowUpper_.data(), dualScale_.data(), dualScaleInverse_.data(),
            dual_.data(), activity_.data(), dualTrial_.data(), activityTrial_.data(),
            steps.globalStep * steps.primalWeight};
        launchLines(rowMatrix(), rowGroup_, rowGrid_, primalTrial_.data(), dualStep, partials_.data(), slot);
        slot += rowGrid_;
        launchHeavy(rowMatrix(), heavyRows_.data(), heavyRowCount_, heavyRowGrid_,
                    primalTrial_.data(), dualStep, partials_.data(), slot);
        slot += heavyRowGrid_;

        // Fixed-order sum of the block partials, then the three scalars home.
        constexpr int kFinalThreads = 256;
        cuda_support::sumPartialsKernel<kSums>
            <<<1, kFinalThreads, kSums * (kFinalThreads / cuda_support::kWarpSize) * sizeof(double), stream>>>(
                partials_.data(), slot, slots_, reduced_.data());
        CUDA_SUPPORT_CHECK_LAUNCH();
        CUDA_SUPPORT_CHECK(cudaMemcpyAsync(
            hostReduced_.data(), reduced_.data(), kSums * sizeof(double),
            cudaMemcpyDeviceToHost, stream));
        // The linesearch decides on the host whether to commit this trial, so
        // the host must wait for these three numbers. This is the only
        // per-iteration synchronisation.
        stream_.synchronize();
        ++profile_.synchronisations;
        profile_.deviceToHostBytes += kSums * static_cast<std::int64_t>(sizeof(double));

        KernelTrialResult result;
        result.primalMovementWeighted = hostReduced_[0];
        result.dualMovementWeighted = hostReduced_[1];
        result.interaction = hostReduced_[2];
        // Same rule as the CPU kernel: a non-finite iterate always surfaces as a
        // non-finite reduction, because the device clip propagates NaN exactly
        // as std::min/std::max do (see pdhg_math.h).
        result.finite = std::isfinite(result.primalMovementWeighted) &&
            std::isfinite(result.dualMovementWeighted) &&
            std::isfinite(result.interaction);
        return result;
    }

    void commit() override {
        // Buffer swaps, never copies; the rejected-trial path never gets here,
        // so the committed iterate is untouched until a trial is accepted.
        primal_.swap(primalTrial_);
        dual_.swap(dualTrial_);
        activity_.swap(activityTrial_);
        ++iteration_;
        hostPrimalValid_ = false;
        hostDualValid_ = false;
    }

    std::int64_t iteration() const noexcept override { return iteration_; }

    void addToAverage(double weight) override {
        if (!(weight > 0.0)) {
            return;
        }
        // The weights are accumulated on the host in the same order as the CPU
        // backend, so the blend fractions are bitwise identical.
        const double newTotal = totalWeight_ + weight;
        const double fraction = weight / newTotal;
        if (vectorGrid_ > 0) {
            cuda::blendKernel<<<vectorGrid_, kBlockSize, 0, stream_.get()>>>(
                averagePrimal_.data(), primal_.data(), columns_,
                averageDual_.data(), dual_.data(), rows_, fraction);
            CUDA_SUPPORT_CHECK_LAUNCH();
        }
        totalWeight_ = newTotal;
        hostAverageValid_ = false;
    }

    bool averageEmpty() const noexcept override { return totalWeight_ == 0.0; }

    void resetAverage() override {
        // All-bits-zero is +0.0, the same as IterateAverage::reset's fill.
        if (averagePrimal_.size() > 0) {
            CUDA_SUPPORT_CHECK(cudaMemsetAsync(averagePrimal_.data(), 0, averagePrimal_.bytes(), stream_.get()));
        }
        if (averageDual_.size() > 0) {
            CUDA_SUPPORT_CHECK(cudaMemsetAsync(averageDual_.data(), 0, averageDual_.bytes(), stream_.get()));
        }
        totalWeight_ = 0.0;
        hostAverageValid_ = false;
    }

    const std::vector<double>& primal() override {
        if (!hostPrimalValid_) {
            snapshot(hostPrimal_, primal_);
            hostPrimalValid_ = true;
        }
        return hostPrimal_;
    }

    const std::vector<double>& dual() override {
        if (!hostDualValid_) {
            snapshot(hostDual_, dual_);
            hostDualValid_ = true;
        }
        return hostDual_;
    }

    const CandidateIterate& average() override {
        if (averageEmpty()) {
            throw std::logic_error("Cannot request an empty iterate average");
        }
        if (!hostAverageValid_) {
            snapshot(hostAverage_.primal, averagePrimal_);
            snapshot(hostAverage_.dual, averageDual_);
            hostAverageValid_ = true;
        }
        return hostAverage_;
    }

    void restartFromAverage() override {
        if (averageEmpty()) {
            throw std::logic_error("Cannot request an empty iterate average");
        }
        const cudaStream_t stream = stream_.get();
        if (primal_.size() > 0) {
            CUDA_SUPPORT_CHECK(cudaMemcpyAsync(primal_.data(), averagePrimal_.data(), primal_.bytes(),
                                               cudaMemcpyDeviceToDevice, stream));
        }
        if (dual_.size() > 0) {
            CUDA_SUPPORT_CHECK(cudaMemcpyAsync(dual_.data(), averageDual_.data(), dual_.bytes(),
                                               cudaMemcpyDeviceToDevice, stream));
        }
        refreshActivity();
        // A device-to-device copy is exact, so a host copy of the average is
        // also a valid host copy of the new iterate.
        if (hostAverageValid_) {
            hostPrimal_ = hostAverage_.primal;
            hostDual_ = hostAverage_.dual;
            hostPrimalValid_ = true;
            hostDualValid_ = true;
        } else {
            hostPrimalValid_ = false;
            hostDualValid_ = false;
        }
    }

    ComputeBackend kind() const noexcept override { return ComputeBackend::Cuda; }
    BackendProfile profile() const noexcept override { return profile_; }

private:
    cuda::LineMatrix rowMatrix() const {
        return cuda::LineMatrix{csrStart_.data(), csrIndex_.data(), csrValue_.data(), rows_, heavyNonzeros_};
    }

    cuda::LineMatrix columnMatrix() const {
        return cuda::LineMatrix{cscStart_.data(), cscIndex_.data(), cscValue_.data(), columns_, heavyNonzeros_};
    }

    // activity = A * x over CSR rows; see CpuPdhgKernel::refreshActivity.
    void refreshActivity() {
        const cuda::StoreProduct store{activity_.data()};
        launchLines(rowMatrix(), rowGroup_, rowGrid_, primal_.data(), store, nullptr, 0);
        launchHeavy(rowMatrix(), heavyRows_.data(), heavyRowCount_, heavyRowGrid_,
                    primal_.data(), store, nullptr, 0);
    }

    template <class Epilogue>
    void launchLines(const cuda::LineMatrix& matrix, int group, int grid, const double* vector,
                     const Epilogue& epilogue, double* partials, int slot) {
        if (matrix.lines == 0 || grid == 0) {
            return;
        }
        const cudaStream_t stream = stream_.get();
        switch (group) {
            case 1:  cuda::lineKernel<1, Epilogue><<<grid, kBlockSize, 0, stream>>>(matrix, vector, epilogue, partials, slots_, slot); break;
            case 2:  cuda::lineKernel<2, Epilogue><<<grid, kBlockSize, 0, stream>>>(matrix, vector, epilogue, partials, slots_, slot); break;
            case 4:  cuda::lineKernel<4, Epilogue><<<grid, kBlockSize, 0, stream>>>(matrix, vector, epilogue, partials, slots_, slot); break;
            case 8:  cuda::lineKernel<8, Epilogue><<<grid, kBlockSize, 0, stream>>>(matrix, vector, epilogue, partials, slots_, slot); break;
            case 16: cuda::lineKernel<16, Epilogue><<<grid, kBlockSize, 0, stream>>>(matrix, vector, epilogue, partials, slots_, slot); break;
            default: cuda::lineKernel<32, Epilogue><<<grid, kBlockSize, 0, stream>>>(matrix, vector, epilogue, partials, slots_, slot); break;
        }
        CUDA_SUPPORT_CHECK_LAUNCH();
    }

    template <class Epilogue>
    void launchHeavy(const cuda::LineMatrix& matrix, const int* heavy, int heavyCount, int grid,
                     const double* vector, const Epilogue& epilogue, double* partials, int slot) {
        if (heavyCount == 0 || grid == 0) {
            return;
        }
        cuda::heavyLineKernel<Epilogue><<<grid, kBlockSize, 0, stream_.get()>>>(
            matrix, cuda::HeavyLines{heavy, heavyCount}, vector, epilogue, partials, slots_, slot);
        CUDA_SUPPORT_CHECK_LAUNCH();
    }

    void snapshot(std::vector<double>& target, const DeviceBuffer<double>& source) {
        const Clock::time_point start = Clock::now();
        cuda_support::download(target, source.data(), source.size(), stream_.get());
        ++profile_.synchronisations;
        profile_.deviceToHostBytes += static_cast<std::int64_t>(source.bytes());
        profile_.snapshotSeconds += secondsSince(start);
    }

    // Declared first so that it is destroyed last: every buffer below is freed
    // while its device is still current.
    cuda_support::DeviceGuard guard_;
    cuda_support::Stream stream_;

    int rows_ = 0;
    int columns_ = 0;
    std::int64_t heavyNonzeros_ = 0;
    int rowGroup_ = 1;
    int columnGroup_ = 1;
    int heavyRowCount_ = 0;
    int heavyColumnCount_ = 0;
    int rowGrid_ = 0;
    int columnGrid_ = 0;
    int heavyRowGrid_ = 0;
    int heavyColumnGrid_ = 0;
    int vectorGrid_ = 0;
    int slots_ = 0;

    DeviceBuffer<Offset> csrStart_;
    DeviceBuffer<int> csrIndex_;
    DeviceBuffer<double> csrValue_;
    DeviceBuffer<Offset> cscStart_;
    DeviceBuffer<int> cscIndex_;
    DeviceBuffer<double> cscValue_;
    DeviceBuffer<int> heavyRows_;
    DeviceBuffer<int> heavyColumns_;

    DeviceBuffer<double> objective_;
    DeviceBuffer<double> variableLower_;
    DeviceBuffer<double> variableUpper_;
    DeviceBuffer<double> rowLower_;
    DeviceBuffer<double> rowUpper_;
    DeviceBuffer<double> primalScale_;
    DeviceBuffer<double> primalScaleInverse_;
    DeviceBuffer<double> dualScale_;
    DeviceBuffer<double> dualScaleInverse_;

    DeviceBuffer<double> primal_;
    DeviceBuffer<double> dual_;
    DeviceBuffer<double> activity_;
    DeviceBuffer<double> primalTrial_;
    DeviceBuffer<double> dualTrial_;
    DeviceBuffer<double> activityTrial_;
    DeviceBuffer<double> averagePrimal_;
    DeviceBuffer<double> averageDual_;

    DeviceBuffer<double> partials_;
    DeviceBuffer<double> reduced_;
    cuda_support::PinnedBuffer<double> hostReduced_;

    std::int64_t iteration_ = 0;
    double totalWeight_ = 0.0;

    std::vector<double> hostPrimal_;
    std::vector<double> hostDual_;
    CandidateIterate hostAverage_;
    bool hostPrimalValid_ = false;
    bool hostDualValid_ = false;
    bool hostAverageValid_ = false;

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

std::unique_ptr<IterationBackend> makeCudaIterationBackend(
    const CompiledLp& problem,
    const DiagonalPreconditioner& preconditioner,
    const CudaBackendConfig& config,
    std::string& error
) {
    const std::string unusable = cuda_support::probeDevice(config.device);
    if (!unusable.empty()) {
        error = unusable;
        return nullptr;
    }
    try {
        return std::make_unique<CudaIterationBackend>(problem, preconditioner, config);
    } catch (const std::exception& failure) {
        error = failure.what();
        return nullptr;
    }
}

}  // namespace pdlp

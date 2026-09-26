#pragma once

#include "pdlp/compiled_lp.h"
#include "pdlp/compute_backend.h"
#include "pdlp/parallel.h"
#include "pdlp/pdhg_kernel.h"
#include "pdlp/pdlp_state.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pdlp {

// The state PDHG touches on every iteration -- the iterate, its trial buffers
// and the running average -- together with the operations that read or write
// it.
//
// This is the seam between the solver's control logic and the hardware it
// runs on. PdlpSolver and FeasibilityPolisher own the algorithm (linesearch,
// restarts, termination, infeasibility detection, polishing) and are written
// once against this interface; a backend owns only data movement and
// arithmetic. The CPU implementation wraps CpuPdhgKernel, PdlpState and
// IterateAverage unchanged, so the CPU solve performs exactly the operations
// it did before this interface existed.
//
// Host-visible views (primal(), dual(), average()) are for the infrequent
// host-side checks. A device backend materialises them on demand, so callers
// must not ask for them on every iteration.
class IterationBackend {
public:
    virtual ~IterationBackend() = default;

    // Installs an iterate in working coordinates, recomputes A*x exactly,
    // restarts the iteration counter at zero and empties the average.
    virtual void reset(const std::vector<double>& primal, const std::vector<double>& dual) = 0;

    // One trial PDHG step from the current iterate; see CpuPdhgKernel::trial.
    // Leaves the current iterate untouched, so a rejected trial is simply
    // followed by another trial.
    [[nodiscard]] virtual KernelTrialResult trial(const StepParameters& steps) = 0;

    // Installs the most recent trial. Only valid immediately after trial().
    virtual void commit() = 0;

    [[nodiscard]] virtual std::int64_t iteration() const noexcept = 0;

    // Folds the current iterate into the running average with `weight`;
    // see IterateAverage::add. Non-positive weights are ignored.
    virtual void addToAverage(double weight) = 0;
    [[nodiscard]] virtual bool averageEmpty() const noexcept = 0;
    virtual void resetAverage() = 0;

    // Host views of the current iterate and of the average (working
    // coordinates). Valid until the next mutating call.
    [[nodiscard]] virtual const std::vector<double>& primal() = 0;
    [[nodiscard]] virtual const std::vector<double>& dual() = 0;
    [[nodiscard]] virtual const CandidateIterate& average() = 0;

    // Replaces the current iterate with the average and recomputes A*x.
    // Requires a non-empty average.
    virtual void restartFromAverage() = 0;

    // Cpu or Cuda; never Auto.
    [[nodiscard]] virtual ComputeBackend kind() const noexcept = 0;
    [[nodiscard]] virtual BackendProfile profile() const noexcept { return {}; }
};

// The CPU backend. `executor` and `plan` follow CpuPdhgKernel's conventions.
// `problem` and `preconditioner` must outlive the backend.
[[nodiscard]] std::unique_ptr<IterationBackend> makeCpuIterationBackend(
    const CompiledLp& problem,
    const DiagonalPreconditioner& preconditioner,
    Executor* executor = nullptr,
    const SpmvPlan* plan = nullptr
);

// Tuning knobs of the CUDA backend. Defaults are what the solver uses; tests
// override them to force every kernel path on small matrices.
struct CudaBackendConfig {
    int device = 0;

    // Lines (rows or columns) with more nonzeros than this are reduced by a
    // whole thread block instead of a thread group, so one dense row cannot
    // serialise a warp while the rest of the grid idles.
    std::int64_t heavyLineNonzeros = 4096;

    // Threads cooperating on one line; 0 derives it from the mean line length.
    int forcedGroupSize = 0;
};

// The CUDA backend, or null with `error` set if it cannot be created (not
// compiled in, no usable device, not enough device memory, a CUDA failure
// during upload). Never falls back on its own; the caller decides.
[[nodiscard]] std::unique_ptr<IterationBackend> makeCudaIterationBackend(
    const CompiledLp& problem,
    const DiagonalPreconditioner& preconditioner,
    const CudaBackendConfig& config,
    std::string& error
);

}  // namespace pdlp

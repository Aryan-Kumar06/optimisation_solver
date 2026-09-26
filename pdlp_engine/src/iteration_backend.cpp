#include "pdlp/iteration_backend.h"

#include "pdlp/iterate_average.h"

#include <stdexcept>
#include <utility>

namespace pdlp {
namespace {

// The CPU backend is the pre-existing solver state, unchanged: CpuPdhgKernel
// for the step, PdlpState for the iterate, IterateAverage for the average.
// Host views are the live vectors themselves, so the solver's host-side checks
// read the iterate in place exactly as they did before this class existed.
class CpuIterationBackend final : public IterationBackend {
public:
    CpuIterationBackend(
        const CompiledLp& problem,
        const DiagonalPreconditioner& preconditioner,
        Executor* executor,
        const SpmvPlan* plan
    )
        : preconditioner_(preconditioner),
          kernel_(problem, executor, plan),
          average_(problem.numColumns(), problem.numRows(), executor) {}

    void reset(const std::vector<double>& primal, const std::vector<double>& dual) override {
        state_.primal = primal;
        state_.dual = dual;
        state_.iteration = 0;
        kernel_.refreshActivity(state_);
        average_.reset();
    }

    KernelTrialResult trial(const StepParameters& steps) override {
        return kernel_.trial(state_, preconditioner_, steps);
    }

    void commit() override { kernel_.commit(state_); }

    std::int64_t iteration() const noexcept override { return state_.iteration; }

    void addToAverage(double weight) override {
        average_.add(state_.primal, state_.dual, weight);
    }

    bool averageEmpty() const noexcept override { return average_.empty(); }

    void resetAverage() override { average_.reset(); }

    const std::vector<double>& primal() override { return state_.primal; }
    const std::vector<double>& dual() override { return state_.dual; }
    const CandidateIterate& average() override { return average_.candidate(); }

    void restartFromAverage() override {
        const CandidateIterate& averaged = average_.candidate();
        state_.primal = averaged.primal;
        state_.dual = averaged.dual;
        kernel_.refreshActivity(state_);
    }

    ComputeBackend kind() const noexcept override { return ComputeBackend::Cpu; }

private:
    const DiagonalPreconditioner& preconditioner_;
    CpuPdhgKernel kernel_;
    IterateAverage average_;
    PdlpState state_;
};

}  // namespace

const char* toString(ComputeBackend backend) noexcept {
    switch (backend) {
        case ComputeBackend::Auto: return "auto";
        case ComputeBackend::Cpu:  return "cpu";
        case ComputeBackend::Cuda: return "cuda";
    }
    return "unknown";
}

std::unique_ptr<IterationBackend> makeCpuIterationBackend(
    const CompiledLp& problem,
    const DiagonalPreconditioner& preconditioner,
    Executor* executor,
    const SpmvPlan* plan
) {
    return std::make_unique<CpuIterationBackend>(problem, preconditioner, executor, plan);
}

}  // namespace pdlp

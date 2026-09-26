#pragma once

#include <cstdint>

#include "pdlp/compiled_lp.h"
#include "pdlp/parallel.h"
#include "pdlp/pdlp_options.h"
#include "pdlp/pdlp_state.h"
#include "pdlp/scaling.h"
#include "pdlp/termination.h"

namespace pdlp {

class IterationBackend;

struct PolishingResult {
    // Always in the original problem's coordinates.
    CandidateIterate candidate;
    CandidateMetrics metrics;
    int iterations = 0;
    std::int64_t stepTrials = 0;
    double hostCheckSeconds = 0.0;   // time in host-side candidate scoring
};

struct PolishingInput {
    const CompiledLp* working = nullptr;         // scaled problem, or original
    const ProblemScaling* scaling = nullptr;     // null when unscaled
    const DiagonalPreconditioner* preconditioner = nullptr;

    CandidateIterate scaledStart;                // starting iterate, working coordinates
    CandidateIterate originalStart;              // the same point, original coordinates
    CandidateMetrics startMetrics;

    StepParameters steps;
    double remainingSeconds = 0.0;

    Executor* executor = nullptr;
    const SpmvPlan* plan = nullptr;

    // Backend to iterate on, bound to *working and *preconditioner. Its state
    // is overwritten. Null builds a CPU backend from executor and plan.
    IterationBackend* backend = nullptr;
};

class FeasibilityPolisher {
public:
    // `checker` must be bound to the original problem.
    [[nodiscard]] static PolishingResult polish(
        const PdlpOptions& options,
        const PolishingInput& input,
        TerminationChecker& checker
    );
};

}  // namespace pdlp

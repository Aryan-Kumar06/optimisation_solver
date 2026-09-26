#pragma once

#include "pdlp/compute_backend.h"

#include <cstdint>

namespace pdlp {

struct PdlpOptions {
    std::int64_t iterationLimit = 200000;
    double timeLimitSeconds = 0.0;

    double primalTolerance = 1e-6;
    double dualTolerance = 1e-6;
    double gapTolerance = 1e-6;

    int terminationCheckFrequency = 100;
    int powerIterationCount = 100;

    // Worker threads. 0 selects hardware_concurrency; 1 forces serial
    // execution. Small models run serially regardless: below
    // parallelNonzeroThreshold the barrier costs more than the work.
    int threadCount = 0;
    std::int64_t parallelNonzeroThreshold = 20000;

    double initialStepSafety = 0.90;
    double minimumStepScale = 1e-6;
    double maximumPrimalWeight = 1e6;
    double minimumPrimalWeight = 1e-6;

    // Ruiz equilibration of A before solving. The solver iterates on the
    // equilibrated problem but always tests termination on the original, so
    // tolerances stay in the caller's units.
    bool useRuizScaling = true;
    int ruizIterations = 10;

    // Infeasibility and unboundedness detection from the iterate sequence.
    // A direction is accepted as a certificate when its distance to the
    // required cone, relative to the certificate's own value, falls below
    // infeasibilityTolerance.
    bool detectInfeasibility = true;
    double infeasibilityTolerance = 1e-8;

    bool useDiagonalPreconditioning = true;

    // PDLP adaptive linesearch. The step size is free to exceed the static
    // Pock-Chambolle bound, which is the single largest reduction in iteration
    // count available; the static bound is used only to initialise it. Turning
    // this off pins the step at that bound.
    bool useAdaptiveLinesearch = true;
    int maximumStepTrials = 60;
    bool useAveraging = true;
    bool useRestarts = true;
    bool useFeasibilityPolishing = true;

    // Restart policy. See RestartController for what each threshold triggers.
    int minimumRestartIterations = 64;
    int maximumRestartIterations = 100000;
    double restartSufficientDecay = 0.20;
    double restartNecessaryDecay = 0.80;
    double restartArtificialFraction = 0.36;

    // Primal weight omega: primal step = eta/omega, dual step = eta*omega.
    //
    // Starts at 1 and is corrected at each restart toward the ratio of dual to
    // primal distance travelled since the previous restart, smoothed in log
    // space by primalWeightSmoothing.
    //
    // Initialising from ||c||/||b|| (as some formulations do) is deliberately
    // not the default: a single very slack one-sided row inflates ||b|| and
    // drives omega orders of magnitude off, which starves the dual step. That
    // measured worse here and could stall trivial models outright.
    double initialPrimalWeight = 1.0;
    double primalWeightSmoothing = 0.50;

    int polishingIterations = 5000;

    // Where the PDHG iteration runs; see ComputeBackend and docs/cuda.md. The
    // algorithm, its options and its termination tests are identical on every
    // backend -- only the per-iteration arithmetic moves.
    //
    // Auto never selects CUDA in a build without it, so this default leaves a
    // CPU-only build behaving exactly as it did before backends existed.
    ComputeBackend backend = ComputeBackend::Auto;
    int cudaDevice = 0;

    // Auto selects CUDA only for matrices with at least this many nonzeros:
    // below it, kernel launch and synchronisation latency outweigh the
    // bandwidth advantage. PROVISIONAL -- chosen conservatively and not yet
    // calibrated on hardware; docs/cuda.md describes how to measure it.
    std::int64_t cudaNonzeroThreshold = 1000000;
};

}  // namespace pdlp

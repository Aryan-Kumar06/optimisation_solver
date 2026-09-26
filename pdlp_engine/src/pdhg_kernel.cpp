#include "pdlp/pdhg_kernel.h"

#include "pdlp/pdhg_math.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <stdexcept>

namespace pdlp {

CpuPdhgKernel::CpuPdhgKernel(
    const CompiledLp& problem,
    Executor* executor,
    const SpmvPlan* plan
)
    : problem_(problem), executor_(executor), plan_(plan) {
    if (executor_ != nullptr && plan_ != nullptr && plan_->parts > 1) {
        parts_ = plan_->parts;
    } else {
        executor_ = nullptr;
        plan_ = nullptr;
        parts_ = 1;
    }
    reductions_.assign(static_cast<std::size_t>(parts_), PartialReduction{});
    primalTrial_.assign(static_cast<std::size_t>(problem.numColumns()), 0.0);
    dualTrial_.assign(static_cast<std::size_t>(problem.numRows()), 0.0);
    activityTrial_.assign(static_cast<std::size_t>(problem.numRows()), 0.0);
}

void CpuPdhgKernel::refreshActivity(PdlpState& state) const {
    problem_.matrix.multiply(state.primal, state.rowActivity, executor_, plan_);
}

// Computes the slice [begin, end) of A^T*y^k and applies the primal prox on the
// same slice, writing the trial iterate. Returns the slice's contribution to
// sum_j dx_j^2 / T_j.
double CpuPdhgKernel::primalHalf(
    const PdlpState& state,
    const DiagonalPreconditioner& preconditioner,
    const StepParameters& steps,
    int begin,
    int end
) noexcept {
    using Offset = SparseMatrix::Offset;
    const SparseMatrix& matrix = problem_.matrix;
    const Offset* const __restrict columnStart = matrix.cscColumnStart().data();
    const int* const __restrict rowIndex = matrix.cscRowIndex().data();
    const double* const __restrict entries = matrix.cscValues().data();

    const double* const __restrict objective = problem_.objective.data();
    const double* const __restrict lower = problem_.variableLower.data();
    const double* const __restrict upper = problem_.variableUpper.data();
    const double* const __restrict scale = preconditioner.primalScale.data();
    const double* const __restrict scaleInverse = preconditioner.primalScaleInverse.data();
    const double* const __restrict dual = state.dual.data();
    const double* const __restrict primal = state.primal.data();
    double* const __restrict trial = primalTrial_.data();

    const double multiplier = steps.globalStep / steps.primalWeight;
    double movementWeighted = 0.0;

    for (int column = begin; column < end; ++column) {
        // A^T*y restricted to this column, with four independent accumulators:
        // one would serialise the reduction on FMA latency.
        Offset k = columnStart[column];
        const Offset last = columnStart[column + 1];
        double a0 = 0.0;
        double a1 = 0.0;
        double a2 = 0.0;
        double a3 = 0.0;
        for (; k + 4 <= last; k += 4) {
            a0 += entries[k] * dual[rowIndex[k]];
            a1 += entries[k + 1] * dual[rowIndex[k + 1]];
            a2 += entries[k + 2] * dual[rowIndex[k + 2]];
            a3 += entries[k + 3] * dual[rowIndex[k + 3]];
        }
        for (; k < last; ++k) {
            a0 += entries[k] * dual[rowIndex[k]];
        }
        const double gradient = objective[column] + ((a0 + a1) + (a2 + a3));

        const double previous = primal[column];
        const double updated = math::primalUpdate(
            previous, multiplier, scale[column], gradient, lower[column], upper[column]);
        const double delta = updated - previous;

        trial[column] = updated;
        movementWeighted += delta * delta * scaleInverse[column];
    }
    return movementWeighted;
}

// Computes the slice [begin, end) of A*x' and applies the dual prox on the same
// slice. The extrapolated activity and A*dx are both recovered from A*x' and the
// carried A*x^k, so the interaction term costs no extra sparse product.
void CpuPdhgKernel::dualHalf(
    const PdlpState& state,
    const DiagonalPreconditioner& preconditioner,
    const StepParameters& steps,
    int begin,
    int end,
    double& movementWeighted,
    double& interaction
) noexcept {
    using Offset = SparseMatrix::Offset;
    const SparseMatrix& matrix = problem_.matrix;
    const Offset* const __restrict rowStart = matrix.csrRowStart().data();
    const int* const __restrict columnIndex = matrix.csrColumnIndex().data();
    const double* const __restrict entries = matrix.csrValues().data();

    const double* const __restrict lower = problem_.rowLower.data();
    const double* const __restrict upper = problem_.rowUpper.data();
    const double* const __restrict scale = preconditioner.dualScale.data();
    const double* const __restrict scaleInverse = preconditioner.dualScaleInverse.data();
    const double* const __restrict trialPrimal = primalTrial_.data();
    const double* const __restrict dual = state.dual.data();
    const double* const __restrict activity = state.rowActivity.data();
    double* const __restrict trialDual = dualTrial_.data();
    double* const __restrict trialActivity = activityTrial_.data();

    const double multiplier = steps.globalStep * steps.primalWeight;
    double movement = 0.0;
    double product = 0.0;

    for (int row = begin; row < end; ++row) {
        Offset k = rowStart[row];
        const Offset last = rowStart[row + 1];
        double a0 = 0.0;
        double a1 = 0.0;
        double a2 = 0.0;
        double a3 = 0.0;
        for (; k + 4 <= last; k += 4) {
            a0 += entries[k] * trialPrimal[columnIndex[k]];
            a1 += entries[k + 1] * trialPrimal[columnIndex[k + 1]];
            a2 += entries[k + 2] * trialPrimal[columnIndex[k + 2]];
            a3 += entries[k + 3] * trialPrimal[columnIndex[k + 3]];
        }
        for (; k < last; ++k) {
            a0 += entries[k] * trialPrimal[columnIndex[k]];
        }
        const double nextActivity = (a0 + a1) + (a2 + a3);   // (A x')_i

        // Extrapolation and the Moreau-form prox; see math::dualUpdate. The
        // branchless form yields exactly 0.0 on an inactive row, so duals of
        // slack rows stay clean instead of accumulating noise across iterations.
        const double previous = dual[row];
        const math::DualUpdate step = math::dualUpdate(
            previous, nextActivity, activity[row], multiplier, scale[row],
            lower[row], upper[row]);

        const double delta = step.value - previous;
        trialDual[row] = step.value;
        trialActivity[row] = nextActivity;

        movement += delta * delta * scaleInverse[row];
        product += delta * step.deltaActivity;
    }

    movementWeighted = movement;
    interaction = product;
}

KernelTrialResult CpuPdhgKernel::trial(
    const PdlpState& state,
    const DiagonalPreconditioner& preconditioner,
    const StepParameters& steps
) {
    const int columns = problem_.numColumns();
    const int rows = problem_.numRows();
    if (state.primal.size() != static_cast<std::size_t>(columns) ||
        state.dual.size() != static_cast<std::size_t>(rows) ||
        state.rowActivity.size() != static_cast<std::size_t>(rows)) {
        throw std::invalid_argument("PDHG state dimensions do not match the LP");
    }

    KernelTrialResult result;

    if (parts_ <= 1) {
        result.primalMovementWeighted =
            primalHalf(state, preconditioner, steps, 0, columns);
        dualHalf(
            state, preconditioner, steps, 0, rows,
            result.dualMovementWeighted, result.interaction
        );
    } else {
        // Workers claim chunks dynamically so that a slow core cannot hold the
        // barrier; see SpmvPlan.
        const std::vector<int>& columnChunk = plan_->columnChunk;
        const int columnChunks = plan_->columnChunkCount();
        std::atomic<int> columnCursor{0};
        executor_->run([&](int worker) {
            double accumulated = 0.0;
            while (true) {
                const int index = columnCursor.fetch_add(1, std::memory_order_relaxed);
                if (index >= columnChunks) {
                    break;
                }
                accumulated += primalHalf(
                    state, preconditioner, steps,
                    columnChunk[static_cast<std::size_t>(index)],
                    columnChunk[static_cast<std::size_t>(index) + 1]
                );
            }
            reductions_[static_cast<std::size_t>(worker)].primal = accumulated;
        });

        // A*x' reads the complete trial primal, so the halves are separated by
        // exactly one barrier.
        const std::vector<int>& rowChunk = plan_->rowChunk;
        const int rowChunks = plan_->rowChunkCount();
        std::atomic<int> rowCursor{0};
        executor_->run([&](int worker) {
            double movement = 0.0;
            double interaction = 0.0;
            while (true) {
                const int index = rowCursor.fetch_add(1, std::memory_order_relaxed);
                if (index >= rowChunks) {
                    break;
                }
                double partialMovement = 0.0;
                double partialInteraction = 0.0;
                dualHalf(
                    state, preconditioner, steps,
                    rowChunk[static_cast<std::size_t>(index)],
                    rowChunk[static_cast<std::size_t>(index) + 1],
                    partialMovement, partialInteraction
                );
                movement += partialMovement;
                interaction += partialInteraction;
            }
            reductions_[static_cast<std::size_t>(worker)].dual = movement;
            reductions_[static_cast<std::size_t>(worker)].interaction = interaction;
        });

        for (const PartialReduction& partial : reductions_) {
            result.primalMovementWeighted += partial.primal;
            result.dualMovementWeighted += partial.dual;
            result.interaction += partial.interaction;
        }
    }

    // A non-finite iterate always surfaces as a non-finite reduction: a NaN
    // propagates through the difference, and an infinite coordinate produces an
    // infinite delta on the iteration that created it. Testing the reductions
    // therefore replaces two full scans of the iterate.
    result.finite = std::isfinite(result.primalMovementWeighted) &&
        std::isfinite(result.dualMovementWeighted) &&
        std::isfinite(result.interaction);

    return result;
}

void CpuPdhgKernel::commit(PdlpState& state) {
    state.primal.swap(primalTrial_);
    state.dual.swap(dualTrial_);
    state.rowActivity.swap(activityTrial_);
    ++state.iteration;
}

}  // namespace pdlp

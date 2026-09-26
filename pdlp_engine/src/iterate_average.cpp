#include "pdlp/iterate_average.h"

#include "pdlp/pdhg_math.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace pdlp {
namespace {

// Below this many coordinates the barrier costs more than the update.
constexpr std::int64_t kParallelThreshold = 1 << 14;

inline void blend(
    double* __restrict target,
    const double* __restrict source,
    double fraction,
    std::int64_t begin,
    std::int64_t end
) noexcept {
    for (std::int64_t i = begin; i < end; ++i) {
        target[i] = math::blended(target[i], source[i], fraction);
    }
}

}  // namespace

IterateAverage::IterateAverage(int numColumns, int numRows, Executor* executor)
    : executor_(executor) {
    average_.primal.assign(static_cast<std::size_t>(numColumns), 0.0);
    average_.dual.assign(static_cast<std::size_t>(numRows), 0.0);
    if (executor_ != nullptr && executor_->threadCount() > 1) {
        parts_ = executor_->threadCount();
    } else {
        executor_ = nullptr;
        parts_ = 1;
    }
}

void IterateAverage::add(
    const std::vector<double>& primal,
    const std::vector<double>& dual,
    double weight
) {
    if (primal.size() != average_.primal.size() ||
        dual.size() != average_.dual.size()) {
        throw std::invalid_argument(
            "IterateAverage received an iterate with the wrong dimension");
    }
    if (!(weight > 0.0)) {
        return;
    }

    const double newTotal = totalWeight_ + weight;
    const double fraction = weight / newTotal;

    const auto columns = static_cast<std::int64_t>(average_.primal.size());
    const auto rows = static_cast<std::int64_t>(average_.dual.size());

    double* const primalTarget = average_.primal.data();
    double* const dualTarget = average_.dual.data();

    if (executor_ == nullptr || columns + rows < kParallelThreshold) {
        blend(primalTarget, primal.data(), fraction, 0, columns);
        blend(dualTarget, dual.data(), fraction, 0, rows);
    } else {
        // A single parallel region covers both vectors: each worker takes its
        // slice of the primal and of the dual, so averaging adds one barrier
        // per iteration rather than two.
        const double* const primalSource = primal.data();
        const double* const dualSource = dual.data();
        const int parts = parts_;
        executor_->run([&](int part) {
            const Range columnRange = evenRange(columns, parts, part);
            blend(primalTarget, primalSource, fraction, columnRange.begin, columnRange.end);
            const Range rowRange = evenRange(rows, parts, part);
            blend(dualTarget, dualSource, fraction, rowRange.begin, rowRange.end);
        });
    }

    totalWeight_ = newTotal;
}

const CandidateIterate& IterateAverage::candidate() const {
    if (empty()) {
        throw std::logic_error("Cannot request an empty iterate average");
    }
    return average_;
}

void IterateAverage::reset() {
    std::fill(average_.primal.begin(), average_.primal.end(), 0.0);
    std::fill(average_.dual.begin(), average_.dual.end(), 0.0);
    totalWeight_ = 0.0;
}

}  // namespace pdlp

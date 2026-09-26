#pragma once

// Per-coordinate PDHG arithmetic, shared by the CPU and CUDA kernels.
//
// Everything after a coordinate's sparse dot product lives here, once, so the
// two backends cannot drift apart: CpuPdhgKernel and the CUDA kernels call the
// same functions on the same operands in the same order. The only intended
// difference between the backends is the summation order inside the dot
// products themselves.
//
// This header compiles as plain C++ and as CUDA; nothing in it may depend on
// the standard library at run time.

#if defined(__CUDACC__)
#define PDLP_HOST_DEVICE __host__ __device__
#else
#define PDLP_HOST_DEVICE
#endif

namespace pdlp {
namespace math {

// Guards the dual prox against a step that has underflowed to exactly zero,
// which would otherwise evaluate 0 * infinity on a one-sided row.
constexpr double kMinimumStep = 1e-300;

// std::min / std::max semantics, spelled out so device code has them too.
//
// These are NOT fmin/fmax. The two families disagree whenever an argument is
// NaN: std::max(NaN, 0.0) evaluates (NaN < 0.0), which is false, and returns
// the NaN, whereas fmax(NaN, 0.0) returns 0.0. The solver detects numerical
// failure by a NaN surviving into the linesearch reductions -- the dual prox
// below relies on max(v - step*upper, 0.0) passing a NaN v through -- so the
// device must evaluate the same comparisons, in the same argument order, as
// the CPU, or it would silently repair iterates the CPU reports as failed.
PDLP_HOST_DEVICE inline double minOf(double a, double b) noexcept {
    return (b < a) ? b : a;
}

PDLP_HOST_DEVICE inline double maxOf(double a, double b) noexcept {
    return (a < b) ? b : a;
}

// std::max(lower, std::min(value, upper)). Infinite bounds are handled by the
// comparisons themselves; no bound is ever replaced by a large finite number.
PDLP_HOST_DEVICE inline double clipped(double value, double lower, double upper) noexcept {
    return maxOf(lower, minOf(value, upper));
}

// Primal prox of the PDHG step for one column:
//   x' = clip(x - (eta/omega) * T_j * (c_j + (A^T y)_j), l_j, u_j).
// `multiplier` is eta/omega and `gradient` is c_j + (A^T y)_j.
PDLP_HOST_DEVICE inline double primalUpdate(
    double previous,
    double multiplier,
    double scale,
    double gradient,
    double lower,
    double upper
) noexcept {
    return clipped(previous - multiplier * scale * gradient, lower, upper);
}

// Dual step for one row, given (A x')_i and the carried (A x^k)_i.
struct DualUpdate {
    double value;          // y'_i
    double deltaActivity;  // (A dx)_i, for the interaction term
};

// Moreau identity for the support function of [lower, upper]:
//   y+ = v - step * clip(v / step, lower, upper)
// rewritten branchlessly and without the division as
//   y+ = max(v - step*upper, 0) + min(v - step*lower, 0),
// with v = y + step * (A x_bar)_i and A x_bar = 2 A x' - A x^k. The two forms
// agree exactly on all three cases, including equality rows and infinite
// bounds, and this one yields exactly 0.0 on an inactive row rather than the
// rounding residue of v - step*(v/step). `multiplier` is eta*omega.
PDLP_HOST_DEVICE inline DualUpdate dualUpdate(
    double previous,
    double nextActivity,
    double previousActivity,
    double multiplier,
    double scale,
    double lower,
    double upper
) noexcept {
    const double deltaActivity = nextActivity - previousActivity;  // (A dx)_i
    const double extrapolated = nextActivity + deltaActivity;      // (A x_bar)_i
    const double step = maxOf(multiplier * scale, kMinimumStep);
    const double v = previous + step * extrapolated;
    const double updated = maxOf(v - step * upper, 0.0) + minOf(v - step * lower, 0.0);
    return DualUpdate{updated, deltaActivity};
}

// avg += w/W * (x - avg); see IterateAverage.
PDLP_HOST_DEVICE inline double blended(double average, double value, double fraction) noexcept {
    return average + fraction * (value - average);
}

}  // namespace math
}  // namespace pdlp

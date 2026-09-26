#pragma once

#include <cstdint>
#include <string>

namespace qp {

// Where the ADMM vector and sparse-product work runs.
//
//   Auto  CUDA when this build has it, a usable device exists, and A and P
//         together have at least AdmmOptions::cudaNonzeroThreshold nonzeros;
//         CPU otherwise. Always CPU in a build without CUDA.
//   Cpu   Always the CPU.
//   Cuda  The hybrid CUDA backend or nothing: if it cannot run, the solve
//         returns InvalidProblem with the reason rather than using the CPU.
//
// The CUDA backend is HYBRID: the KKT factorisation and triangular solves stay
// on the CPU (see docs/cuda.md), so every iteration moves the x-update
// right-hand side to the host and the solution back.
enum class ComputeBackend {
    Auto,
    Cpu,
    Cuda
};

[[nodiscard]] const char* toString(ComputeBackend backend) noexcept;

// Transfer and setup accounting for a backend. All zero on the CPU.
struct BackendProfile {
    double setupSeconds = 0.0;        // device init, allocation, problem upload
    double snapshotSeconds = 0.0;     // downloads for host-side termination checks
    std::int64_t hostToDeviceBytes = 0;
    std::int64_t deviceToHostBytes = 0;
    std::int64_t synchronisations = 0;  // host waits on the device
};

struct CudaAvailability {
    bool compiled = false;
    bool usable = false;
    std::string deviceName;
    std::string reason;
    int runtimeVersion = 0;  // CUDA runtime, e.g. 12080 for 12.8; 0 when not compiled
    int driverVersion = 0;   // highest CUDA version the driver supports; 0 if unknown
};

[[nodiscard]] CudaAvailability cudaAvailability(int device = 0);

}  // namespace qp

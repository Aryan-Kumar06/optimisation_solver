#pragma once

#include <cstdint>
#include <string>

namespace pdlp {

// Where the PDHG iteration runs.
//
//   Auto  CUDA when this build has it, a usable device exists, and the matrix
//         has at least PdlpOptions::cudaNonzeroThreshold nonzeros; CPU
//         otherwise. Always CPU in a build without CUDA.
//   Cpu   Always the CPU kernel.
//   Cuda  The CUDA kernel or nothing: if CUDA is not compiled in, or no usable
//         device exists, the solve returns InvalidProblem with the reason
//         rather than quietly running on the CPU.
enum class ComputeBackend {
    Auto,
    Cpu,
    Cuda
};

[[nodiscard]] const char* toString(ComputeBackend backend) noexcept;

// Transfer and setup accounting for a backend. All zero on the CPU.
struct BackendProfile {
    double setupSeconds = 0.0;        // device init, allocation, problem upload
    double snapshotSeconds = 0.0;     // downloads of iterates for host-side checks
    std::int64_t hostToDeviceBytes = 0;
    std::int64_t deviceToHostBytes = 0;
    std::int64_t synchronisations = 0;  // host waits on the device
};

struct CudaAvailability {
    bool compiled = false;   // this build contains the CUDA backend
    bool usable = false;     // compiled, and the requested device can be used
    std::string deviceName;  // when usable
    std::string reason;      // why not usable, when it is not
    int runtimeVersion = 0;  // CUDA runtime, e.g. 12080 for 12.8; 0 when not compiled
    int driverVersion = 0;   // highest CUDA version the driver supports; 0 if unknown
};

// Probes the CUDA runtime for `device`. In a build without CUDA this returns
// compiled == false without touching any driver.
[[nodiscard]] CudaAvailability cudaAvailability(int device = 0);

}  // namespace pdlp

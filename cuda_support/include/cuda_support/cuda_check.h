#pragma once

// Error handling for the CUDA runtime and the CUDA math libraries.
//
// Included only from CUDA translation units: nothing in a CPU-only build sees
// this header. Every failure becomes a CudaError carrying the failing call, the
// source location and the library's own status string, because "CUDA failed"
// with no context is not actionable on a machine the author has never seen.
//
// The macros exist only to capture __FILE__/__LINE__ and the call text; they
// contain no algorithmic logic.

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#if defined(CUDA_SUPPORT_WITH_CUSPARSE)
#include <cusparse.h>
#endif

namespace cuda_support {

class CudaError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

[[noreturn]] inline void throwCudaError(
    const char* library,
    const char* call,
    const char* file,
    int line,
    const std::string& status
) {
    throw CudaError(
        std::string(library) + " call failed: " + call + " at " + file + ":" +
        std::to_string(line) + ": " + status);
}

inline void check(cudaError_t status, const char* call, const char* file, int line) {
    if (status != cudaSuccess) {
        throwCudaError("CUDA runtime", call, file, line,
              std::string(cudaGetErrorName(status)) + " (" + cudaGetErrorString(status) + ")");
    }
}

#if defined(CUDA_SUPPORT_WITH_CUSPARSE)
inline void check(cusparseStatus_t status, const char* call, const char* file, int line) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throwCudaError("cuSPARSE", call, file, line,
              std::string(cusparseGetErrorName(status)) + " (" +
                  cusparseGetErrorString(status) + ")");
    }
}
#endif

}  // namespace cuda_support

#define CUDA_SUPPORT_CHECK(call) \
    ::cuda_support::check((call), #call, __FILE__, __LINE__)

// Kernel launches report configuration errors lazily; this surfaces them at the
// launch site instead of at some later, unrelated synchronisation point. It
// does not synchronise.
#define CUDA_SUPPORT_CHECK_LAUNCH() \
    ::cuda_support::check(cudaGetLastError(), "kernel launch", __FILE__, __LINE__)

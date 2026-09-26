// Compiled INSTEAD of the CUDA sources when PDLP_ENABLE_CUDA is off, so a
// CPU-only build carries the same API and no CUDA dependency. Nothing here
// touches a driver.

#include "pdlp/compute_backend.h"
#include "pdlp/iteration_backend.h"

namespace pdlp {

CudaAvailability cudaAvailability(int /*device*/) {
    CudaAvailability availability;
    availability.compiled = false;
    availability.usable = false;
    availability.reason = "this build of pdlp_engine was configured without CUDA "
                          "(reconfigure with -DPDLP_ENABLE_CUDA=ON or "
                          "-DOPTIMSOLVER_ENABLE_CUDA=ON)";
    return availability;
}

std::unique_ptr<IterationBackend> makeCudaIterationBackend(
    const CompiledLp& /*problem*/,
    const DiagonalPreconditioner& /*preconditioner*/,
    const CudaBackendConfig& config,
    std::string& error
) {
    error = cudaAvailability(config.device).reason;
    return nullptr;
}

}  // namespace pdlp

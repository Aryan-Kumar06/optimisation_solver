// Compiled INSTEAD of the CUDA sources when QP_ENABLE_CUDA is off, so a
// CPU-only build carries the same API and no CUDA dependency. Nothing here
// touches a driver.

#include "qp/admm_backend.h"
#include "qp/compute_backend.h"

namespace qp {

CudaAvailability cudaAvailability(int /*device*/) {
    CudaAvailability availability;
    availability.compiled = false;
    availability.usable = false;
    availability.reason = "this build of qp_engine was configured without CUDA "
                          "(reconfigure with -DQP_ENABLE_CUDA=ON or "
                          "-DOPTIMSOLVER_ENABLE_CUDA=ON)";
    return availability;
}

std::unique_ptr<AdmmBackend> makeCudaAdmmBackend(
    const QpModel& /*model*/,
    int device,
    std::string& error
) {
    error = cudaAvailability(device).reason;
    return nullptr;
}

}  // namespace qp

#pragma once

// Owning wrappers for device memory, pinned host memory, streams and events.
//
// Every solver-owned CUDA resource lives in one of these, so an exception
// thrown halfway through setup (out of memory, a bad status from a library)
// releases everything allocated before it. Destructors never throw: a failure
// while freeing is unrecoverable and reporting it from a destructor would
// terminate the process.

#include "cuda_support/cuda_check.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace cuda_support {

// count * sizeof(T), refusing to wrap. A wrapped size would allocate a small
// buffer and every later copy into it would be an out-of-bounds write.
template <typename T>
[[nodiscard]] std::size_t byteCount(std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw CudaError("Device allocation size overflows size_t");
    }
    return count * sizeof(T);
}

// Accumulates a memory requirement without overflow, for the up-front check
// against free device memory.
class ByteBudget {
public:
    template <typename T>
    void add(std::size_t count) {
        const std::size_t bytes = byteCount<T>(count);
        if (bytes > std::numeric_limits<std::size_t>::max() - total_) {
            throw CudaError("Device memory requirement overflows size_t");
        }
        total_ += bytes;
    }
    [[nodiscard]] std::size_t total() const noexcept { return total_; }

private:
    std::size_t total_ = 0;
};

// Sets the current device for its lifetime and restores the previous one.
class DeviceGuard {
public:
    explicit DeviceGuard(int device) {
        CUDA_SUPPORT_CHECK(cudaGetDevice(&previous_));
        CUDA_SUPPORT_CHECK(cudaSetDevice(device));
    }
    ~DeviceGuard() { cudaSetDevice(previous_); }

    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

private:
    int previous_ = 0;
};

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { allocate(count); }
    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          count_(std::exchange(other.count_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    // A zero-length buffer holds a null pointer and is never passed to a kernel
    // that would dereference it: every launch site skips empty ranges.
    void allocate(std::size_t count) {
        release();
        if (count == 0) {
            return;
        }
        void* raw = nullptr;
        CUDA_SUPPORT_CHECK(cudaMalloc(&raw, byteCount<T>(count)));
        data_ = static_cast<T*>(raw);
        count_ = count;
    }

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return count_ * sizeof(T); }

    void swap(DeviceBuffer& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(count_, other.count_);
    }

private:
    void release() noexcept {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
        data_ = nullptr;
        count_ = 0;
    }

    T* data_ = nullptr;
    std::size_t count_ = 0;
};

// Page-locked host memory. Only small, frequently read results (the linesearch
// scalars) live here, allowing asynchronous transfers without pageable staging.
template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(std::size_t count) {
        if (count == 0) {
            return;
        }
        void* raw = nullptr;
        CUDA_SUPPORT_CHECK(cudaMallocHost(&raw, byteCount<T>(count)));
        data_ = static_cast<T*>(raw);
        count_ = count;
    }
    ~PinnedBuffer() {
        if (data_ != nullptr) {
            cudaFreeHost(data_);
        }
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          count_(std::exchange(other.count_, 0)) {}
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) {
                cudaFreeHost(data_);
            }
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }
    [[nodiscard]] T& operator[](std::size_t i) noexcept { return data_[i]; }
    [[nodiscard]] const T& operator[](std::size_t i) const noexcept { return data_[i]; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

class Stream {
public:
    Stream() { CUDA_SUPPORT_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    ~Stream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }
    void synchronize() const { CUDA_SUPPORT_CHECK(cudaStreamSynchronize(stream_)); }

private:
    cudaStream_t stream_ = nullptr;
};

class Event {
public:
    Event() { CUDA_SUPPORT_CHECK(cudaEventCreate(&event_)); }
    ~Event() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    void record(cudaStream_t stream) { CUDA_SUPPORT_CHECK(cudaEventRecord(event_, stream)); }
    void synchronize() const { CUDA_SUPPORT_CHECK(cudaEventSynchronize(event_)); }
    [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

    // Milliseconds between two recorded events; both must have completed.
    [[nodiscard]] static float elapsedMilliseconds(const Event& start, const Event& stop) {
        float ms = 0.0f;
        CUDA_SUPPORT_CHECK(cudaEventElapsedTime(&ms, start.event_, stop.event_));
        return ms;
    }

private:
    cudaEvent_t event_ = nullptr;
};

// Host -> device, asynchronous on `stream`. The source must stay alive and
// unmodified until the stream is synchronised.
template <typename T>
void uploadAsync(DeviceBuffer<T>& target, const std::vector<T>& source, cudaStream_t stream) {
    if (source.size() != target.size()) {
        throw CudaError("uploadAsync: size mismatch between host vector and device buffer");
    }
    if (source.empty()) {
        return;
    }
    CUDA_SUPPORT_CHECK(cudaMemcpyAsync(
        target.data(), source.data(), target.bytes(), cudaMemcpyHostToDevice, stream));
}

template <typename T>
[[nodiscard]] DeviceBuffer<T> uploadNew(const std::vector<T>& source, cudaStream_t stream) {
    DeviceBuffer<T> buffer(source.size());
    uploadAsync(buffer, source, stream);
    return buffer;
}

// Setup uploads from temporary staging vectors must complete before the vector
// is destroyed or reused. Pageable cudaMemcpyAsync is not a lifetime guarantee.
template <typename T>
[[nodiscard]] DeviceBuffer<T> uploadNewAndWait(const std::vector<T>& source, cudaStream_t stream) {
    auto buffer = uploadNew(source, stream);
    CUDA_SUPPORT_CHECK(cudaStreamSynchronize(stream));
    return buffer;
}

// Device -> host into a std::vector; returns only once the data has arrived.
template <typename T>
void download(std::vector<T>& target, const T* source, std::size_t count, cudaStream_t stream) {
    target.resize(count);
    if (count == 0) {
        return;
    }
    CUDA_SUPPORT_CHECK(cudaMemcpyAsync(
        target.data(), source, byteCount<T>(count), cudaMemcpyDeviceToHost, stream));
    CUDA_SUPPORT_CHECK(cudaStreamSynchronize(stream));
}

// Why a device cannot be used, or the empty string if it can. Creates the
// primary context as a side effect, which is the expensive part of first use
// and is therefore charged to setup rather than to the first iteration.
[[nodiscard]] inline std::string probeDevice(int device, std::string* name = nullptr) {
    int count = 0;
    const cudaError_t countStatus = cudaGetDeviceCount(&count);
    if (countStatus != cudaSuccess) {
        cudaGetLastError();  // consume the (non-sticky) error so later calls start clean
        return std::string("no usable CUDA driver/device: ") + cudaGetErrorString(countStatus);
    }
    if (count <= 0) {
        return "no CUDA device present";
    }
    if (device < 0 || device >= count) {
        return "CUDA device " + std::to_string(device) + " requested but only " +
               std::to_string(count) + " device(s) present";
    }
    cudaDeviceProp properties{};
    const cudaError_t propertyStatus = cudaGetDeviceProperties(&properties, device);
    if (propertyStatus != cudaSuccess) {
        cudaGetLastError();
        return std::string("cannot query CUDA device: ") + cudaGetErrorString(propertyStatus);
    }
    // cudaDeviceProp::computeMode was removed in CUDA 13; the attribute query
    // works on every supported toolkit (12.8+).
    int computeMode = cudaComputeModeDefault;
    const cudaError_t modeStatus = cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, device);
    if (modeStatus != cudaSuccess) {
        cudaGetLastError();
        return std::string("cannot query CUDA device: ") + cudaGetErrorString(modeStatus);
    }
    if (computeMode == cudaComputeModeProhibited) {
        return "CUDA device " + std::to_string(device) + " is in prohibited compute mode";
    }
    if (name != nullptr) {
        *name = properties.name;
    }
    int previous = 0;
    if (cudaGetDevice(&previous) != cudaSuccess) {
        cudaGetLastError();
        previous = 0;
    }
    const cudaError_t setStatus = cudaSetDevice(device);
    const cudaError_t initStatus = setStatus == cudaSuccess ? cudaFree(nullptr) : setStatus;
    cudaSetDevice(previous);
    if (initStatus != cudaSuccess) {
        cudaGetLastError();
        return std::string("cannot initialise CUDA device: ") + cudaGetErrorString(initStatus);
    }
    return {};
}

}  // namespace cuda_support

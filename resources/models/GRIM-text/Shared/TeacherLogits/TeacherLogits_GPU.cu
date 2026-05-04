#include "TeacherLogits_GPU.hpp"

#include <sstream>

namespace GRIM::TeacherLogits {

Buffer::~Buffer() {
    release(*this);
}

Buffer::Buffer(Buffer&& other) noexcept
    : device(other.device), capacity(other.capacity) {
    other.device = nullptr;
    other.capacity = 0;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        release(*this);
        device = other.device;
        capacity = other.capacity;
        other.device = nullptr;
        other.capacity = 0;
    }
    return *this;
}

namespace {
constexpr auto kModule = "TeacherLogits";

bool allocate(Buffer& buf, std::size_t elements) {
    if (elements == 0) {
        return false;
    }
    if (buf.device && buf.capacity >= elements) {
        return true;
    }
    release(buf);
    const std::size_t bytes = elements * sizeof(float);
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&buf.device), bytes);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "[TeacherLogits] cudaMalloc(" << bytes << ") failed: "
            << cudaGetErrorString(err);
        GRIM::Logging::EmitModuleError(kModule, oss.str());
        buf.device = nullptr;
        buf.capacity = 0;
        return false;
    }
    buf.capacity = elements;
    return true;
}
}  // namespace

bool ensureCapacity(Buffer& buf, std::size_t tokens, int vocab, cudaStream_t /*stream*/) {
    if (tokens == 0 || vocab <= 0) {
        GRIM::Logging::EmitModuleWarning(kModule, "[TeacherLogits] ensureCapacity called with empty shape");
        return false;
    }
    const std::size_t elements = tokens * static_cast<std::size_t>(vocab);
    const bool ok = allocate(buf, elements);
    if (ok) {
        std::ostringstream oss;
        oss << "[TeacherLogits] capacity=" << buf.capacity << " floats (tokens=" << tokens
            << ", vocab=" << vocab << ")";
        GRIM::Logging::EmitModuleInfo(kModule, oss.str());
    }
    return ok;
}

bool copyFromDevice(Buffer& buf,
                    const float* src,
                    std::size_t tokens,
                    int vocab,
                    cudaStream_t stream) {
    if (!ensureCapacity(buf, tokens, vocab, stream)) {
        return false;
    }
    const std::size_t elements = tokens * static_cast<std::size_t>(vocab);
    const std::size_t bytes = elements * sizeof(float);
    cudaError_t err = cudaMemcpyAsync(buf.device, src, bytes, cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "[TeacherLogits] D2D copy failed: " << cudaGetErrorString(err);
        GRIM::Logging::EmitModuleError(kModule, oss.str());
        return false;
    }
    return true;
}

bool copyFromHost(Buffer& buf,
                  const float* src_host,
                  std::size_t tokens,
                  int vocab,
                  cudaStream_t stream) {
    if (!ensureCapacity(buf, tokens, vocab, stream)) {
        return false;
    }
    const std::size_t elements = tokens * static_cast<std::size_t>(vocab);
    const std::size_t bytes = elements * sizeof(float);
    cudaError_t err = cudaMemcpyAsync(buf.device, src_host, bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "[TeacherLogits] H2D copy failed: " << cudaGetErrorString(err);
        GRIM::Logging::EmitModuleError(kModule, oss.str());
        return false;
    }
    return true;
}

void release(Buffer& buf) {
    if (buf.device) {
        cudaFree(buf.device);
    }
    buf.device = nullptr;
    buf.capacity = 0;
}

}  // namespace GRIM::TeacherLogits

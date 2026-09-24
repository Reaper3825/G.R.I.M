#pragma once
// Host-only CUDA stand-in. Include this directory ONLY for this unit test.
#include <cstddef>
#include <cstdlib>

using cudaError_t = int;
using cudaStream_t = void*;
struct FakeEvent { bool ready = false; };
using cudaEvent_t = FakeEvent*;
inline constexpr int cudaSuccess = 0;
inline constexpr int cudaErrorNotReady = 34;
inline constexpr int cudaEventDisableTiming = 2;
inline int fake_device = 0;
inline bool fake_free_failure = false;
inline bool fake_event_failure = false;
inline bool fake_allocation_failure = false;
inline FakeEvent* last_event = nullptr;
inline int events_alive = 0;
inline cudaError_t cudaGetDevice(int* device) { *device = fake_device; return cudaSuccess; }
inline cudaError_t cudaMemGetInfo(size_t* available, size_t* total) {
    *available = *total = 1024; return cudaSuccess;
}
inline const char* cudaGetErrorString(cudaError_t) { return "fake_cuda_error"; }
inline cudaError_t cudaMalloc(void** ptr, size_t bytes) {
    if (fake_allocation_failure) return 2;
    *ptr = std::malloc(bytes); return *ptr ? cudaSuccess : 2;
}
inline cudaError_t cudaFree(void* ptr) {
    if (fake_free_failure) return 1;
    std::free(ptr); return cudaSuccess;
}
inline cudaError_t cudaFreeAsync(void* ptr, cudaStream_t) { return cudaFree(ptr); }
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned) {
    if (fake_event_failure) return 1;
    *event = last_event = new FakeEvent;
    ++events_alive;
    return cudaSuccess;
}
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaEventQuery(cudaEvent_t event) {
    return event->ready ? cudaSuccess : cudaErrorNotReady;
}
inline cudaError_t cudaEventDestroy(cudaEvent_t event) {
    delete event; --events_alive; return cudaSuccess;
}

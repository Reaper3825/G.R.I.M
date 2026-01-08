#pragma once
#include <cuda_runtime.h>

template<int MaxCallbacks, typename... Args>
class GPUMulticastDelegate
{
public:
    using Callback = void(*)(Args...);

    __device__ bool Add(Callback cb)
    {
        int idx = atomicAdd(&count, 1);
        if (idx >= MaxCallbacks) {
            // overflow; undo
            atomicSub(&count, 1);
            return false;
        }
        callbacks[idx] = cb;
        return true;
    }

    __device__ void Clear()
    {
        count = 0;
    }

    __device__ void Broadcast(Args... args) const
    {
        // Every thread executes all callbacks… that’s how GPU “multicast” works.
        for (int i = 0; i < count; i++) {
            callbacks[i](args...);
        }
    }

    __device__ int Count() const { return count; }

private:
    Callback callbacks[MaxCallbacks] = {};
    int count = 0;
};


// -------------------------------
// SINGLE-CAST GPU DELEGATE
// -------------------------------
template<typename... Args>
class GPUSingleDelegate
{
public:
    using Callback = void(*)(Args...);

    __device__ void Bind(Callback callback)
    {
        cb = callback;
    }

    __device__ void Unbind()
    {
        cb = nullptr;
    }

    __device__ bool IsBound() const
    {
        return cb != nullptr;
    }

    __device__ void Execute(Args... args) const
    {
        if (cb) cb(args...);
    }

private:
    Callback cb = nullptr;
};

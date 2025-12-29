//ABSOLUTELY SHOULD NOT BE USED WITH GRIM-text

#pragma once
#include <functional>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <future>
#include <thread>

class DelegateHandle
{
public:
    DelegateHandle() : id(0) {}
    explicit DelegateHandle(std::uint64_t handleId) : id(handleId) {}

    bool IsValid() const { return id != 0; }
    bool operator==(const DelegateHandle& other) const { return id == other.id; }

private:
    friend struct std::hash<DelegateHandle>;
    std::uint64_t id;
};

namespace std {
    template<> struct hash<DelegateHandle> {
        std::size_t operator()(const DelegateHandle& h) const noexcept {
            return std::hash<std::uint64_t>{}(h.id);
        }
    };
}

// Generic multicast delegate template (similar to Unreal Engine's delegates)
template<typename... Args>
class MulticastDelegate
{
public:
    using Callback = std::function<void(Args...)>;

    // Add a listener and return a handle for later removal
    DelegateHandle Add(const Callback& func)
    {
        DelegateHandle handle(++lastId);
        listeners[handle] = func;
        return handle;
    }

    // Remove a specific listener by handle
    bool Remove(const DelegateHandle& handle)
    {
        return listeners.erase(handle) > 0;
    }

    // Clear all listeners
    void Clear()
    {
        listeners.clear();
    }

    // Broadcast synchronously - all listeners called on current thread
    void Broadcast(Args... args) const
    {
        for (const auto& [_, func] : listeners)
        {
            func(args...);
        }
    }

    // Broadcast asynchronously - all listeners called in parallel on thread pool
    void BroadcastAsync(Args... args) const
    {
        std::vector<std::future<void>> futures;
        futures.reserve(listeners.size());

        for (const auto& [_, func] : listeners)
        {
            // Launch each listener in its own thread
            futures.push_back(std::async(std::launch::async, func, args...));
        }

        // Optionally wait for all to complete (detached by default when futures go out of scope)
        // If you need to wait, you can iterate: for (auto& f : futures) f.wait();
    }

    // Broadcast with fire-and-forget async behavior (don't wait for completion)
    void BroadcastAsyncDetached(Args... args) const
    {
        for (const auto& pair : listeners)
        {
            const auto& callback = pair.second;
            std::thread([callback, args...]() { callback(args...); }).detach();
        }
    }

    bool IsEmpty() const { return listeners.empty(); }
    size_t Count() const { return listeners.size(); }

private:
    std::unordered_map<DelegateHandle, Callback> listeners;
    std::uint64_t lastId = 0;
};

// Single-cast delegate (only one listener at a time)
template<typename... Args>
class Delegate
{
public:
    using Callback = std::function<void(Args...)>;

    void Bind(const Callback& func)
    {
        callback = func;
    }

    void Unbind()
    {
        callback = nullptr;
    }

    bool IsBound() const
    {
        return callback != nullptr;
    }

    void Execute(Args... args) const
    {
        if (callback)
            callback(args...);
    }

    void ExecuteAsync(Args... args) const
    {
        if (callback)
            std::async(std::launch::async, callback, args...);
    }

    void ExecuteAsyncDetached(Args... args) const
    {
        if (callback)
            std::thread([cb = callback, args...]() { cb(args...); }).detach();
    }

private:
    Callback callback;
};

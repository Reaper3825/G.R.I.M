#include "../../Shared/Diagnostics/MemoryAllocationTracker.hpp"
#include <cassert>
#include <iostream>
#include <memory>

int main() {
    using namespace GRIM::MemoryAccounting;
    void* ptr = nullptr;
    cudaMallocOrThrow(&ptr, 4096, "consumer_A", Kind::Gradient);
    {
        auto owner = std::shared_ptr<void>(ptr, [](void* p) { GRIM::MemoryAccounting::free(p); });
        auto alias = owner;
        assert(registry().snapshot(0).bytes == 4096);
        owner.reset();
        assert(registry().snapshot(0).count == 1);
    }
    assert(registry().snapshot(0).bytes == 0);

    cudaMallocOrThrow(&ptr, 32, "async_temp");
    assert(freeAsync(ptr, nullptr) == cudaSuccess);
    auto snapshot = registry().snapshot(0);
    assert(snapshot.bytes == 32 && snapshot.pending_free_bytes == 32);
    last_event->ready = true;
    assert(registry().snapshot(0).bytes == 0);
    assert(events_alive == 0);

    // Failed frees must not hide a live allocation.
    cudaMallocOrThrow(&ptr, 64, "failed_free");
    fake_free_failure = true;
    assert(GRIM::MemoryAccounting::free(ptr) != cudaSuccess);
    assert(registry().snapshot(0).bytes == 64);
    fake_free_failure = false;
    GRIM::MemoryAccounting::free(ptr);
    assert(registry().snapshot(0).bytes == 0);
    assert(registry().snapshot(0).observation_errors == 1);

    // Per-device isolation and reclassification do not allocate/count twice.
    fake_device = 1;
    cudaMallocOrThrow(&ptr, 128, "engine_accumulator");
    classify(ptr, Kind::EngineGradient);
    snapshot = registry().snapshot(1);
    assert(snapshot.bytes == 128 && snapshot.count == 1);
    assert(snapshot.by_owner.at({Kind::EngineGradient, "engine_accumulator"}).bytes == 128);
    assert(registry().snapshot(0).bytes == 0);
    GRIM::MemoryAccounting::free(ptr);
    fake_device = 0;

    // A stale release must not erase a new allocation at the same address.
    Registry isolated;
    int address = 0;
    isolated.allocated(&address, 10, 0, "old");
    const auto old_id = isolated.identity(&address);
    cudaEvent_t pending;
    cudaEventCreateWithFlags(&pending, cudaEventDisableTiming);
    isolated.released(&address, old_id, pending);
    isolated.allocated(&address, 20, 0, "new");
    isolated.released(&address, old_id);
    assert(isolated.snapshot(0).bytes == 20);
    assert(events_alive == 0);
    isolated.released(&address, isolated.identity(&address));
    assert(isolated.snapshot(0).bytes == 0);

    // The same race is possible between cudaFree returning and bookkeeping.
    isolated.allocated(&address, 30, 0, "sync_old");
    const auto sync_id = isolated.identity(&address);
    isolated.allocated(&address, 40, 0, "sync_new");
    isolated.released(&address, sync_id);
    assert(isolated.snapshot(0).bytes == 40);
    assert(isolated.snapshot(0).observation_errors == 0);
    isolated.released(&address, isolated.identity(&address));

    // Observability failure must preserve CUDA success and flag incomplete data.
    cudaMallocOrThrow(&ptr, 256, "unobservable_async");
    const auto id = registry().identity(ptr);
    fake_event_failure = true;
    assert(freeAsync(ptr, nullptr) == cudaSuccess);
    fake_event_failure = false;
    assert(registry().snapshot(0).bytes == 256);
    assert(registry().snapshot(0).observation_errors == 2);
    registry().released(ptr, id); // fake backend already freed it

    fake_allocation_failure = true;
    bool threw = false;
    try { cudaMallocOrThrow(&ptr, 1024, "failed_allocation"); }
    catch (const std::runtime_error&) { threw = true; }
    fake_allocation_failure = false;
    assert(threw && registry().snapshot(0).bytes == 0);

    void* foreign = std::malloc(16);
    assert(GRIM::MemoryAccounting::free(foreign) == cudaSuccess);
    assert(registry().snapshot(0).bytes == 0);
    std::cout << "Memory accounting lifetime tests passed\n";
}

//======================================================//
//  ScratchBlock_GPU.hpp
//  Pinned Memory Scratch Blocks for Zero-Copy GPU Transfers
//  
//  Provides togglable pinned memory pools for async CPU→GPU
//  transfers. Designed for multi-model orchestration where
//  not every model needs internal intermediate buffers.
//  
//  Features:
//  - Double-buffered pinned memory (true async transfers)
//  - Runtime enable/disable (no overhead when disabled)
//  - Thread-safe block acquisition
//  
//  Performance:
//  - 2-3x faster transfers vs pageable memory
//  - True async: GPU trains batch N while CPU pads batch N+1
//  - Zero implicit synchronization
//  
//  Author: GRIM Team
//  Date: December 2025
//======================================================//

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <mutex>
#include <atomic>

namespace GRIM {
namespace ScratchBlock {

//======================================================//
//  Configuration
//======================================================//

struct ScratchBlockConfig {
    bool enabled = true;                  // Enable/disable scratch blocks
    size_t max_tokens_per_block = 16384;  // Max tokens per scratch block
    size_t num_blocks = 2;                // Number of blocks (2 = double buffer)
    bool use_write_combined = false;      // Use write-combined memory (faster for write-once data)
    
    ScratchBlockConfig() = default;
};

//======================================================//
//  Scratch Block Handle
//======================================================//

struct ScratchBlockHandle {
    int* data;                    // Pinned memory pointer
    size_t capacity_tokens;       // Capacity in tokens
    size_t capacity_bytes;        // Capacity in bytes
    uint32_t block_id;            // Block identifier
    bool is_pinned;               // True if pinned memory
    
    ScratchBlockHandle() 
        : data(nullptr)
        , capacity_tokens(0)
        , capacity_bytes(0)
        , block_id(0)
        , is_pinned(false) {}
};

//======================================================//
//  Scratch Block Pool
//======================================================//

class ScratchBlockPool {
public:
    explicit ScratchBlockPool(const ScratchBlockConfig& config = ScratchBlockConfig());
    ~ScratchBlockPool();
    
    // Disable copy
    ScratchBlockPool(const ScratchBlockPool&) = delete;
    ScratchBlockPool& operator=(const ScratchBlockPool&) = delete;
    
    // Move support
    ScratchBlockPool(ScratchBlockPool&&) noexcept;
    ScratchBlockPool& operator=(ScratchBlockPool&&) noexcept;
    
    //--------------------------------------------------//
    // Configuration
    //--------------------------------------------------//
    
    // Enable/disable at runtime (for multi-model orchestration)
    void setEnabled(bool enabled);
    bool isEnabled() const { return config_.enabled; }
    
    // Check if pool is ready
    bool isInitialized() const { return initialized_; }
    
    // Get current configuration
    const ScratchBlockConfig& getConfig() const { return config_; }
    
    //--------------------------------------------------//
    // Block Acquisition
    //--------------------------------------------------//
    
    // Acquire a block for use (thread-safe)
    // Returns handle with pinned memory (throws if disabled/uninitialized/out of blocks)
    // min_tokens: minimum required capacity
    ScratchBlockHandle acquire(size_t min_tokens);
    
    // Release a block back to the pool (thread-safe)
    void release(const ScratchBlockHandle& handle);
    
    // Check if a specific block is available
    bool isAvailable(uint32_t block_id) const;
    
    //--------------------------------------------------//
    // Statistics
    //--------------------------------------------------//
    
    struct Stats {
        size_t total_acquisitions = 0;
        size_t total_releases = 0;
        size_t pinned_hits = 0;        // Acquisitions using pinned memory
        size_t current_in_use = 0;     // Currently acquired blocks
        size_t peak_in_use = 0;        // Peak concurrent usage
    };
    
    Stats getStats() const;
    void resetStats();
    
    //--------------------------------------------------//
    // Memory Info
    //--------------------------------------------------//
    
    size_t getTotalPinnedMemoryBytes() const;
    size_t getAvailableBlocks() const;
    
private:
    struct Block {
        int* data;
        size_t capacity_tokens;
        bool is_pinned;
        std::atomic<bool> in_use;
        
        Block() : data(nullptr), capacity_tokens(0), is_pinned(false), in_use(false) {}
        
        // Move constructor - std::atomic is not copyable/movable, so we manually handle it
        Block(Block&& other) noexcept 
            : data(other.data)
            , capacity_tokens(other.capacity_tokens)
            , is_pinned(other.is_pinned)
            , in_use(other.in_use.load(std::memory_order_relaxed))
        {
            other.data = nullptr;
            other.capacity_tokens = 0;
            other.is_pinned = false;
            other.in_use.store(false, std::memory_order_relaxed);
        }
        
        // Move assignment
        Block& operator=(Block&& other) noexcept {
            if (this != &other) {
                data = other.data;
                capacity_tokens = other.capacity_tokens;
                is_pinned = other.is_pinned;
                in_use.store(other.in_use.load(std::memory_order_relaxed), std::memory_order_relaxed);
                
                other.data = nullptr;
                other.capacity_tokens = 0;
                other.is_pinned = false;
                other.in_use.store(false, std::memory_order_relaxed);
            }
            return *this;
        }
        
        // Disable copy
        Block(const Block&) = delete;
        Block& operator=(const Block&) = delete;
    };
    
    ScratchBlockConfig config_;
    std::vector<Block> blocks_;
    bool initialized_;
    mutable std::mutex mutex_;
    
    // Statistics
    mutable Stats stats_;
    
    // Initialization
    bool initializeBlocks();
    void cleanupBlocks();
    
    // Find available block
    int findAvailableBlock() const;
};

//======================================================//
//  RAII Scratch Block Guard
//======================================================//

class ScratchBlockGuard {
public:
    ScratchBlockGuard(ScratchBlockPool& pool, size_t min_tokens)
        : pool_(pool)
        , handle_(pool.acquire(min_tokens))
        , released_(false) {}
    
    ~ScratchBlockGuard() {
        if (!released_) {
            pool_.release(handle_);
        }
    }
    
    // Disable copy
    ScratchBlockGuard(const ScratchBlockGuard&) = delete;
    ScratchBlockGuard& operator=(const ScratchBlockGuard&) = delete;
    
    // Move support
    ScratchBlockGuard(ScratchBlockGuard&& other) noexcept
        : pool_(other.pool_)
        , handle_(other.handle_)
        , released_(other.released_) {
        other.released_ = true;
    }
    
    // Access handle
    const ScratchBlockHandle& handle() const { return handle_; }
    int* data() const { return handle_.data; }
    size_t capacity() const { return handle_.capacity_tokens; }
    bool isPinned() const { return handle_.is_pinned; }
    
    // Manual release (if needed before destruction)
    void release() {
        if (!released_) {
            pool_.release(handle_);
            released_ = true;
        }
    }
    
private:
    ScratchBlockPool& pool_;
    ScratchBlockHandle handle_;
    bool released_;
};

} // namespace ScratchBlock
} // namespace GRIM

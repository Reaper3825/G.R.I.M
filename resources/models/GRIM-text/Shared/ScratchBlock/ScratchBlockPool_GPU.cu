//======================================================//
//  ScratchBlockPool_GPU.cu
//  Implementation of Pinned Memory Scratch Blocks
//======================================================//

#include "ScratchBlockPool_GPU.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace GRIM {
namespace ScratchBlock {

//======================================================//
//  ScratchBlockPool Implementation
//======================================================//

ScratchBlockPool::ScratchBlockPool(const ScratchBlockConfig& config)
    : config_(config)
    , initialized_(false)
{
    if (config_.enabled) {
        initialized_ = initializeBlocks();
    }
}

ScratchBlockPool::~ScratchBlockPool() {
    cleanupBlocks();
}

ScratchBlockPool::ScratchBlockPool(ScratchBlockPool&& other) noexcept
    : config_(other.config_)
    , blocks_(std::move(other.blocks_))
    , initialized_(other.initialized_)
    , stats_(other.stats_)
{
    other.initialized_ = false;
}

ScratchBlockPool& ScratchBlockPool::operator=(ScratchBlockPool&& other) noexcept {
    if (this != &other) {
        cleanupBlocks();
        
        config_ = other.config_;
        blocks_ = std::move(other.blocks_);
        initialized_ = other.initialized_;
        stats_ = other.stats_;
        
        other.initialized_ = false;
    }
    return *this;
}

void ScratchBlockPool::setEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (enabled == config_.enabled) {
        return;  // No change
    }
    
    config_.enabled = enabled;
    
    if (enabled && !initialized_) {
        // Re-initialize blocks
        initialized_ = initializeBlocks();
    } else if (!enabled) {
        // Keep blocks allocated but mark as disabled
        // This allows quick re-enable without reallocation
    }
}

ScratchBlockHandle ScratchBlockPool::acquire(size_t min_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    stats_.total_acquisitions++;
    
    // If disabled or not initialized, fail loud
    if (!config_.enabled || !initialized_) {
        std::fprintf(stderr, "ScratchBlockPool FATAL: acquire called while disabled or uninitialized\n");
        throw std::runtime_error("ScratchBlockPool: acquire called while disabled or uninitialized");
    }
    
    // Find available block with sufficient byte capacity
    int block_idx = -1;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        const size_t block_bytes = blocks_[i].capacity_tokens * sizeof(int);
        if (!blocks_[i].in_use.load() && block_bytes >= min_bytes) {
            bool expected = false;
            if (blocks_[i].in_use.compare_exchange_strong(expected, true)) {
                block_idx = static_cast<int>(i);
                break;
            }
        }
    }
    
    // No available block - fail loud
    if (block_idx == -1) {
        std::fprintf(stderr, "ScratchBlockPool FATAL: No available blocks (min_bytes=%zu)\n", min_bytes);
        throw std::runtime_error("ScratchBlockPool: No available blocks");
    }
    
    // Use pinned block
    stats_.pinned_hits++;
    stats_.current_in_use++;
    stats_.peak_in_use = std::max(stats_.peak_in_use, stats_.current_in_use);
    
    Block& block = blocks_[block_idx];
    
    ScratchBlockHandle handle;
    handle.data = block.data;
    handle.capacity_tokens = block.capacity_tokens;
    handle.capacity_bytes = block.capacity_tokens * sizeof(int);
    handle.block_id = static_cast<uint32_t>(block_idx);
    handle.is_pinned = block.is_pinned;
    
    return handle;
}

void ScratchBlockPool::release(const ScratchBlockHandle& handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    stats_.total_releases++;
    
    // Release pinned block
    if (handle.block_id < blocks_.size()) {
        blocks_[handle.block_id].in_use.store(false);
        if (stats_.current_in_use > 0) {
            stats_.current_in_use--;
        }
    }
}

ScratchBlockPool::Stats ScratchBlockPool::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void ScratchBlockPool::resetStats() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_ = Stats{};
}

size_t ScratchBlockPool::getTotalPinnedMemoryBytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t total = 0;
    for (const auto& block : blocks_) {
        if (block.is_pinned && block.data) {
            total += block.capacity_tokens * sizeof(int);
        }
    }
    return total;
}

size_t ScratchBlockPool::getAvailableBlocks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t available = 0;
    for (const auto& block : blocks_) {
        if (!block.in_use.load()) {
            available++;
        }
    }
    return available;
}

bool ScratchBlockPool::initializeBlocks() {
    // Clear any existing blocks
    blocks_.clear();
    
    // Reserve space and construct blocks in-place (avoids copy/move issues with std::atomic)
    blocks_.reserve(config_.num_blocks);
    
    const size_t bytes_per_block = config_.max_tokens_per_block * sizeof(int);
    
    for (size_t i = 0; i < config_.num_blocks; ++i) {
        // Emplace default-constructed block
        blocks_.emplace_back();
        Block& block = blocks_.back();
        
        // Allocate pinned memory
        unsigned int flags = cudaHostAllocDefault;
        if (config_.use_write_combined) {
            flags |= cudaHostAllocWriteCombined;
        }
        
        cudaError_t err = cudaMallocHost(
            reinterpret_cast<void**>(&block.data),
            bytes_per_block,
            flags
        );
        
        if (err != cudaSuccess) {
            // Allocation failed - cleanup already-allocated blocks before throwing
            for (size_t j = 0; j < blocks_.size(); ++j) {
                if (blocks_[j].data) {
                    cudaFreeHost(blocks_[j].data);
                    blocks_[j].data = nullptr;
                }
            }
            blocks_.clear();
            throw std::runtime_error(
                "ScratchBlockPool: cudaMallocHost failed for block " + std::to_string(i)
                + " (" + std::to_string(bytes_per_block) + " bytes): "
                + cudaGetErrorString(err));
        }
        
        block.capacity_tokens = config_.max_tokens_per_block;
        block.is_pinned = true;
        block.in_use.store(false);
    }
    
    return true;
}

void ScratchBlockPool::cleanupBlocks() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& block : blocks_) {
        if (block.data && block.is_pinned) {
            cudaFreeHost(block.data);
            block.data = nullptr;
        }
    }
    
    blocks_.clear();
    initialized_ = false;
}

} // namespace ScratchBlock
} // namespace GRIM

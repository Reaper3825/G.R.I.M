#pragma once

#include "../storage/storage_manager.hpp"
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace GRIM {

// ─────────────────────────────────────────────────────────
//  FileTransferManager — chunked binary transfer protocol
//
//  Upload  (device→hub):
//    1. Device sends TransferRequest JSON
//    2. Hub replies TransferResponse JSON (accepted + transfer_id)
//    3. Device streams binary frames: [4-byte chunk_index BE] [payload ≤ 64KB]
//    4. Hub reassembles → temp file → SHA-256 verify → atomic move
//    5. Hub sends TransferComplete JSON
//
//  Download (hub→device): same protocol, reversed direction.
//  One active transfer per device.
// ─────────────────────────────────────────────────────────

enum class TransferDirection { Upload, Download };

struct ActiveTransfer {
    std::string transfer_id;
    std::string device_id;
    std::string relative_path;
    std::string expected_sha256;
    TransferDirection direction = TransferDirection::Upload;

    uint64_t total_size      = 0;
    uint64_t received_bytes  = 0;
    uint32_t total_chunks    = 0;
    uint32_t received_chunks = 0;
    uint32_t chunk_size      = 65536;

    std::string temp_path; // staging area path

    std::chrono::steady_clock::time_point started_at;
    std::chrono::steady_clock::time_point last_activity;

    float progress() const {
        if (total_size == 0) return 0.0f;
        return static_cast<float>(received_bytes) / static_cast<float>(total_size);
    }
};

using TransferCompleteCallback = std::function<void(const std::string& transfer_id,
                                                     const std::string& file_id,
                                                     bool success,
                                                     const std::string& error)>;

class FileTransferManager {
public:
    explicit FileTransferManager(StorageManager& storage,
                                 uint32_t max_chunk_size = 65536,
                                 uint32_t transfer_timeout_sec = 300);

    // Begin an upload transfer (device→hub).
    // Returns transfer_id on acceptance; throws on rejection.
    std::string beginUpload(const std::string& device_id,
                            const std::string& relative_path,
                            uint64_t total_size,
                            const std::string& sha256,
                            uint32_t chunk_size);

    // Process one binary chunk for transfer_id.
    // Chunk format: [4-byte big-endian chunk_index][payload ≤ chunk_size]
    // Returns true when all chunks have been received and file is finalized.
    bool processChunk(const std::string& transfer_id,
                      const uint8_t* data,
                      size_t length);

    // Begin a download transfer (hub→device).
    // Returns transfer_id; caller must stream chunks via getNextChunk().
    std::string beginDownload(const std::string& device_id,
                              const std::string& file_id);

    // Fill buffer with next chunk. Returns payload size (0 = done).
    size_t getNextChunk(const std::string& transfer_id,
                        uint8_t* buffer,
                        size_t buffer_size);

    // Cancel a transfer.
    void cancelTransfer(const std::string& transfer_id);

    // Garbage-collect transfers that have been idle too long.
    void purgeTimedOut();

    // Query active transfers
    const ActiveTransfer* findTransfer(const std::string& transfer_id) const;
    bool hasActiveTransfer(const std::string& device_id) const;
    std::vector<ActiveTransfer> listActive() const;

    // Completion notification
    void setCompletionCallback(TransferCompleteCallback cb) { on_complete_ = std::move(cb); }

private:
    static std::string generateTransferID();
    void finalizeUpload(ActiveTransfer& transfer);
    void cleanupTempFile(const ActiveTransfer& transfer);

    StorageManager& storage_;
    uint32_t max_chunk_size_;
    uint32_t transfer_timeout_sec_;

    mutable std::mutex mu_;
    std::unordered_map<std::string, ActiveTransfer> transfers_;
    TransferCompleteCallback on_complete_;
};

} // namespace GRIM

#include "file_transfer_manager.hpp"
#include "logger.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>

namespace GRIM {

static constexpr const char* TAG = "FileTransferMgr";

// ─── Helpers ─────────────────────────────────────────────

std::string FileTransferManager::generateTransferID() {
    static std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<uint32_t> dist(0, 15);
    const char hex[] = "0123456789abcdef";

    std::string id = "xfer-";
    id.reserve(21);
    for (int i = 0; i < 16; ++i) id += hex[dist(gen)];
    return id;
}

// Read 4-byte big-endian uint32 from buffer
static uint32_t readU32BE(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24) |
           (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8)  |
           (static_cast<uint32_t>(p[3]));
}

// Write 4-byte big-endian uint32 into buffer
static void writeU32BE(uint8_t* p, uint32_t val) {
    p[0] = static_cast<uint8_t>((val >> 24) & 0xFF);
    p[1] = static_cast<uint8_t>((val >> 16) & 0xFF);
    p[2] = static_cast<uint8_t>((val >>  8) & 0xFF);
    p[3] = static_cast<uint8_t>( val        & 0xFF);
}

// ─── Construction ────────────────────────────────────────

FileTransferManager::FileTransferManager(StorageManager& storage,
                                         uint32_t max_chunk_size,
                                         uint32_t transfer_timeout_sec)
    : storage_(storage)
    , max_chunk_size_(max_chunk_size)
    , transfer_timeout_sec_(transfer_timeout_sec)
{}

// ─── Upload ──────────────────────────────────────────────

std::string FileTransferManager::beginUpload(const std::string& device_id,
                                              const std::string& relative_path,
                                              uint64_t total_size,
                                              const std::string& sha256,
                                              uint32_t chunk_size) {
    std::lock_guard<std::mutex> lock(mu_);

    // One active transfer per device
    for (const auto& [id, t] : transfers_) {
        if (t.device_id == device_id)
            throw std::runtime_error("Device " + device_id + " already has an active transfer: " + id);
    }

    if (chunk_size > max_chunk_size_)
        throw std::runtime_error("chunk_size " + std::to_string(chunk_size) +
                                 " exceeds maximum " + std::to_string(max_chunk_size_));

    ActiveTransfer transfer;
    transfer.transfer_id    = generateTransferID();
    transfer.device_id      = device_id;
    transfer.relative_path  = relative_path;
    transfer.expected_sha256 = sha256;
    transfer.direction      = TransferDirection::Upload;
    transfer.total_size     = total_size;
    transfer.chunk_size     = chunk_size;
    transfer.total_chunks   = static_cast<uint32_t>((total_size + chunk_size - 1) / chunk_size);
    transfer.received_bytes = 0;
    transfer.received_chunks = 0;

    // Stage in temp directory
    std::string temp_dir = storage_.storageRoot() + "/.transfers/";
    std::filesystem::create_directories(temp_dir);
    transfer.temp_path = temp_dir + transfer.transfer_id + ".part";

    auto now = std::chrono::steady_clock::now();
    transfer.started_at    = now;
    transfer.last_activity = now;

    // Pre-allocate temp file
    {
        std::ofstream f(transfer.temp_path, std::ios::binary | std::ios::trunc);
        if (!f)
            throw std::runtime_error("Cannot create temp file: " + transfer.temp_path);
    }

    std::string tid = transfer.transfer_id;
    transfers_.emplace(tid, std::move(transfer));
    LOG_DEBUG(TAG, "Upload started: " + tid + " (" + std::to_string(total_size) + " bytes, " +
              std::to_string(chunk_size) + " chunk size)");
    return tid;
}

bool FileTransferManager::processChunk(const std::string& transfer_id,
                                        const uint8_t* data,
                                        size_t length) {
    std::lock_guard<std::mutex> lock(mu_);

    auto it = transfers_.find(transfer_id);
    if (it == transfers_.end())
        throw std::runtime_error("Unknown transfer_id: " + transfer_id);

    ActiveTransfer& t = it->second;
    if (t.direction != TransferDirection::Upload)
        throw std::runtime_error("processChunk called on non-upload transfer: " + transfer_id);

    // Binary frame format: [4-byte chunk_index BE][payload]
    if (length < 4)
        throw std::runtime_error("Chunk too small (< 4 bytes header): " + transfer_id);

    uint32_t chunk_index = readU32BE(data);
    const uint8_t* payload = data + 4;
    size_t payload_len = length - 4;

    if (payload_len > t.chunk_size)
        throw std::runtime_error("Chunk payload " + std::to_string(payload_len) +
                                 " exceeds chunk_size " + std::to_string(t.chunk_size));

    // Write payload at correct offset in temp file
    uint64_t offset = static_cast<uint64_t>(chunk_index) * t.chunk_size;
    {
        std::ofstream f(t.temp_path, std::ios::binary | std::ios::in | std::ios::out);
        if (!f)
            throw std::runtime_error("Cannot open temp file for chunk write: " + t.temp_path);
        f.seekp(static_cast<std::streamoff>(offset));
        f.write(reinterpret_cast<const char*>(payload), static_cast<std::streamsize>(payload_len));
    }

    t.received_bytes += payload_len;
    t.received_chunks++;
    t.last_activity = std::chrono::steady_clock::now();

    // All chunks received?
    if (t.received_chunks >= t.total_chunks) {
        finalizeUpload(t);
        std::string tid = t.transfer_id;
        transfers_.erase(it);
        return true;
    }

    return false;
}

void FileTransferManager::finalizeUpload(ActiveTransfer& transfer) {
    // Verify SHA-256
    std::string actual_hash = StorageManager::calculateSHA256(
        std::filesystem::path(transfer.temp_path));

    if (actual_hash != transfer.expected_sha256) {
        cleanupTempFile(transfer);
        std::string error = "SHA-256 mismatch: expected=" + transfer.expected_sha256 +
                            " actual=" + actual_hash;
        LOG_ERROR(TAG, "Transfer " + transfer.transfer_id + " failed: " + error);
        if (on_complete_) {
            on_complete_(transfer.transfer_id, "", false, error);
        }
        throw std::runtime_error("Transfer " + transfer.transfer_id + ": " + error);
    }

    // Read temp file into memory for StorageManager (which does atomic write)
    std::ifstream f(transfer.temp_path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot read temp file for finalization: " + transfer.temp_path);

    std::vector<uint8_t> data(transfer.total_size);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(transfer.total_size));

    std::string file_id = storage_.storeFile(transfer.relative_path, data, transfer.device_id);

    // Cleanup temp
    cleanupTempFile(transfer);

    LOG_DEBUG(TAG, "Transfer " + transfer.transfer_id + " completed → file_id=" + file_id);
    if (on_complete_) {
        on_complete_(transfer.transfer_id, file_id, true, "");
    }
}

// ─── Download ────────────────────────────────────────────

std::string FileTransferManager::beginDownload(const std::string& device_id,
                                                const std::string& file_id) {
    std::lock_guard<std::mutex> lock(mu_);

    for (const auto& [id, t] : transfers_) {
        if (t.device_id == device_id)
            throw std::runtime_error("Device " + device_id + " already has an active transfer: " + id);
    }

    std::filesystem::path file_path = storage_.retrieveFile(file_id);
    uint64_t file_size = std::filesystem::file_size(file_path);

    ActiveTransfer transfer;
    transfer.transfer_id   = generateTransferID();
    transfer.device_id     = device_id;
    transfer.relative_path = file_path.string(); // full path for reading
    transfer.direction     = TransferDirection::Download;
    transfer.total_size    = file_size;
    transfer.chunk_size    = max_chunk_size_;
    transfer.total_chunks  = static_cast<uint32_t>((file_size + max_chunk_size_ - 1) / max_chunk_size_);
    transfer.received_chunks = 0; // repurposed: chunks sent
    transfer.received_bytes  = 0;

    auto now = std::chrono::steady_clock::now();
    transfer.started_at    = now;
    transfer.last_activity = now;

    std::string tid = transfer.transfer_id;
    transfers_.emplace(tid, std::move(transfer));
    LOG_DEBUG(TAG, "Download started: " + tid + " (" + std::to_string(file_size) + " bytes)");
    return tid;
}

size_t FileTransferManager::getNextChunk(const std::string& transfer_id,
                                          uint8_t* buffer,
                                          size_t buffer_size) {
    std::lock_guard<std::mutex> lock(mu_);

    auto it = transfers_.find(transfer_id);
    if (it == transfers_.end())
        throw std::runtime_error("Unknown transfer_id: " + transfer_id);

    ActiveTransfer& t = it->second;
    if (t.direction != TransferDirection::Download)
        throw std::runtime_error("getNextChunk called on non-download transfer: " + transfer_id);

    if (t.received_chunks >= t.total_chunks) {
        // Transfer complete — remove
        transfers_.erase(it);
        return 0;
    }

    // Read next chunk from file
    uint64_t offset = static_cast<uint64_t>(t.received_chunks) * t.chunk_size;
    uint64_t remaining = t.total_size - offset;
    size_t payload_len = static_cast<size_t>(std::min(static_cast<uint64_t>(t.chunk_size), remaining));

    if (buffer_size < payload_len + 4)
        throw std::runtime_error("Buffer too small for chunk + 4-byte header");

    // 4-byte chunk index header
    writeU32BE(buffer, t.received_chunks);

    std::ifstream f(t.relative_path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot read file for download: " + t.relative_path);
    f.seekg(static_cast<std::streamoff>(offset));
    f.read(reinterpret_cast<char*>(buffer + 4), static_cast<std::streamsize>(payload_len));

    t.received_chunks++;
    t.received_bytes += payload_len;
    t.last_activity = std::chrono::steady_clock::now();

    return payload_len + 4; // total frame size
}

// ─── Cancel / purge ──────────────────────────────────────

void FileTransferManager::cancelTransfer(const std::string& transfer_id) {
    std::lock_guard<std::mutex> lock(mu_);

    auto it = transfers_.find(transfer_id);
    if (it == transfers_.end()) return;

    cleanupTempFile(it->second);
    LOG_DEBUG(TAG, "Transfer cancelled: " + transfer_id);
    if (on_complete_) {
        on_complete_(transfer_id, "", false, "cancelled");
    }
    transfers_.erase(it);
}

void FileTransferManager::purgeTimedOut() {
    std::lock_guard<std::mutex> lock(mu_);
    auto now = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(transfer_timeout_sec_);

    for (auto it = transfers_.begin(); it != transfers_.end(); ) {
        if (now - it->second.last_activity > timeout) {
            LOG_DEBUG(TAG, "Transfer timed out: " + it->first);
            cleanupTempFile(it->second);
            if (on_complete_) {
                on_complete_(it->first, "", false, "timeout");
            }
            it = transfers_.erase(it);
        } else {
            ++it;
        }
    }
}

// ─── Queries ─────────────────────────────────────────────

const ActiveTransfer* FileTransferManager::findTransfer(const std::string& transfer_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = transfers_.find(transfer_id);
    if (it == transfers_.end()) return nullptr;
    return &it->second;
}

bool FileTransferManager::hasActiveTransfer(const std::string& device_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& [id, t] : transfers_) {
        if (t.device_id == device_id) return true;
    }
    return false;
}

std::vector<ActiveTransfer> FileTransferManager::listActive() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<ActiveTransfer> result;
    result.reserve(transfers_.size());
    for (const auto& [id, t] : transfers_) {
        result.push_back(t);
    }
    return result;
}

// ─── Internal ────────────────────────────────────────────

void FileTransferManager::cleanupTempFile(const ActiveTransfer& transfer) {
    if (!transfer.temp_path.empty()) {
        std::filesystem::remove(transfer.temp_path);
    }
}

} // namespace GRIM

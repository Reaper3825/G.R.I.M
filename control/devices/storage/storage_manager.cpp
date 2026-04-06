#include "storage_manager.hpp"
#include "memory/atomic_writer.hpp"
#include "logger.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>

// SHA-256 — use platform crypto or a minimal header-only impl.
// For now, use a simple hash placeholder that can be swapped for
// OpenSSL/CommonCrypto when available.
#ifdef __APPLE__
#include <CommonCrypto/CommonDigest.h>
#elif defined(_WIN32)
#include <wincrypt.h>
#pragma comment(lib, "advapi32.lib")
#else
// Linux: link against -lcrypto (OpenSSL)
#include <openssl/sha.h>
#endif

namespace GRIM {

static constexpr const char* TAG = "StorageManager";

// ─── Path security ───────────────────────────────────────

void StorageManager::sanitizePath(const std::string& path) {
    if (path.empty())
        throw std::runtime_error("StorageManager: empty path rejected");

    // Null bytes
    if (path.find('\0') != std::string::npos)
        throw std::runtime_error("StorageManager: null bytes in path rejected");

    // Absolute paths
    if (path.front() == '/' || path.front() == '\\')
        throw std::runtime_error("StorageManager: absolute path rejected: " + path);
#ifdef _WIN32
    if (path.size() >= 2 && path[1] == ':')
        throw std::runtime_error("StorageManager: absolute path rejected: " + path);
#endif

    // Path traversal
    if (path.find("..") != std::string::npos)
        throw std::runtime_error("StorageManager: path traversal '..' rejected: " + path);
}

// ─── Helpers ─────────────────────────────────────────────

std::string StorageManager::generateUUID() {
    static std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<uint32_t> dist(0, 15);

    const char hex[] = "0123456789abcdef";
    const char* pattern = "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx";

    std::string uuid;
    uuid.reserve(36);
    for (const char* p = pattern; *p; ++p) {
        if (*p == 'x') {
            uuid += hex[dist(gen)];
        } else if (*p == 'y') {
            uuid += hex[(dist(gen) & 0x3) | 0x8];
        } else {
            uuid += *p;
        }
    }
    return uuid;
}

std::string StorageManager::nowISO8601() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &time);
#else
    gmtime_r(&time, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

std::string StorageManager::inferContentType(const std::string& filename) {
    auto ext_pos = filename.rfind('.');
    if (ext_pos == std::string::npos) return "application/octet-stream";
    std::string ext = filename.substr(ext_pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".txt")  return "text/plain";
    if (ext == ".html") return "text/html";
    if (ext == ".json") return "application/json";
    if (ext == ".xml")  return "application/xml";
    if (ext == ".pdf")  return "application/pdf";
    if (ext == ".png")  return "image/png";
    if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
    if (ext == ".gif")  return "image/gif";
    if (ext == ".mp4")  return "video/mp4";
    if (ext == ".mp3")  return "audio/mpeg";
    if (ext == ".zip")  return "application/zip";
    if (ext == ".csv")  return "text/csv";
    if (ext == ".md")   return "text/markdown";
    return "application/octet-stream";
}

// ─── SHA-256 ─────────────────────────────────────────────

std::string StorageManager::calculateSHA256(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("StorageManager::calculateSHA256: cannot open " + path.string());

    constexpr size_t BUF_SIZE = 65536;
    char buffer[BUF_SIZE];
    unsigned char hash[32];

#ifdef __APPLE__
    CC_SHA256_CTX ctx;
    CC_SHA256_Init(&ctx);
    while (file.read(buffer, BUF_SIZE) || file.gcount()) {
        CC_SHA256_Update(&ctx, buffer, static_cast<CC_LONG>(file.gcount()));
    }
    CC_SHA256_Final(hash, &ctx);
#elif defined(_WIN32)
    HCRYPTPROV hProv = 0;
    HCRYPTHASH hHash = 0;
    CryptAcquireContext(&hProv, nullptr, nullptr, PROV_RSA_AES, CRYPT_VERIFYCONTEXT);
    CryptCreateHash(hProv, CALG_SHA_256, 0, 0, &hHash);
    while (file.read(buffer, BUF_SIZE) || file.gcount()) {
        CryptHashData(hHash, reinterpret_cast<BYTE*>(buffer), static_cast<DWORD>(file.gcount()), 0);
    }
    DWORD hashLen = 32;
    CryptGetHashParam(hHash, HP_HASHVAL, hash, &hashLen, 0);
    CryptDestroyHash(hHash);
    CryptReleaseContext(hProv, 0);
#else
    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    while (file.read(buffer, BUF_SIZE) || file.gcount()) {
        SHA256_Update(&ctx, buffer, file.gcount());
    }
    SHA256_Final(hash, &ctx);
#endif

    // Convert to hex
    char hex_str[65];
    for (int i = 0; i < 32; ++i) {
        std::snprintf(hex_str + i * 2, 3, "%02x", hash[i]);
    }
    hex_str[64] = '\0';
    return hex_str;
}

// ─── Construction ────────────────────────────────────────

StorageManager::StorageManager(const std::string& storage_root)
    : storage_root_(storage_root)
    , index_(storage_root)
{
    std::filesystem::create_directories(storage_root_);
    LOG_DEBUG(TAG, "Storage root: " + storage_root_);
}

// ─── File operations ─────────────────────────────────────

std::string StorageManager::storeFile(const std::string& relative_path,
                                      const std::vector<uint8_t>& data,
                                      const std::string& source_device_id) {
    sanitizePath(relative_path);

    std::filesystem::path full_path = std::filesystem::path(storage_root_) / relative_path;

    // Write file atomically
    AtomicWriter::write(full_path.string(), data.data(), data.size());

    // Compute hash
    std::string hash = calculateSHA256(full_path);

    // Create index entry
    StorageEntry entry;
    entry.file_id          = generateUUID();
    entry.relative_path    = relative_path;
    entry.size_bytes       = data.size();
    entry.sha256_hash      = hash;
    entry.content_type     = inferContentType(relative_path);
    entry.source_device_id = source_device_id;
    entry.created_at       = nowISO8601();
    entry.modified_at      = entry.created_at;

    index_.addEntry(std::move(entry));
    LOG_DEBUG(TAG, "Stored file: " + relative_path + " (" + std::to_string(data.size()) + " bytes)");

    return entry.file_id;
}

std::filesystem::path StorageManager::retrieveFile(const std::string& file_id) const {
    const StorageEntry* entry = index_.findById(file_id);
    if (!entry)
        throw std::runtime_error("StorageManager::retrieveFile: file_id '" + file_id + "' not found");

    std::filesystem::path full_path = std::filesystem::path(storage_root_) / entry->relative_path;
    if (!std::filesystem::exists(full_path))
        throw std::runtime_error("StorageManager::retrieveFile: file missing from disk: " +
                                 full_path.string());

    return full_path;
}

void StorageManager::deleteFile(const std::string& file_id) {
    const StorageEntry* entry = index_.findById(file_id);
    if (!entry)
        throw std::runtime_error("StorageManager::deleteFile: file_id '" + file_id + "' not found");

    std::filesystem::path full_path = std::filesystem::path(storage_root_) / entry->relative_path;
    std::filesystem::remove(full_path);

    index_.removeEntry(file_id);
    LOG_DEBUG(TAG, "Deleted file: " + file_id);
}

void StorageManager::createDirectory(const std::string& relative_path) {
    sanitizePath(relative_path);
    std::filesystem::path full_path = std::filesystem::path(storage_root_) / relative_path;
    std::filesystem::create_directories(full_path);
    LOG_DEBUG(TAG, "Created directory: " + relative_path);
}

uint64_t StorageManager::getDiskUsage() const {
    uint64_t total = 0;
    for (const auto& entry : index_.listAll()) {
        total += entry.size_bytes;
    }
    return total;
}

} // namespace GRIM

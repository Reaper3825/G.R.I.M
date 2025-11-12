//======================================================//
//  GRIM Streaming Downloader
//  Memory-efficient streaming downloads with automatic cleanup
//  
//  Features:
//  - Stream download → FlatBuffer → Disk pipeline
//  - Automatic memory management and cleanup
//  - Configurable memory limits
//  - Zero-copy where possible
//  
//  Version: 1.0.0
//======================================================//

#pragma once

#include <string>
#include <memory>
#include <functional>
#include <atomic>
#include <curl/curl.h>
#include <flatbuffers/flatbuffers.h>

namespace GRIM {
namespace Training {

//======================================================//
//  Memory-Aware Stream Buffer
//======================================================//

struct StreamBuffer {
    static constexpr size_t DEFAULT_CHUNK_SIZE = 64 * 1024;  // 64KB chunks
    static constexpr size_t MAX_BUFFER_SIZE = 10 * 1024 * 1024;  // 10MB max per download
    
    std::string data;
    size_t bytes_written = 0;
    bool overflow = false;
    
    void clear() {
        data.clear();
        bytes_written = 0;
        overflow = false;
    }
    
    bool canAcceptMore() const {
        return !overflow && data.size() < MAX_BUFFER_SIZE;
    }
};

//======================================================//
//  Download Context for CURL callbacks
//======================================================//

struct DownloadContext {
    StreamBuffer* buffer = nullptr;
    std::function<void(const char*, size_t)> chunk_callback;  // Process chunks on-the-fly
    std::atomic<bool>* cancel_flag = nullptr;
    size_t total_downloaded = 0;
    
    // For streaming directly to FlatBuffer
    flatbuffers::FlatBufferBuilder* fb_builder = nullptr;
    std::vector<flatbuffers::Offset<flatbuffers::String>>* fb_chunks = nullptr;
};

//======================================================//
//  Streaming Downloader - Memory-Efficient
//======================================================//

class StreamingDownloader {
public:
    StreamingDownloader() {
        curl_global_init(CURL_GLOBAL_DEFAULT);
    }
    
    ~StreamingDownloader() {
        curl_global_cleanup();
    }
    
    // Stream download with automatic memory management
    struct DownloadResult {
        bool success = false;
        std::string content;  // Only populated if store_in_memory = true
        std::string error;
        size_t bytes_downloaded = 0;
        long http_code = 0;
        double download_time_ms = 0;
    };
    
    // Download with callback for each chunk (streaming mode - memory efficient)
    DownloadResult downloadStreaming(
        const std::string& url,
        std::function<void(const char*, size_t)> chunk_callback,
        int timeout_seconds = 30,
        std::atomic<bool>* cancel_flag = nullptr
    ) {
        DownloadResult result;
        
        CURL* curl = curl_easy_init();
        if (!curl) {
            result.error = "Failed to initialize CURL";
            return result;
        }
        
        StreamBuffer buffer;
        DownloadContext context;
        context.buffer = &buffer;
        context.chunk_callback = chunk_callback;
        context.cancel_flag = cancel_flag;
        
        // Configure CURL for streaming
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, streamingCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &context);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, static_cast<long>(timeout_seconds));
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 5L);
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, 128L * 1024L);  // 128KB internal buffer
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
        
        // Perform download
        auto start = std::chrono::steady_clock::now();
        CURLcode res = curl_easy_perform(curl);
        auto end = std::chrono::steady_clock::now();
        
        result.download_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        result.bytes_downloaded = context.total_downloaded;
        
        if (res == CURLE_OK) {
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &result.http_code);
            result.success = (result.http_code >= 200 && result.http_code < 400);
            
            if (!result.success) {
                result.error = "HTTP error " + std::to_string(result.http_code);
            }
        } else {
            result.error = curl_easy_strerror(res);
        }
        
        curl_easy_cleanup(curl);
        
        // Buffer is automatically cleared when it goes out of scope
        return result;
    }
    
    // Download and store in memory (for small files only)
    bool downloadToMemory(
        const std::string& url,
        std::string& output,
        size_t max_size = 10 * 1024 * 1024  // 10MB default limit
    ) {
        output.clear();
        size_t total_size = 0;
        
        auto chunk_callback = [&](const char* data, size_t size) {
            total_size += size;
            if (total_size > max_size) {
                return;  // Stop accepting data
            }
            output.append(data, size);
        };
        
        DownloadResult result = downloadStreaming(url, chunk_callback);
        return result.success && !output.empty();
    }

private:private:
    // Streaming callback - processes chunks as they arrive
    static size_t streamingCallback(char* data, size_t size, size_t nmemb, void* userp) {
        size_t realsize = size * nmemb;
        DownloadContext* ctx = static_cast<DownloadContext*>(userp);
        
        // Check for cancellation
        if (ctx->cancel_flag && ctx->cancel_flag->load()) {
            return 0;  // Abort download
        }
        
        // Check buffer overflow
        if (!ctx->buffer->canAcceptMore()) {
            ctx->buffer->overflow = true;
            return 0;  // Stop download
        }
        
        // Process chunk immediately if callback provided
        if (ctx->chunk_callback) {
            try {
                ctx->chunk_callback(data, realsize);
            } catch (...) {
                return 0;  // Abort on callback error
            }
        }
        
        ctx->total_downloaded += realsize;
        ctx->buffer->bytes_written += realsize;
        
        return realsize;
    }
};

//======================================================//
//  Memory Statistics Tracker
//======================================================//

class MemoryTracker {
public:
    static MemoryTracker& instance() {
        static MemoryTracker tracker;
        return tracker;
    }
    
    void allocate(size_t bytes) {
        current_usage_.fetch_add(bytes);
        size_t curr = current_usage_.load();
        
        // Update peak atomically
        size_t peak = peak_usage_.load();
        while (curr > peak && !peak_usage_.compare_exchange_weak(peak, curr)) {
            peak = peak_usage_.load();
        }
    }
    
    void deallocate(size_t bytes) {
        current_usage_.fetch_sub(bytes);
    }
    
    size_t getCurrentUsage() const {
        return current_usage_.load();
    }
    
    size_t getPeakUsage() const {
        return peak_usage_.load();
    }
    
    void reset() {
        current_usage_.store(0);
        peak_usage_.store(0);
    }
    
    // Check if we're approaching limits (40GB soft limit)
    bool isNearLimit() const {
        constexpr size_t SOFT_LIMIT = 40ULL * 1024 * 1024 * 1024;  // 40GB
        return current_usage_.load() > SOFT_LIMIT;
    }

private:
    std::atomic<size_t> current_usage_{0};
    std::atomic<size_t> peak_usage_{0};
    
    MemoryTracker() = default;
};

//======================================================//
//  RAII Memory Guard - Auto cleanup
//======================================================//

class MemoryGuard {
public:
    explicit MemoryGuard(size_t bytes) : bytes_(bytes) {
        MemoryTracker::instance().allocate(bytes_);
    }
    
    ~MemoryGuard() {
        MemoryTracker::instance().deallocate(bytes_);
    }
    
    // Non-copyable
    MemoryGuard(const MemoryGuard&) = delete;
    MemoryGuard& operator=(const MemoryGuard&) = delete;
    
    // Moveable
    MemoryGuard(MemoryGuard&& other) noexcept : bytes_(other.bytes_) {
        other.bytes_ = 0;
    }

private:
    size_t bytes_;
};

} // namespace Training
} // namespace GRIM

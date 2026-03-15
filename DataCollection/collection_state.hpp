//======================================================//
//  GRIM Collection State Manager
//  Tracks what data has been collected, from which sources,
//  to prevent duplicate downloads and repeated processing.
//
//  Features:
//  - Persistent state tracking (survives restarts)
//  - URL deduplication (hash-based)
//  - Content deduplication (hash-based)
//  - Source completion tracking
//  - Download queue persistence
//  - HuggingFace dataset tracking
//
//  Author: GRIM Development Team
//  Date: December 2025
//======================================================//

#pragma once

#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <functional>

namespace GRIM {
namespace DataCollection {

namespace fs = std::filesystem;
using json = nlohmann::json;

//======================================================//
//  Content Hash Utilities
//======================================================//

inline uint64_t computeContentHash(const std::string& content) {
    // FNV-1a 64-bit hash for fast deduplication
    uint64_t hash = 14695981039346656037ULL;
    for (char c : content) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

inline uint64_t computeUrlHash(const std::string& url) {
    return computeContentHash(url);
}

//======================================================//
//  Source Collection Record
//======================================================//

struct SourceRecord {
    std::string url;
    std::string name;
    std::string source_type;
    int64_t last_collected_timestamp = 0;
    int64_t last_successful_timestamp = 0;
    size_t total_entries_collected = 0;
    size_t successful_entries = 0;
    size_t failed_attempts = 0;
    bool enabled = true;
    
    // Pagination tracking for APIs
    std::string last_cursor;
    int last_page = 0;
    int last_offset = 0;
    
    json toJson() const {
        return {
            {"url", url},
            {"name", name},
            {"source_type", source_type},
            {"last_collected_timestamp", last_collected_timestamp},
            {"last_successful_timestamp", last_successful_timestamp},
            {"total_entries_collected", total_entries_collected},
            {"successful_entries", successful_entries},
            {"failed_attempts", failed_attempts},
            {"enabled", enabled},
            {"last_cursor", last_cursor},
            {"last_page", last_page},
            {"last_offset", last_offset}
        };
    }
    
    static SourceRecord fromJson(const json& j) {
        SourceRecord rec;
        rec.url = j.value("url", "");
        rec.name = j.value("name", "");
        rec.source_type = j.value("source_type", "");
        rec.last_collected_timestamp = j.value("last_collected_timestamp", 0LL);
        rec.last_successful_timestamp = j.value("last_successful_timestamp", 0LL);
        rec.total_entries_collected = j.value("total_entries_collected", 0ULL);
        rec.successful_entries = j.value("successful_entries", 0ULL);
        rec.failed_attempts = j.value("failed_attempts", 0ULL);
        rec.enabled = j.value("enabled", true);
        rec.last_cursor = j.value("last_cursor", "");
        rec.last_page = j.value("last_page", 0);
        rec.last_offset = j.value("last_offset", 0);
        return rec;
    }
};

//======================================================//
//  HuggingFace Download Record
//======================================================//

struct HFDownloadRecord {
    std::string dataset_id;
    std::string display_name;
    std::string output_dir;
    std::string split;
    std::string config;
    int64_t download_timestamp = 0;
    int64_t completion_timestamp = 0;
    size_t files_downloaded = 0;
    size_t total_bytes = 0;
    std::string status;  // "pending", "downloading", "completed", "failed"
    std::string error_message;
    int retry_count = 0;
    
    json toJson() const {
        return {
            {"dataset_id", dataset_id},
            {"display_name", display_name},
            {"output_dir", output_dir},
            {"split", split},
            {"config", config},
            {"download_timestamp", download_timestamp},
            {"completion_timestamp", completion_timestamp},
            {"files_downloaded", files_downloaded},
            {"total_bytes", total_bytes},
            {"status", status},
            {"error_message", error_message},
            {"retry_count", retry_count}
        };
    }
    
    static HFDownloadRecord fromJson(const json& j) {
        HFDownloadRecord rec;
        rec.dataset_id = j.value("dataset_id", "");
        rec.display_name = j.value("display_name", "");
        rec.output_dir = j.value("output_dir", "");
        rec.split = j.value("split", "train");
        rec.config = j.value("config", "");
        rec.download_timestamp = j.value("download_timestamp", 0LL);
        rec.completion_timestamp = j.value("completion_timestamp", 0LL);
        rec.files_downloaded = j.value("files_downloaded", 0ULL);
        rec.total_bytes = j.value("total_bytes", 0ULL);
        rec.status = j.value("status", "pending");
        rec.error_message = j.value("error_message", "");
        rec.retry_count = j.value("retry_count", 0);
        return rec;
    }
};

//======================================================//
//  UI Configuration State
//======================================================//

struct UIConfigState {
    // Collection settings
    int fetch_limit = 100;
    int vocab_size = 50000;
    float verification_threshold = 0.7f;
    int max_hf_results = 4;
    
    // Source filters
    std::string active_source_filter = "all";
    std::string active_status_filter = "all";
    
    // HuggingFace
    std::string hf_search_query;
    int hf_category_index = 0;
    
    json toJson() const {
        return {
            {"fetch_limit", fetch_limit},
            {"vocab_size", vocab_size},
            {"verification_threshold", verification_threshold},
            {"max_hf_results", max_hf_results},
            {"active_source_filter", active_source_filter},
            {"active_status_filter", active_status_filter},
            {"hf_search_query", hf_search_query},
            {"hf_category_index", hf_category_index}
        };
    }
    
    static UIConfigState fromJson(const json& j) {
        UIConfigState cfg;
        cfg.fetch_limit = j.value("fetch_limit", 100);
        cfg.vocab_size = j.value("vocab_size", 50000);
        cfg.verification_threshold = j.value("verification_threshold", 0.7f);
        cfg.max_hf_results = j.value("max_hf_results", 4);
        cfg.active_source_filter = j.value("active_source_filter", "all");
        cfg.active_status_filter = j.value("active_status_filter", "all");
        cfg.hf_search_query = j.value("hf_search_query", "");
        cfg.hf_category_index = j.value("hf_category_index", 0);
        return cfg;
    }
};

//======================================================//
//  Collection State Manager
//======================================================//

class CollectionStateManager {
public:
    CollectionStateManager();
    explicit CollectionStateManager(const std::string& state_dir);
    ~CollectionStateManager();
    
    // Initialize with state directory (creates if not exists)
    bool initialize(const std::string& state_dir);
    
    // Load state from disk
    bool loadState();
    
    // Save state to disk
    bool saveState();
    
    // ========== URL Deduplication ==========
    
    // Check if URL has been collected before
    bool hasCollectedUrl(const std::string& url) const;
    
    // Mark URL as collected (with optional source type for tracking)
    void markUrlCollected(const std::string& url, const std::string& source_type = "");
    
    // Get all collected URLs for a source type
    std::vector<std::string> getCollectedUrlsForSource(const std::string& source_type) const;
    
    // ========== Content Deduplication ==========
    
    // Check if content has been seen before (by hash)
    bool hasSeenContent(const std::string& content) const;
    bool hasSeenContentHash(uint64_t hash) const;
    
    // Mark content as seen
    void markContentSeen(const std::string& content);
    void markContentSeenHash(uint64_t hash);
    
    // Get duplicate count
    size_t getDuplicateCount() const;
    
    // ========== Merge Tracking (separate from collection) ==========
    
    // Check if content has already been merged into training data
    bool hasMergedContent(const std::string& content) const;
    bool hasMergedContentHash(uint64_t hash) const;
    
    // Mark content as merged into training data
    void markContentMerged(const std::string& content);
    void markContentMergedHash(uint64_t hash);
    
    // ========== Structured Content Tracking ==========
    
    // Check if content has already been structured (Q/A generated by LLM)
    bool hasStructuredContent(const std::string& content) const;
    bool hasStructuredContentHash(uint64_t hash) const;
    
    // Mark content as structured
    void markContentStructured(const std::string& content);
    void markContentStructuredHash(uint64_t hash);
    
    // ========== Source Tracking ==========
    
    // Get source record
    std::optional<SourceRecord> getSourceRecord(const std::string& url) const;
    
    // Update source record
    void updateSourceRecord(const SourceRecord& record);
    
    // Mark source as completed for this session
    void markSourceCompleted(const std::string& url, size_t entries_collected);
    
    // Mark source as failed
    void markSourceFailed(const std::string& url, const std::string& error = "");
    
    // Get all source records
    std::vector<SourceRecord> getAllSourceRecords() const;
    
    // Check if source needs refresh (based on time since last collection)
    bool sourceNeedsRefresh(const std::string& url, int64_t refresh_interval_seconds = 86400) const;
    
    // Clear all state (reset everything)
    void clear();
    
    // ========== HuggingFace Download Tracking ==========
    
    // Check if HF dataset has been downloaded
    bool hasDownloadedHFDataset(const std::string& dataset_id) const;
    
    // Get HF download record
    std::optional<HFDownloadRecord> getHFDownloadRecord(const std::string& dataset_id) const;
    
    // Add/update HF download record
    void updateHFDownloadRecord(const HFDownloadRecord& record);
    
    // Mark HF download as completed
    void markHFDownloadCompleted(const std::string& dataset_id, size_t files, size_t bytes);
    
    // Mark HF download as failed
    void markHFDownloadFailed(const std::string& dataset_id, const std::string& error);
    
    // Get all HF download records
    std::vector<HFDownloadRecord> getAllHFDownloads() const;
    
    // Get pending HF downloads
    std::vector<HFDownloadRecord> getPendingHFDownloads() const;
    
    // Remove HF download from queue
    void removeHFDownload(const std::string& dataset_id);
    
    // Clear completed HF downloads
    void clearCompletedHFDownloads();
    
    // ========== Download Queue ==========
    
    // Add to queue (returns false if already in queue or downloaded)
    bool addToDownloadQueue(const std::string& dataset_id, const std::string& display_name);
    
    // Check if in queue
    bool isInDownloadQueue(const std::string& dataset_id) const;
    
    // Get queue size
    size_t getQueueSize() const;
    
    // ========== UI Config Persistence ==========
    
    // Get UI config
    UIConfigState getUIConfig() const;
    
    // Save UI config
    void saveUIConfig(const UIConfigState& config);
    
    // ========== Statistics ==========
    
    // Get total unique URLs collected
    size_t getTotalUniqueUrls() const;
    
    // Get total unique content hashes
    size_t getTotalUniqueContent() const;
    
    // Get collection statistics summary
    json getStatisticsSummary() const;
    
    // ========== Maintenance ==========
    
    // Clear old entries (older than days)
    void clearOldEntries(int days_old = 30);
    
    // Reset all state
    void resetAll();
    
    // Export state to JSON
    json exportState() const;
    
    // Import state from JSON
    bool importState(const json& state);

private:
    std::string state_dir_;
    mutable std::mutex mutex_;
    
    // State data
    std::unordered_set<uint64_t> collected_url_hashes_;
    std::unordered_set<uint64_t> content_hashes_;
    std::unordered_set<uint64_t> merged_content_hashes_;  // Tracks what was merged into training data
    std::unordered_set<uint64_t> structured_content_hashes_;  // Tracks what was structured (Q/A generated)
    std::unordered_map<std::string, SourceRecord> source_records_;
    std::unordered_map<std::string, HFDownloadRecord> hf_downloads_;
    std::unordered_map<std::string, std::unordered_set<std::string>> urls_by_source_;  // URLs grouped by source type
    UIConfigState ui_config_;
    
    // Statistics
    size_t total_duplicates_found_ = 0;
    int64_t state_last_saved_ = 0;
    
    // File paths
    std::string getUrlHashesPath() const;
    std::string getContentHashesPath() const;
    std::string getMergedHashesPath() const;
    std::string getStructuredHashesPath() const;
    std::string getSourceRecordsPath() const;
    std::string getHFDownloadsPath() const;
    std::string getUIConfigPath() const;
    std::string getStatisticsPath() const;
    
    // Internal helpers
    bool saveUrlHashes() const;
    bool loadUrlHashes();
    bool saveContentHashes() const;
    bool loadContentHashes();
    bool saveMergedHashes() const;
    bool loadMergedHashes();
    bool saveStructuredHashes() const;
    bool loadStructuredHashes();
    bool saveSourceRecords() const;
    bool loadSourceRecords();
    bool saveHFDownloads() const;
    bool loadHFDownloads();
    bool saveUIConfigInternal() const;
    bool loadUIConfig();
    
    int64_t getCurrentTimestamp() const;
};

//======================================================//
//  Implementation
//======================================================//

inline CollectionStateManager::CollectionStateManager() {}

inline CollectionStateManager::CollectionStateManager(const std::string& state_dir) {
    initialize(state_dir);
}

inline CollectionStateManager::~CollectionStateManager() {
    saveState();
}

inline bool CollectionStateManager::initialize(const std::string& state_dir) {
    std::lock_guard<std::mutex> lock(mutex_);
    state_dir_ = state_dir;
    
    try {
        fs::create_directories(state_dir_);
    } catch (const std::exception& e) {
        std::cerr << "[CollectionState] Failed to create state dir: " << e.what() << std::endl;
        return false;
    }
    
    return loadState();
}

inline bool CollectionStateManager::loadState() {
    bool success = true;
    success &= loadUrlHashes();
    success &= loadContentHashes();
    success &= loadMergedHashes();
    success &= loadStructuredHashes();
    success &= loadSourceRecords();
    success &= loadHFDownloads();
    success &= loadUIConfig();
    return success;
}

inline bool CollectionStateManager::saveState() {
    std::lock_guard<std::mutex> lock(mutex_);
    bool success = true;
    success &= saveUrlHashes();
    success &= saveContentHashes();
    success &= saveMergedHashes();
    success &= saveStructuredHashes();
    success &= saveSourceRecords();
    success &= saveHFDownloads();
    success &= saveUIConfigInternal();
    state_last_saved_ = getCurrentTimestamp();
    return success;
}

// ========== URL Deduplication ==========

inline bool CollectionStateManager::hasCollectedUrl(const std::string& url) const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t hash = computeUrlHash(url);
    return collected_url_hashes_.count(hash) > 0;
}

inline void CollectionStateManager::markUrlCollected(const std::string& url, const std::string& source_type) {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t hash = computeUrlHash(url);
    collected_url_hashes_.insert(hash);
    
    // Also track by source type if provided
    if (!source_type.empty()) {
        urls_by_source_[source_type].insert(url);
    }
}

// ========== Content Deduplication ==========

inline bool CollectionStateManager::hasSeenContent(const std::string& content) const {
    return hasSeenContentHash(computeContentHash(content));
}

inline bool CollectionStateManager::hasSeenContentHash(uint64_t hash) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return content_hashes_.count(hash) > 0;
}

inline void CollectionStateManager::markContentSeen(const std::string& content) {
    markContentSeenHash(computeContentHash(content));
}

inline void CollectionStateManager::markContentSeenHash(uint64_t hash) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto result = content_hashes_.insert(hash);
    if (!result.second) {
        total_duplicates_found_++;
    }
}

inline size_t CollectionStateManager::getDuplicateCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_duplicates_found_;
}

// ========== Merge Tracking ==========

inline bool CollectionStateManager::hasMergedContent(const std::string& content) const {
    return hasMergedContentHash(computeContentHash(content));
}

inline bool CollectionStateManager::hasMergedContentHash(uint64_t hash) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return merged_content_hashes_.count(hash) > 0;
}

inline void CollectionStateManager::markContentMerged(const std::string& content) {
    markContentMergedHash(computeContentHash(content));
}

inline void CollectionStateManager::markContentMergedHash(uint64_t hash) {
    std::lock_guard<std::mutex> lock(mutex_);
    merged_content_hashes_.insert(hash);
}

// ========== Structured Content Tracking ==========

inline bool CollectionStateManager::hasStructuredContent(const std::string& content) const {
    return hasStructuredContentHash(computeContentHash(content));
}

inline bool CollectionStateManager::hasStructuredContentHash(uint64_t hash) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return structured_content_hashes_.count(hash) > 0;
}

inline void CollectionStateManager::markContentStructured(const std::string& content) {
    markContentStructuredHash(computeContentHash(content));
}

inline void CollectionStateManager::markContentStructuredHash(uint64_t hash) {
    std::lock_guard<std::mutex> lock(mutex_);
    structured_content_hashes_.insert(hash);
}

// ========== Source Tracking ==========

inline std::optional<SourceRecord> CollectionStateManager::getSourceRecord(const std::string& url) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = source_records_.find(url);
    if (it != source_records_.end()) {
        return it->second;
    }
    return std::nullopt;
}

inline void CollectionStateManager::updateSourceRecord(const SourceRecord& record) {
    std::lock_guard<std::mutex> lock(mutex_);
    source_records_[record.url] = record;
}

inline void CollectionStateManager::markSourceCompleted(const std::string& url, size_t entries_collected) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& record = source_records_[url];
    record.url = url;
    record.last_collected_timestamp = getCurrentTimestamp();
    record.last_successful_timestamp = getCurrentTimestamp();
    record.total_entries_collected += entries_collected;
    record.successful_entries += entries_collected;
}

inline void CollectionStateManager::markSourceFailed(const std::string& url, const std::string& error) {
    (void)error;
    std::lock_guard<std::mutex> lock(mutex_);
    auto& record = source_records_[url];
    record.url = url;
    record.last_collected_timestamp = getCurrentTimestamp();
    record.failed_attempts++;
}

inline std::vector<SourceRecord> CollectionStateManager::getAllSourceRecords() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<SourceRecord> records;
    records.reserve(source_records_.size());
    for (const auto& [url, record] : source_records_) {
        records.push_back(record);
    }
    return records;
}

inline bool CollectionStateManager::sourceNeedsRefresh(const std::string& url, int64_t refresh_interval_seconds) const {
    auto record = getSourceRecord(url);
    if (!record) return true;
    
    int64_t now = getCurrentTimestamp();
    return (now - record->last_successful_timestamp) > refresh_interval_seconds;
}

// ========== HuggingFace Download Tracking ==========

inline bool CollectionStateManager::hasDownloadedHFDataset(const std::string& dataset_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = hf_downloads_.find(dataset_id);
    return it != hf_downloads_.end() && it->second.status == "completed";
}

inline std::optional<HFDownloadRecord> CollectionStateManager::getHFDownloadRecord(const std::string& dataset_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = hf_downloads_.find(dataset_id);
    if (it != hf_downloads_.end()) {
        return it->second;
    }
    return std::nullopt;
}

inline void CollectionStateManager::updateHFDownloadRecord(const HFDownloadRecord& record) {
    std::lock_guard<std::mutex> lock(mutex_);
    hf_downloads_[record.dataset_id] = record;
}

inline void CollectionStateManager::markHFDownloadCompleted(const std::string& dataset_id, size_t files, size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& record = hf_downloads_[dataset_id];
    record.dataset_id = dataset_id;
    record.status = "completed";
    record.completion_timestamp = getCurrentTimestamp();
    record.files_downloaded = files;
    record.total_bytes = bytes;
    record.error_message.clear();
}

inline void CollectionStateManager::markHFDownloadFailed(const std::string& dataset_id, const std::string& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& record = hf_downloads_[dataset_id];
    record.dataset_id = dataset_id;
    record.status = "failed";
    record.error_message = error;
    record.retry_count++;
}

inline std::vector<HFDownloadRecord> CollectionStateManager::getAllHFDownloads() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<HFDownloadRecord> records;
    records.reserve(hf_downloads_.size());
    for (const auto& [id, record] : hf_downloads_) {
        records.push_back(record);
    }
    return records;
}

inline std::vector<HFDownloadRecord> CollectionStateManager::getPendingHFDownloads() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<HFDownloadRecord> records;
    for (const auto& [id, record] : hf_downloads_) {
        if (record.status == "pending") {
            records.push_back(record);
        }
    }
    return records;
}

inline void CollectionStateManager::removeHFDownload(const std::string& dataset_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    hf_downloads_.erase(dataset_id);
}

inline void CollectionStateManager::clearCompletedHFDownloads() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = hf_downloads_.begin(); it != hf_downloads_.end(); ) {
        if (it->second.status == "completed") {
            it = hf_downloads_.erase(it);
        } else {
            ++it;
        }
    }
}

// ========== Download Queue ==========

inline bool CollectionStateManager::addToDownloadQueue(const std::string& dataset_id, const std::string& display_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Check if already downloaded or in queue
    auto it = hf_downloads_.find(dataset_id);
    if (it != hf_downloads_.end()) {
        if (it->second.status == "completed") {
            return false;  // Already downloaded
        }
        if (it->second.status == "pending" || it->second.status == "downloading") {
            return false;  // Already in queue
        }
    }
    
    // Add to queue
    HFDownloadRecord record;
    record.dataset_id = dataset_id;
    record.display_name = display_name.empty() ? dataset_id : display_name;
    record.status = "pending";
    record.download_timestamp = getCurrentTimestamp();
    hf_downloads_[dataset_id] = record;
    
    return true;
}

inline bool CollectionStateManager::isInDownloadQueue(const std::string& dataset_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = hf_downloads_.find(dataset_id);
    if (it != hf_downloads_.end()) {
        return it->second.status == "pending" || it->second.status == "downloading";
    }
    return false;
}

inline size_t CollectionStateManager::getQueueSize() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t count = 0;
    for (const auto& [id, record] : hf_downloads_) {
        if (record.status == "pending") {
            count++;
        }
    }
    return count;
}

// ========== UI Config ==========

inline UIConfigState CollectionStateManager::getUIConfig() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ui_config_;
}

inline void CollectionStateManager::saveUIConfig(const UIConfigState& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    ui_config_ = config;
    saveUIConfigInternal();
}

// ========== Statistics ==========

inline size_t CollectionStateManager::getTotalUniqueUrls() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return collected_url_hashes_.size();
}

inline size_t CollectionStateManager::getTotalUniqueContent() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return content_hashes_.size();
}

inline json CollectionStateManager::getStatisticsSummary() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {
        {"total_unique_urls", collected_url_hashes_.size()},
        {"total_unique_content", content_hashes_.size()},
        {"total_duplicates_found", total_duplicates_found_},
        {"total_sources", source_records_.size()},
        {"total_hf_downloads", hf_downloads_.size()},
        {"state_last_saved", state_last_saved_}
    };
}

// ========== Maintenance ==========

inline void CollectionStateManager::resetAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    collected_url_hashes_.clear();
    content_hashes_.clear();
    merged_content_hashes_.clear();
    structured_content_hashes_.clear();
    source_records_.clear();
    hf_downloads_.clear();
    urls_by_source_.clear();
    ui_config_ = UIConfigState();
    total_duplicates_found_ = 0;
}

inline void CollectionStateManager::clear() {
    resetAll();
    // Also delete state files
    if (!state_dir_.empty()) {
        try {
            fs::remove(getUrlHashesPath());
            fs::remove(getContentHashesPath());
            fs::remove(getMergedHashesPath());
            fs::remove(getStructuredHashesPath());
            fs::remove(getSourceRecordsPath());
            fs::remove(getHFDownloadsPath());
            fs::remove(getUIConfigPath());
            fs::remove(getStatisticsPath());
        } catch (...) {
            // Ignore file deletion errors
        }
    }
}

inline json CollectionStateManager::exportState() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    json state;
    state["version"] = "1.0.0";
    state["exported_at"] = getCurrentTimestamp();
    
    // Export URL hashes as array
    json url_hashes = json::array();
    for (uint64_t hash : collected_url_hashes_) {
        url_hashes.push_back(hash);
    }
    state["url_hashes"] = url_hashes;
    
    // Export content hashes
    json content_hashes = json::array();
    for (uint64_t hash : content_hashes_) {
        content_hashes.push_back(hash);
    }
    state["content_hashes"] = content_hashes;
    
    // Export source records
    json sources = json::array();
    for (const auto& [url, record] : source_records_) {
        sources.push_back(record.toJson());
    }
    state["source_records"] = sources;
    
    // Export HF downloads
    json hf_downloads = json::array();
    for (const auto& [id, record] : hf_downloads_) {
        hf_downloads.push_back(record.toJson());
    }
    state["hf_downloads"] = hf_downloads;
    
    state["ui_config"] = ui_config_.toJson();
    state["statistics"] = {
        {"total_duplicates_found", total_duplicates_found_}
    };
    
    return state;
}

inline bool CollectionStateManager::importState(const json& state) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        // Clear existing state
        collected_url_hashes_.clear();
        content_hashes_.clear();
        source_records_.clear();
        hf_downloads_.clear();
        
        // Import URL hashes
        if (state.contains("url_hashes")) {
            for (const auto& hash : state["url_hashes"]) {
                collected_url_hashes_.insert(hash.get<uint64_t>());
            }
        }
        
        // Import content hashes
        if (state.contains("content_hashes")) {
            for (const auto& hash : state["content_hashes"]) {
                content_hashes_.insert(hash.get<uint64_t>());
            }
        }
        
        // Import source records
        if (state.contains("source_records")) {
            for (const auto& j : state["source_records"]) {
                SourceRecord record = SourceRecord::fromJson(j);
                source_records_[record.url] = record;
            }
        }
        
        // Import HF downloads
        if (state.contains("hf_downloads")) {
            for (const auto& j : state["hf_downloads"]) {
                HFDownloadRecord record = HFDownloadRecord::fromJson(j);
                hf_downloads_[record.dataset_id] = record;
            }
        }
        
        // Import UI config
        if (state.contains("ui_config")) {
            ui_config_ = UIConfigState::fromJson(state["ui_config"]);
        }
        
        // Import statistics
        if (state.contains("statistics")) {
            total_duplicates_found_ = state["statistics"].value("total_duplicates_found", 0ULL);
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[CollectionState] Failed to import state: " << e.what() << std::endl;
        return false;
    }
}

// ========== Private Helpers ==========

inline std::string CollectionStateManager::getUrlHashesPath() const {
    return (fs::path(state_dir_) / "url_hashes.bin").string();
}

inline std::string CollectionStateManager::getContentHashesPath() const {
    return (fs::path(state_dir_) / "content_hashes.bin").string();
}

inline std::string CollectionStateManager::getMergedHashesPath() const {
    return (fs::path(state_dir_) / "merged_hashes.bin").string();
}

inline std::string CollectionStateManager::getStructuredHashesPath() const {
    return (fs::path(state_dir_) / "structured_hashes.bin").string();
}

inline std::string CollectionStateManager::getSourceRecordsPath() const {
    return (fs::path(state_dir_) / "source_records.json").string();
}

inline std::string CollectionStateManager::getHFDownloadsPath() const {
    return (fs::path(state_dir_) / "hf_downloads.json").string();
}

inline std::string CollectionStateManager::getUIConfigPath() const {
    return (fs::path(state_dir_) / "ui_config.json").string();
}

inline std::string CollectionStateManager::getStatisticsPath() const {
    return (fs::path(state_dir_) / "statistics.json").string();
}

inline bool CollectionStateManager::saveUrlHashes() const {
    try {
        std::ofstream file(getUrlHashesPath(), std::ios::binary);
        if (!file) return false;
        
        size_t count = collected_url_hashes_.size();
        file.write(reinterpret_cast<const char*>(&count), sizeof(count));
        
        for (uint64_t hash : collected_url_hashes_) {
            file.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
        }
        
        return true;
    } catch (...) {
        return false;
    }
}

inline bool CollectionStateManager::loadUrlHashes() {
    try {
        std::ifstream file(getUrlHashesPath(), std::ios::binary);
        if (!file) return true;  // OK if not exists
        
        size_t count;
        file.read(reinterpret_cast<char*>(&count), sizeof(count));
        
        for (size_t i = 0; i < count; ++i) {
            uint64_t hash;
            file.read(reinterpret_cast<char*>(&hash), sizeof(hash));
            collected_url_hashes_.insert(hash);
        }
        
        return true;
    } catch (...) {
        return false;
    }
}

inline bool CollectionStateManager::saveContentHashes() const {
    try {
        std::ofstream file(getContentHashesPath(), std::ios::binary);
        if (!file) return false;
        
        size_t count = content_hashes_.size();
        file.write(reinterpret_cast<const char*>(&count), sizeof(count));
        
        for (uint64_t hash : content_hashes_) {
            file.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
        }
        
        return true;
    } catch (...) {
        return false;
    }
}

inline bool CollectionStateManager::loadContentHashes() {
    try {
        std::ifstream file(getContentHashesPath(), std::ios::binary);
        if (!file) return true;  // OK if not exists
        
        size_t count;
        file.read(reinterpret_cast<char*>(&count), sizeof(count));
        
        for (size_t i = 0; i < count; ++i) {
            uint64_t hash;
            file.read(reinterpret_cast<char*>(&hash), sizeof(hash));
            content_hashes_.insert(hash);
        }
        
        return true;
    } catch (...) {
        return false;
    }
}

inline bool CollectionStateManager::saveMergedHashes() const {
    try {
        std::ofstream file(getMergedHashesPath(), std::ios::binary);
        if (!file) return false;
        
        size_t count = merged_content_hashes_.size();
        file.write(reinterpret_cast<const char*>(&count), sizeof(count));
        
        for (uint64_t hash : merged_content_hashes_) {
            file.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
        }
        
        return true;
    } catch (...) {
        return false;
    }
}

inline bool CollectionStateManager::loadMergedHashes() {
    try {
        std::ifstream file(getMergedHashesPath(), std::ios::binary);
        if (!file) return true;  // OK if not exists
        
        size_t count;
        file.read(reinterpret_cast<char*>(&count), sizeof(count));
        
        for (size_t i = 0; i < count; ++i) {
            uint64_t hash;
            file.read(reinterpret_cast<char*>(&hash), sizeof(hash));
            merged_content_hashes_.insert(hash);
        }
        
        return true;
    } catch (...) {
        return false;
    }
}

inline bool CollectionStateManager::saveStructuredHashes() const {
    try {
        std::ofstream file(getStructuredHashesPath(), std::ios::binary);
        if (!file) return false;
        
        size_t count = structured_content_hashes_.size();
        file.write(reinterpret_cast<const char*>(&count), sizeof(count));
        
        for (uint64_t hash : structured_content_hashes_) {
            file.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
        }
        
        return true;
    } catch (...) {
        return false;
    }
}

inline bool CollectionStateManager::loadStructuredHashes() {
    try {
        std::ifstream file(getStructuredHashesPath(), std::ios::binary);
        if (!file) return true;  // OK if not exists
        
        size_t count;
        file.read(reinterpret_cast<char*>(&count), sizeof(count));
        
        for (size_t i = 0; i < count; ++i) {
            uint64_t hash;
            file.read(reinterpret_cast<char*>(&hash), sizeof(hash));
            structured_content_hashes_.insert(hash);
        }
        
        return true;
    } catch (...) {
        return false;
    }
}

inline bool CollectionStateManager::saveSourceRecords() const {
    try {
        json j = json::array();
        for (const auto& [url, record] : source_records_) {
            j.push_back(record.toJson());
        }
        
        std::ofstream file(getSourceRecordsPath());
        if (!file) return false;
        
        file << j.dump(2);
        return true;
    } catch (...) {
        return false;
    }
}

inline bool CollectionStateManager::loadSourceRecords() {
    try {
        std::ifstream file(getSourceRecordsPath());
        if (!file) return true;  // OK if not exists
        
        json j;
        file >> j;
        
        for (const auto& item : j) {
            SourceRecord record = SourceRecord::fromJson(item);
            source_records_[record.url] = record;
        }
        
        return true;
    } catch (...) {
        return false;
    }
}

inline bool CollectionStateManager::saveHFDownloads() const {
    try {
        json j = json::array();
        for (const auto& [id, record] : hf_downloads_) {
            j.push_back(record.toJson());
        }
        
        std::ofstream file(getHFDownloadsPath());
        if (!file) return false;
        
        file << j.dump(2);
        return true;
    } catch (...) {
        return false;
    }
}

inline bool CollectionStateManager::loadHFDownloads() {
    try {
        std::ifstream file(getHFDownloadsPath());
        if (!file) return true;  // OK if not exists
        
        json j;
        file >> j;
        
        for (const auto& item : j) {
            HFDownloadRecord record = HFDownloadRecord::fromJson(item);
            hf_downloads_[record.dataset_id] = record;
        }
        
        return true;
    } catch (...) {
        return false;
    }
}

inline bool CollectionStateManager::saveUIConfigInternal() const {
    try {
        std::ofstream file(getUIConfigPath());
        if (!file) return false;
        
        file << ui_config_.toJson().dump(2);
        return true;
    } catch (...) {
        return false;
    }
}

inline bool CollectionStateManager::loadUIConfig() {
    try {
        std::ifstream file(getUIConfigPath());
        if (!file) return true;  // OK if not exists
        
        json j;
        file >> j;
        
        ui_config_ = UIConfigState::fromJson(j);
        return true;
    } catch (...) {
        return false;
    }
}

inline int64_t CollectionStateManager::getCurrentTimestamp() const {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// Global instance accessor
inline CollectionStateManager& getCollectionStateManager() {
    static CollectionStateManager instance;
    return instance;
}

} // namespace DataCollection
} // namespace GRIM

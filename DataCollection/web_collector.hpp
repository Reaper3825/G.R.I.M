//======================================================//
//  GRIM Web Data Collector - Enhanced Version
//  Fetches data from URLs specified in source_data.json
//  
//  Features:
//  - Multiple source types (GitHub, ArXiv, Wikipedia, etc.)
//  - JSON configuration loading
//  - Rate limiting and retry logic
//  - Concurrent requests with thread pool
//  - Content filtering and validation
//  - Progress tracking
//  - Error handling and logging
//  - URL/Content deduplication via CollectionStateManager
//  
//  Author: GRIM Development Team
//  Date: November 4, 2025
//  Version: 2.1.0 - Added persistent state tracking
//======================================================//

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <chrono>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <atomic>
#include <thread>
#include <fstream>
#include <sstream>
#include <iostream>
#include <curl/curl.h>

// For JSON parsing
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// FlatBuffers for efficient serialization
#include "web_training_data_generated.h"
#include "checkpoint_data_generated.h"
#include <flatbuffers/flatbuffers.h>

// High-performance data collection components
#include "streaming_downloader.hpp"
#include "async_crawler.hpp"
#include "html_extractor.hpp"

// AI Config for paths
#include "../resources/models/GRIM-text/Shared/HyperParameters/HyperParameters_GPU.hpp"

// Persistent state tracking for deduplication
#include "collection_state.hpp"

namespace GRIM {
namespace Training {

// Import types from streaming/crawler
using CrawlConfig = AsyncCrawler::CrawlConfig;

//======================================================//
//  Fetcher Type Enum (Internal - for API routing)
//  This determines HOW to fetch, not WHAT category the data is
//======================================================//

enum class FetcherType {
    HTML_CRAWL = 0,    // Default: Generic HTML crawling
    GITHUB_API,        // GitHub REST API
    ARXIV_API,         // ArXiv API
    WIKIPEDIA_API,     // Wikipedia API
    STACKOVERFLOW_API, // StackExchange API
    REDDIT_API,        // Reddit JSON API
    NEWS_API,          // News API
    TECH_DOCS          // Technical documentation sites
};

// Legacy SourceType alias for backward compatibility
using SourceType = FetcherType;

// Convert explicit fetcher string to FetcherType
inline FetcherType fetcherTypeFromString(const std::string& str) {
    if (str == "github" || str == "github_api") return FetcherType::GITHUB_API;
    if (str == "arxiv" || str == "arxiv_api") return FetcherType::ARXIV_API;
    if (str == "wikipedia" || str == "wikipedia_api") return FetcherType::WIKIPEDIA_API;
    if (str == "stackoverflow" || str == "stack_exchange") return FetcherType::STACKOVERFLOW_API;
    if (str == "reddit" || str == "reddit_api") return FetcherType::REDDIT_API;
    if (str == "news_api") return FetcherType::NEWS_API;
    if (str == "tech_docs") return FetcherType::TECH_DOCS;
    // Default to HTML crawl for any other value
    return FetcherType::HTML_CRAWL;
}

// Auto-detect fetcher type from URL patterns
inline FetcherType detectFetcherFromUrl(const std::string& url) {
    if (url.find("api.github.com") != std::string::npos || 
        url.find("github.com") != std::string::npos) {
        return FetcherType::GITHUB_API;
    }
    if (url.find("arxiv.org") != std::string::npos) {
        return FetcherType::ARXIV_API;
    }
    if (url.find("wikipedia.org") != std::string::npos) {
        return FetcherType::WIKIPEDIA_API;
    }
    if (url.find("stackexchange.com") != std::string::npos ||
        url.find("stackoverflow.com") != std::string::npos) {
        return FetcherType::STACKOVERFLOW_API;
    }
    if (url.find("reddit.com") != std::string::npos) {
        return FetcherType::REDDIT_API;
    }
    if (url.find("newsapi.org") != std::string::npos) {
        return FetcherType::NEWS_API;
    }
    // Default to HTML crawl
    return FetcherType::HTML_CRAWL;
}

inline std::string fetcherTypeToString(FetcherType type) {
    switch (type) {
        case FetcherType::GITHUB_API: return "github_api";
        case FetcherType::ARXIV_API: return "arxiv_api";
        case FetcherType::WIKIPEDIA_API: return "wikipedia_api";
        case FetcherType::STACKOVERFLOW_API: return "stackoverflow_api";
        case FetcherType::REDDIT_API: return "reddit_api";
        case FetcherType::NEWS_API: return "news_api";
        case FetcherType::TECH_DOCS: return "tech_docs";
        case FetcherType::HTML_CRAWL:
        default: return "html_crawl";
    }
}

// Legacy alias for backward compatibility
inline SourceType sourceTypeFromString(const std::string& str) { return fetcherTypeFromString(str); }
inline std::string sourceTypeToString(SourceType type) { return fetcherTypeToString(type); }

//======================================================//
//  Content Filter Configuration
//======================================================//

struct ContentFilter {
    int min_length = 100;
    int max_length = 50000;
    int min_words = 20;
    
    std::vector<std::string> required_keywords;
    std::vector<std::string> excluded_keywords;
    std::vector<std::string> file_types;
    std::vector<std::string> path_patterns;
    std::vector<std::string> exclude_patterns;
    
    // Type-specific filters
    int min_stars = 0;           // GitHub
    int min_score = 0;           // StackOverflow, Reddit
    int min_year = 2000;         // ArXiv
    std::vector<std::string> categories;
    std::vector<std::string> tags;
    
    bool passes(const std::string& content) const {
        if (content.length() < static_cast<size_t>(min_length)) return false;
        if (content.length() > static_cast<size_t>(max_length)) return false;
        
        // Count words
        int word_count = 0;
        bool in_word = false;
        for (char c : content) {
            if (std::isspace(c)) {
                in_word = false;
            } else if (!in_word) {
                word_count++;
                in_word = true;
            }
        }
        if (word_count < min_words) return false;
        
        // Check required keywords
        for (const auto& keyword : required_keywords) {
            if (content.find(keyword) == std::string::npos) {
                return false;
            }
        }
        
        // Check excluded keywords
        for (const auto& keyword : excluded_keywords) {
            if (content.find(keyword) != std::string::npos) {
                return false;
            }
        }
        
        return true;
    }
};

//======================================================//
//  Data Source Configuration
//======================================================//

struct DataSource {
    std::string name;
    std::string url;
    
    // User-defined category (from JSON) - defaults to "miscellaneous"
    std::string source_type_str = "miscellaneous";
    
    // Internal fetcher type - auto-detected from URL or explicit "fetcher" field
    FetcherType fetcher_type = FetcherType::HTML_CRAWL;
    
    // Legacy alias for backward compatibility
    SourceType source_type = SourceType::HTML_CRAWL;
    
    bool enabled = true;
    int priority = 5;  // 1-10, higher = more trusted
    
    bool requires_auth = false;
    std::string api_key_env;  // Environment variable name
    std::string api_key;      // Actual key (loaded from env)
    
    int fetch_limit = -1;  // -1 = not set, use dynamic calculation
    int crawl_depth = 2;  // Default to depth 2 for article crawling
    
    ContentFilter filter;
    
    // Auto-detect fetcher from URL if not explicitly set
    void autoDetectFetcher() {
        fetcher_type = detectFetcherFromUrl(url);
        source_type = fetcher_type;  // Keep legacy alias in sync
    }
    
    // Load API key from environment
    void loadApiKey() {
        if (!api_key_env.empty()) {
            const char* env_val = std::getenv(api_key_env.c_str());
            if (env_val) {
                api_key = env_val;
            }
        }
    }
    
    // Calculate dynamic fetch_limit based on crawl depth
    // Only applies if user hasn't explicitly set a fetch_limit
    void applyDynamicFetchLimit() {
        if (fetch_limit == -1) {  // Only adjust if not explicitly set
            switch (crawl_depth) {
                case 1:
                    fetch_limit = 100;   // Single page
                    break;
                case 2:
                    fetch_limit = 500;   // Can discover 100-500 pages at depth 2
                    break;
                case 3:
                    fetch_limit = 2000;  // Can discover 1000+ pages at depth 3
                    break;
                default:
                    fetch_limit = crawl_depth * 1000;  // Scale exponentially for depth > 3
                    break;
            }
        }
    }
};

//======================================================//
//  Collected Raw Data Entry
//======================================================//

struct RawDataEntry {
    std::string content;
    std::string source_url;
    std::string source_name;
    
    // User-defined category string (from JSON source config)
    std::string source_type_str = "miscellaneous";
    
    // Legacy enum for backward compatibility
    SourceType source_type = SourceType::HTML_CRAWL;
    
    int source_priority = 5;
    
    std::string author;
    std::string title;
    uint64_t publish_date = 0;
    uint64_t fetch_date = 0;
    
    std::string metadata_json;  // Additional metadata as JSON
    
    // Q/A fields — populated when entry is a question/answer pair
    std::string question;   // Raw question text (for FlatBuffer input_text)
    std::string answer;     // Raw answer text (for FlatBuffer target_text)
    
    // Semantic tags (POS, NER, content type, etc.)
    std::vector<std::string> tags;
    
    // Generate unique ID
    std::string generateId() const {
        // Simple hash-based ID
        std::hash<std::string> hasher;
        size_t hash = hasher(source_url + content.substr(0, std::min<size_t>(100, content.length())));
        return std::to_string(hash);
    }
};

//======================================================//
//  Collection Statistics
//======================================================//

struct CollectionStats {
    uint32_t total_sources = 0;
    uint32_t enabled_sources = 0;
    uint32_t total_fetched = 0;
    uint32_t successful = 0;
    uint32_t failed = 0;
    uint32_t filtered_out = 0;
    uint32_t duplicates_skipped = 0;  // URLs already collected
    uint32_t content_duplicates = 0;  // Same content from different URLs
    
    std::chrono::milliseconds total_time{0};
    std::chrono::milliseconds avg_request_time{0};
    
    std::unordered_map<SourceType, uint32_t> per_source_type;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    
    void reset() {
        total_sources = 0;
        enabled_sources = 0;
        total_fetched = 0;
        successful = 0;
        failed = 0;
        filtered_out = 0;
        duplicates_skipped = 0;
        content_duplicates = 0;
        total_time = std::chrono::milliseconds(0);
        avg_request_time = std::chrono::milliseconds(0);
        per_source_type.clear();
        errors.clear();
        warnings.clear();
    }
    
    std::string toString() const {
        std::ostringstream oss;
        oss << "Collection Statistics:\n";
        oss << "  Sources: " << enabled_sources << "/" << total_sources << " enabled\n";
        oss << "  Fetched: " << total_fetched << " (" << successful << " success, " 
            << failed << " failed, " << filtered_out << " filtered)\n";
        oss << "  Duplicates: " << duplicates_skipped << " URLs skipped, " 
            << content_duplicates << " content duplicates\n";
        oss << "  Time: " << total_time.count() << " ms total, " 
            << avg_request_time.count() << " ms avg/request\n";
        oss << "  By Source Type:\n";
        for (const auto& [type, count] : per_source_type) {
            oss << "    " << sourceTypeToString(type) << ": " << count << "\n";
        }
        if (!errors.empty()) {
            oss << "  Errors: " << errors.size() << "\n";
        }
        if (!warnings.empty()) {
            oss << "  Warnings: " << warnings.size() << "\n";
        }
        return oss.str();
    }
};

//======================================================//
//  Web Data Collector Configuration
//======================================================//

struct CollectorConfig {
    std::vector<DataSource> sources;
    
    std::string output_dir;
    int max_entries_per_source = 100;
    int max_new_entries_per_run = 5000;  // Global limit: stop after collecting this many NEW entries total
    int timeout_seconds = 15;  // Faster timeout (was 30)
    int rate_limit_delay_ms = 100;  // 10x faster (was 1000)
    int max_retries = 3;
    int max_concurrent_requests = 50;  // 10x more parallel (was 5) - OPTIMIZED FOR 128GB RAM
    
    std::string user_agent = "GRIM-DataCollector/2.0";
    
    bool save_as_jsonl = true;
    bool save_metadata = true;
    bool verbose = true;
    
    std::string log_file;  // Loaded from ai_config.json by constructors
};

//======================================================//
//  Enhanced Web Data Collector
//======================================================//

class WebDataCollector {
public:
    WebDataCollector();
    explicit WebDataCollector(const CollectorConfig& config);
    ~WebDataCollector();
    
    // Load configuration from JSON file
    bool loadConfigFromJson(const std::string& json_path);
    
    // Main collection entry point
    size_t collectData();
    
    // Add/remove sources programmatically
    void addSource(const DataSource& source);
    void removeSource(const std::string& source_name);
    void enableSource(const std::string& source_name, bool enable);
    
    // Configuration
    void setOutputDir(const std::string& dir) { config_.output_dir = dir; }
    void setVerbose(bool verbose) { config_.verbose = verbose; }
    void setMaxConcurrent(int max_concurrent) { config_.max_concurrent_requests = max_concurrent; }
    
    // Statistics
    const CollectionStats& getStats() const { return stats_; }
    void resetStats() { stats_.reset(); }
    
    // Retrieve collected data
    const std::vector<RawDataEntry>& getCollectedData() const { return collected_data_; }
    
    // Save collected data
    bool saveToJsonl(const std::string& output_path);
    bool saveToFlatBuffer(const std::string& output_path);  // Requires FlatBuffer serialization
    bool saveCheckpoint(const std::string& checkpoint_path);  // Save current progress
    
    // Load checkpoint data and merge with existing collected_data_
    bool loadCheckpoint(const std::string& checkpoint_path);
    
    // Merge multiple checkpoint files into collected_data_
    bool mergeCheckpoints(const std::vector<std::string>& checkpoint_paths);
    
    // State management - persistent tracking of collected URLs and content
    void initializeStateManager(const std::string& stateDir);
    bool hasCollectedUrl(const std::string& url) const;
    bool hasSeenContent(const std::string& content) const;
    void markUrlCollected(const std::string& url, const std::string& sourceType);
    void markContentSeen(const std::string& content);
    size_t getUniqueUrlCount() const;
    size_t getUniqueContentCount() const;
    void saveCollectionState();
    void clearCollectionState();
    
private:
    CollectorConfig config_;
    CollectionStats stats_;
    std::vector<RawDataEntry> collected_data_;
    
    // Persistent state manager for deduplication
    mutable std::unique_ptr<DataCollection::CollectionStateManager> stateManager_;
    
    // Function map for fetcher types - extensible design
    std::unordered_map<FetcherType, std::function<std::vector<RawDataEntry>(const DataSource&)>> source_fetchers_;
    
    std::mutex data_mutex_;
    std::mutex log_mutex_;
    std::ofstream log_file_;
    
    // libcurl handle
    CURL* curl_handle_;
    
    // High-performance components
    std::unique_ptr<StreamingDownloader> downloader_;
    std::unique_ptr<AsyncCrawler> crawler_;
    
    // Static CURL write callback
    static size_t curlWriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
        try {
            size_t realsize = size * nmemb;
            std::string* str = (std::string*)userp;
            
            // Limit total size to 20MB to prevent memory issues
            const size_t MAX_SIZE = 20 * 1024 * 1024;
            if (str->size() + realsize > MAX_SIZE) {
                // Return 0 to stop the transfer
                return 0;
            }
            
            str->append((char*)contents, realsize);
            return realsize;
        } catch (...) {
            // Return 0 to stop the transfer on any error
            return 0;
        }
    }
    
    // Fetching methods for different source types
    std::vector<RawDataEntry> fetchFromSource(const DataSource& source);
    std::vector<RawDataEntry> fetchGitHub(const DataSource& source);
    std::vector<RawDataEntry> fetchArXiv(const DataSource& source);
    std::vector<RawDataEntry> fetchWikipedia(const DataSource& source);
    std::vector<RawDataEntry> fetchStackOverflow(const DataSource& source);
    std::vector<RawDataEntry> fetchReddit(const DataSource& source);
    std::vector<RawDataEntry> fetchNewsAPI(const DataSource& source);
    std::vector<RawDataEntry> fetchTechDocs(const DataSource& source);
    std::vector<RawDataEntry> fetchCustom(const DataSource& source);
    
    // HTTP helpers
    std::string httpGet(const std::string& url, 
                       const std::unordered_map<std::string, std::string>& headers = {});
    bool downloadFile(const std::string& url, const std::string& output_path);
    
    // Utilities
    void log(const std::string& message, const std::string& level = "INFO");
    void logError(const std::string& error);
    void logWarning(const std::string& warning);
    
    // Source fetcher registration - allows adding new fetcher types dynamically
    void registerSourceFetcher(FetcherType type, std::function<std::vector<RawDataEntry>(const DataSource&)> fetcher);
    void initializeDefaultFetchers();
    
    std::string getCurrentTimestamp() const;
    uint64_t getCurrentUnixTime() const;
    
    // Rate limiting
    void applyRateLimit(const std::string& domain);
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> last_request_time_;
    
    // Progress tracking
    std::atomic<size_t> progress_current_{0};
    std::atomic<size_t> progress_total_{0};
    float updateProgress(size_t current, size_t total);
    
    // Progress callback for external reporting
    std::function<void(float)> progress_callback_;
public:
    void setProgressCallback(std::function<void(float)> callback) {
        progress_callback_ = callback;
    }
};

//======================================================//
//  Implementation: WebDataCollector
//======================================================//

WebDataCollector::WebDataCollector()
    : config_(), stats_(), curl_handle_(nullptr)
{
    auto snapshot = GRIM::Config::loadAiConfigSnapshot();
    if (snapshot && snapshot->has_grim_paths) {
        if (!snapshot->grim_text_checkpoints.empty()) {
            config_.output_dir = snapshot->grim_text_checkpoints;
        }
        if (!snapshot->grim_text_collector_log.empty()) {
            config_.log_file = snapshot->grim_text_collector_log;
        }
    }

    if (config_.output_dir.empty()) {
        config_.output_dir = GRIM::Config::getCheckpointDir();
    }

    if (config_.log_file.empty()) {
        config_.log_file = GRIM::Config::getCollectorLogPath();
    }

    if (snapshot && snapshot->has_data_collection) {
        config_.max_new_entries_per_run = snapshot->data_collection_max_new_entries_per_run;
    }
    
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl_handle_ = curl_easy_init();
    if (!curl_handle_) {
        logError("Failed to initialize CURL");
    }
    
    // Initialize high-performance components
    downloader_ = std::make_unique<StreamingDownloader>();
    crawler_ = std::make_unique<AsyncCrawler>();
    
    // Initialize state manager for deduplication
    std::string stateDir = config_.output_dir + "/collection_state";
    stateManager_ = std::make_unique<DataCollection::CollectionStateManager>(stateDir);
    
    // Initialize source fetcher map
    initializeDefaultFetchers();
    
    // Open log file if configured
    if (!config_.log_file.empty()) {
        log_file_.open(config_.log_file, std::ios::app);
        if (log_file_.is_open()) {
            log("WebDataCollector initialized (default constructor)");
            log("State manager loaded with " + std::to_string(stateManager_->getTotalUniqueUrls()) + " unique URLs");
            log("Global limit: " + std::to_string(config_.max_new_entries_per_run) + " max new entries per run");
        }
    }
}

WebDataCollector::WebDataCollector(const CollectorConfig& config)
    : config_(config), stats_(), curl_handle_(nullptr)
{
    auto snapshot = GRIM::Config::loadAiConfigSnapshot();
    if (snapshot && snapshot->has_grim_paths) {
        if (config_.output_dir.empty() && !snapshot->grim_text_checkpoints.empty()) {
            config_.output_dir = snapshot->grim_text_checkpoints;
        }
        if (config_.log_file.empty() && !snapshot->grim_text_collector_log.empty()) {
            config_.log_file = snapshot->grim_text_collector_log;
        }
    }

    // If output_dir not set in config, use ai_config default/fallback.
    if (config_.output_dir.empty()) {
        config_.output_dir = GRIM::Config::getCheckpointDir();
    }
    
    // If log_file not set in config, use ai_config default/fallback.
    if (config_.log_file.empty()) {
        config_.log_file = GRIM::Config::getCollectorLogPath();
    }
    
    // If max_new_entries_per_run not explicitly set, load from the snapshot-owned
    // data_collection view rather than reparsing ai_config.json through a second loader.
    if (config_.max_new_entries_per_run == 5000 && snapshot && snapshot->has_data_collection) {  // Check if still default
        config_.max_new_entries_per_run = snapshot->data_collection_max_new_entries_per_run;
    }
    
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl_handle_ = curl_easy_init();
    if (!curl_handle_) {
        logError("Failed to initialize CURL");
    }
    
    // Initialize high-performance components
    downloader_ = std::make_unique<StreamingDownloader>();
    crawler_ = std::make_unique<AsyncCrawler>();
    
    // Initialize state manager for deduplication
    std::string stateDir = config_.output_dir + "/collection_state";
    stateManager_ = std::make_unique<DataCollection::CollectionStateManager>(stateDir);
    
    // Initialize source fetcher map
    initializeDefaultFetchers();
    
    if (!config_.log_file.empty()) {
        log_file_.open(config_.log_file, std::ios::app);
        if (log_file_.is_open()) {
            log("WebDataCollector initialized");
            log("State manager loaded with " + std::to_string(stateManager_->getTotalUniqueUrls()) + " unique URLs");
            log("Global limit: " + std::to_string(config_.max_new_entries_per_run) + " max new entries per run");
        }
    }
}

WebDataCollector::~WebDataCollector() {
    // Save state before shutting down
    if (stateManager_) {
        stateManager_->saveState();
    }
    
    if (curl_handle_) {
        curl_easy_cleanup(curl_handle_);
    }
    curl_global_cleanup();
    
    if (log_file_.is_open()) {
        log("WebDataCollector shutting down");
        log_file_.close();
    }
}

bool WebDataCollector::loadConfigFromJson(const std::string& json_path) {
    try {
        std::ifstream file(json_path);
        if (!file.is_open()) {
            logError("Failed to open config file: " + json_path);
            return false;
        }
        
        json config_json;
        file >> config_json;
        
        // Parse collection settings
        if (config_json.contains("collection_settings")) {
            auto settings = config_json["collection_settings"];
            if (settings.contains("max_entries_per_source"))
                config_.max_entries_per_source = settings["max_entries_per_source"];
            if (settings.contains("max_new_entries_per_run"))
                config_.max_new_entries_per_run = settings["max_new_entries_per_run"];
            if (settings.contains("timeout_seconds"))
                config_.timeout_seconds = settings["timeout_seconds"];
            if (settings.contains("rate_limit_delay_ms"))
                config_.rate_limit_delay_ms = settings["rate_limit_delay_ms"];
            if (settings.contains("max_retries"))
                config_.max_retries = settings["max_retries"];
            if (settings.contains("concurrent_requests"))
                config_.max_concurrent_requests = settings["concurrent_requests"];
            if (settings.contains("user_agent"))
                config_.user_agent = settings["user_agent"];
        }
        
        // Parse data sources
        if (config_json.contains("data_sources")) {
            config_.sources.clear();
            for (const auto& source_json : config_json["data_sources"]) {
                DataSource source;
                source.name = source_json.value("name", "");
                source.url = source_json.value("url", "");
                
                // source_type is now optional - defaults to "miscellaneous"
                source.source_type_str = source_json.value("source_type", "miscellaneous");
                
                // Fetcher is ALWAYS auto-detected from URL pattern
                source.autoDetectFetcher();
                source.source_type = source.fetcher_type;  // Keep legacy alias in sync
                
                source.enabled = source_json.value("enabled", true);
                source.priority = source_json.value("priority", 5);
                source.requires_auth = source_json.value("requires_auth", false);
                source.api_key_env = source_json.value("api_key_env", "");
                source.fetch_limit = source_json.value("fetch_limit", -1);  // -1 = use dynamic calculation
                source.crawl_depth = source_json.value("crawl_depth", 2);  // Default to depth 2 for HTML crawling
                
                // Load API key from environment
                source.loadApiKey();
                
                // Parse content filter
                if (source_json.contains("content_filter")) {
                    auto filter_json = source_json["content_filter"];
                    if (filter_json.contains("min_length"))
                        source.filter.min_length = filter_json["min_length"];
                    if (filter_json.contains("max_length"))
                        source.filter.max_length = filter_json["max_length"];
                    if (filter_json.contains("min_stars"))
                        source.filter.min_stars = filter_json["min_stars"];
                    if (filter_json.contains("path_patterns")) {
                        for (const auto& p : filter_json["path_patterns"])
                            source.filter.path_patterns.push_back(p.get<std::string>());
                    }
                    if (filter_json.contains("exclude_patterns")) {
                        for (const auto& p : filter_json["exclude_patterns"])
                            source.filter.exclude_patterns.push_back(p.get<std::string>());
                    }
                    // ... parse other filter fields
                }
                
                config_.sources.push_back(source);
            }
        }
        
        // Parse storage settings
        if (config_json.contains("storage_settings")) {
            auto storage = config_json["storage_settings"];
            if (storage.contains("output_dir"))
                config_.output_dir = storage["output_dir"];
            if (storage.contains("output_format")) {
                std::string format = storage["output_format"];
                config_.save_as_jsonl = (format == "jsonl");
            }
        }
        
        // Parse logging settings
        if (config_json.contains("logging")) {
            auto logging = config_json["logging"];
            if (logging.contains("log_file"))
                config_.log_file = logging["log_file"];
            if (logging.contains("verbose"))
                config_.verbose = logging["verbose"];
        }
        
        log("Configuration loaded from " + json_path);
        log("Loaded " + std::to_string(config_.sources.size()) + " data sources");
        
        return true;
    } catch (const std::exception& e) {
        logError("Failed to parse JSON config: " + std::string(e.what()));
        return false;
    }
}

size_t WebDataCollector::collectData() {
    log("=== Starting Data Collection ===");
    stats_.reset();
    collected_data_.clear();
    
    auto start_time = std::chrono::steady_clock::now();
    
    stats_.total_sources = config_.sources.size();
    
    // Count enabled sources
    for (const auto& source : config_.sources) {
        if (source.enabled) {
            stats_.enabled_sources++;
        }
    }
    
    log("Processing " + std::to_string(stats_.enabled_sources) + " enabled sources");
    log("Global limit: " + std::to_string(config_.max_new_entries_per_run) + " max new entries");
    
    // Track total NEW entries collected across all sources
    size_t total_new_entries = 0;
    
    // Process each source
    progress_total_ = stats_.enabled_sources;
    progress_current_ = 0;
    
    for (const auto& source : config_.sources) {
        // Check global limit before processing next source
        if (static_cast<int>(total_new_entries) >= config_.max_new_entries_per_run) {
            log("\n⚠ Reached global limit of " + std::to_string(config_.max_new_entries_per_run) + 
                " new entries. Stopping collection.");
            break;
        }
        
        if (!source.enabled) {
            log("Skipping disabled source: " + source.name);
            continue;
        }
        
        // Skip sources that were recently collected (avoids re-fetching known data)
        if (stateManager_) {
            const int64_t REFRESH_INTERVAL_SECONDS = 86400;  // 24 hours
            if (!stateManager_->sourceNeedsRefresh(source.url, REFRESH_INTERVAL_SECONDS)) {
                auto record = stateManager_->getSourceRecord(source.url);
                int64_t hours_ago = 0;
                if (record) {
                    int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count();
                    hours_ago = (now - record->last_successful_timestamp) / 3600;
                }
                log("[SKIP] " + source.name + ": Collected " + std::to_string(hours_ago) + 
                    "h ago, refresh interval is " + std::to_string(REFRESH_INTERVAL_SECONDS / 3600) + "h");
                progress_current_++;
                updateProgress(progress_current_, progress_total_);
                continue;
            }
        }
        
        log("\n--- Fetching from: " + source.name + " ---");
        log("URL: " + source.url);
        log("Type: " + sourceTypeToString(source.source_type));
        log("Fetch limit: " + std::to_string(source.fetch_limit) + ", Crawl depth: " + std::to_string(source.crawl_depth));
        log("Progress: " + std::to_string(total_new_entries) + "/" + std::to_string(config_.max_new_entries_per_run) + " new entries");
        
        try {
            auto start_time = std::chrono::steady_clock::now();
            std::vector<RawDataEntry> entries = fetchFromSource(source);
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
            
            // Filter out duplicates using state manager
            size_t new_entries = 0;
            size_t url_duplicates = 0;
            size_t content_duplicates = 0;
            
            {
                std::lock_guard<std::mutex> lock(data_mutex_);
                std::string sourceTypeName = sourceTypeToString(source.source_type);
                
                // Calculate how many more we can accept before hitting global limit
                int remaining_capacity = config_.max_new_entries_per_run - static_cast<int>(total_new_entries);
                
                for (const auto& entry : entries) {
                    // Stop if we've hit global limit
                    if (static_cast<int>(new_entries) >= remaining_capacity) {
                        log("[LIMIT] Reached global max_new_entries_per_run limit, stopping source");
                        break;
                    }
                    
                    // Check if URL has already been collected
                    if (stateManager_ && stateManager_->hasCollectedUrl(entry.source_url)) {
                        url_duplicates++;
                        continue;
                    }
                    
                    // Check if content has already been seen
                    if (stateManager_ && stateManager_->hasSeenContent(entry.content)) {
                        content_duplicates++;
                        continue;
                    }
                    
                    // New unique entry - add it
                    collected_data_.push_back(entry);
                    new_entries++;
                    
                    // Mark as collected for future deduplication
                    if (stateManager_) {
                        stateManager_->markUrlCollected(entry.source_url, sourceTypeName);
                        stateManager_->markContentSeen(entry.content);
                    }
                }
                
                // Update total new entries count
                total_new_entries += new_entries;
                
                stats_.successful += new_entries;
                stats_.duplicates_skipped += url_duplicates;
                stats_.content_duplicates += content_duplicates;
                stats_.per_source_type[source.source_type] += new_entries;
            }
            
            // Track source completion in state manager for refresh gating
            if (stateManager_) {
                stateManager_->markSourceCompleted(source.url, new_entries);
                // Persist source metadata for tracking
                auto existingRecord = stateManager_->getSourceRecord(source.url);
                if (existingRecord) {
                    auto updated = *existingRecord;
                    updated.name = source.name;
                    updated.source_type = source.source_type_str;
                    stateManager_->updateSourceRecord(updated);
                }
            }
            
            if (new_entries > 0) {
                log("[SUCCESS] " + source.name + ": " + std::to_string(new_entries) + " new entries in " + std::to_string(duration) + "s");
                if (url_duplicates > 0 || content_duplicates > 0) {
                    log("  Skipped: " + std::to_string(url_duplicates) + " URL duplicates, " + 
                        std::to_string(content_duplicates) + " content duplicates");
                }
                // Log sample content length
                size_t avg_length = 0;
                for (const auto& entry : entries) {
                    avg_length += entry.content.length();
                }
                if (!entries.empty()) {
                    avg_length /= entries.size();
                }
                log("  Average content length: " + std::to_string(avg_length) + " chars");
            } else {
                if (url_duplicates > 0 || content_duplicates > 0) {
                    log("[SKIP] " + source.name + ": All " + std::to_string(entries.size()) + 
                        " entries were duplicates (collected previously)");
                } else {
                    logWarning("[EMPTY] " + source.name + ": No entries collected in " + std::to_string(duration) + "s");
                }
            }
            
        } catch (const std::exception& e) {
            logError("[FAILED] " + source.name + ": " + e.what());
            stats_.failed++;
            if (stateManager_) {
                stateManager_->markSourceFailed(source.url, e.what());
            }
        } catch (...) {
            logError("[FAILED] " + source.name + ": Unknown error");
            stats_.failed++;
            if (stateManager_) {
                stateManager_->markSourceFailed(source.url, "Unknown error");
            }
        }
        
        progress_current_++;
        updateProgress(progress_current_, progress_total_);
        
        // Save checkpoint every 5 sources (also saves state)
        if (progress_current_ % 5 == 0) {
            std::string checkpoint_file = "data/checkpoint_" + std::to_string(progress_current_) + ".ckpt";
            if (saveCheckpoint(checkpoint_file)) {
                log("Checkpoint saved: " + checkpoint_file + " (" + std::to_string(collected_data_.size()) + " entries)");
            }
            // Also save state manager state periodically
            if (stateManager_) {
                stateManager_->saveState();
                log("Collection state saved (" + std::to_string(stateManager_->getTotalUniqueUrls()) + " unique URLs)");
            }
        }
        
        // Apply rate limiting
        try {
            applyRateLimit(source.url);
        } catch (...) {
            // Ignore rate limit errors
        }
    }
    
    // Save final state
    if (stateManager_) {
        stateManager_->saveState();
    }
    
    auto end_time = std::chrono::steady_clock::now();
    stats_.total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (stats_.successful > 0) {
        stats_.avg_request_time = stats_.total_time / stats_.successful;
    }
    
    stats_.total_fetched = collected_data_.size();
    
    log("\n=== Collection Complete ===");
    log(stats_.toString());
    
    return collected_data_.size();
}

std::vector<RawDataEntry> WebDataCollector::fetchFromSource(const DataSource& source) {
    // Look up fetcher by FetcherType (auto-detected or explicit from config)
    auto it = source_fetchers_.find(source.fetcher_type);
    
    if (it != source_fetchers_.end()) {
        // Found registered fetcher
        log("[SOURCE] Fetcher: " + fetcherTypeToString(source.fetcher_type) + 
            ", Category: " + source.source_type_str + ", Name: " + source.name);
        return it->second(source);
    } else {
        // Default to HTML crawl for unknown fetcher types
        log("[SOURCE] HTML_CRAWL (fallback), Category: " + source.source_type_str + ", Name: " + source.name);
        return fetchCustom(source);
    }
}

void WebDataCollector::registerSourceFetcher(FetcherType type, std::function<std::vector<RawDataEntry>(const DataSource&)> fetcher) {
    source_fetchers_[type] = fetcher;
    log("Registered fetcher: " + fetcherTypeToString(type));
}

void WebDataCollector::initializeDefaultFetchers() {
    // Register API-based fetchers (keyed by FetcherType)
    source_fetchers_[FetcherType::GITHUB_API] = [this](const DataSource& s) { return fetchGitHub(s); };
    source_fetchers_[FetcherType::ARXIV_API] = [this](const DataSource& s) { return fetchArXiv(s); };
    source_fetchers_[FetcherType::WIKIPEDIA_API] = [this](const DataSource& s) { return fetchWikipedia(s); };
    source_fetchers_[FetcherType::STACKOVERFLOW_API] = [this](const DataSource& s) { return fetchStackOverflow(s); };
    source_fetchers_[FetcherType::REDDIT_API] = [this](const DataSource& s) { return fetchReddit(s); };
    source_fetchers_[FetcherType::NEWS_API] = [this](const DataSource& s) { return fetchNewsAPI(s); };
    source_fetchers_[FetcherType::TECH_DOCS] = [this](const DataSource& s) { return fetchTechDocs(s); };
    
    // HTML crawl is the default fallback for all other sources
    source_fetchers_[FetcherType::HTML_CRAWL] = [this](const DataSource& s) { return fetchCustom(s); };
}

std::vector<RawDataEntry> WebDataCollector::fetchGitHub(const DataSource& source) {
    std::vector<RawDataEntry> entries;
    log("Fetching from GitHub: " + source.url);
    
    // Build headers with authentication
    std::unordered_map<std::string, std::string> headers;
    headers["User-Agent"] = config_.user_agent;
    headers["Accept"] = "application/vnd.github.v3+json";
    
    if (source.requires_auth && !source.api_key.empty()) {
        headers["Authorization"] = "token " + source.api_key;
    }
    
    // Fetch from GitHub API
    std::string response = httpGet(source.url, headers);
    if (response.empty()) {
        logError("Failed to fetch from GitHub: " + source.url);
        return entries;
    }
    
    try {
        // Check if this is a GitHub API endpoint
        if (source.url.find("api.github.com") != std::string::npos) {
            json data = json::parse(response);
            
            // Handle single repository response
            if (data.is_object() && data.contains("name")) {
                std::string repo_url = data.value("html_url", source.url);
                
                if (!(stateManager_ && stateManager_->hasCollectedUrl(repo_url))) {
                    RawDataEntry entry;
                    entry.content = data.value("description", "") + "\n\n" + 
                                   data.value("readme", "");
                    entry.source_url = repo_url;
                    entry.author = data.value("owner", json::object()).value("login", "Unknown");
                    entry.source_name = source.name;
                    entry.source_type = source.source_type;
                    entry.source_priority = source.priority;
                    entry.fetch_date = getCurrentUnixTime();
                    entry.metadata_json = data.dump();
                    
                    if (source.filter.passes(entry.content)) {
                        entries.push_back(entry);
                    } else {
                        stats_.filtered_out++;
                    }
                }
            }
            // Handle array of repositories
            else if (data.is_array()) {
                for (const auto& item : data) {
                    if (entries.size() >= static_cast<size_t>(config_.max_entries_per_source)) break;
                    
                    std::string repo_url = item.value("html_url", source.url);
                    if (stateManager_ && stateManager_->hasCollectedUrl(repo_url)) {
                        continue;
                    }
                    
                    RawDataEntry entry;
                    entry.content = item.value("description", "") + "\n\n" + 
                                   item.value("readme", "");
                    entry.source_url = repo_url;
                    entry.author = item.value("owner", json::object()).value("login", "Unknown");
                    entry.source_name = source.name;
                    entry.source_type = source.source_type;
                    entry.source_priority = source.priority;
                    entry.fetch_date = getCurrentUnixTime();
                    entry.metadata_json = item.dump();
                    
                    if (source.filter.passes(entry.content)) {
                        entries.push_back(entry);
                    } else {
                        stats_.filtered_out++;
                    }
                }
            }
        } else {
            // Direct markdown/text content
            RawDataEntry entry;
            entry.content = response;
            entry.source_url = source.url;
            entry.author = "Unknown";
            entry.source_name = source.name;
            entry.source_type = source.source_type;
            entry.source_priority = source.priority;
            entry.fetch_date = getCurrentUnixTime();
            
            if (source.filter.passes(entry.content)) {
                entries.push_back(entry);
            } else {
                stats_.filtered_out++;
            }
        }
    } catch (const json::parse_error& e) {
        logError("JSON parse error for GitHub source: " + std::string(e.what()));
    }
    
    return entries;
}

std::vector<RawDataEntry> WebDataCollector::fetchArXiv(const DataSource& source) {
    std::vector<RawDataEntry> entries;
    log("Fetching ArXiv papers via streaming...");
    
    try {
        // Resume from last offset to avoid re-fetching known papers
        int startOffset = 0;
        if (stateManager_) {
            auto record = stateManager_->getSourceRecord(source.url);
            if (record && record->last_offset > 0) {
                startOffset = record->last_offset;
                log("[ArXiv] Resuming from offset " + std::to_string(startOffset));
            }
        }
        
        int maxResults = std::min(source.fetch_limit, 100);
        
        // Parse search_query from the configured source URL so user's
        // category/topic filters are respected (e.g. cat:cs.AI+OR+cat:cs.CL)
        std::string search_query = "all";
        size_t sq_pos = source.url.find("search_query=");
        if (sq_pos != std::string::npos) {
            sq_pos += 13; // length of "search_query="
            size_t sq_end = source.url.find('&', sq_pos);
            search_query = (sq_end != std::string::npos) 
                ? source.url.substr(sq_pos, sq_end - sq_pos)
                : source.url.substr(sq_pos);
        }
        
        std::string api_url = "http://export.arxiv.org/api/query?search_query=" + search_query +
            "&start=" + std::to_string(startOffset) + 
            "&max_results=" + std::to_string(maxResults) +
            "&sortBy=submittedDate&sortOrder=descending";
        
        std::string response;
        if (!downloader_->downloadToMemory(api_url, response, 5 * 1024 * 1024)) {
            logError("Failed to download from ArXiv API");
            return entries;
        }
        
        size_t skipped_known = 0;
        
        // Simple XML parsing - extract entries
        size_t pos = 0;
        while ((pos = response.find("<entry>", pos)) != std::string::npos && 
               entries.size() < static_cast<size_t>(source.fetch_limit)) {
            size_t end = response.find("</entry>", pos);
            if (end == std::string::npos) break;
            
            std::string entry_xml = response.substr(pos, end - pos + 8);
            
            auto extract = [&](const std::string& tag) {
                std::string open = "<" + tag + ">";
                std::string close = "</" + tag + ">";
                size_t s = entry_xml.find(open);
                size_t e = entry_xml.find(close);
                if (s != std::string::npos && e != std::string::npos) {
                    return entry_xml.substr(s + open.length(), e - s - open.length());
                }
                return std::string();
            };
            
            std::string title = extract("title");
            std::string summary = extract("summary");
            std::string id = extract("id");
            
            // Skip papers we've already collected
            if (stateManager_ && stateManager_->hasCollectedUrl(id)) {
                skipped_known++;
                pos = end + 8;
                continue;
            }
            
            std::string content = title + "\n\n" + summary;
            
            if (content.length() > 100 && source.filter.passes(content)) {
                RawDataEntry entry;
                entry.content = std::move(content);
                entry.source_url = std::move(id);
                entry.title = std::move(title);
                entry.source_name = source.name;
                entry.source_type = source.source_type;
                entry.source_priority = source.priority;
                entry.fetch_date = getCurrentUnixTime();
                entries.push_back(std::move(entry));
            } else {
                stats_.filtered_out++;
            }
            
            pos = end + 8;
        }
        
        // Advance pagination offset for next run
        if (stateManager_) {
            DataCollection::SourceRecord record;
            auto existing = stateManager_->getSourceRecord(source.url);
            if (existing) record = *existing;
            record.url = source.url;
            record.name = source.name;
            record.source_type = "arxiv_api";
            record.last_offset = startOffset + maxResults;
            stateManager_->updateSourceRecord(record);
        }
        
        log("Fetched " + std::to_string(entries.size()) + " ArXiv papers" +
            (skipped_known > 0 ? " (skipped " + std::to_string(skipped_known) + " already collected)" : ""));
    } catch (const std::exception& e) {
        logError("Exception in fetchArXiv: " + std::string(e.what()));
    }
    
    return entries;
}

std::vector<RawDataEntry> WebDataCollector::fetchWikipedia(const DataSource& source) {
    std::vector<RawDataEntry> entries;
    log("Fetching Wikipedia articles via streaming...");
    
    try {
        std::string api_url = "https://en.wikipedia.org/w/api.php?action=query&list=random&rnlimit=";
        api_url += std::to_string(std::min(source.fetch_limit, 50));
        api_url += "&rnnamespace=0&format=json";
        
        std::string response;
        if (!downloader_->downloadToMemory(api_url, response, 2 * 1024 * 1024)) {
            logError("Failed to fetch Wikipedia articles");
            return entries;
        }
        
        // Extract page IDs (simple parsing)
        size_t pos = response.find("\"random\":[");
        if (pos == std::string::npos) return entries;
        
        size_t wiki_skipped = 0;
        
        pos = response.find("\"id\":", pos);
        while (pos != std::string::npos && entries.size() < static_cast<size_t>(source.fetch_limit)) {
            pos += 5;
            size_t end = response.find_first_of(",}", pos);
            if (end == std::string::npos) break;
            
            std::string page_id = response.substr(pos, end - pos);
            
            // Check if this article was already collected BEFORE making the HTTP request
            std::string article_url = "https://en.wikipedia.org/?curid=" + page_id;
            if (stateManager_ && stateManager_->hasCollectedUrl(article_url)) {
                wiki_skipped++;
                pos = response.find("\"id\":", pos);
                continue;
            }
            
            // Fetch content
            std::string content_url = "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&pageids=";
            content_url += page_id + "&explaintext=1&format=json";
            
            std::string content_response;
            if (downloader_->downloadToMemory(content_url, content_response, 5 * 1024 * 1024)) {
                size_t extract_pos = content_response.find("\"extract\":\"");
                if (extract_pos != std::string::npos) {
                    extract_pos += 11;
                    size_t extract_end = content_response.find("\"", extract_pos);
                    if (extract_end != std::string::npos) {
                        std::string extract = content_response.substr(extract_pos, extract_end - extract_pos);
                        
                        // Unescape
                        size_t esc = 0;
                        while ((esc = extract.find("\\n", esc)) != std::string::npos) {
                            extract.replace(esc, 2, "\n");
                            esc++;
                        }
                        
                        if (extract.length() > 200 && source.filter.passes(extract)) {
                            RawDataEntry entry;
                            entry.content = std::move(extract);
                            entry.source_url = "https://en.wikipedia.org/?curid=" + page_id;
                            entry.source_name = source.name;
                            entry.source_type = source.source_type;
                            entry.source_priority = source.priority;
                            entry.fetch_date = getCurrentUnixTime();
                            entries.push_back(std::move(entry));
                        } else {
                            stats_.filtered_out++;
                        }
                    }
                }
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            pos = response.find("\"id\":", pos);
        }
        
        log("Fetched " + std::to_string(entries.size()) + " Wikipedia articles" +
            (wiki_skipped > 0 ? " (skipped " + std::to_string(wiki_skipped) + " already collected)" : ""));
    } catch (const std::exception& e) {
        logError("Exception in fetchWikipedia: " + std::string(e.what()));
    }
    
    return entries;
}

std::vector<RawDataEntry> WebDataCollector::fetchStackOverflow(const DataSource& source) {
    std::vector<RawDataEntry> entries;
    log("Fetching StackOverflow via streaming...");
    
    try {
        // Resume from last page to avoid re-fetching same top questions
        int startPage = 1;
        if (stateManager_) {
            auto record = stateManager_->getSourceRecord(source.url);
            if (record && record->last_page > 0) {
                startPage = record->last_page + 1;
                log("[SO] Resuming from page " + std::to_string(startPage));
            }
        }
        
        int pageSize = std::min(source.fetch_limit, 100);
        std::string api_url = "https://api.stackexchange.com/2.3/questions?order=desc&sort=votes&site=stackoverflow&pagesize=";
        api_url += std::to_string(pageSize) + "&page=" + std::to_string(startPage) + "&filter=withbody";
        
        std::string response;
        if (!downloader_->downloadToMemory(api_url, response, 10 * 1024 * 1024)) {
            logError("Failed to fetch StackOverflow");
            return entries;
        }
        
        size_t skipped_known = 0;
        
        // Helper to extract a JSON string value from a raw JSON object substring
        auto extractStr = [](const std::string& obj, const std::string& field) -> std::string {
            std::string search = "\"" + field + "\":\"";
            size_t s = obj.find(search);
            if (s == std::string::npos) return {};
            s += search.length();
            size_t e = obj.find("\"", s);
            return (e != std::string::npos) ? obj.substr(s, e - s) : std::string();
        };
        
        // Helper to extract a JSON integer value from a raw JSON object substring
        auto extractInt = [](const std::string& obj, const std::string& field) -> int64_t {
            std::string search = "\"" + field + "\":";
            size_t s = obj.find(search);
            if (s == std::string::npos) return -1;
            s += search.length();
            // Skip whitespace
            while (s < obj.size() && (obj[s] == ' ' || obj[s] == '\t')) ++s;
            if (s >= obj.size() || (!std::isdigit(obj[s]) && obj[s] != '-')) return -1;
            size_t e = s;
            while (e < obj.size() && (std::isdigit(obj[e]) || obj[e] == '-')) ++e;
            return std::stoll(obj.substr(s, e - s));
        };
        
        // First pass: parse questions and collect accepted_answer_ids for batching
        struct ParsedQuestion {
            std::string title;
            std::string body;
            std::string link;
            int64_t accepted_answer_id = -1;
        };
        std::vector<ParsedQuestion> parsed_questions;
        std::vector<int64_t> answer_ids_to_fetch;
        
        size_t items_pos = response.find("\"items\":[");
        if (items_pos == std::string::npos) return entries;
        
        size_t pos = items_pos + 9;
        while (parsed_questions.size() < static_cast<size_t>(source.fetch_limit)) {
            pos = response.find("{", pos);
            if (pos == std::string::npos || pos > response.find("]", items_pos)) break;
            
            size_t obj_end = response.find("}", pos);
            if (obj_end == std::string::npos) break;
            
            std::string obj = response.substr(pos, obj_end - pos + 1);
            
            std::string link = extractStr(obj, "link");
            
            // Skip questions we've already collected
            if (!link.empty() && stateManager_ && stateManager_->hasCollectedUrl(link)) {
                skipped_known++;
                pos = obj_end + 1;
                continue;
            }
            
            std::string title = extractStr(obj, "title");
            std::string body = extractStr(obj, "body");
            
            if (!title.empty() && !body.empty()) {
                ParsedQuestion pq;
                pq.title = std::move(title);
                pq.body = std::move(body);
                pq.link = std::move(link);
                pq.accepted_answer_id = extractInt(obj, "accepted_answer_id");
                
                if (pq.accepted_answer_id > 0) {
                    answer_ids_to_fetch.push_back(pq.accepted_answer_id);
                }
                parsed_questions.push_back(std::move(pq));
            }
            
            pos = obj_end + 1;
        }
        
        // Batch-fetch accepted answers (up to 100 IDs per request to respect SE API)
        std::unordered_map<int64_t, std::string> answer_bodies;
        
        for (size_t batch_start = 0; batch_start < answer_ids_to_fetch.size(); batch_start += 100) {
            size_t batch_end = std::min(batch_start + 100, answer_ids_to_fetch.size());
            
            // Build semicolon-separated ID list: /answers/1;2;3
            std::string id_list;
            for (size_t i = batch_start; i < batch_end; ++i) {
                if (!id_list.empty()) id_list += ";";
                id_list += std::to_string(answer_ids_to_fetch[i]);
            }
            
            std::string answers_url = "https://api.stackexchange.com/2.3/answers/" + id_list
                + "?site=stackoverflow&filter=withbody";
            
            std::string answers_response;
            if (!downloader_->downloadToMemory(answers_url, answers_response, 10 * 1024 * 1024)) {
                log("[SO] Warning: failed to fetch answer batch, continuing without answers");
                continue;
            }
            
            // Parse answer bodies from response
            size_t a_items_pos = answers_response.find("\"items\":[");
            if (a_items_pos == std::string::npos) continue;
            
            size_t a_pos = a_items_pos + 9;
            while (true) {
                a_pos = answers_response.find("{", a_pos);
                if (a_pos == std::string::npos || a_pos > answers_response.find("]", a_items_pos)) break;
                
                size_t a_obj_end = answers_response.find("}", a_pos);
                if (a_obj_end == std::string::npos) break;
                
                std::string a_obj = answers_response.substr(a_pos, a_obj_end - a_pos + 1);
                
                int64_t answer_id = extractInt(a_obj, "answer_id");
                std::string a_body = extractStr(a_obj, "body");
                
                if (answer_id > 0 && !a_body.empty()) {
                    answer_bodies[answer_id] = std::move(a_body);
                }
                
                a_pos = a_obj_end + 1;
            }
        }
        
        size_t with_answers = 0;
        
        // Build entries with Q/A format when answer is available
        for (auto& pq : parsed_questions) {
            std::string question_text = pq.title + "\n\n" + pq.body;
            std::string content;
            std::string answer_text;
            
            auto it = (pq.accepted_answer_id > 0) ? answer_bodies.find(pq.accepted_answer_id) : answer_bodies.end();
            if (it != answer_bodies.end()) {
                answer_text = it->second;
                content = "Q: " + question_text + "\n\nA: " + answer_text;
                with_answers++;
            } else {
                content = "Q: " + question_text;
            }
            
            if (source.filter.passes(content)) {
                RawDataEntry entry;
                entry.content = std::move(content);
                entry.source_url = std::move(pq.link);
                entry.title = pq.title;
                entry.question = std::move(question_text);
                entry.answer = std::move(answer_text);
                entry.source_name = source.name;
                entry.source_type = source.source_type;
                entry.source_priority = source.priority;
                entry.fetch_date = getCurrentUnixTime();
                entries.push_back(std::move(entry));
            } else {
                stats_.filtered_out++;
            }
        }
        
        // Advance page for next run
        if (stateManager_) {
            DataCollection::SourceRecord record;
            auto existing = stateManager_->getSourceRecord(source.url);
            if (existing) record = *existing;
            record.url = source.url;
            record.name = source.name;
            record.source_type = "stackoverflow_api";
            record.last_page = startPage;
            stateManager_->updateSourceRecord(record);
        }
        
        log("Fetched " + std::to_string(entries.size()) + " StackOverflow Q/A entries (" +
            std::to_string(with_answers) + " with accepted answers)" +
            (skipped_known > 0 ? " (skipped " + std::to_string(skipped_known) + " already collected)" : ""));
    } catch (const std::exception& e) {
        logError("Exception in fetchStackOverflow: " + std::string(e.what()));
    }
    
    return entries;
}

std::vector<RawDataEntry> WebDataCollector::fetchReddit(const DataSource& source) {
    std::vector<RawDataEntry> entries;
    log("Fetching Reddit via streaming...");
    
    try {
        std::string api_url = source.url;
        if (api_url.find(".json") == std::string::npos) {
            if (api_url.back() == '/') api_url.pop_back();
            api_url += ".json?limit=" + std::to_string(std::min(source.fetch_limit, 100));
        }
        
        std::string response;
        if (!downloader_->downloadToMemory(api_url, response, 10 * 1024 * 1024)) {
            logError("Failed to fetch Reddit");
            return entries;
        }
        
        size_t children_pos = response.find("\"children\":[");
        if (children_pos == std::string::npos) return entries;
        
        size_t pos = children_pos + 12;
        while (entries.size() < static_cast<size_t>(source.fetch_limit)) {
            pos = response.find("\"data\":{", pos);
            if (pos == std::string::npos || pos > response.find("]", children_pos)) break;
            
            pos += 8;
            size_t data_end = pos;
            int brace_count = 1;
            while (data_end < response.length() && brace_count > 0) {
                if (response[data_end] == '{') brace_count++;
                else if (response[data_end] == '}') brace_count--;
                data_end++;
            }
            
            std::string data_obj = response.substr(pos, data_end - pos - 1);
            
            auto extract = [&](const std::string& field) {
                std::string search = "\"" + field + "\":\"";
                size_t s = data_obj.find(search);
                if (s == std::string::npos) return std::string();
                s += search.length();
                size_t e = s;
                while (e < data_obj.length() && data_obj[e] != '\"') {
                    if (data_obj[e] == '\\') e++;
                    e++;
                }
                return data_obj.substr(s, e - s);
            };
            
            std::string title = extract("title");
            std::string selftext = extract("selftext");
            std::string permalink = extract("permalink");
            std::string author = extract("author");
            
            if (!title.empty()) {
                std::string post_url = "https://www.reddit.com" + permalink;
                
                if (stateManager_ && stateManager_->hasCollectedUrl(post_url)) {
                    pos = data_end;
                    continue;
                }
                
                std::string content = "Title: " + title + "\n\n" + selftext;
                if (content.length() > 100 && source.filter.passes(content)) {
                    RawDataEntry entry;
                    entry.content = std::move(content);
                    entry.source_url = std::move(post_url);
                    entry.title = std::move(title);
                    entry.author = std::move(author);
                    entry.source_name = source.name;
                    entry.source_type = source.source_type;
                    entry.source_priority = source.priority;
                    entry.fetch_date = getCurrentUnixTime();
                    entries.push_back(std::move(entry));
                } else {
                    stats_.filtered_out++;
                }
            }
            
            pos = data_end;
        }
        
        log("Fetched " + std::to_string(entries.size()) + " Reddit posts");
    } catch (const std::exception& e) {
        logError("Exception in fetchReddit: " + std::string(e.what()));
    }
    
    return entries;
}

std::vector<RawDataEntry> WebDataCollector::fetchNewsAPI(const DataSource& source) {
    std::vector<RawDataEntry> entries;
    log("Fetching news via streaming...");
    
    try {
        std::string response;
        if (!downloader_->downloadToMemory(source.url, response, 5 * 1024 * 1024)) {
            logError("Failed to fetch News API");
            return entries;
        }
        
        size_t articles_pos = response.find("\"articles\":[");
        if (articles_pos == std::string::npos) return entries;
        
        size_t pos = articles_pos + 12;
        while (entries.size() < static_cast<size_t>(config_.max_entries_per_source)) {
            pos = response.find("{", pos);
            if (pos == std::string::npos || pos > response.find("]", articles_pos)) break;
            
            size_t obj_end = pos + 1;
            int brace_count = 1;
            while (obj_end < response.length() && brace_count > 0) {
                if (response[obj_end] == '{') brace_count++;
                else if (response[obj_end] == '}') brace_count--;
                obj_end++;
            }
            
            std::string article = response.substr(pos, obj_end - pos);
            
            auto extract = [&](const std::string& field) {
                std::string search = "\"" + field + "\":\"";
                size_t s = article.find(search);
                if (s == std::string::npos) return std::string();
                s += search.length();
                size_t e = s;
                while (e < article.length() && article[e] != '\"') {
                    if (article[e] == '\\') e++;
                    e++;
                }
                return article.substr(s, e - s);
            };
            
            std::string title = extract("title");
            std::string description = extract("description");
            std::string content = extract("content");
            std::string url = extract("url");
            std::string author = extract("author");
            
            if (!title.empty()) {
                if (stateManager_ && stateManager_->hasCollectedUrl(url)) {
                    pos = obj_end;
                    continue;
                }
                
                std::string full_content = title + "\n\n" + description + "\n\n" + content;
                if (source.filter.passes(full_content)) {
                    RawDataEntry entry;
                    entry.content = std::move(full_content);
                    entry.source_url = std::move(url);
                    entry.author = std::move(author);
                    entry.source_name = source.name;
                    entry.source_type = source.source_type;
                    entry.source_priority = source.priority;
                    entry.fetch_date = getCurrentUnixTime();
                    entries.push_back(std::move(entry));
                } else {
                    stats_.filtered_out++;
                }
            }
            
            pos = obj_end;
        }
        
        log("Fetched " + std::to_string(entries.size()) + " news articles");
    } catch (const std::exception& e) {
        logError("Exception in fetchNewsAPI: " + std::string(e.what()));
    }
    
    return entries;
}

std::vector<RawDataEntry> WebDataCollector::fetchTechDocs(const DataSource& source) {
    // Route all tech docs through custom HTML fetch (async crawler removed)
    return fetchCustom(source);
}

std::vector<RawDataEntry> WebDataCollector::fetchCustom(const DataSource& source) {
    std::vector<RawDataEntry> entries;
    log("[HTML] Fetching custom source with streaming and HTML extraction...");
    
    try {
        // Determine if we should crawl or just fetch single page
        bool should_crawl = (source.crawl_depth > 1 && source.fetch_limit > 1);
        
        if (should_crawl) {
            log("[HTML] Simple synchronous crawl (depth=" + std::to_string(source.crawl_depth) + 
                ", limit=" + std::to_string(source.fetch_limit) + ")...");
        } else {
            log("[HTML] Fetching single page with streaming...");
        }
        
        std::string raw_html;
        if (downloader_->downloadToMemory(source.url, raw_html, 10 * 1024 * 1024)) {
            log("[HTML] Downloaded " + std::to_string(raw_html.length()) + " bytes, extracting text...");
            
            // Check if this is plain text or HTML
            bool is_html = (raw_html.find("<html") != std::string::npos || 
                           raw_html.find("<!DOCTYPE") != std::string::npos ||
                           raw_html.find("<HTML") != std::string::npos);
            
            std::string clean_content;
            std::string title;
            
            if (!is_html) {
                // Plain text file - use directly
                log("[HTML] Detected plain text format, using directly");
                clean_content = raw_html;
            } else {
                // HTML content - extract text
                auto extracted = HTMLExtractor::extract(raw_html);
                
                log("[HTML] Extraction results: blocks=" + std::to_string(extracted.text_blocks_found) + 
                    ", text_len=" + std::to_string(extracted.main_text.length()) + 
                    ", has_content=" + std::string(extracted.has_article_content ? "true" : "false"));
                
                if (!extracted.has_article_content || extracted.main_text.empty()) {
                    log("[HTML] ✗ No article content found in page");
                    if (!extracted.main_text.empty() && extracted.main_text.length() < 500) {
                        log("[HTML] DEBUG - First 200 chars: " + extracted.main_text.substr(0, std::min<size_t>(200, extracted.main_text.length())));
                    }
                    stats_.filtered_out++;
                } else {
                    log("[HTML] ✓ Extracted " + std::to_string(extracted.text_blocks_found) + 
                        " text blocks, " + std::to_string(extracted.main_text.length()) + " chars");
                    
                    clean_content = extracted.main_text;
                    title = extracted.title;
                    if (!title.empty()) {
                        clean_content = title + "\n\n" + clean_content;
                        log("[HTML] Title: " + title);
                    }
                }
            }
            
            // If we have content (either from HTML extraction or plain text), process it
            if (!clean_content.empty() && source.filter.passes(clean_content)) {
                RawDataEntry entry;
                entry.content = std::move(clean_content);
                entry.source_url = source.url;
                entry.title = title;
                entry.source_name = source.name;
                entry.source_type = source.source_type;
                entry.source_priority = source.priority;
                entry.fetch_date = getCurrentUnixTime();
                
                entries.push_back(std::move(entry));
                log("[HTML] ✓ Single page passed filter");
            } else {
                log("[HTML] ✗ Single page failed content filter");
                stats_.filtered_out++;
            }
            
            // If crawling enabled and we have HTML, extract and fetch article links
            if (should_crawl && is_html && entries.size() < static_cast<size_t>(source.fetch_limit)) {
                log("[HTML] Extracting article links for crawling...");
                auto links = LinkExtractor::extractLinks(raw_html, source.url);
                log("[HTML] Found " + std::to_string(links.size()) + " links");
                
                // Filter links based on source patterns
                std::vector<std::string> valid_links;
                size_t urls_skipped_duplicate = 0;
                for (const auto& link : links) {
                    // Pre-fetch URL duplicate check - skip URLs we already have
                    if (stateManager_ && stateManager_->hasCollectedUrl(link)) {
                        urls_skipped_duplicate++;
                        continue;  // Don't waste time fetching known URLs
                    }
                    
                    // Apply path patterns if specified
                    bool matches_pattern = source.filter.path_patterns.empty();
                    for (const auto& pattern : source.filter.path_patterns) {
                        if (link.find(pattern) != std::string::npos) {
                            matches_pattern = true;
                            break;
                        }
                    }
                    
                    // Check exclude patterns
                    bool excluded = false;
                    for (const auto& pattern : source.filter.exclude_patterns) {
                        if (link.find(pattern) != std::string::npos) {
                            excluded = true;
                            break;
                        }
                    }
                    
                    if (matches_pattern && !excluded) {
                        valid_links.push_back(link);
                    }
                    
                    // Stop after we have enough candidate URLs
                    if (valid_links.size() >= static_cast<size_t>(source.fetch_limit - entries.size())) {
                        break;
                    }
                }
                
                if (urls_skipped_duplicate > 0) {
                    log("[HTML] Skipped " + std::to_string(urls_skipped_duplicate) + " already-collected URLs");
                }
                
                log("[HTML] Crawling " + std::to_string(std::min(valid_links.size(), 
                    static_cast<size_t>(source.fetch_limit - entries.size()))) + " article pages...");
                
                // Fetch article pages with error tracking and timeout
                int consecutive_download_failures = 0;  // actual network/download failures only
                int consecutive_filter_misses = 0;       // content filter rejections (not errors)
                const int MAX_CONSECUTIVE_DOWNLOAD_FAILURES = 10;
                const int MAX_CONSECUTIVE_FILTER_MISSES = 20;  // non-article pages are expected
                const int MAX_CRAWL_TIME_SECONDS = 180;  // 3 minutes max per source
                auto crawl_start = std::chrono::steady_clock::now();
                
                for (size_t i = 0; i < valid_links.size(); ++i) {
                    if (entries.size() >= static_cast<size_t>(source.fetch_limit)) break;
                    if (consecutive_download_failures >= MAX_CONSECUTIVE_DOWNLOAD_FAILURES) {
                        log("[HTML] ⚠ Stopping crawl - too many consecutive download failures");
                        break;
                    }
                    if (consecutive_filter_misses >= MAX_CONSECUTIVE_FILTER_MISSES) {
                        log("[HTML] ⚠ Stopping crawl - too many consecutive content filter misses");
                        break;
                    }
                    
                    // Check for timeout
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::steady_clock::now() - crawl_start).count();
                    if (elapsed > MAX_CRAWL_TIME_SECONDS) {
                        log("[HTML] ⚠ Stopping crawl - max time limit reached (" + 
                            std::to_string(elapsed) + "s)");
                        break;
                    }
                    
                    const auto& article_url = valid_links[i];
                    
                    try {
                        // Pre-fetch URL check for crawled article pages
                        if (stateManager_ && stateManager_->hasCollectedUrl(article_url)) {
                            continue;
                        }
                        
                        std::string article_html;
                        // Rate limit between requests to avoid server bans
                        if (i > 0 && config_.rate_limit_delay_ms > 0) {
                            std::this_thread::sleep_for(
                                std::chrono::milliseconds(config_.rate_limit_delay_ms));
                        }
                        
                        if (downloader_->downloadToMemory(article_url, article_html, 10 * 1024 * 1024)) {
                            auto article_extracted = HTMLExtractor::extract(article_html);
                            
                            if (article_extracted.has_article_content && 
                                !article_extracted.main_text.empty() &&
                                source.filter.passes(article_extracted.main_text)) {
                                
                                std::string article_content = article_extracted.main_text;
                                if (!article_extracted.title.empty()) {
                                    article_content = article_extracted.title + "\n\n" + article_content;
                                }
                                
                                RawDataEntry article_entry;
                                article_entry.content = std::move(article_content);
                                article_entry.source_url = article_url;
                                article_entry.title = article_extracted.title;
                                article_entry.source_name = source.name;
                                article_entry.source_type = source.source_type;
                                article_entry.source_priority = source.priority;
                                article_entry.fetch_date = getCurrentUnixTime();
                                
                                entries.push_back(std::move(article_entry));
                                consecutive_download_failures = 0;
                                consecutive_filter_misses = 0;
                            } else {
                                // Content filter miss — page downloaded OK but wasn't useful content
                                consecutive_filter_misses++;
                                consecutive_download_failures = 0; // download succeeded
                            }
                        } else {
                            consecutive_download_failures++;
                        }
                    } catch (const std::exception& e) {
                        logError("[HTML] Article fetch error: " + std::string(e.what()));
                        consecutive_download_failures++;
                    }
                }
                
                log("[HTML] ✓ Crawled total: " + std::to_string(entries.size()) + " pages");
            }
        } else {
            log("[HTML] ✗ Failed to download single page");
        }
    } catch (const std::exception& e) {
        logError("[HTML] Exception in fetchCustom: " + std::string(e.what()));
    }
    
    return entries;
}

std::string WebDataCollector::httpGet(const std::string& url,
                                     const std::unordered_map<std::string, std::string>& headers) {
    if (!curl_handle_) {
        log("ERROR: CURL handle not initialized");
        return "";
    }

    std::string response;
    
    try {
        // Reset CURL handle to clean state
        curl_easy_reset(curl_handle_);
        
        curl_easy_setopt(curl_handle_, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl_handle_, CURLOPT_WRITEFUNCTION, curlWriteCallback);
        curl_easy_setopt(curl_handle_, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl_handle_, CURLOPT_TIMEOUT, static_cast<long>(config_.timeout_seconds));
        curl_easy_setopt(curl_handle_, CURLOPT_CONNECTTIMEOUT, 10L);  // 10 second connect timeout
        curl_easy_setopt(curl_handle_, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl_handle_, CURLOPT_MAXREDIRS, 5L);  // Limit redirects
        curl_easy_setopt(curl_handle_, CURLOPT_USERAGENT, config_.user_agent.c_str());
        curl_easy_setopt(curl_handle_, CURLOPT_SSL_VERIFYPEER, 0L);
        curl_easy_setopt(curl_handle_, CURLOPT_SSL_VERIFYHOST, 0L);
        curl_easy_setopt(curl_handle_, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(curl_handle_, CURLOPT_LOW_SPEED_TIME, 30L);  // Abort if slower than low speed for 30s
        curl_easy_setopt(curl_handle_, CURLOPT_LOW_SPEED_LIMIT, 100L);  // 100 bytes/sec minimum

        // Add custom headers
        struct curl_slist* header_list = nullptr;
        for (const auto& [key, value] : headers) {
            std::string header = key + ": " + value;
            header_list = curl_slist_append(header_list, header.c_str());
        }
        
        if (header_list) {
            curl_easy_setopt(curl_handle_, CURLOPT_HTTPHEADER, header_list);
        }

        CURLcode res = curl_easy_perform(curl_handle_);
        
        if (header_list) {
            curl_slist_free_all(header_list);
        }

        if (res != CURLE_OK) {
            log("ERROR: CURL request failed: " + std::string(curl_easy_strerror(res)));
            response.clear();
            return "";
        }
        
        // Check HTTP response code
        long http_code = 0;
        curl_easy_getinfo(curl_handle_, CURLINFO_RESPONSE_CODE, &http_code);
        if (http_code >= 400) {
            log("ERROR: HTTP error " + std::to_string(http_code) + " for: " + url);
            response.clear();
            return "";
        }

        log("HTTP GET: " + url + " (" + std::to_string(response.length()) + " bytes)");
        
    } catch (const std::exception& e) {
        log("ERROR: Exception in httpGet: " + std::string(e.what()));
        response.clear();
    } catch (...) {
        log("ERROR: Unknown exception in httpGet");
        response.clear();
    }
    
    return response;
}

void WebDataCollector::applyRateLimit(const std::string& domain) {
    // Extract domain from URL
    std::string extracted_domain = domain;
    size_t pos = domain.find("://");
    if (pos != std::string::npos) {
        extracted_domain = domain.substr(pos + 3);
        pos = extracted_domain.find("/");
        if (pos != std::string::npos) {
            extracted_domain = extracted_domain.substr(0, pos);
        }
    }
    
    auto now = std::chrono::steady_clock::now();
    auto it = last_request_time_.find(extracted_domain);
    
    if (it != last_request_time_.end()) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second);
        if (elapsed.count() < config_.rate_limit_delay_ms) {
            int sleep_ms = config_.rate_limit_delay_ms - elapsed.count();
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        }
    }
    
    last_request_time_[extracted_domain] = std::chrono::steady_clock::now();
}

void WebDataCollector::log(const std::string& message, const std::string& level) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    std::string log_msg = "[" + level + "] " + message;
    
    if (config_.verbose) {
        std::cout << log_msg << std::endl;
    }
    
    if (log_file_.is_open()) {
        log_file_ << getCurrentTimestamp() << " " << log_msg << std::endl;
        log_file_.flush();
    }
}

void WebDataCollector::logError(const std::string& error) {
    log(error, "ERROR");
    stats_.errors.push_back(error);
}

void WebDataCollector::logWarning(const std::string& warning) {
    log(warning, "WARN");
    stats_.warnings.push_back(warning);
}

std::string WebDataCollector::getCurrentTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

uint64_t WebDataCollector::getCurrentUnixTime() const {
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
}

float WebDataCollector::updateProgress(size_t current, size_t total) {
    float percent = 0.0f;
    if (total > 0) {
        percent = (static_cast<float>(current) / total) * 100.0f;
    }
    
    if (config_.verbose && total > 0) {
        std::cout << "\rProgress: " << current << "/" << total 
                  << " (" << std::fixed << std::setprecision(1) << percent << "%)" << std::flush;
        if (current == total) {
            std::cout << std::endl;
        }
        
        // Call progress callback if set
        if (progress_callback_) {
            progress_callback_(percent);
        }
    }
    
    return percent;
}

void WebDataCollector::addSource(const DataSource& source) {
    DataSource adjustedSource = source;
    
    // Ensure fetcher_type is correctly detected from URL
    adjustedSource.autoDetectFetcher();
    
    // Apply dynamic fetch_limit based on crawl_depth
    adjustedSource.applyDynamicFetchLimit();
    
    config_.sources.push_back(adjustedSource);
    log("Added source: " + source.name + 
        " (fetcher=" + fetcherTypeToString(adjustedSource.fetcher_type) +
        ", crawl_depth=" + std::to_string(adjustedSource.crawl_depth) + 
        ", fetch_limit=" + std::to_string(adjustedSource.fetch_limit) + ")");
}

void WebDataCollector::removeSource(const std::string& source_name) {
    auto it = std::remove_if(config_.sources.begin(), config_.sources.end(),
                            [&](const DataSource& s) { return s.name == source_name; });
    if (it != config_.sources.end()) {
        config_.sources.erase(it, config_.sources.end());
        log("Removed source: " + source_name);
    }
}

void WebDataCollector::enableSource(const std::string& source_name, bool enable) {
    for (auto& source : config_.sources) {
        if (source.name == source_name) {
            source.enabled = enable;
            log((enable ? "Enabled" : "Disabled") + std::string(" source: ") + source_name);
            break;
        }
    }
}

bool WebDataCollector::saveToJsonl(const std::string& output_path) {
    std::ofstream file(output_path);
    if (!file.is_open()) {
        logError("Failed to open output file: " + output_path);
        return false;
    }
    
    for (const auto& entry : collected_data_) {
        // Serialize entry to JSON line
        nlohmann::json j;
        j["content"] = entry.content;
        j["source_url"] = entry.source_url;
        j["source_type"] = sourceTypeToString(entry.source_type);
        j["author"] = entry.author;
        j["title"] = entry.title;
        j["publish_date"] = entry.publish_date;
        j["fetch_date"] = entry.fetch_date;
        
        // Add metadata if present (struct stores it as `metadata_json`)
        if (!entry.metadata_json.empty()) {
            j["metadata"] = entry.metadata_json;
        }
        
        file << j.dump() << std::endl;
    }
    
    file.close();
    log("Saved " + std::to_string(collected_data_.size()) + " entries to " + output_path);
    return true;
}

bool WebDataCollector::saveToFlatBuffer(const std::string& output_path) {
    log("Saving to FlatBuffer: " + output_path);
    
    // Initialize FlatBuffer builder
    flatbuffers::FlatBufferBuilder builder(1024 * 1024 * 10); // 10MB initial
    
    // Build training examples
    std::vector<flatbuffers::Offset<GRIMWebTraining::TrainingExample>> examples;
    examples.reserve(collected_data_.size());
    
    uint64_t now = std::chrono::system_clock::now().time_since_epoch().count();
    
    for (const auto& entry : collected_data_) {
        // Create source info
        auto url_offset = builder.CreateString(entry.source_url);
        auto domain_offset = builder.CreateString("");  // TODO: Extract domain from URL
        auto author_offset = builder.CreateString(entry.author);
        auto title_offset = builder.CreateString(entry.title);
        
        // Map C++ FetcherType enum to FlatBuffer SourceType enum
        GRIMWebTraining::SourceType fb_source_type;
        switch (entry.source_type) {
            case FetcherType::NEWS_API: fb_source_type = GRIMWebTraining::SourceType_NEWS_API; break;
            case FetcherType::GITHUB_API: fb_source_type = GRIMWebTraining::SourceType_GITHUB; break;
            case FetcherType::TECH_DOCS: fb_source_type = GRIMWebTraining::SourceType_TECH_DOCS; break;
            case FetcherType::WIKIPEDIA_API: fb_source_type = GRIMWebTraining::SourceType_WIKIPEDIA; break;
            case FetcherType::ARXIV_API: fb_source_type = GRIMWebTraining::SourceType_ARXIV; break;
            case FetcherType::STACKOVERFLOW_API: fb_source_type = GRIMWebTraining::SourceType_STACKOVERFLOW; break;
            case FetcherType::REDDIT_API: fb_source_type = GRIMWebTraining::SourceType_REDDIT; break;
            case FetcherType::HTML_CRAWL: fb_source_type = GRIMWebTraining::SourceType_CUSTOM; break;
            default: fb_source_type = GRIMWebTraining::SourceType_UNKNOWN; break;
        }
        
        auto source_info = GRIMWebTraining::CreateSourceInfo(
            builder,
            url_offset,
            fb_source_type,
            domain_offset,
            author_offset,
            title_offset,
            entry.publish_date,
            entry.fetch_date ? entry.fetch_date : now
        );
        
        // Create verification info (basic for now)
        auto verification = GRIMWebTraining::CreateVerificationInfo(
            builder,
            GRIMWebTraining::VerificationStatus_UNVERIFIED,
            static_cast<float>(entry.source_priority) / 10.0f // Convert priority to score
        );
        
        // Create content metadata
        uint32_t word_count = 0;
        bool in_word = false;
        for (char c : entry.content) {
            if (std::isspace(c)) in_word = false;
            else if (!in_word) { word_count++; in_word = true; }
        }
        
        auto lang_offset = builder.CreateString("en");
        auto metadata = GRIMWebTraining::CreateContentMetadata(
            builder,
            static_cast<uint32_t>(entry.content.length()),
            word_count,
            0, 0, // sentence/paragraph count
            lang_offset
        );
        
        // Create training example
        auto raw_text_offset = builder.CreateString(entry.content);
        auto example_id_offset = builder.CreateString(entry.generateId());
        
        // Populate Q/A fields when available
        auto input_text_offset = entry.question.empty()
            ? flatbuffers::Offset<flatbuffers::String>(0)
            : builder.CreateString(entry.question);
        auto target_text_offset = entry.answer.empty()
            ? flatbuffers::Offset<flatbuffers::String>(0)
            : builder.CreateString(entry.answer);
        
        auto example = GRIMWebTraining::CreateTrainingExample(
            builder,
            raw_text_offset,
            0, // cleaned_text
            0, // token_ids (will be added during preprocessing)
            0, // token_count
            source_info,
            verification,
            metadata,
            input_text_offset, target_text_offset, 0, 0, // input/target/context/prompt
            example_id_offset,
            now
        );
        
        examples.push_back(example);
    }
    
    // Create session info
    auto session_id_offset = builder.CreateString("session_" + std::to_string(now));
    auto examples_offset = builder.CreateVector(examples);
    
    // Create collection session - match FlatBuffer schema signature
    auto session = GRIMWebTraining::CreateCollectionSession(
        builder,
        session_id_offset,  // session_id
        now,                // start_time
        now,                // end_time
        0,                  // source_urls
        0,                  // num_sources
        0,                  // total_fetched
        0,                  // successful
        0,                  // failed
        0,                  // total_verified
        0,                  // verification_passed
        0,                  // verification_failed
        0.0f,               // avg_reliability_score
        0,                  // config_json
        0,                  // errors
        0                   // warnings
    );
    
    // Finish buffer
    builder.Finish(session);
    
    // Write to file
    std::ofstream out(output_path, std::ios::binary);
    if (!out.is_open()) {
        log("ERROR: Cannot open output file: " + output_path);
        return false;
    }
    
    out.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
    out.close();
    
    log("Saved " + std::to_string(collected_data_.size()) + " entries to FlatBuffer (" + 
        std::to_string(builder.GetSize() / 1024.0 / 1024.0) + " MB)");
    
    return true;
}

bool WebDataCollector::saveCheckpoint(const std::string& checkpoint_path) {
    try {
        // Create data directory if it doesn't exist
        std::filesystem::path path(checkpoint_path);
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }
        
        // Build FlatBuffer
        flatbuffers::FlatBufferBuilder builder(1024 * 1024); // 1MB initial
        std::vector<flatbuffers::Offset<GRIMCheckpoint::CheckpointEntry>> entries;
        
        // Convert collected data to FlatBuffer entries
        for (const auto& entry : collected_data_) {
            auto content = builder.CreateString(entry.content);
            auto source_url = builder.CreateString(entry.source_url);
            auto source_type = builder.CreateString(sourceTypeToString(entry.source_type));
            
            // Calculate content hash for deduplication
            uint64_t content_hash = std::hash<std::string>{}(entry.content);
            
            auto fb_entry = GRIMCheckpoint::CreateCheckpointEntry(builder,
                content,
                source_url,
                source_type,
                0.7f, // default reliability
                static_cast<uint64_t>(std::time(nullptr)),
                content_hash
            );
            entries.push_back(fb_entry);
        }
        
        // Create checkpoint
        auto entries_vector = builder.CreateVector(entries);
        auto session_id = builder.CreateString("session_" + std::to_string(std::time(nullptr)));
        
        auto checkpoint = GRIMCheckpoint::CreateCheckpoint(builder,
            entries_vector,
            static_cast<uint32_t>(collected_data_.size()),
            static_cast<uint32_t>(collected_data_.size()),
            static_cast<uint64_t>(std::time(nullptr)),
            session_id
        );
        
        builder.Finish(checkpoint);
        
        // Write to binary FlatBuffer file
        std::ofstream file(checkpoint_path, std::ios::binary);
        if (!file.is_open()) {
            logError("Failed to open checkpoint file: " + checkpoint_path);
            return false;
        }
        
        file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
        file.close();
        
        if (config_.verbose) {
            std::cout << "[Checkpoint] Saved " << collected_data_.size() 
                     << " entries to " << checkpoint_path << " (" 
                     << (builder.GetSize() / 1024) << " KB FlatBuffer)" << std::endl;
        }
        
        return true;
    } catch (const std::exception& e) {
        logError("Exception saving checkpoint: " + std::string(e.what()));
        return false;
    }
}

//======================================================//
// Load checkpoint and merge into collected_data_
//======================================================//
bool WebDataCollector::loadCheckpoint(const std::string& checkpoint_path) {
    try {
        std::ifstream file(checkpoint_path, std::ios::binary);
        if (!file.is_open()) {
            if (config_.verbose) {
                std::cerr << "[ERROR] Failed to open checkpoint file: " << checkpoint_path << std::endl;
            }
            return false;
        }
        
        // Read entire FlatBuffer file
        file.seekg(0, std::ios::end);
        size_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<uint8_t> buffer(fileSize);
        file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
        file.close();
        
        // Parse FlatBuffer
        auto checkpoint = GRIMCheckpoint::GetCheckpoint(buffer.data());
        if (!checkpoint || !checkpoint->entries()) {
            if (config_.verbose) {
                std::cerr << "[ERROR] Invalid checkpoint format: " << checkpoint_path << std::endl;
            }
            return false;
        }
        
        size_t loaded_count = 0;
        
        // Convert FlatBuffer entries to RawDataEntry
        for (const auto* fb_entry : *checkpoint->entries()) {
            if (!fb_entry || !fb_entry->content() || !fb_entry->source_url()) continue;
            
            RawDataEntry entry;
            entry.content = fb_entry->content()->str();
            entry.source_url = fb_entry->source_url()->str();
            entry.source_type = sourceTypeFromString(fb_entry->source_type() ? fb_entry->source_type()->str() : "unknown");
            entry.source_priority = static_cast<int>(fb_entry->reliability_score() * 10.0f); // Convert back to priority
            entry.fetch_date = fb_entry->fetch_timestamp();
            entry.source_name = "checkpoint"; // Not stored in minimal schema
            
            // Add to collected_data_ (thread-safe)
            {
                std::lock_guard<std::mutex> lock(data_mutex_);
                collected_data_.push_back(entry);
            }
            loaded_count++;
        }
        
        if (config_.verbose) {
            std::cout << "[INFO] Loaded " << loaded_count << " entries from FlatBuffer checkpoint: " 
                      << checkpoint_path << " (" << (fileSize / 1024) << " KB)" << std::endl;
        }
        return true;
        
    } catch (const std::exception& e) {
        if (config_.verbose) {
            std::cerr << "[ERROR] Exception loading checkpoint: " << e.what() << std::endl;
        }
        return false;
    }
}

//======================================================//
// Merge multiple checkpoint files
//======================================================//
bool WebDataCollector::mergeCheckpoints(const std::vector<std::string>& checkpoint_paths) {
    size_t total_loaded = 0;
    size_t failed_count = 0;
    std::vector<std::string> successfully_loaded;
    
    for (const auto& path : checkpoint_paths) {
        size_t before = collected_data_.size();
        if (loadCheckpoint(path)) {
            size_t loaded = collected_data_.size() - before;
            total_loaded += loaded;
            successfully_loaded.push_back(path);
        } else {
            failed_count++;
        }
    }
    
    if (config_.verbose) {
        std::cout << "[INFO] Merged checkpoints: " << total_loaded << " total entries, " 
                  << failed_count << " files failed" << std::endl;
    }
    
    // Delete successfully loaded checkpoint files to avoid re-processing
    for (const auto& path : successfully_loaded) {
        try {
            if (std::remove(path.c_str()) == 0) {
                if (config_.verbose) {
                    std::cout << "[INFO] Deleted checkpoint file: " << path << std::endl;
                }
            } else {
                if (config_.verbose) {
                    std::cerr << "[WARN] Failed to delete checkpoint file: " << path << std::endl;
                }
            }
        } catch (const std::exception& e) {
            if (config_.verbose) {
                std::cerr << "[WARN] Exception deleting checkpoint: " << e.what() << std::endl;
            }
        }
    }
    
    return failed_count == 0;
}

//======================================================//
// State Manager Methods - Persistent deduplication tracking
//======================================================//

void WebDataCollector::initializeStateManager(const std::string& stateDir) {
    stateManager_ = std::make_unique<DataCollection::CollectionStateManager>(stateDir);
    log("State manager initialized at: " + stateDir);
    log("Loaded " + std::to_string(stateManager_->getTotalUniqueUrls()) + " unique URLs");
    log("Loaded " + std::to_string(stateManager_->getTotalUniqueContent()) + " unique content hashes");
}

bool WebDataCollector::hasCollectedUrl(const std::string& url) const {
    if (!stateManager_) return false;
    return stateManager_->hasCollectedUrl(url);
}

bool WebDataCollector::hasSeenContent(const std::string& content) const {
    if (!stateManager_) return false;
    return stateManager_->hasSeenContent(content);
}

void WebDataCollector::markUrlCollected(const std::string& url, const std::string& sourceType) {
    if (stateManager_) {
        stateManager_->markUrlCollected(url, sourceType);
    }
}

void WebDataCollector::markContentSeen(const std::string& content) {
    if (stateManager_) {
        stateManager_->markContentSeen(content);
    }
}

size_t WebDataCollector::getUniqueUrlCount() const {
    if (!stateManager_) return 0;
    return stateManager_->getTotalUniqueUrls();
}

size_t WebDataCollector::getUniqueContentCount() const {
    if (!stateManager_) return 0;
    return stateManager_->getTotalUniqueContent();
}

void WebDataCollector::saveCollectionState() {
    if (stateManager_) {
        stateManager_->saveState();
        log("Collection state saved");
    }
}

void WebDataCollector::clearCollectionState() {
    if (stateManager_) {
        stateManager_->clear();
        log("Collection state cleared");
    }
}

} // namespace Training
} // namespace GRIM

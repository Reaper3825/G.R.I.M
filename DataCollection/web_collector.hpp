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
//  
//  Author: GRIM Development Team
//  Date: November 4, 2025
//  Version: 2.0.0
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
#include "../../control/ai_config_paths.hpp"

namespace GRIM {
namespace Training {

// Import types from streaming/crawler
using CrawlConfig = AsyncCrawler::CrawlConfig;

//======================================================//
//  Source Type Enum
//======================================================//

enum class SourceType {
    UNKNOWN = 0,
    NEWS_API = 1,
    GITHUB = 2,
    TECH_DOCS = 3,
    WIKIPEDIA = 4,
    ARXIV = 5,
    STACKOVERFLOW = 6,
    REDDIT = 7,
    CUSTOM = 8,
    GUTENBERG = 9,
    OPEN_BOOKS = 10,
    JSTOR_OA = 11,
    PHILOSOPHY = 12,
    CLASSICAL_TEXTS = 13,
    ACADEMIC_PAPERS = 14,
    LOGIC = 15,
    THEORETICAL_REASONING = 16,
    THEORETICAL_SCIENCE = 17,
    GRAMMAR = 18,
    RHETORIC = 19,
    LINGUISTICS = 20,
    SPEECH_CORPUS = 21,
    HARDWARE_SPECS = 22,
    ERUDITE_WRITING = 23
};

inline SourceType sourceTypeFromString(const std::string& str) {
    if (str == "news_api") return SourceType::NEWS_API;
    if (str == "github") return SourceType::GITHUB;
    if (str == "tech_docs") return SourceType::TECH_DOCS;
    if (str == "wikipedia") return SourceType::WIKIPEDIA;
    if (str == "arxiv") return SourceType::ARXIV;
    if (str == "stackoverflow") return SourceType::STACKOVERFLOW;
    if (str == "reddit") return SourceType::REDDIT;
    if (str == "gutenberg") return SourceType::GUTENBERG;
    if (str == "open_books") return SourceType::OPEN_BOOKS;
    if (str == "jstor_oa") return SourceType::JSTOR_OA;
    if (str == "philosophy") return SourceType::PHILOSOPHY;
    if (str == "classical_texts") return SourceType::CLASSICAL_TEXTS;
    if (str == "academic_papers") return SourceType::ACADEMIC_PAPERS;
    if (str == "logic") return SourceType::LOGIC;
    if (str == "theoretical_reasoning") return SourceType::THEORETICAL_REASONING;
    if (str == "theoretical_science") return SourceType::THEORETICAL_SCIENCE;
    if (str == "grammar") return SourceType::GRAMMAR;
    if (str == "rhetoric") return SourceType::RHETORIC;
    if (str == "linguistics") return SourceType::LINGUISTICS;
    if (str == "speech_corpus") return SourceType::SPEECH_CORPUS;
    if (str == "hardware_specs") return SourceType::HARDWARE_SPECS;
    if (str == "erudite_writing") return SourceType::ERUDITE_WRITING;
    if (str == "custom") return SourceType::CUSTOM;
    return SourceType::UNKNOWN;
}

inline std::string sourceTypeToString(SourceType type) {
    switch (type) {
        case SourceType::NEWS_API: return "news_api";
        case SourceType::GITHUB: return "github";
        case SourceType::TECH_DOCS: return "tech_docs";
        case SourceType::WIKIPEDIA: return "wikipedia";
        case SourceType::ARXIV: return "arxiv";
        case SourceType::STACKOVERFLOW: return "stackoverflow";
        case SourceType::REDDIT: return "reddit";
        case SourceType::CUSTOM: return "custom";
        case SourceType::GUTENBERG: return "gutenberg";
        case SourceType::OPEN_BOOKS: return "open_books";
        case SourceType::JSTOR_OA: return "jstor_oa";
        case SourceType::PHILOSOPHY: return "philosophy";
        case SourceType::CLASSICAL_TEXTS: return "classical_texts";
        case SourceType::ACADEMIC_PAPERS: return "academic_papers";
        case SourceType::LOGIC: return "logic";
        case SourceType::THEORETICAL_REASONING: return "theoretical_reasoning";
        case SourceType::THEORETICAL_SCIENCE: return "theoretical_science";
        case SourceType::GRAMMAR: return "grammar";
        case SourceType::RHETORIC: return "rhetoric";
        case SourceType::LINGUISTICS: return "linguistics";
        case SourceType::SPEECH_CORPUS: return "speech_corpus";
        case SourceType::HARDWARE_SPECS: return "hardware_specs";
        case SourceType::ERUDITE_WRITING: return "erudite_writing";
        default: return "unknown";
    }
}

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
    SourceType source_type = SourceType::UNKNOWN;
    bool enabled = true;
    int priority = 5;  // 1-10, higher = more trusted
    
    bool requires_auth = false;
    std::string api_key_env;  // Environment variable name
    std::string api_key;      // Actual key (loaded from env)
    
    int fetch_limit = 100;
    int crawl_depth = 2;  // Default to depth 2 for article crawling
    
    ContentFilter filter;
    
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
    // Deeper crawls discover exponentially more pages, so we need higher limits
    void applyDynamicFetchLimit() {
        if (fetch_limit == 100) {  // Only adjust if still at default
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
    SourceType source_type = SourceType::UNKNOWN;
    int source_priority = 5;
    
    std::string author;
    std::string title;
    uint64_t publish_date = 0;
    uint64_t fetch_date = 0;
    
    std::string metadata_json;  // Additional metadata as JSON
    
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
    int timeout_seconds = 15;  // Faster timeout (was 30)
    int rate_limit_delay_ms = 100;  // 10x faster (was 1000)
    int max_retries = 3;
    int max_concurrent_requests = 25;  // 10x more parallel (was 5) - OPTIMIZED FOR 128GB RAM
    
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
    
private:
    CollectorConfig config_;
    CollectionStats stats_;
    std::vector<RawDataEntry> collected_data_;
    
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
    // Load checkpoint directory from ai_config
    config_.output_dir = GRIM::Config::getCheckpointDir();
    
    // Load collector log path from ai_config
    config_.log_file = GRIM::Config::getCollectorLogPath();
    
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl_handle_ = curl_easy_init();
    if (!curl_handle_) {
        logError("Failed to initialize CURL");
    }
    
    // Initialize high-performance components
    downloader_ = std::make_unique<StreamingDownloader>();
    crawler_ = std::make_unique<AsyncCrawler>();
    
    // Open log file if configured
    if (!config_.log_file.empty()) {
        log_file_.open(config_.log_file, std::ios::app);
        if (log_file_.is_open()) {
            log("WebDataCollector initialized (default constructor)");
        }
    }
}

WebDataCollector::WebDataCollector(const CollectorConfig& config)
    : config_(config), stats_(), curl_handle_(nullptr)
{
    // If output_dir not set in config, use ai_config default
    if (config_.output_dir.empty()) {
        config_.output_dir = GRIM::Config::getCheckpointDir();
    }
    
    // If log_file not set in config, use ai_config default
    if (config_.log_file.empty()) {
        config_.log_file = GRIM::Config::getCollectorLogPath();
    }
    
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl_handle_ = curl_easy_init();
    if (!curl_handle_) {
        logError("Failed to initialize CURL");
    }
    
    // Initialize high-performance components
    downloader_ = std::make_unique<StreamingDownloader>();
    crawler_ = std::make_unique<AsyncCrawler>();
    
    if (!config_.log_file.empty()) {
        log_file_.open(config_.log_file, std::ios::app);
        if (log_file_.is_open()) {
            log("WebDataCollector initialized");
        }
    }
}

WebDataCollector::~WebDataCollector() {
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
#ifdef HAVE_NLOHMANN_JSON
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
                source.source_type = sourceTypeFromString(source_json.value("source_type", "unknown"));
                source.enabled = source_json.value("enabled", true);
                source.priority = source_json.value("priority", 5);
                source.requires_auth = source_json.value("requires_auth", false);
                source.api_key_env = source_json.value("api_key_env", "");
                source.fetch_limit = source_json.value("fetch_limit", 100);
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
#else
    logError("JSON support not available (HAVE_NLOHMANN_JSON not defined)");
    return false;
#endif
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
    
    // Process each source
    progress_total_ = stats_.enabled_sources;
    progress_current_ = 0;
    
    for (const auto& source : config_.sources) {
        if (!source.enabled) {
            log("Skipping disabled source: " + source.name);
            continue;
        }
        
        log("\n--- Fetching from: " + source.name + " ---");
        log("URL: " + source.url);
        log("Type: " + sourceTypeToString(source.source_type));
        log("Fetch limit: " + std::to_string(source.fetch_limit) + ", Crawl depth: " + std::to_string(source.crawl_depth));
        
        try {
            auto start_time = std::chrono::steady_clock::now();
            std::vector<RawDataEntry> entries = fetchFromSource(source);
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
            
            {
                std::lock_guard<std::mutex> lock(data_mutex_);
                collected_data_.insert(collected_data_.end(), entries.begin(), entries.end());
                stats_.successful += entries.size();
                stats_.per_source_type[source.source_type] += entries.size();
            }
            
            if (entries.size() > 0) {
                log("[SUCCESS] " + source.name + ": " + std::to_string(entries.size()) + " entries in " + std::to_string(duration) + "s");
                // Log sample content length
                size_t avg_length = 0;
                for (const auto& entry : entries) {
                    avg_length += entry.content.length();
                }
                avg_length /= entries.size();
                log("  Average content length: " + std::to_string(avg_length) + " chars");
            } else {
                logWarning("[EMPTY] " + source.name + ": No entries collected in " + std::to_string(duration) + "s");
            }
            
        } catch (const std::exception& e) {
            logError("[FAILED] " + source.name + ": " + e.what());
            stats_.failed++;
        } catch (...) {
            logError("[FAILED] " + source.name + ": Unknown error");
            stats_.failed++;
        }
        
        progress_current_++;
        updateProgress(progress_current_, progress_total_);
        
        // Save checkpoint every 5 sources
        if (progress_current_ % 5 == 0) {
            std::string checkpoint_file = "data/checkpoint_" + std::to_string(progress_current_) + ".ckpt";
            if (saveCheckpoint(checkpoint_file)) {
                log("Checkpoint saved: " + checkpoint_file + " (" + std::to_string(collected_data_.size()) + " entries)");
            }
        }
        
        // Apply rate limiting
        try {
            applyRateLimit(source.url);
        } catch (...) {
            // Ignore rate limit errors
        }
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
    std::string type_name;
    switch (source.source_type) {
        case SourceType::GITHUB:
            type_name = "GITHUB_API";
            log("[SOURCE] Starting " + type_name + ": " + source.name);
            return fetchGitHub(source);
        case SourceType::ARXIV:
            type_name = "ARXIV_API";
            log("[SOURCE] Starting " + type_name + ": " + source.name);
            return fetchArXiv(source);
        case SourceType::WIKIPEDIA:
            type_name = "WIKIPEDIA_API";
            log("[SOURCE] Starting " + type_name + ": " + source.name);
            return fetchWikipedia(source);
        case SourceType::STACKOVERFLOW:
            type_name = "STACKOVERFLOW_API";
            log("[SOURCE] Starting " + type_name + ": " + source.name);
            return fetchStackOverflow(source);
        case SourceType::REDDIT:
            type_name = "REDDIT_API";
            log("[SOURCE] Starting " + type_name + ": " + source.name);
            return fetchReddit(source);
        case SourceType::NEWS_API:
            type_name = "NEWS_API";
            log("[SOURCE] Starting " + type_name + ": " + source.name);
            return fetchNewsAPI(source);
        case SourceType::TECH_DOCS:
            type_name = "TECH_DOCS";
            log("[SOURCE] Starting " + type_name + ": " + source.name);
            return fetchTechDocs(source);
        case SourceType::CUSTOM:
        case SourceType::GUTENBERG:
        case SourceType::OPEN_BOOKS:
        case SourceType::JSTOR_OA:
        case SourceType::PHILOSOPHY:
        case SourceType::CLASSICAL_TEXTS:
        case SourceType::ACADEMIC_PAPERS:
        case SourceType::LOGIC:
        case SourceType::THEORETICAL_REASONING:
        case SourceType::THEORETICAL_SCIENCE:
        case SourceType::GRAMMAR:
        case SourceType::RHETORIC:
        case SourceType::LINGUISTICS:
        case SourceType::SPEECH_CORPUS:
        case SourceType::HARDWARE_SPECS:
        case SourceType::ERUDITE_WRITING:
            type_name = "HTML_CRAWL";
            log("[SOURCE] Starting " + type_name + " (depth=" + std::to_string(source.crawl_depth) + "): " + source.name);
            return fetchCustom(source);
        default:
            // All other unknown types should try HTML crawling
            type_name = "HTML_CRAWL_FALLBACK";
            log("[SOURCE] Starting " + type_name + " (unknown type, trying HTML): " + source.name);
            return fetchCustom(source);
    }
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
                RawDataEntry entry;
                entry.content = data.value("description", "") + "\n\n" + 
                               data.value("readme", "");
                entry.source_url = data.value("html_url", source.url);
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
            // Handle array of repositories
            else if (data.is_array()) {
                for (const auto& item : data) {
                    if (entries.size() >= static_cast<size_t>(config_.max_entries_per_source)) break;
                    
                    RawDataEntry entry;
                    entry.content = item.value("description", "") + "\n\n" + 
                                   item.value("readme", "");
                    entry.source_url = item.value("html_url", source.url);
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
        std::string api_url = "http://export.arxiv.org/api/query?search_query=all&start=0&max_results=";
        api_url += std::to_string(std::min(source.fetch_limit, 100));
        
        std::string response;
        if (!downloader_->downloadToMemory(api_url, response, 5 * 1024 * 1024)) {
            logError("Failed to download from ArXiv API");
            return entries;
        }
        
        // Simple XML parsing - extract entries
        size_t pos = 0;
        while ((pos = response.find("<entry>", pos)) != std::string::npos && 
               entries.size() < static_cast<size_t>(source.fetch_limit)) {
            size_t end = response.find("</entry>", pos);
            if (end == std::string::npos) break;
            
            std::string entry_xml = response.substr(pos, end - pos + 8);
            
            // Extract title, summary, id
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
        
        log("Fetched " + std::to_string(entries.size()) + " ArXiv papers");
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
        
        pos = response.find("\"id\":", pos);
        while (pos != std::string::npos && entries.size() < static_cast<size_t>(source.fetch_limit)) {
            pos += 5;
            size_t end = response.find_first_of(",}", pos);
            if (end == std::string::npos) break;
            
            std::string page_id = response.substr(pos, end - pos);
            
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
        
        log("Fetched " + std::to_string(entries.size()) + " Wikipedia articles");
    } catch (const std::exception& e) {
        logError("Exception in fetchWikipedia: " + std::string(e.what()));
    }
    
    return entries;
}

std::vector<RawDataEntry> WebDataCollector::fetchStackOverflow(const DataSource& source) {
    std::vector<RawDataEntry> entries;
    log("Fetching StackOverflow via streaming...");
    
    try {
        std::string api_url = "https://api.stackexchange.com/2.3/questions?order=desc&sort=votes&site=stackoverflow&pagesize=";
        api_url += std::to_string(std::min(source.fetch_limit, 100)) + "&filter=withbody";
        
        std::string response;
        if (!downloader_->downloadToMemory(api_url, response, 10 * 1024 * 1024)) {
            logError("Failed to fetch StackOverflow");
            return entries;
        }
        
        // Simple parsing
        size_t items_pos = response.find("\"items\":[");
        if (items_pos == std::string::npos) return entries;
        
        size_t pos = items_pos + 9;
        while (entries.size() < static_cast<size_t>(source.fetch_limit)) {
            pos = response.find("{", pos);
            if (pos == std::string::npos || pos > response.find("]", items_pos)) break;
            
            size_t obj_end = response.find("}", pos);
            if (obj_end == std::string::npos) break;
            
            std::string obj = response.substr(pos, obj_end - pos + 1);
            
            auto extract = [&](const std::string& field) {
                std::string search = "\"" + field + "\":\"";
                size_t s = obj.find(search);
                if (s == std::string::npos) return std::string();
                s += search.length();
                size_t e = obj.find("\"", s);
                return (e != std::string::npos) ? obj.substr(s, e - s) : std::string();
            };
            
            std::string title = extract("title");
            std::string body = extract("body");
            std::string link = extract("link");
            
            if (!title.empty() && !body.empty()) {
                std::string content = "Question: " + title + "\n\n" + body;
                if (source.filter.passes(content)) {
                    RawDataEntry entry;
                    entry.content = std::move(content);
                    entry.source_url = std::move(link);
                    entry.title = std::move(title);
                    entry.source_name = source.name;
                    entry.source_type = source.source_type;
                    entry.source_priority = source.priority;
                    entry.fetch_date = getCurrentUnixTime();
                    entries.push_back(std::move(entry));
                } else {
                    stats_.filtered_out++;
                }
            }
            
            pos = obj_end + 1;
        }
        
        log("Fetched " + std::to_string(entries.size()) + " StackOverflow questions");
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
                std::string content = "Title: " + title + "\n\n" + selftext;
                if (content.length() > 100 && source.filter.passes(content)) {
                    RawDataEntry entry;
                    entry.content = std::move(content);
                    entry.source_url = "https://www.reddit.com" + permalink;
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
                for (const auto& link : links) {
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
                    
                    // Stop after we have enough
                    if (valid_links.size() >= static_cast<size_t>(source.fetch_limit - entries.size())) {
                        break;
                    }
                }
                
                log("[HTML] Crawling " + std::to_string(std::min(valid_links.size(), 
                    static_cast<size_t>(source.fetch_limit - entries.size()))) + " article pages...");
                
                // Fetch article pages with error tracking and timeout
                int consecutive_failures = 0;
                const int MAX_CONSECUTIVE_FAILURES = 5;
                const int MAX_CRAWL_TIME_SECONDS = 180;  // 3 minutes max per source
                auto crawl_start = std::chrono::steady_clock::now();
                
                for (size_t i = 0; i < valid_links.size(); ++i) {
                    if (entries.size() >= static_cast<size_t>(source.fetch_limit)) break;
                    if (consecutive_failures >= MAX_CONSECUTIVE_FAILURES) {
                        log("[HTML] ⚠ Stopping crawl - too many consecutive failures");
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
                        std::string article_html;
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
                                consecutive_failures = 0; // Reset on success
                            } else {
                                consecutive_failures++;
                            }
                        } else {
                            consecutive_failures++;
                        }
                    } catch (const std::exception& e) {
                        logError("[HTML] Article fetch error: " + std::string(e.what()));
                        consecutive_failures++;
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
    
    // Apply dynamic fetch_limit based on crawl_depth
    adjustedSource.applyDynamicFetchLimit();
    
    config_.sources.push_back(adjustedSource);
    log("Added source: " + source.name + 
        " (crawl_depth=" + std::to_string(adjustedSource.crawl_depth) + 
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
        // TODO: Serialize to JSON line
        // For now, simple output
        file << "{\"content\":\"...\"}" << std::endl;
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
        
        // Map C++ enum to FlatBuffer enum
        GRIMWebTraining::SourceType fb_source_type;
        switch (entry.source_type) {
            case SourceType::NEWS_API: fb_source_type = GRIMWebTraining::SourceType_NEWS_API; break;
            case SourceType::GITHUB: fb_source_type = GRIMWebTraining::SourceType_GITHUB; break;
            case SourceType::TECH_DOCS: fb_source_type = GRIMWebTraining::SourceType_TECH_DOCS; break;
            case SourceType::WIKIPEDIA: fb_source_type = GRIMWebTraining::SourceType_WIKIPEDIA; break;
            case SourceType::ARXIV: fb_source_type = GRIMWebTraining::SourceType_ARXIV; break;
            case SourceType::STACKOVERFLOW: fb_source_type = GRIMWebTraining::SourceType_STACKOVERFLOW; break;
            case SourceType::REDDIT: fb_source_type = GRIMWebTraining::SourceType_REDDIT; break;
            case SourceType::CUSTOM: fb_source_type = GRIMWebTraining::SourceType_CUSTOM; break;
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
        
        auto example = GRIMWebTraining::CreateTrainingExample(
            builder,
            raw_text_offset,
            0, // cleaned_text
            0, // token_ids (will be added during preprocessing)
            0, // token_count
            source_info,
            verification,
            metadata,
            0, 0, 0, 0, // input/target/context/prompt
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

} // namespace Training
} // namespace GRIM

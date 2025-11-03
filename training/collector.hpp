#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <chrono>

namespace grim {
namespace training {

/**
 * @brief Data source configuration for online data collection
 */
struct DataSource {
    std::string url;
    std::string source_type;  // "news_api", "github", "tech_docs", etc.
    bool requires_auth;
    std::string api_key;
    int priority;  // Higher = more trusted
};

/**
 * @brief Metadata for collected raw data
 */
struct RawDataEntry {
    std::string content;
    std::string source_url;
    std::string source_type;
    std::string author;
    std::chrono::system_clock::time_point timestamp;
    std::string metadata_json;  // Additional metadata as JSON
};

/**
 * @brief Configuration for the data collection process
 */
struct CollectorConfig {
    std::vector<DataSource> sources;
    std::string output_dir = "data/raw";
    size_t max_entries_per_source = 100;
    int timeout_seconds = 30;
    bool save_as_jsonl = true;  // If false, saves as .txt
    std::vector<std::string> keyword_filters;  // Optional content filters
};

/**
 * @brief Stage 1: Online Data Collector
 * 
 * Fetches fresh online data from pre-approved sources for potential training.
 * Stores raw text and metadata for further verification and processing.
 */
class Collector {
public:
    Collector();
    explicit Collector(const CollectorConfig& config);
    ~Collector();

    /**
     * @brief Main entry point: Fetch online data from configured sources
     * 
     * Pulls from pre-approved sources (news APIs, tech docs, GitHub READMEs, etc.)
     * and stores raw text with metadata in data/raw/
     * 
     * @return Number of entries successfully collected
     */
    size_t fetch_online_data();

    /**
     * @brief Add a data source to the collector
     */
    void add_source(const DataSource& source);

    /**
     * @brief Load sources from a configuration file
     */
    bool load_sources_from_file(const std::string& config_path);

    /**
     * @brief Set output directory for raw data
     */
    void set_output_dir(const std::string& dir);

    /**
     * @brief Get collection statistics
     */
    struct Stats {
        size_t total_fetched = 0;
        size_t successful = 0;
        size_t failed = 0;
        std::chrono::milliseconds total_time{0};
    };
    Stats get_stats() const;

    /**
     * @brief Clear collected statistics
     */
    void reset_stats();

private:
    /**
     * @brief Fetch data from a single source
     */
    std::vector<RawDataEntry> fetch_from_source(const DataSource& source);

    /**
     * @brief Fetch from News API
     */
    std::vector<RawDataEntry> fetch_news_api(const DataSource& source);

    /**
     * @brief Fetch from GitHub (READMEs, documentation)
     */
    std::vector<RawDataEntry> fetch_github(const DataSource& source);

    /**
     * @brief Fetch from technical documentation sites
     */
    std::vector<RawDataEntry> fetch_tech_docs(const DataSource& source);

    /**
     * @brief Save entries to disk (JSONL or TXT format)
     */
    bool save_entries(const std::vector<RawDataEntry>& entries, const std::string& filename);

    /**
     * @brief Apply keyword filters to content
     */
    bool passes_filters(const std::string& content) const;

    /**
     * @brief Generate unique filename for output
     */
    std::string generate_output_filename(const std::string& source_type) const;

    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace training
} // namespace grim

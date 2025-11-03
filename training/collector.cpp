#include "collector.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace grim {
namespace training {

// PIMPL implementation
class Collector::Impl {
public:
    CollectorConfig config;
    Stats stats;
    CURL* curl_handle = nullptr;

    Impl() {
        curl_global_init(CURL_GLOBAL_DEFAULT);
        curl_handle = curl_easy_init();
    }

    ~Impl() {
        if (curl_handle) {
            curl_easy_cleanup(curl_handle);
        }
        curl_global_cleanup();
    }

    // CURL write callback
    static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
        ((std::string*)userp)->append((char*)contents, size * nmemb);
        return size * nmemb;
    }

    std::string http_get(const std::string& url, const std::string& api_key = "") {
        if (!curl_handle) return "";

        std::string response;
        curl_easy_setopt(curl_handle, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl_handle, CURLOPT_TIMEOUT, config.timeout_seconds);
        curl_easy_setopt(curl_handle, CURLOPT_FOLLOWLOCATION, 1L);

        struct curl_slist* headers = nullptr;
        if (!api_key.empty()) {
            std::string auth_header = "Authorization: Bearer " + api_key;
            headers = curl_slist_append(headers, auth_header.c_str());
            headers = curl_slist_append(headers, "User-Agent: GRIM-Training-Bot/1.0");
            curl_easy_setopt(curl_handle, CURLOPT_HTTPHEADER, headers);
        }

        CURLcode res = curl_easy_perform(curl_handle);
        
        if (headers) {
            curl_slist_free_all(headers);
        }

        if (res != CURLE_OK) {
            return "";
        }

        return response;
    }
};

Collector::Collector() : pImpl(std::make_unique<Impl>()) {
    pImpl->config = CollectorConfig{};
}

Collector::Collector(const CollectorConfig& config) : pImpl(std::make_unique<Impl>()) {
    pImpl->config = config;
}

Collector::~Collector() = default;

size_t Collector::fetch_online_data() {
    auto start_time = std::chrono::steady_clock::now();
    
    // Create output directory if it doesn't exist
    fs::create_directories(pImpl->config.output_dir);
    
    pImpl->stats = Stats{};  // Reset stats
    
    for (const auto& source : pImpl->config.sources) {
        try {
            auto entries = fetch_from_source(source);
            
            if (!entries.empty()) {
                std::string filename = generate_output_filename(source.source_type);
                if (save_entries(entries, filename)) {
                    pImpl->stats.successful += entries.size();
                } else {
                    pImpl->stats.failed += entries.size();
                }
            }
            
            pImpl->stats.total_fetched += entries.size();
            
        } catch (const std::exception& e) {
            pImpl->stats.failed++;
            // Log error (could integrate with logger.hpp)
        }
    }
    
    auto end_time = std::chrono::steady_clock::now();
    pImpl->stats.total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    return pImpl->stats.successful;
}

void Collector::add_source(const DataSource& source) {
    pImpl->config.sources.push_back(source);
}

bool Collector::load_sources_from_file(const std::string& config_path) {
    try {
        std::ifstream file(config_path);
        if (!file.is_open()) return false;
        
        json j;
        file >> j;
        
        if (j.contains("sources") && j["sources"].is_array()) {
            for (const auto& src : j["sources"]) {
                DataSource source;
                source.url = src.value("url", "");
                source.source_type = src.value("type", "generic");
                source.requires_auth = src.value("requires_auth", false);
                source.api_key = src.value("api_key", "");
                source.priority = src.value("priority", 5);
                
                if (!source.url.empty()) {
                    add_source(source);
                }
            }
        }
        
        return true;
    } catch (...) {
        return false;
    }
}

void Collector::set_output_dir(const std::string& dir) {
    pImpl->config.output_dir = dir;
}

Collector::Stats Collector::get_stats() const {
    return pImpl->stats;
}

void Collector::reset_stats() {
    pImpl->stats = Stats{};
}

std::vector<RawDataEntry> Collector::fetch_from_source(const DataSource& source) {
    if (source.source_type == "news_api") {
        return fetch_news_api(source);
    } else if (source.source_type == "github") {
        return fetch_github(source);
    } else if (source.source_type == "tech_docs") {
        return fetch_tech_docs(source);
    }
    return {};
}

std::vector<RawDataEntry> Collector::fetch_news_api(const DataSource& source) {
    std::vector<RawDataEntry> entries;
    
    std::string response = pImpl->http_get(source.url, source.api_key);
    if (response.empty()) return entries;
    
    try {
        json data = json::parse(response);
        
        if (data.contains("articles") && data["articles"].is_array()) {
            for (const auto& article : data["articles"]) {
                RawDataEntry entry;
                
                std::string title = article.value("title", "");
                std::string description = article.value("description", "");
                std::string content = article.value("content", "");
                
                entry.content = title + "\n\n" + description + "\n\n" + content;
                entry.source_url = article.value("url", source.url);
                entry.source_type = "news_api";
                entry.author = article.value("author", "Unknown");
                entry.timestamp = std::chrono::system_clock::now();
                entry.metadata_json = article.dump();
                
                if (passes_filters(entry.content)) {
                    entries.push_back(entry);
                }
                
                if (entries.size() >= pImpl->config.max_entries_per_source) {
                    break;
                }
            }
        }
    } catch (const json::parse_error&) {
        // Failed to parse JSON
    }
    
    return entries;
}

std::vector<RawDataEntry> Collector::fetch_github(const DataSource& source) {
    std::vector<RawDataEntry> entries;
    
    // Example: Fetch README.md from a repository
    std::string response = pImpl->http_get(source.url, source.api_key);
    if (response.empty()) return entries;
    
    try {
        // If it's GitHub API endpoint
        if (source.url.find("api.github.com") != std::string::npos) {
            json data = json::parse(response);
            
            RawDataEntry entry;
            entry.content = data.value("content", "");
            entry.source_url = source.url;
            entry.source_type = "github";
            entry.author = data.value("author", "Unknown");
            entry.timestamp = std::chrono::system_clock::now();
            entry.metadata_json = data.dump();
            
            if (passes_filters(entry.content)) {
                entries.push_back(entry);
            }
        } else {
            // Direct markdown content
            RawDataEntry entry;
            entry.content = response;
            entry.source_url = source.url;
            entry.source_type = "github";
            entry.author = "Unknown";
            entry.timestamp = std::chrono::system_clock::now();
            
            if (passes_filters(entry.content)) {
                entries.push_back(entry);
            }
        }
    } catch (...) {
        // Handle errors
    }
    
    return entries;
}

std::vector<RawDataEntry> Collector::fetch_tech_docs(const DataSource& source) {
    std::vector<RawDataEntry> entries;
    
    std::string response = pImpl->http_get(source.url, source.api_key);
    if (response.empty()) return entries;
    
    RawDataEntry entry;
    entry.content = response;
    entry.source_url = source.url;
    entry.source_type = "tech_docs";
    entry.author = "Documentation";
    entry.timestamp = std::chrono::system_clock::now();
    
    if (passes_filters(entry.content)) {
        entries.push_back(entry);
    }
    
    return entries;
}

bool Collector::save_entries(const std::vector<RawDataEntry>& entries, const std::string& filename) {
    try {
        std::string filepath = pImpl->config.output_dir + "/" + filename;
        std::ofstream outfile(filepath);
        
        if (!outfile.is_open()) return false;
        
        if (pImpl->config.save_as_jsonl) {
            // Save as JSONL (one JSON object per line)
            for (const auto& entry : entries) {
                json j;
                j["content"] = entry.content;
                j["source_url"] = entry.source_url;
                j["source_type"] = entry.source_type;
                j["author"] = entry.author;
                j["timestamp"] = std::chrono::system_clock::to_time_t(entry.timestamp);
                j["metadata"] = json::parse(entry.metadata_json.empty() ? "{}" : entry.metadata_json);
                
                outfile << j.dump() << "\n";
            }
        } else {
            // Save as plain text
            for (const auto& entry : entries) {
                outfile << "=== SOURCE: " << entry.source_url << " ===\n";
                outfile << "=== AUTHOR: " << entry.author << " ===\n";
                outfile << "=== TYPE: " << entry.source_type << " ===\n";
                outfile << entry.content << "\n\n";
                outfile << "---\n\n";
            }
        }
        
        outfile.close();
        return true;
        
    } catch (...) {
        return false;
    }
}

bool Collector::passes_filters(const std::string& content) const {
    // If no filters, pass everything
    if (pImpl->config.keyword_filters.empty()) return true;
    
    // Check if content contains any of the keywords
    for (const auto& keyword : pImpl->config.keyword_filters) {
        if (content.find(keyword) != std::string::npos) {
            return true;
        }
    }
    
    return false;
}

std::string Collector::generate_output_filename(const std::string& source_type) const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << source_type << "_" 
       << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    
    if (pImpl->config.save_as_jsonl) {
        ss << ".jsonl";
    } else {
        ss << ".txt";
    }
    
    return ss.str();
}

} // namespace training
} // namespace grim

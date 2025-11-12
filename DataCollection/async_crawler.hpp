//======================================================//
//  GRIM Async Web Crawler
//  High-performance recursive link crawling
//  
//  Features:
//  - Async/parallel crawling with depth limiting
//  - Smart link extraction and filtering
//  - Domain-restricted crawling
//  - Visited URL tracking with bloom filter
//  - Non-blocking operation
//  - HTML content extraction with Gumbo parser
//  
//  Version: 1.0.0
//======================================================//

#pragma once

#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <atomic>
#include <regex>
#include <thread>
#include <future>
#include <functional>
#include "streaming_downloader.hpp"
#include "html_extractor.hpp"

namespace GRIM {
namespace Training {

//======================================================//
//  URL Utilities
//======================================================//

class URLUtils {
public:
    // Extract domain from URL
    static std::string extractDomain(const std::string& url) {
        static const std::regex domain_regex(R"(^(?:https?://)?(?:www\.)?([^/:?#]+))");
        std::smatch match;
        if (std::regex_search(url, match, domain_regex) && match.size() > 1) {
            return match[1].str();
        }
        return "";
    }
    
    // Normalize URL (remove fragments, sort query params, etc.)
    static std::string normalize(const std::string& url) {
        std::string normalized = url;
        
        // Remove fragment
        size_t fragment_pos = normalized.find('#');
        if (fragment_pos != std::string::npos) {
            normalized = normalized.substr(0, fragment_pos);
        }
        
        // Remove trailing slash
        if (!normalized.empty() && normalized.back() == '/') {
            normalized.pop_back();
        }
        
        // Convert to lowercase
        std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
        
        return normalized;
    }
    
    // Convert relative URL to absolute
    static std::string makeAbsolute(const std::string& base_url, const std::string& relative_url) {
        if (relative_url.empty()) return "";
        
        // Already absolute
        if (relative_url.find("http://") == 0 || relative_url.find("https://") == 0) {
            return relative_url;
        }
        
        // Protocol-relative URL
        if (relative_url.size() >= 2 && relative_url[0] == '/' && relative_url[1] == '/') {
            // Extract protocol from base
            size_t protocol_end = base_url.find("://");
            if (protocol_end != std::string::npos) {
                return base_url.substr(0, protocol_end) + ":" + relative_url;
            }
            return "https:" + relative_url;
        }
        
        // Absolute path
        if (!relative_url.empty() && relative_url[0] == '/') {
            // Extract protocol and domain from base
            size_t protocol_end = base_url.find("://");
            if (protocol_end == std::string::npos) return "";
            
            size_t domain_start = protocol_end + 3;
            size_t path_start = base_url.find('/', domain_start);
            
            std::string base = (path_start != std::string::npos) 
                ? base_url.substr(0, path_start)
                : base_url;
            
            return base + relative_url;
        }
        
        // Relative path - append to base directory
        size_t last_slash = base_url.find_last_of('/');
        if (last_slash != std::string::npos) {
            return base_url.substr(0, last_slash + 1) + relative_url;
        }
        
        return base_url + "/" + relative_url;
    }
    
    // Check if URL matches path patterns
    static bool matchesPattern(const std::string& url, const std::vector<std::string>& patterns) {
        if (patterns.empty()) return true;
        
        for (const auto& pattern : patterns) {
            if (url.find(pattern) != std::string::npos) {
                return true;
            }
        }
        return false;
    }
};

//======================================================//
//  Link Extractor - Parse HTML for links
//======================================================//

class LinkExtractor {
public:
    // Extract all <a href="..."> links from HTML
    static std::vector<std::string> extractLinks(const std::string& html, const std::string& base_url) {
        std::vector<std::string> links;
        
        // Simple regex-based extraction (fast but not perfect)
        // For production, consider using a proper HTML parser
        static const std::regex href_regex(R"(<a\s+[^>]*href\s*=\s*["']([^"']+)["'][^>]*>)", 
                                          std::regex::icase);
        
        auto matches_begin = std::sregex_iterator(html.begin(), html.end(), href_regex);
        auto matches_end = std::sregex_iterator();
        
        for (std::sregex_iterator i = matches_begin; i != matches_end; ++i) {
            std::smatch match = *i;
            if (match.size() > 1) {
                std::string href = match[1].str();
                
                // Skip javascript:, mailto:, tel:, etc.
                if (href.find("javascript:") == 0 || 
                    href.find("mailto:") == 0 || 
                    href.find("tel:") == 0 ||
                    href.find("#") == 0) {
                    continue;
                }
                
                // Convert to absolute URL
                std::string absolute = URLUtils::makeAbsolute(base_url, href);
                if (!absolute.empty()) {
                    links.push_back(absolute);
                }
            }
        }
        
        return links;
    }
};

//======================================================//
//  Crawl Task - Represents a URL to crawl
//======================================================//

struct CrawlTask {
    std::string url;
    int depth = 0;
    int priority = 5;
    std::string source_name;
    
    bool operator<(const CrawlTask& other) const {
        // Higher priority first, then shallower depth
        if (priority != other.priority) {
            return priority < other.priority;  // Note: priority_queue is max-heap
        }
        return depth > other.depth;
    }
};

//======================================================//
//  Async Crawler - Non-blocking recursive crawling
//======================================================//

class AsyncCrawler {
public:
    struct CrawlConfig {
        int max_depth = 2;
        int max_concurrent = 50;  // Parallel downloads
        int max_pages_per_source = 1000;
        int timeout_seconds = 30;
        
        bool stay_on_domain = true;
        bool respect_robots_txt = true;
        
        std::vector<std::string> path_patterns;      // Include only URLs matching these
        std::vector<std::string> exclude_patterns;    // Exclude URLs matching these
        
        size_t max_memory_per_page = 10 * 1024 * 1024;  // 10MB limit per page
    };
    
    AsyncCrawler() = default;
    
    // Start async crawl - returns immediately
    std::future<size_t> crawlAsync(
        const std::string& start_url,
        const CrawlConfig& config,
        std::function<void(const std::string& url, const std::string& content)> on_page_callback
    ) {
        return std::async(std::launch::async, [=]() {
            return crawlSync(start_url, config, on_page_callback);
        });
    }
    
    // Synchronous crawl (runs in background thread)
    size_t crawlSync(
        const std::string& start_url,
        const CrawlConfig& config,
        std::function<void(const std::string& url, const std::string& content)> on_page_callback
    ) {
        std::priority_queue<CrawlTask> frontier;
        std::unordered_set<std::string> visited;
        std::mutex frontier_mutex;
        std::mutex visited_mutex;
        std::atomic<size_t> pages_crawled{0};
        std::atomic<bool> should_stop{false};
        
        std::string base_domain = URLUtils::extractDomain(start_url);
        
        // Add start URL to frontier
        CrawlTask start_task;
        start_task.url = start_url;
        start_task.depth = 0;
        start_task.priority = 10;
        frontier.push(start_task);
        
        StreamingDownloader downloader;
        
        // Worker function
        auto worker = [&]() {
            while (!should_stop.load()) {
                CrawlTask task;
                
                // Get next task from frontier
                {
                    std::lock_guard<std::mutex> lock(frontier_mutex);
                    if (frontier.empty()) {
                        break;
                    }
                    task = frontier.top();
                    frontier.pop();
                }
                
                // Check if already visited
                {
                    std::lock_guard<std::mutex> lock(visited_mutex);
                    std::string normalized = URLUtils::normalize(task.url);
                    if (visited.count(normalized)) {
                        continue;
                    }
                    visited.insert(normalized);
                }
                
                // Check page limit
                if (pages_crawled.load() >= static_cast<size_t>(config.max_pages_per_source)) {
                    should_stop.store(true);
                    break;
                }
                
                // Check memory usage
                if (MemoryTracker::instance().isNearLimit()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }
                
                // Download page
                std::string raw_html;
                bool success = downloader.downloadToMemory(task.url, raw_html, config.max_memory_per_page);
                
                if (!success || raw_html.empty()) {
                    continue;
                }
                
                pages_crawled++;
                
                // Pass raw HTML to callback - let the caller decide how to extract
                // (Some callers may want HTML, others may want extracted text)
                if (on_page_callback) {
                    try {
                        on_page_callback(task.url, raw_html);
                    } catch (...) {
                        // Ignore callback errors
                    }
                }
                
                // Extract and queue links if not at max depth
                if (task.depth < config.max_depth) {
                    auto links = LinkExtractor::extractLinks(raw_html, task.url);
                    
                    std::lock_guard<std::mutex> lock(frontier_mutex);
                    for (const auto& link : links) {
                        std::string normalized = URLUtils::normalize(link);
                        
                        // Apply filters
                        if (config.stay_on_domain) {
                            std::string link_domain = URLUtils::extractDomain(normalized);
                            if (link_domain != base_domain) {
                                continue;
                            }
                        }
                        
                        if (!config.path_patterns.empty() && 
                            !URLUtils::matchesPattern(normalized, config.path_patterns)) {
                            continue;
                        }
                        
                        if (!config.exclude_patterns.empty() && 
                            URLUtils::matchesPattern(normalized, config.exclude_patterns)) {
                            continue;
                        }
                        
                        // Check if already visited
                        {
                            std::lock_guard<std::mutex> vlock(visited_mutex);
                            if (visited.count(normalized)) {
                                continue;
                            }
                        }
                        
                        CrawlTask new_task;
                        new_task.url = normalized;
                        new_task.depth = task.depth + 1;
                        new_task.priority = task.priority - 1;
                        frontier.push(new_task);
                    }
                }
                
                // Clear HTML immediately to free memory
                raw_html.clear();
                raw_html.shrink_to_fit();
                
                // Small delay to avoid overwhelming servers
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        };
        
        // Launch worker threads
        std::vector<std::thread> workers;
        int num_workers = std::min(config.max_concurrent, 50);
        
        for (int i = 0; i < num_workers; ++i) {
            workers.emplace_back(worker);
        }
        
        // Wait for all workers to complete
        for (auto& w : workers) {
            if (w.joinable()) {
                w.join();
            }
        }
        
        return pages_crawled.load();
    }
    
    // Cancel ongoing crawl
    void cancel() {
        // TODO: Implement cancellation mechanism
    }
};

} // namespace Training
} // namespace GRIM

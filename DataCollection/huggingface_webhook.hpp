#pragma once

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <optional>

namespace GRIM {
namespace DataCollection {

// Dataset metadata from Hugging Face
struct HFDatasetInfo {
    std::string id;              // e.g., "openai/gsm8k"
    std::string author;          // Dataset author/organization
    std::string name;            // Dataset name
    std::string description;     // Short description
    int downloads = 0;           // Download count
    int likes = 0;               // Number of likes
    std::string lastModified;    // Last update timestamp
    std::vector<std::string> tags; // Tags (e.g., "text-generation", "qa")
    std::vector<std::string> configs; // Available configurations
    std::vector<std::string> splits; // Available splits (train/test/validation)
    size_t sizeBytes = 0;        // Approximate size in bytes
};

// Download progress callback
using ProgressCallback = std::function<void(size_t downloaded, size_t total, const std::string& status)>;

// Hugging Face API client for dataset operations
class HuggingFaceWebhook {
public:
    HuggingFaceWebhook();
    ~HuggingFaceWebhook();

    // Authentication
    void setApiToken(const std::string& token);
    bool isAuthenticated() const;

    // Dataset search and browsing
    std::vector<HFDatasetInfo> searchDatasets(
        const std::string& query,
        int limit = 20,
        const std::string& filter = "" // e.g., "task_categories:text-generation"
    );

    std::optional<HFDatasetInfo> getDatasetInfo(const std::string& datasetId);

    // Dataset download
    bool downloadDataset(
        const std::string& datasetId,
        const std::string& outputDir,
        const std::string& split = "train", // "train", "test", "validation", or "all"
        const std::string& config = "",     // Configuration name (if applicable)
        ProgressCallback progressCallback = nullptr
    );

    // Stream dataset directly without full download (for large datasets)
    bool streamDataset(
        const std::string& datasetId,
        const std::string& outputFile,
        const std::string& split = "train",
        const std::string& config = "",
        int maxSamples = -1,              // -1 for unlimited
        ProgressCallback progressCallback = nullptr
    );

    // Convert downloaded dataset to GRIM training format
    bool convertToGRIMFormat(
        const std::string& inputPath,
        const std::string& outputPath,
        const std::string& format = "auto" // "json", "jsonl", "parquet", "csv", "auto"
    );

    // Cache management
    void setCacheDir(const std::string& cacheDir);
    std::string getCacheDir() const;
    void clearCache();
    size_t getCacheSize() const;

    // Error handling
    std::string getLastError() const;
    bool hasError() const;

    // Download tracking - prevent duplicate downloads across sessions
    bool isDatasetDownloaded(const std::string& datasetId) const;
    void markDatasetDownloaded(const std::string& datasetId);
    void clearDownloadHistory();
    size_t getDownloadedCount() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;

    // Internal helpers
    std::string buildApiUrl(const std::string& endpoint) const;
    std::string makeRequest(
        const std::string& url,
        const std::string& method = "GET",
        const std::string& body = ""
    );
    bool downloadFile(
        const std::string& url,
        const std::string& outputPath,
        ProgressCallback progressCallback = nullptr
    );
    std::vector<std::string> getDatasetFiles(
        const std::string& datasetId,
        const std::string& split,
        const std::string& config
    );
};

// Utility functions
namespace HFUtils {
    // Parse Hugging Face dataset URL
    // e.g., "https://huggingface.co/datasets/openai/gsm8k" -> "openai/gsm8k"
    std::string parseDatasetId(const std::string& url);

    // Format size for display (bytes -> "1.2 GB")
    std::string formatSize(size_t bytes);

    // Validate dataset ID format
    bool isValidDatasetId(const std::string& datasetId);

    // Get default cache directory
    std::string getDefaultCacheDir();
}

} // namespace DataCollection
} // namespace GRIM

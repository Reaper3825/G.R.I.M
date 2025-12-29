#include "huggingface_webhook.hpp"
#include "collection_state.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <regex>
#include <nlohmann/json.hpp>
#include <curl/curl.h>
#include <zip.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace GRIM {
namespace DataCollection {

// Private implementation
struct HuggingFaceWebhook::Impl {
    std::string apiToken;
    std::string cacheDir;
    std::string lastError;
    std::string downloadedDatasetsPath;
    std::unordered_set<std::string> downloadedDatasets;
    
    static constexpr const char* API_BASE = "https://huggingface.co";
    static constexpr const char* API_DATASETS = "https://datasets-server.huggingface.co";
    
    Impl() {
        cacheDir = HFUtils::getDefaultCacheDir();
        fs::create_directories(cacheDir);
        downloadedDatasetsPath = (fs::path(cacheDir) / "downloaded_datasets.json").string();
        loadDownloadedDatasets();
    }
    
    void loadDownloadedDatasets() {
        try {
            if (fs::exists(downloadedDatasetsPath)) {
                std::ifstream file(downloadedDatasetsPath);
                json j;
                file >> j;
                if (j.contains("datasets") && j["datasets"].is_array()) {
                    for (const auto& id : j["datasets"]) {
                        downloadedDatasets.insert(id.get<std::string>());
                    }
                }
                std::cout << "[HF] Loaded " << downloadedDatasets.size() << " previously downloaded datasets" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[HF] Failed to load downloaded datasets list: " << e.what() << std::endl;
        }
    }
    
    void saveDownloadedDatasets() {
        try {
            json j;
            j["datasets"] = json::array();
            for (const auto& id : downloadedDatasets) {
                j["datasets"].push_back(id);
            }
            j["last_updated"] = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            std::ofstream file(downloadedDatasetsPath);
            file << j.dump(2);
        } catch (const std::exception& e) {
            std::cerr << "[HF] Failed to save downloaded datasets list: " << e.what() << std::endl;
        }
    }
    
    bool isDatasetDownloaded(const std::string& datasetId) const {
        return downloadedDatasets.count(datasetId) > 0;
    }
    
    void markDatasetDownloaded(const std::string& datasetId) {
        downloadedDatasets.insert(datasetId);
        saveDownloadedDatasets();
    }
};

// Helper function to extract zip archives
static bool extractZipArchive(const std::string& zipPath, const std::string& outputDir) {
    int error = 0;
    zip_t* archive = zip_open(zipPath.c_str(), ZIP_RDONLY, &error);
    
    if (!archive) {
        zip_error_t zipError;
        zip_error_init_with_code(&zipError, error);
        std::cout << "[HF EXTRACT] Failed to open zip: " << zip_error_strerror(&zipError) << std::endl;
        zip_error_fini(&zipError);
        return false;
    }
    
    zip_int64_t numEntries = zip_get_num_entries(archive, 0);
    std::cout << "[HF EXTRACT] Extracting " << numEntries << " files from " << zipPath << std::endl;
    
    for (zip_int64_t i = 0; i < numEntries; i++) {
        const char* name = zip_get_name(archive, i, 0);
        if (!name) {
            std::cout << "[HF EXTRACT] Failed to get name for entry " << i << std::endl;
            continue;
        }
        
        // Skip directories
        std::string entryName(name);
        if (entryName.back() == '/' || entryName.back() == '\\') {
            continue;
        }
        
        // Open the file in the archive
        zip_file_t* file = zip_fopen_index(archive, i, 0);
        if (!file) {
            std::cout << "[HF EXTRACT] Failed to open entry: " << name << std::endl;
            continue;
        }
        
        // Create output path
        fs::path outputPath = fs::path(outputDir) / name;
        
        // Create parent directories if needed
        fs::create_directories(outputPath.parent_path());
        
        // Read and write file
        std::ofstream outFile(outputPath, std::ios::binary);
        if (!outFile) {
            std::cout << "[HF EXTRACT] Failed to create output file: " << outputPath << std::endl;
            zip_fclose(file);
            continue;
        }
        
        char buffer[8192];
        zip_int64_t bytesRead;
        while ((bytesRead = zip_fread(file, buffer, sizeof(buffer))) > 0) {
            outFile.write(buffer, bytesRead);
        }
        
        outFile.close();
        zip_fclose(file);
        
        std::cout << "[HF EXTRACT] Extracted: " << name << std::endl;
    }
    
    zip_close(archive);
    std::cout << "[HF EXTRACT] Successfully extracted archive" << std::endl;
    return true;
}

HuggingFaceWebhook::HuggingFaceWebhook()
    : pImpl(std::make_unique<Impl>()) {
}

HuggingFaceWebhook::~HuggingFaceWebhook() = default;

void HuggingFaceWebhook::setApiToken(const std::string& token) {
    pImpl->apiToken = token;
}

bool HuggingFaceWebhook::isAuthenticated() const {
    return !pImpl->apiToken.empty();
}

std::string HuggingFaceWebhook::buildApiUrl(const std::string& endpoint) const {
    return std::string(pImpl->API_BASE) + endpoint;
}

// Callback for curl to write response data
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t totalSize = size * nmemb;
    std::string* response = static_cast<std::string*>(userp);
    response->append(static_cast<char*>(contents), totalSize);
    return totalSize;
}

// Callback for curl to write file data
static size_t WriteFileCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t totalSize = size * nmemb;
    std::ofstream* file = static_cast<std::ofstream*>(userp);
    file->write(static_cast<char*>(contents), totalSize);
    return totalSize;
}

// Progress callback structure
struct ProgressData {
    ProgressCallback callback;
    std::string status;
};

// Callback for curl progress updates
static int ProgressCallbackCurl(void* clientp, curl_off_t dltotal, curl_off_t dlnow,
                                curl_off_t ultotal, curl_off_t ulnow) {
    if (clientp) {
        ProgressData* data = static_cast<ProgressData*>(clientp);
        if (data->callback) {
            data->callback(static_cast<size_t>(dlnow), 
                          static_cast<size_t>(dltotal), 
                          data->status);
        }
    }
    return 0; // Return 0 to continue
}

std::string HuggingFaceWebhook::makeRequest(
    const std::string& url,
    const std::string& method,
    const std::string& body
) {
    pImpl->lastError.clear();
    
    CURL* curl = curl_easy_init();
    if (!curl) {
        pImpl->lastError = "Failed to initialize curl";
        return "";
    }
    
    std::string response;
    struct curl_slist* headers = nullptr;
    
    // Set URL
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    
    // Set method
    if (method == "POST") {
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        if (!body.empty()) {
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
        }
    } else if (method == "GET") {
        curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
    }
    
    // Add authorization header if token is set
    if (!pImpl->apiToken.empty()) {
        std::string authHeader = "Authorization: Bearer " + pImpl->apiToken;
        headers = curl_slist_append(headers, authHeader.c_str());
    }
    
    // Add User-Agent
    headers = curl_slist_append(headers, "User-Agent: GRIM-HuggingFace/1.0");
    
    if (headers) {
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    }
    
    // Set write callback
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    
    // Follow redirects
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    
    // SSL verification
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    
    // Perform request
    CURLcode res = curl_easy_perform(curl);
    
    if (res != CURLE_OK) {
        pImpl->lastError = "Request failed: " + std::string(curl_easy_strerror(res));
        response.clear();
    } else {
        // Check HTTP response code
        long httpCode = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
        if (httpCode >= 400) {
            pImpl->lastError = "HTTP error: " + std::to_string(httpCode);
            response.clear();
        }
    }
    
    // Cleanup
    if (headers) {
        curl_slist_free_all(headers);
    }
    curl_easy_cleanup(curl);
    
    return response;
}

bool HuggingFaceWebhook::downloadFile(
    const std::string& url,
    const std::string& outputPath,
    ProgressCallback progressCallback
) {
    pImpl->lastError.clear();
    
    // Debug: Log the download URL
    std::cout << "[HF DOWNLOAD] URL: " << url << std::endl;
    std::cout << "[HF DOWNLOAD] Output: " << outputPath << std::endl;
    
    CURL* curl = curl_easy_init();
    if (!curl) {
        pImpl->lastError = "Failed to initialize curl";
        return false;
    }
    
    // Open output file
    std::ofstream outFile(outputPath, std::ios::binary);
    if (!outFile.is_open()) {
        curl_easy_cleanup(curl);
        pImpl->lastError = "Failed to open output file";
        return false;
    }
    
    struct curl_slist* headers = nullptr;
    ProgressData progressData;
    progressData.callback = progressCallback;
    progressData.status = "Downloading...";
    
    // Set URL
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    
    // Add authorization header if token is set
    if (!pImpl->apiToken.empty()) {
        std::string authHeader = "Authorization: Bearer " + pImpl->apiToken;
        headers = curl_slist_append(headers, authHeader.c_str());
    }
    
    // Add User-Agent
    headers = curl_slist_append(headers, "User-Agent: GRIM-HuggingFace/1.0");
    
    if (headers) {
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    }
    
    // Set write callback
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteFileCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &outFile);
    
    // Set progress callback
    if (progressCallback) {
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, ProgressCallbackCurl);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &progressData);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    }
    
    // Follow redirects
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    
    // SSL verification
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    
    // Perform request
    CURLcode res = curl_easy_perform(curl);
    
    outFile.close();
    
    bool success = false;
    if (res != CURLE_OK) {
        pImpl->lastError = "Download failed: " + std::string(curl_easy_strerror(res));
        std::cout << "[HF DOWNLOAD] CURL error: " << pImpl->lastError << std::endl;
    } else {
        // Check HTTP response code
        long httpCode = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
        std::cout << "[HF DOWNLOAD] HTTP code: " << httpCode << std::endl;
        if (httpCode >= 400) {
            if (httpCode == 401) {
                pImpl->lastError = "HTTP 401: Authentication required - Please add your HuggingFace token";
            } else if (httpCode == 403) {
                pImpl->lastError = "HTTP 403: Access forbidden - Dataset may be private or gated";
            } else if (httpCode == 404) {
                pImpl->lastError = "HTTP 404: File not found";
            } else {
                pImpl->lastError = "HTTP error: " + std::to_string(httpCode);
            }
        } else {
            success = true;
        }
    }
    
    // Cleanup
    if (headers) {
        curl_slist_free_all(headers);
    }
    curl_easy_cleanup(curl);
    
    // Delete partial file on failure
    if (!success && fs::exists(outputPath)) {
        fs::remove(outputPath);
    }
    
    return success;
}

std::vector<HFDatasetInfo> HuggingFaceWebhook::searchDatasets(
    const std::string& query,
    int limit,
    const std::string& filter
) {
    std::vector<HFDatasetInfo> results;
    
    // Build search URL
    std::string url = std::string(pImpl->API_BASE) + "/api/datasets";
    url += "?search=" + query;
    url += "&limit=" + std::to_string(limit);
    if (!filter.empty()) {
        url += "&filter=" + filter;
    }
    
    std::string response = makeRequest(url);
    if (response.empty()) {
        return results;
    }
    
    try {
        auto j = json::parse(response);
        
        for (const auto& item : j) {
            HFDatasetInfo info;
            info.id = item.value("id", "");
            info.author = item.value("author", "");
            info.downloads = item.value("downloads", 0);
            info.likes = item.value("likes", 0);
            info.lastModified = item.value("lastModified", "");
            
            if (item.contains("tags")) {
                for (const auto& tag : item["tags"]) {
                    info.tags.push_back(tag.get<std::string>());
                }
            }
            
            results.push_back(info);
        }
    } catch (const std::exception& e) {
        pImpl->lastError = "Failed to parse search results: " + std::string(e.what());
    }
    
    return results;
}

std::optional<HFDatasetInfo> HuggingFaceWebhook::getDatasetInfo(const std::string& datasetId) {
    std::string url = std::string(pImpl->API_BASE) + "/api/datasets/" + datasetId;
    
    std::string response = makeRequest(url);
    if (response.empty()) {
        return std::nullopt;
    }
    
    try {
        auto j = json::parse(response);
        
        HFDatasetInfo info;
        info.id = j.value("id", datasetId);
        info.author = j.value("author", "");
        info.name = j.value("cardData", json::object()).value("pretty_name", "");
        info.description = j.value("description", "");
        info.downloads = j.value("downloads", 0);
        info.likes = j.value("likes", 0);
        info.lastModified = j.value("lastModified", "");
        
        if (j.contains("tags")) {
            for (const auto& tag : j["tags"]) {
                info.tags.push_back(tag.get<std::string>());
            }
        }
        
        return info;
        
    } catch (const std::exception& e) {
        pImpl->lastError = "Failed to parse dataset info: " + std::string(e.what());
        return std::nullopt;
    }
}

bool HuggingFaceWebhook::downloadDataset(
    const std::string& datasetId,
    const std::string& outputDir,
    const std::string& split,
    const std::string& config,
    ProgressCallback progressCallback
) {
    pImpl->lastError.clear();
    
    // Check if already downloaded
    if (pImpl->isDatasetDownloaded(datasetId)) {
        std::cout << "[HF DOWNLOAD] Dataset already downloaded: " << datasetId << std::endl;
        
        // Check if output directory exists and has files
        if (fs::exists(outputDir) && !fs::is_empty(outputDir)) {
            std::cout << "[HF DOWNLOAD] Output directory exists with files, skipping download" << std::endl;
            pImpl->lastError = "Dataset already downloaded: " + datasetId;
            
            if (progressCallback) {
                progressCallback(1, 1, "Already downloaded");
            }
            return true;  // Return true since we have the data
        }
        
        // Directory doesn't exist or is empty, remove from downloaded list and re-download
        std::cout << "[HF DOWNLOAD] Output directory missing/empty, re-downloading..." << std::endl;
    }
    
    // Create output directory
    fs::create_directories(outputDir);
    
    // Get dataset files
    auto files = getDatasetFiles(datasetId, split, config);
    if (files.empty()) {
        pImpl->lastError = "No files found for dataset";
        return false;
    }
    
    // Download each file
    size_t fileIndex = 0;
    std::vector<std::string> downloadedArchives;
    
    for (const auto& fileUrl : files) {
        fileIndex++;
        
        // Extract filename from URL (check for fragment first, used for LFS files)
        std::string actualUrl = fileUrl;
        std::string filename;
        size_t fragmentPos = fileUrl.find('#');
        if (fragmentPos != std::string::npos) {
            // LFS file with filename in fragment
            actualUrl = fileUrl.substr(0, fragmentPos);
            std::string pathInFragment = fileUrl.substr(fragmentPos + 1);
            size_t lastSlash = pathInFragment.find_last_of('/');
            filename = (lastSlash != std::string::npos) 
                ? pathInFragment.substr(lastSlash + 1) : pathInFragment;
        } else {
            // Regular file - extract from URL path
            size_t lastSlash = fileUrl.find_last_of('/');
            filename = (lastSlash != std::string::npos) 
                ? fileUrl.substr(lastSlash + 1) : "data.parquet";
        }
        
        fs::path outputPath = fs::path(outputDir) / filename;
        
        // Skip if file already exists with non-zero size
        if (fs::exists(outputPath) && fs::file_size(outputPath) > 0) {
            std::cout << "[HF DOWNLOAD] File already exists, skipping: " << filename << std::endl;
            if (progressCallback) {
                progressCallback(fileIndex, files.size(), "Skipping existing file " + filename);
            }
            continue;
        }
        
        std::cout << "[HF DOWNLOAD] URL: " << actualUrl << std::endl;
        std::cout << "[HF DOWNLOAD] Filename: " << filename << std::endl;
        std::cout << "[HF DOWNLOAD] Output: " << outputPath.string() << std::endl;
        
        if (progressCallback) {
            progressCallback(fileIndex - 1, files.size(), 
                "Downloading file " + std::to_string(fileIndex) + "/" + std::to_string(files.size()));
        }
        
        if (!downloadFile(actualUrl, outputPath.string(), progressCallback)) {
            pImpl->lastError = "Failed to download file: " + filename;
            return false;
        }
        
        // Track archive files for later extraction and deletion
        if (filename.find(".zip") != std::string::npos ||
            filename.find(".tar") != std::string::npos ||
            filename.find(".gz") != std::string::npos) {
            downloadedArchives.push_back(outputPath.string());
        }
    }
    
    // Extract archives before cleanup
    for (const auto& archivePath : downloadedArchives) {
        std::cout << "[HF DOWNLOAD] Extracting archive: " << archivePath << std::endl;
        
        // Only handle .zip files for now (tar/gz would need different libraries)
        if (archivePath.find(".zip") != std::string::npos) {
            if (!extractZipArchive(archivePath, outputDir)) {
                std::cout << "[HF DOWNLOAD] Warning: Failed to extract " << archivePath << std::endl;
                // Continue anyway - don't fail the whole download
            }
        } else {
            std::cout << "[HF DOWNLOAD] Skipping non-zip archive (tar/gz support not yet implemented): " 
                      << archivePath << std::endl;
        }
    }
    
    // Clean up archive files after extraction
    for (const auto& archivePath : downloadedArchives) {
        try {
            if (fs::exists(archivePath)) {
                fs::remove(archivePath);
                std::cout << "[HF DOWNLOAD] Cleaned up archive: " << archivePath << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "[HF DOWNLOAD] Failed to remove archive " << archivePath << ": " << e.what() << std::endl;
        }
    }
    
    // Mark dataset as downloaded to prevent re-downloading
    pImpl->markDatasetDownloaded(datasetId);
    std::cout << "[HF DOWNLOAD] Marked dataset as downloaded: " << datasetId << std::endl;
    
    if (progressCallback) {
        progressCallback(files.size(), files.size(), "Download complete");
    }
    
    return true;
}

bool HuggingFaceWebhook::streamDataset(
    const std::string& datasetId,
    const std::string& outputFile,
    const std::string& split,
    const std::string& config,
    int maxSamples,
    ProgressCallback progressCallback
) {
    // Use the datasets-server API for streaming
    std::string url = std::string(pImpl->API_DATASETS) + "/rows";
    url += "?dataset=" + datasetId;
    url += "&config=" + (config.empty() ? "default" : config);
    url += "&split=" + split;
    
    std::string response = makeRequest(url);
    if (response.empty()) {
        return false;
    }
    
    try {
        auto j = json::parse(response);
        
        std::ofstream outFile(outputFile);
        if (!outFile.is_open()) {
            pImpl->lastError = "Failed to open output file";
            return false;
        }
        
        int samplesWritten = 0;
        if (j.contains("rows")) {
            for (const auto& row : j["rows"]) {
                if (maxSamples > 0 && samplesWritten >= maxSamples) {
                    break;
                }
                
                outFile << row.dump() << "\n";
                samplesWritten++;
                
                if (progressCallback && samplesWritten % 100 == 0) {
                    progressCallback(samplesWritten, maxSamples > 0 ? maxSamples : samplesWritten, 
                        "Streaming samples...");
                }
            }
        }
        
        outFile.close();
        
        if (progressCallback) {
            progressCallback(samplesWritten, samplesWritten, "Stream complete");
        }
        
        return samplesWritten > 0;
        
    } catch (const std::exception& e) {
        pImpl->lastError = "Failed to stream dataset: " + std::string(e.what());
        return false;
    }
}

std::vector<std::string> HuggingFaceWebhook::getDatasetFiles(
    const std::string& datasetId,
    const std::string& split,
    const std::string& config
) {
    std::vector<std::string> files;
    
    // Query the dataset API for file list
    const std::string datasetBase = std::string(pImpl->API_BASE) + "/datasets/";
    std::string url = std::string(pImpl->API_BASE) + "/api/datasets/" + datasetId + "/tree/main?recursive=1";
    
    std::string response = makeRequest(url);
    if (response.empty()) {
        pImpl->lastError = "Failed to fetch dataset tree from API";
        return files;
    }
    
    try {
        auto j = json::parse(response);
        
        // Debug: Check the structure of the response
        std::cout << "[HF DEBUG] Response structure: " << j.dump(2).substr(0, 500) << std::endl;
        
        // The API might return either an array directly or an object with a "tree" field
        json itemsList;
        if (j.is_array()) {
            itemsList = j;
        } else if (j.contains("tree") && j["tree"].is_array()) {
            itemsList = j["tree"];
        } else {
            pImpl->lastError = "Unexpected API response format - not an array or tree object";
            return files;
        }
        
        // Look for parquet, csv, json, or txt data files
        for (const auto& item : itemsList) {
            if (item.value("type", "") == "file") {
                std::string path = item.value("path", "");
                
                // Check if it's a data file (parquet, csv, json, txt, zip, tar, gz)
                bool isDataFile = path.find(".parquet") != std::string::npos ||
                                 path.find(".csv") != std::string::npos ||
                                 path.find(".json") != std::string::npos ||
                                 path.find(".jsonl") != std::string::npos ||
                                 path.find(".txt") != std::string::npos ||
                                 path.find(".zip") != std::string::npos ||
                                 path.find(".tar") != std::string::npos ||
                                 path.find(".gz") != std::string::npos;
                
                if (!isDataFile) continue;
                
                // Skip README and other non-data files
                if (path.find("README") != std::string::npos || 
                    path.find(".gitattributes") != std::string::npos ||
                    path.find(".md") != std::string::npos) {
                    continue;
                }
                
                // Filter by split if specified (check for common split names)
                if (!split.empty()) {
                    bool matchesSplit = path.find(split) != std::string::npos ||
                                       path.find("train") != std::string::npos ||  // Default to train if split in path
                                       (split == "train" && path.find("test") == std::string::npos && 
                                        path.find("valid") == std::string::npos && path.find("dev") == std::string::npos);
                    if (!matchesSplit) continue;
                }
                
                // Filter by config if specified
                if (!config.empty() && path.find(config) == std::string::npos) {
                    continue;
                }
                
                // Construct download URL (check if file uses LFS)
                std::string downloadUrl;
                if (item.contains("lfs") && item["lfs"].contains("oid")) {
                    // For LFS files, still use the huggingface resolve endpoint so we get
                    // a signed redirect (avoids DNS issues with cdn-lfs host in some envs)
                    downloadUrl = datasetBase + datasetId + "/resolve/main/" + path;
                    
                    std::cout << "[HF LFS] Detected LFS file: " << path << std::endl;
                    std::cout << "[HF LFS] OID: " << item["lfs"]["oid"].get<std::string>() << std::endl;
                    std::cout << "[HF LFS] Using resolve URL (signed redirect): " << downloadUrl << std::endl;
                } else {
                    // Regular file
                    downloadUrl = datasetBase + datasetId + "/resolve/main/" + path;
                }
                files.push_back(downloadUrl);
            }
        }
        
        // If no files found, set a helpful error message
        if (files.empty()) {
            pImpl->lastError = "No compatible data files found. Available files in repo shown in debug output.";
        }
        
    } catch (const std::exception& e) {
        pImpl->lastError = "Failed to parse file list: " + std::string(e.what());
    }
    
    return files;
}

bool HuggingFaceWebhook::convertToGRIMFormat(
    const std::string& inputPath,
    const std::string& outputPath,
    const std::string& format
) {
    // TODO: Implement conversion from various formats to GRIM's JSONL format
    // This would parse parquet/json/csv and convert to the expected format
    pImpl->lastError = "Format conversion not yet implemented";
    return false;
}

void HuggingFaceWebhook::setCacheDir(const std::string& cacheDir) {
    pImpl->cacheDir = cacheDir;
    fs::create_directories(cacheDir);
}

std::string HuggingFaceWebhook::getCacheDir() const {
    return pImpl->cacheDir;
}

void HuggingFaceWebhook::clearCache() {
    if (fs::exists(pImpl->cacheDir)) {
        fs::remove_all(pImpl->cacheDir);
        fs::create_directories(pImpl->cacheDir);
    }
}

size_t HuggingFaceWebhook::getCacheSize() const {
    size_t totalSize = 0;
    
    if (fs::exists(pImpl->cacheDir)) {
        for (const auto& entry : fs::recursive_directory_iterator(pImpl->cacheDir)) {
            if (fs::is_regular_file(entry)) {
                totalSize += fs::file_size(entry);
            }
        }
    }
    
    return totalSize;
}

std::string HuggingFaceWebhook::getLastError() const {
    return pImpl->lastError;
}

bool HuggingFaceWebhook::hasError() const {
    return !pImpl->lastError.empty();
}

bool HuggingFaceWebhook::isDatasetDownloaded(const std::string& datasetId) const {
    return pImpl->isDatasetDownloaded(datasetId);
}

void HuggingFaceWebhook::markDatasetDownloaded(const std::string& datasetId) {
    pImpl->markDatasetDownloaded(datasetId);
}

void HuggingFaceWebhook::clearDownloadHistory() {
    pImpl->downloadedDatasets.clear();
    pImpl->saveDownloadedDatasets();
}

size_t HuggingFaceWebhook::getDownloadedCount() const {
    return pImpl->downloadedDatasets.size();
}

// Utility functions
namespace HFUtils {

std::string parseDatasetId(const std::string& url) {
    // Match patterns like:
    // https://huggingface.co/datasets/openai/gsm8k
    // huggingface.co/datasets/openai/gsm8k
    // openai/gsm8k
    
    std::regex pattern(R"((?:https?://)?(?:www\.)?(?:huggingface\.co/datasets/)?([^/]+/[^/?#]+))");
    std::smatch match;
    
    if (std::regex_search(url, match, pattern) && match.size() > 1) {
        return match[1].str();
    }
    
    // If already in correct format
    if (url.find('/') != std::string::npos && url.find("http") == std::string::npos) {
        return url;
    }
    
    return "";
}

std::string formatSize(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unitIndex = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unitIndex < 4) {
        size /= 1024.0;
        unitIndex++;
    }
    
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%.2f %s", size, units[unitIndex]);
    return std::string(buffer);
}

bool isValidDatasetId(const std::string& datasetId) {
    // Format: author/dataset-name
    std::regex pattern(R"(^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$)");
    return std::regex_match(datasetId, pattern);
}

std::string getDefaultCacheDir() {
    fs::path cacheDir;
    
#ifdef _WIN32
    const char* appData = std::getenv("LOCALAPPDATA");
    if (appData) {
        cacheDir = fs::path(appData) / "GRIM" / "huggingface_cache";
    } else {
        cacheDir = fs::temp_directory_path() / "grim_hf_cache";
    }
#else
    const char* home = std::getenv("HOME");
    if (home) {
        cacheDir = fs::path(home) / ".cache" / "grim" / "huggingface";
    } else {
        cacheDir = fs::temp_directory_path() / "grim_hf_cache";
    }
#endif
    
    return cacheDir.string();
}

} // namespace HFUtils

} // namespace DataCollection
} // namespace GRIM

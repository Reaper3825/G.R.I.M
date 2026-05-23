//======================================================//
//  GRIM Data Pipeline - Unified Tool
//  Collect → Verify → Merge → Tokenize
//  
//  Single executable for entire data preparation pipeline
//  Enhanced with persistent state tracking for deduplication
//======================================================//

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#endif

#include <iostream>
#include <iomanip>
#include <filesystem>
#include <regex>
#include <fstream>
#include <chrono>
#include <unordered_set>
#include <thread>
#include <mutex>
#include <atomic>
#include <future>
#include <queue>
#include "web_collector.hpp"
#include "verifier.hpp"
#include "data_preprocessor.hpp"
#include "data_splitter.hpp"
#include "collection_state.hpp"  // Persistent state tracking
#include "data_structurer.hpp"   // LLM-powered data structuring (Q/A generation)
#include "control/training_control_generated.h"
#include "resources/models/GRIM-text/Shared/HyperParameters/HyperParameters_GPU.hpp"  // For loading paths from ai_config.json
#include "checkpoint_data_generated.h"  // FlatBuffers checkpoint schema
#include <flatbuffers/flatbuffers.h>
#include <system_error>
#include "training_paths.hpp"
#include <zip.h>
#include <poppler/cpp/poppler-document.h>
#include <poppler/cpp/poppler-page.h>

using namespace GRIM::Training;
namespace fs = std::filesystem;

namespace {
    std::string g_checkpoint_dir;
    std::string g_verified_dir; 
    std::string g_output_dir = "data";
    std::string g_raw_dir;
    bool g_force_rebuild = false;  // Skip deduplication state for fresh rebuild
    bool g_skip_structuring = false;  // Skip LLM structuring step
    std::vector<std::string> g_qa_jsonl_paths;  // External Q/A JSONL files for merge ingestion
    GRIM::DataCollection::DataStructuringConfig g_structuring_config;  // LLM structuring config
    
    // Global state manager for tracking processed files
    std::unique_ptr<GRIM::DataCollection::CollectionStateManager> g_stateManager;

    // Extract a .zip archive to a destination directory (best-effort, skips dirs).
    bool extractZipArchive(const fs::path& zipPath, const fs::path& outputDir, size_t& filesExtracted) {
        filesExtracted = 0;
        int error = 0;
        zip_t* archive = zip_open(zipPath.string().c_str(), ZIP_RDONLY, &error);
        if (!archive) {
            zip_error_t zipError;
            zip_error_init_with_code(&zipError, error);
            std::cerr << "  [HF ZIP] Failed to open " << zipPath << ": " << zip_error_strerror(&zipError) << "\n";
            zip_error_fini(&zipError);
            return false;
        }

        zip_int64_t numEntries = zip_get_num_entries(archive, 0);
        fs::create_directories(outputDir);

        for (zip_int64_t i = 0; i < numEntries; i++) {
            const char* name = zip_get_name(archive, i, 0);
            if (!name) continue;

            std::string entryName(name);
            if (entryName.back() == '/' || entryName.back() == '\\') continue; // skip dirs

            zip_file_t* file = zip_fopen_index(archive, i, 0);
            if (!file) continue;

            fs::path outPath = outputDir / entryName;
            fs::create_directories(outPath.parent_path());

            std::ofstream outFile(outPath, std::ios::binary);
            if (!outFile.is_open()) {
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
            filesExtracted++;
        }

        zip_close(archive);
        return true;
    }

    bool loadJsonlFile(const fs::path& filePath,
                       const std::string& sourceTag,
                       std::vector<VerifiedEntry>& out,
                       size_t& badLines) {
        badLines = 0;
        std::ifstream file(filePath);
        if (!file.is_open()) return false;

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            try {
                nlohmann::json j = nlohmann::json::parse(line);
                VerifiedEntry ve;
                if (j.contains("content") && j["content"].is_string()) {
                    ve.content = j["content"].get<std::string>();
                } else if (j.is_string()) {
                    ve.content = j.get<std::string>();
                } else if (j.contains("text") && j["text"].is_string()) {
                    ve.content = j["text"].get<std::string>();
                } else {
                    badLines++;
                    continue;
                }
                ve.source_url = j.value("source_url", sourceTag);
                ve.source_type = j.value("source_type", "huggingface");
                ve.reliability_score = j.value("reliability_score", 0.8f);
                ve.verification_time = j.value("verification_time", (time_t)0);
                out.push_back(std::move(ve));
            } catch (...) {
                badLines++;
                continue;
            }
        }
        return true;
    }

    // Ingest external Q/A and conversational JSONL files into cleaned_texts.
    // Supported line formats:
    //   {"question": "...", "answer": "..."}              → "Q: ...\n\nA: ..."
    //   {"conversation": [{"role":"human","content":"..."}, {"role":"assistant","content":"..."}]}
    //                                                      → "Human: ...\n\nAssistant: ..."
    //   {"content": "..."}                                 → verbatim (already formatted)
    void ingestQaJsonlFiles(const std::vector<std::string>& paths,
                            std::vector<std::string>& cleaned_texts,
                            size_t& total_ingested,
                            size_t& total_bad_lines,
                            size_t& total_skipped) {
        total_ingested = 0;
        total_bad_lines = 0;
        total_skipped = 0;

        for (const auto& path : paths) {
            if (!fs::exists(path)) {
                std::cerr << "  [Q/A] Warning: file not found: " << path << "\n";
                continue;
            }

            // Skip if already processed (unless force rebuild)
            if (!g_force_rebuild && g_stateManager && g_stateManager->hasCollectedUrl(path)) {
                std::cout << "  [Q/A] Skipping already processed: " << path << "\n";
                total_skipped++;
                continue;
            }

            std::ifstream file(path);
            if (!file.is_open()) {
                std::cerr << "  [Q/A] Warning: cannot open: " << path << "\n";
                continue;
            }

            size_t file_ingested = 0;
            size_t file_bad = 0;
            std::string line;
            while (std::getline(file, line)) {
                if (line.empty()) continue;
                try {
                    nlohmann::json j = nlohmann::json::parse(line);

                    std::string content;

                    if (j.contains("question") && j["question"].is_string() &&
                        j.contains("answer") && j["answer"].is_string()) {
                        // Q/A pair
                        content = "Q: " + j["question"].get<std::string>()
                                + "\n\nA: " + j["answer"].get<std::string>();
                    } else if (j.contains("conversation") && j["conversation"].is_array()) {
                        // Multi-turn conversation
                        for (const auto& turn : j["conversation"]) {
                            if (!turn.contains("role") || !turn.contains("content")) continue;
                            std::string role = turn["role"].get<std::string>();
                            std::string text = turn["content"].get<std::string>();
                            if (!content.empty()) content += "\n\n";
                            if (role == "human" || role == "user") {
                                content += "Human: " + text;
                            } else if (role == "assistant" || role == "bot") {
                                content += "Assistant: " + text;
                            } else {
                                content += role + ": " + text;
                            }
                        }
                    } else if (j.contains("content") && j["content"].is_string()) {
                        // Already formatted content
                        content = j["content"].get<std::string>();
                    } else {
                        file_bad++;
                        continue;
                    }

                    if (!content.empty()) {
                        cleaned_texts.push_back(std::move(content));
                        file_ingested++;
                    }
                } catch (...) {
                    file_bad++;
                    continue;
                }
            }

            // Mark file as processed
            if (g_stateManager) {
                g_stateManager->markUrlCollected(path, "qa_jsonl");
            }

            std::cout << "  [Q/A] " << fs::path(path).filename().string()
                      << ": " << file_ingested << " entries";
            if (file_bad > 0) std::cout << " (" << file_bad << " bad lines)";
            std::cout << "\n";

            total_ingested += file_ingested;
            total_bad_lines += file_bad;
        }
    }

    // Extract text from PDF using poppler C++ API
    bool loadPdfFile(const fs::path& filePath,
                     const std::string& sourceTag,
                     std::vector<VerifiedEntry>& out) {
        try {
            std::unique_ptr<poppler::document> doc(poppler::document::load_from_file(filePath.string()));
            if (!doc) {
                std::cerr << "  [HF PDF ERROR] Failed to load: " << filePath.filename().string() << " (null doc)\n";
                return false;
            }
            if (doc->is_locked()) {
                std::cerr << "  [HF PDF ERROR] PDF is locked: " << filePath.filename().string() << "\n";
                return false;
            }

        std::string fullText;
        fullText.reserve(50000);  // Pre-allocate for typical paper size

        int numPages = doc->pages();
        for (int i = 0; i < numPages; i++) {
            std::unique_ptr<poppler::page> page(doc->create_page(i));
            if (!page) continue;
            
            // Extract text with layout preservation
            poppler::byte_array textBytes = page->text().to_utf8();
            if (!textBytes.empty()) {
                fullText.append(textBytes.data(), textBytes.size());
                fullText.push_back('\n');
            }
        }

        // Skip if too little text extracted (likely scanned/image PDF)
        if (fullText.size() < 200) return false;
        
        // Basic cleanup: normalize whitespace
        std::string cleaned;
        cleaned.reserve(fullText.size());
        bool lastWasSpace = false;
        for (char c : fullText) {
            if (c == '\r') continue;
            if (std::isspace(static_cast<unsigned char>(c))) {
                if (!lastWasSpace) {
                    cleaned += (c == '\n') ? '\n' : ' ';
                    lastWasSpace = true;
                }
            } else {
                cleaned += c;
                lastWasSpace = false;
            }
        }
        fullText = std::move(cleaned);
        
        // Split very long documents into chunks for better training
        const size_t MAX_CHUNK_SIZE = 8000;  // ~2000 tokens roughly
        
        if (fullText.size() <= MAX_CHUNK_SIZE) {
            VerifiedEntry ve;
            ve.content = std::move(fullText);
            ve.source_url = sourceTag;
            ve.source_type = "huggingface_pdf";
            ve.reliability_score = 0.70f;
            ve.verification_time = 0;
            out.push_back(std::move(ve));
        } else {
            // Split into chunks at paragraph boundaries
            size_t pos = 0;
            int chunkNum = 0;
            while (pos < fullText.size()) {
                size_t chunkEnd = std::min(pos + MAX_CHUNK_SIZE, fullText.size());
                
                // Try to break at paragraph boundary
                if (chunkEnd < fullText.size()) {
                    size_t breakPos = fullText.rfind("\n\n", chunkEnd);
                    if (breakPos != std::string::npos && breakPos > pos + MAX_CHUNK_SIZE / 2) {
                        chunkEnd = breakPos + 2;
                    } else {
                        breakPos = fullText.rfind("\n", chunkEnd);
                        if (breakPos != std::string::npos && breakPos > pos + MAX_CHUNK_SIZE / 2) {
                            chunkEnd = breakPos + 1;
                        }
                    }
                }
                
                std::string chunk = fullText.substr(pos, chunkEnd - pos);
                if (chunk.size() > 100) {  // Skip tiny chunks
                    VerifiedEntry ve;
                    ve.content = std::move(chunk);
                    ve.source_url = sourceTag + "#chunk" + std::to_string(chunkNum++);
                    ve.source_type = "huggingface_pdf";
                    ve.reliability_score = 0.70f;
                    ve.verification_time = 0;
                    out.push_back(std::move(ve));
                }
                pos = chunkEnd;
            }
        }
        return true;
        } catch (const std::exception& e) {
            std::cerr << "  [HF PDF ERROR] Exception loading " << filePath.filename().string() << ": " << e.what() << "\n";
            return false;
        } catch (...) {
            std::cerr << "  [HF PDF ERROR] Unknown exception loading " << filePath.filename().string() << "\n";
            return false;
        }
    }

    bool loadTextFile(const fs::path& filePath,
                      const std::string& sourceTag,
                      std::vector<VerifiedEntry>& out) {
        std::ifstream file(filePath);
        if (!file.is_open()) return false;
        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
        if (content.empty()) return false;
        VerifiedEntry ve;
        ve.content = content;
        ve.source_url = sourceTag;
        ve.source_type = "huggingface";
        ve.reliability_score = 0.7f;
        ve.verification_time = 0;
        out.push_back(std::move(ve));
        return true;
    }

    // Structure to hold PDF processing result
    struct PdfResult {
        std::string filePath;
        std::string sourceTag;
        std::vector<VerifiedEntry> entries;
        bool success;
        int64_t durationMs;
    };

    // Ingest any HuggingFace downloads under <output_dir>/huggingface/*
    // Enhanced: Uses state manager to track processed files and skip duplicates
    // PARALLEL: Processes PDFs using thread pool for massive speedup
    void ingestHuggingFaceDownloads(const fs::path& outputDir,
                                    std::vector<VerifiedEntry>& aggregated,
                                    size_t& totalFiles,
                                    size_t& totalExtracted,
                                    size_t& totalBadLines,
                                    size_t& totalSkipped) {
        totalFiles = 0;
        totalExtracted = 0;
        totalBadLines = 0;
        totalSkipped = 0;

        fs::path hf_root = outputDir / "huggingface";
        std::cout << "  [HF DEBUG] Checking path: " << hf_root.string() << "\n";
        if (!fs::exists(hf_root)) {
            std::cout << "  [HF DEBUG] Path does not exist, skipping\n";
            return;
        }
        std::cout << "  [HF DEBUG] Path exists, proceeding with ingestion\n";

        // Load limits from ai_config.json
        size_t maxPdfsPerDataset = 2000;  // Default
        size_t maxPdfsTotal = 10000;      // Default
        try {
            std::ifstream configFile("ai_config.json");
            if (configFile.is_open()) {
                nlohmann::json config = nlohmann::json::parse(configFile);
                if (config.contains("data_collection")) {
                    auto& dc = config["data_collection"];
                    maxPdfsPerDataset = dc.value("max_huggingface_pdfs_per_dataset", 2000);
                    maxPdfsTotal = dc.value("max_huggingface_pdfs_total", 10000);
                }
            }
        } catch (...) {}
        std::cout << "  [HF PDF] Limits: " << maxPdfsPerDataset << " per dataset, " << maxPdfsTotal << " total\n";

        // Use moderate parallelism - 4 threads balances speed vs stability
        // More threads = faster but higher crash risk from bad PDFs
        const unsigned int numThreads = 4;
        std::cout << "  [HF PDF] Processing PDFs with " << numThreads << " threads (resumable on crash)\n";

        size_t globalPdfCount = 0;  // Track total PDFs processed across all datasets

        // To avoid re-extracting on every run, drop archives into __unzipped/<zipname>/
        for (const auto& datasetDir : fs::directory_iterator(hf_root)) {
            if (!datasetDir.is_directory()) continue;
            if (globalPdfCount >= maxPdfsTotal) {
                std::cout << "  [HF PDF] Global limit reached (" << maxPdfsTotal << "), stopping\n";
                break;
            }
            const std::string datasetName = datasetDir.path().filename().string();
            fs::path unzipRoot = datasetDir.path() / "__unzipped__";

            // Extract any zip archives (sequential - IO bound)
            for (const auto& item : fs::recursive_directory_iterator(datasetDir.path())) {
                if (!item.is_regular_file()) continue;
                if (item.path().extension() == ".zip") {
                    fs::path targetDir = unzipRoot / item.path().stem();
                    if (!fs::exists(targetDir) || fs::is_empty(targetDir)) {
                        size_t extracted = 0;
                        if (extractZipArchive(item.path(), targetDir, extracted)) {
                            totalExtracted += extracted;
                            std::cout << "  [HF Merge] Extracted " << extracted
                                      << " files from " << item.path().filename().string()
                                      << " into " << targetDir.string() << "\n";
                        }
                    }
                }
            }

            // Collect files into separate lists by type
            std::vector<std::pair<fs::path, std::string>> pdfFiles;  // path, sourceTag
            std::vector<std::pair<fs::path, std::string>> jsonlFiles;
            std::vector<std::pair<fs::path, std::string>> txtFiles;
            
            for (const auto& item : fs::recursive_directory_iterator(datasetDir.path())) {
                if (!item.is_regular_file()) continue;
                if (item.path().extension() == ".zip") continue;

                std::string ext = item.path().extension().string();
                std::string sourceTag = "huggingface://" + datasetName + "/" + item.path().filename().string();
                std::string filePath = item.path().string();
                
                // Check if this file was already processed using state manager
                if (g_stateManager && g_stateManager->hasCollectedUrl(filePath)) {
                    totalSkipped++;
                    continue;
                }

                if (ext == ".jsonl") {
                    jsonlFiles.emplace_back(item.path(), sourceTag);
                } else if (ext == ".txt") {
                    txtFiles.emplace_back(item.path(), sourceTag);
                } else if (ext == ".pdf" || ext == ".PDF") {
                    // Apply per-dataset limit
                    if (pdfFiles.size() < maxPdfsPerDataset) {
                        pdfFiles.emplace_back(item.path(), sourceTag);
                    }
                }
            }
            
            // Also apply global limit
            size_t pdfsToProcess = std::min(pdfFiles.size(), maxPdfsTotal - globalPdfCount);
            if (pdfsToProcess < pdfFiles.size()) {
                std::cout << "  [HF PDF] Limiting to " << pdfsToProcess << " PDFs (global limit)\n";
                pdfFiles.resize(pdfsToProcess);
            }
            
            size_t totalToProcess = pdfFiles.size() + jsonlFiles.size() + txtFiles.size();
            std::cout << "  [HF Ingest] " << datasetName << ": " 
                      << pdfFiles.size() << " PDFs, "
                      << jsonlFiles.size() << " JSONLs, "
                      << txtFiles.size() << " TXTs "
                      << "(" << totalSkipped << " already processed)\n";

            // Process JSONL files (sequential - fast)
            for (const auto& [path, sourceTag] : jsonlFiles) {
                size_t bad = 0;
                if (loadJsonlFile(path, sourceTag, aggregated, bad)) {
                    totalFiles++;
                    totalBadLines += bad;
                    if (g_stateManager) g_stateManager->markUrlCollected(path.string(), "huggingface");
                }
            }
            
            // Process TXT files (sequential - fast)
            for (const auto& [path, sourceTag] : txtFiles) {
                if (loadTextFile(path, sourceTag, aggregated)) {
                    totalFiles++;
                    if (g_stateManager) g_stateManager->markUrlCollected(path.string(), "huggingface");
                }
            }

            // Process PDFs in PARALLEL using thread pool
            if (!pdfFiles.empty()) {
                std::cout << "  [HF PDF] Processing " << pdfFiles.size() << " PDFs in parallel...\n";
                
                std::atomic<size_t> pdfIndex{0};
                std::atomic<size_t> successCount{0};
                std::atomic<size_t> failCount{0};
                std::atomic<size_t> lastProgressAt{0};
                std::mutex progressMutex;  // Only for progress output
                
                // Thread-local storage for results (no mutex contention!)
                std::vector<std::vector<PdfResult>> threadResults(numThreads);
                for (auto& v : threadResults) {
                    v.reserve(pdfFiles.size() / numThreads + 100);
                }
                
                auto startTime = std::chrono::steady_clock::now();
                
                // Worker function - single-threaded for safety
                auto worker = [&](unsigned int threadId) {
                    auto& localResults = threadResults[threadId];
                    size_t saveCounter = 0;  // Save state every 100 PDFs
                    
                    while (true) {
                        size_t idx = pdfIndex.fetch_add(1);
                        if (idx >= pdfFiles.size()) break;
                        
                        const auto& [path, sourceTag] = pdfFiles[idx];
                        
                        PdfResult result;
                        result.filePath = path.string();
                        result.sourceTag = sourceTag;
                        
                        auto t0 = std::chrono::steady_clock::now();
                        result.success = loadPdfFile(path, sourceTag, result.entries);
                        result.durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0).count();
                        
                        if (result.success) {
                            successCount.fetch_add(1);
                            // Mark URL as collected IMMEDIATELY so crash preserves progress
                            if (g_stateManager) {
                                g_stateManager->markUrlCollected(path.string(), "huggingface");
                            }
                        } else {
                            failCount.fetch_add(1);
                            // Still mark failed PDFs so we don't retry them
                            if (g_stateManager) {
                                g_stateManager->markUrlCollected(path.string(), "huggingface_failed");
                            }
                        }
                        
                        // Store result in thread-local vector
                        localResults.push_back(std::move(result));
                        
                        // Save state every 100 PDFs for crash recovery
                        if (++saveCounter >= 100) {
                            if (g_stateManager) g_stateManager->saveState();
                            saveCounter = 0;
                        }
                        
                        // Progress update every 500 PDFs
                        size_t done = successCount.load() + failCount.load();
                        size_t lastDone = lastProgressAt.load();
                        if ((done - lastDone >= 500 || done == pdfFiles.size()) && 
                            lastProgressAt.compare_exchange_weak(lastDone, done)) {
                            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                                std::chrono::steady_clock::now() - startTime).count();
                            float pct = 100.0f * done / pdfFiles.size();
                            float rate = elapsed > 0 ? (float)done / elapsed : 0;
                            std::lock_guard<std::mutex> lock(progressMutex);
                            std::cout << "  [HF PDF] " << done << "/" << pdfFiles.size() 
                                      << " (" << std::fixed << std::setprecision(1) << pct << "%) "
                                      << "- " << successCount.load() << " ok, " << failCount.load() << " failed "
                                      << "(" << std::setprecision(1) << rate << " PDFs/sec)\n";
                        }
                    }
                };
                
                // Launch threads
                std::vector<std::thread> threads;
                threads.reserve(numThreads);
                for (unsigned int i = 0; i < numThreads; i++) {
                    threads.emplace_back(worker, i);
                }
                
                // Wait for all threads
                for (auto& t : threads) {
                    t.join();
                }
                
                auto totalElapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - startTime).count();
                std::cout << "  [HF PDF] Completed " << pdfFiles.size() << " PDFs in " 
                          << totalElapsed << "s (" << successCount.load() << " succeeded, "
                          << failCount.load() << " failed)\n";
                
                // Merge all thread results (sequential, after workers complete)
                size_t totalEntries = 0;
                for (auto& localResults : threadResults) {
                    for (auto& result : localResults) {
                        if (result.success && !result.entries.empty()) {
                            totalFiles++;
                            totalEntries += result.entries.size();
                            aggregated.insert(aggregated.end(), 
                                             std::make_move_iterator(result.entries.begin()),
                                             std::make_move_iterator(result.entries.end()));
                            // URL already marked in worker loop - no need to duplicate
                        }
                    }
                }
                std::cout << "  [HF PDF] Added " << totalEntries << " text entries from PDFs\n";
                
                // Update global counter
                globalPdfCount += pdfFiles.size();
                
                // Final state save
                if (g_stateManager) g_stateManager->saveState();
            }
        }
    }
}

// Forward declarations
int runCollect(const std::string& config_path, std::function<void(float)> progressCallback);
int runVerify(const std::string& raw_dir, const std::string& verified_dir);
int runMerge(const std::string& checkpoint_dir, const std::string& verified_dir,
             const std::string& output_dir, bool skip_verification,
             bool retrain_vocab, int vocab_size, std::function<void(float)> progressCallback);

static void printUsage() {
    std::cout << "GRIM Data Pipeline - Unified data preparation tool\n\n";
    std::cout << "Usage: grim_data_pipeline [MODE] [OPTIONS]\n\n";
    std::cout << "Modes:\n";
    std::cout << "  collect      Collect data from web sources\n";
    std::cout << "  verify       Verify collected data quality\n";
    std::cout << "  merge        Merge checkpoints and prepare for training\n";
    std::cout << "  full         Run complete pipeline (collect → verify → merge)\n\n";
    std::cout << "Options:\n";
    std::cout << "  --config <path>          Source configuration (default: source_data.json)\n";
    std::cout << "  --checkpoint-dir <dir>   Directory with checkpoint files\n";
    std::cout << "  --raw-dir <dir>          Raw data directory (default: data/raw)\n";
    std::cout << "  --verified-dir <dir>     Verified data directory (default: data/verified)\n";
    std::cout << "  --output-dir <dir>       Output directory (default: data)\n";
    std::cout << "  --skip-verification      Skip re-verification when merging\n";
    std::cout << "  --force-rebuild          Ignore deduplication state, rebuild from all verified data\n";
    std::cout << "  --retrain-vocab          Force retrain tokenizer (ignores existing vocab)\n";
    std::cout << "  --vocab-size <size>      Vocabulary size for tokenizer (default: 50000)\n";
    std::cout << "  --qa-jsonl <path>        External Q/A JSONL file (repeatable)\n";
    std::cout << "  --skip-structuring       Skip LLM data structuring step\n\n";
    std::cout << "Examples:\n";
    std::cout << "  grim_data_pipeline collect --config source_data.json\n";
    std::cout << "  grim_data_pipeline verify --raw-dir data/raw\n";
    std::cout << "  grim_data_pipeline merge --checkpoint-dir ../../../data\n";
    std::cout << "  grim_data_pipeline merge --retrain-vocab --vocab-size 30000\n";
    std::cout << "  grim_data_pipeline full --config source_data.json\n";
}

void updateCollectionProgress(float progress, const std::string& phase = "Collecting") {
    try {
        flatbuffers::FlatBufferBuilder builder(1024);
        auto phase_str = builder.CreateString(phase);
        auto error_str = builder.CreateString("");
        
        auto stats = GRIMText::Control::CreateTrainingStats(builder,
            0, 0, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, progress, phase_str, error_str, 0, 0);
        
        auto config = GRIMText::Control::CreateTrainingConfig(builder);
        
        int64_t timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        auto status_response = GRIMText::Control::CreateStatusResponse(builder,
            GRIMText::Control::TrainingState_Collecting,
            stats, config, true, timestamp);
        
        builder.Finish(status_response);
        
        auto statusPath = GRIM::Training::getTrainingStatusFilePath();
        std::error_code mkdirErr;
        fs::create_directories(statusPath.parent_path(), mkdirErr);

        std::ofstream statusFile(statusPath, std::ios::binary);
        if (statusFile.is_open()) {
            statusFile.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
        }
    } catch (...) {}
}

int  runCollect(const std::string& config_path) {
    return runCollect(config_path, nullptr);
}

int runCollect(const std::string& config_path, std::function<void(float)> progressCallback) {
    std::cout << "=== COLLECTING DATA ===\n\n";
    
    if (!fs::exists(config_path)) {
        std::cerr << "ERROR: Config file not found: " << config_path << "\n";
        return 1;
    }
    
    std::cout << "[1/2] Loading sources from " << config_path << "...\n";
    
    WebDataCollector collector;
    collector.setVerbose(true);
    
    // Wire up progress callback if provided
    if (progressCallback) {
        collector.setProgressCallback(progressCallback);
    }
    
    // Use the unified config loader — it handles fetcher auto-detection,
    // content filters, crawl_depth, source_type_str, and all other fields.
    // The old manual JSON parsing was missing fetcher_type/autoDetectFetcher(),
    // causing ALL sources to fall through to HTML_CRAWL regardless of type.
    if (!collector.loadConfigFromJson(config_path)) {
        std::cerr << "ERROR: Failed to load config from " << config_path << "\n";
        return 1;
    }
    
    std::string resolved_raw_dir = g_raw_dir.empty() ? GRIM::Config::getCollectedDir() : g_raw_dir;
    if (resolved_raw_dir.empty()) {
        resolved_raw_dir = "data/raw";
    }
    fs::path raw_dir_path(resolved_raw_dir);
    std::error_code mkdirErr;
    fs::create_directories(raw_dir_path, mkdirErr);

    collector.setOutputDir(resolved_raw_dir);

    std::cout << "[2/2] Collecting data...\n";
    size_t collected = collector.collectData();
    std::cout << "\xE2\x9C\x93 Collected entries: " << collected << "\n";

    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    std::string stamp = std::to_string(timestamp);

    fs::path raw_output_path = raw_dir_path / ("collected_" + stamp + ".jsonl");
    if (collector.saveToJsonl(raw_output_path.string())) {
        std::cout << "\xE2\x9C\x93 Saved raw JSONL: " << raw_output_path.string() << "\n";
    } else {
        std::cerr << "WARNING: Failed to save raw JSONL output." << std::endl;
    }

    if (!g_checkpoint_dir.empty()) {
        fs::path checkpoint_dir = g_checkpoint_dir;
        fs::create_directories(checkpoint_dir, mkdirErr);
        fs::path checkpoint_path = checkpoint_dir / ("checkpoint_" + stamp + ".ckpt");
        if (collector.saveCheckpoint(checkpoint_path.string())) {
            std::cout << "\xE2\x9C\x93 Saved checkpoint: " << checkpoint_path.string() << "\n";
        } else {
            std::cerr << "WARNING: Failed to save checkpoint." << std::endl;
        }
    }

    const auto& stats = collector.getStats();
    std::cout << stats.toString() << std::endl;
    std::cout << "=== COLLECTION COMPLETE ===\n\n";
    return 0;
}

int runVerify(const std::string& raw_dir, const std::string& verified_dir) {
    try {
        std::cout << "=== VERIFYING DATA ===\n\n";
        std::cout << "[Verify] Input raw_dir param: " << (raw_dir.empty() ? "(empty)" : raw_dir) << "\n";
        std::cout << "[Verify] Input verified_dir param: " << (verified_dir.empty() ? "(empty)" : verified_dir) << "\n";
        std::cout.flush();

    std::string resolved_raw_dir = raw_dir.empty() ? g_raw_dir : raw_dir;
    if (resolved_raw_dir.empty()) {
        resolved_raw_dir = GRIM::Config::getCollectedDir();
    }
    if (resolved_raw_dir.empty()) {
        resolved_raw_dir = "data/raw";
    }

    std::string resolved_verified_dir = verified_dir.empty() ? g_verified_dir : verified_dir;
    if (resolved_verified_dir.empty()) {
        resolved_verified_dir = GRIM::Config::getVerifiedDir();
    }
    if (resolved_verified_dir.empty()) {
        resolved_verified_dir = "data/verified";
    }

    std::cout << "[Verify] Resolved raw_dir: " << resolved_raw_dir << "\n";
    std::cout << "[Verify] Resolved verified_dir: " << resolved_verified_dir << "\n";
    std::cout.flush();

    if (!fs::exists(resolved_raw_dir)) {
        std::cerr << "ERROR: Raw directory not found: " << resolved_raw_dir << "\n";
        return 1;
    }
    fs::create_directories(resolved_verified_dir);
    
    // Initialize state manager for tracking verified content
    std::string checkpoint_dir = GRIM::Config::getCheckpointDir();
    if (checkpoint_dir.empty()) checkpoint_dir = "data/checkpoints";
    std::string stateDir = checkpoint_dir + "/collection_state";
    if (!g_stateManager) {
        g_stateManager = std::make_unique<GRIM::DataCollection::CollectionStateManager>(stateDir);
        std::cout << "[State] Loaded tracking data for deduplication\n";
    }

    std::cout << "[Verify] Creating verifier config...\n";
    std::cout.flush();
    
    Config verifier_config;
    verifier_config.input_dir = resolved_raw_dir;
    verifier_config.output_dir = resolved_verified_dir;
    verifier_config.verbose_logging = true;
    verifier_config.save_rejected = true;

    std::cout << "[Verify] Constructing Verifier object...\n";
    std::cout.flush();
    
    Verifier verifier(verifier_config);

    std::cout << "[Verify] Verifier constructed successfully\n";
    std::cout.flush();

    auto unverified_entries = verifier.load_unverified_entries();
    if (unverified_entries.empty()) {
        std::cout << "No new entries to verify in " << resolved_raw_dir << "\n";
        return 0;
    }

    std::cout << "Loaded " << unverified_entries.size() << " unverified entries.\n";
    std::cout << "Verifying entries...\n";
    
    auto verified_entries = verifier.verify_entries(unverified_entries, 
        [](float progress, size_t processed, size_t total) {
            std::cout << "\r  Progress: " << std::fixed << std::setprecision(1) << progress 
                      << "% (" << processed << "/" << total << ")" << std::flush;
        });
    
    std::cout << "\n\xE2\x9C\x93 Verification complete: " << verified_entries.size() << " entries passed filters.\n";

    if (!verifier.save_verified_entries(verified_entries)) {
        std::cerr << "ERROR: Failed to save verified entries to " << resolved_verified_dir << "\n";
        return 1;
    }

    auto stats = verifier.get_stats();
    stats.writeSummaryToLog();
    std::cout << "Saved verification summary.\n";
    std::cout << "High quality: " << stats.high_quality_count
              << ", Medium: " << stats.medium_quality_count
              << ", Low: " << stats.low_quality_count << "\n";
    
    return 0;
    
    } catch (const std::exception& e) {
        std::cerr << "\n[VERIFY ERROR] Fatal exception: " << e.what() << "\n";
        std::cerr << "[VERIFY ERROR] Verification failed. Check paths and model files.\n";
        return 1;
    } catch (...) {
        std::cerr << "\n[VERIFY ERROR] Unknown fatal exception during verification.\n";
        return 1;
    }
}

int runVerify_Cleanup(const std::string& raw_dir) {
    // Separated cleanup function to avoid crash during verification
    std::string resolved_raw_dir = raw_dir.empty() ? g_raw_dir : raw_dir;
    if (resolved_raw_dir.empty()) {
        resolved_raw_dir = GRIM::Config::getCollectedDir();
    }
    if (resolved_raw_dir.empty() || !fs::exists(resolved_raw_dir)) {
        return 0;
    }
    
    // Clean up processed collected files to prevent reprocessing duplicates
    std::cout << "\n[Cleanup] Removing processed collected files...\n";
    int files_removed = 0;
    try {
        for (const auto& entry : fs::directory_iterator(resolved_raw_dir)) {
            if (entry.is_regular_file() && 
                (entry.path().extension() == ".jsonl" || entry.path().extension() == ".json")) {
                std::string filename = entry.path().filename().string();
                if (filename.find("collected_") == 0) {
                    fs::remove(entry.path());
                    files_removed++;
                }
            }
        }
        std::cout << "\xE2\x9C\x93 Removed " << files_removed << " collected files\n";
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to clean up some collected files: " << e.what() << "\n";
    }
    
    std::cout << "=== VERIFICATION COMPLETE ===\n\n";
    return 0;
}

int runMerge(const std::string& checkpoint_dir, const std::string& verified_dir,
             const std::string& output_dir, bool skip_verification,
             bool retrain_vocab, int vocab_size, std::function<void(float)> progressCallback) {
    try {
        (void)skip_verification;
        (void)retrain_vocab;
        (void)vocab_size;
        
        // Current phase for UI status
        std::string currentPhase = "Loading";

        auto reportProgress = [&](float p) {
            if (progressCallback) {
                // Clamp to [0, 100] to avoid UI weirdness
                if (p < 0.0f) p = 0.0f;
                if (p > 100.0f) p = 100.0f;
                progressCallback(p);
            }
            // Also write to status file with phase label
            updateCollectionProgress(p, currentPhase);
        };

        std::cout << "=== MERGING DATA ===\n\n";

        // Ensure UI shows activity immediately for merge-only / merge-rebuild modes.
        reportProgress(1.0f);

    std::string resolved_checkpoint_dir = checkpoint_dir.empty() ? g_checkpoint_dir : checkpoint_dir;
    std::string resolved_verified_dir = verified_dir.empty() ? g_verified_dir : verified_dir;
    std::string resolved_output_dir = output_dir.empty() ? g_output_dir : output_dir;

    if (resolved_checkpoint_dir.empty()) {
        resolved_checkpoint_dir = GRIM::Config::getCheckpointDir();
        if (resolved_checkpoint_dir.empty()) {
            resolved_checkpoint_dir = "data/checkpoints";
        }
    }
    if (resolved_verified_dir.empty()) {
        resolved_verified_dir = GRIM::Config::getVerifiedDir();
        if (resolved_verified_dir.empty()) {
            resolved_verified_dir = "data/verified";
        }
    }
    if (resolved_output_dir.empty()) {
        // Get the training data directory from ai_config.json (same location DataLoader expects)
        auto snapshot = GRIM::Config::loadAiConfigSnapshot();
        if (snapshot && snapshot->has_grim_paths && !snapshot->grim_text_training_data.empty()) {
            resolved_output_dir = fs::path(snapshot->grim_text_training_data).parent_path().string();
            std::cout << "  Using training data directory from config: " << resolved_output_dir << "\n";
        } else {
            resolved_output_dir = "data";
        }
    }
    
    // Initialize state manager for tracking processed content
    std::string stateDir = resolved_checkpoint_dir + "/collection_state";
    if (!g_stateManager) {
        g_stateManager = std::make_unique<GRIM::DataCollection::CollectionStateManager>(stateDir);
        std::cout << "[State] Loaded " << g_stateManager->getTotalUniqueUrls() << " tracked URLs, "
                  << g_stateManager->getTotalUniqueContent() << " content hashes\n";
    }
    
    // Auto-detect missing training data: if we have content hashes but no .grmt files,
    // the user likely deleted training data and wants to rebuild
    if (!g_force_rebuild && g_stateManager && g_stateManager->getTotalUniqueContent() > 0) {
        fs::path training_grmt = fs::path(resolved_output_dir) / "training_data.grmt";
        if (!fs::exists(training_grmt)) {
            std::cout << "[Auto-Rebuild] Training data missing but " 
                      << g_stateManager->getTotalUniqueContent() 
                      << " content hashes found.\n";
            std::cout << "  Enabling force-rebuild to regenerate training data.\n\n";
            g_force_rebuild = true;
        }
    }

    // Load checkpoint files if provided (FlatBuffer only)
    std::vector<std::string> checkpoint_files;
    if (!resolved_checkpoint_dir.empty() && fs::exists(resolved_checkpoint_dir)) {
        for (const auto& entry : fs::directory_iterator(resolved_checkpoint_dir)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                std::string ext = entry.path().extension().string();
                // Only load .ckpt FlatBuffer files
                if (filename.substr(0, 11) == "checkpoint_" && ext == ".ckpt") {
                    checkpoint_files.push_back(entry.path().string());
                }
            }
        }
    }

    currentPhase = "Loading";
    std::cout << "[1/7] Loading data...\n";
    std::cout << "  Checkpoints: " << checkpoint_files.size() << "\n";
    reportProgress(5.0f);

    // Load existing verified entries from JSONL files
    std::vector<VerifiedEntry> existing_verified;

    if (fs::exists(resolved_verified_dir)) {
        for (const auto& entry : fs::directory_iterator(resolved_verified_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".jsonl") {
                std::ifstream file(entry.path());
                std::string line;

                while (std::getline(file, line)) {
                    if (line.empty()) continue;

                    try {
                        nlohmann::json j = nlohmann::json::parse(line);

                        VerifiedEntry ve;
                        ve.content = j["content"].get<std::string>();
                        ve.source_url = j["source_url"].get<std::string>();
                        ve.source_type = j["source_type"].get<std::string>();
                        ve.reliability_score = j.value("reliability_score", 0.8f);
                        ve.verification_time = j.value("verification_time", (time_t)0);

                        existing_verified.push_back(ve);
                    } catch (...) {
                        continue;  // Skip malformed entries
                    }
                }
            }
        }
    }

    reportProgress(10.0f);

    std::cout << "  Verified entries: " << existing_verified.size() << "\n";

    // Ingest HuggingFace downloads (jsonl/txt and zip archives) under output_dir/huggingface
    std::vector<VerifiedEntry> hf_entries;
    size_t hf_files = 0, hf_extracted = 0, hf_bad = 0, hf_skipped = 0;
    ingestHuggingFaceDownloads(fs::path(resolved_output_dir), hf_entries, hf_files, hf_extracted, hf_bad, hf_skipped);
    if (!hf_entries.empty() || hf_skipped > 0) {
        std::cout << "  HuggingFace entries: " << hf_entries.size()
                  << " (files: " << hf_files << ", extracted: " << hf_extracted
                  << ", skipped lines: " << hf_bad;
        if (hf_skipped > 0) {
            std::cout << ", already processed: " << hf_skipped;
        }
        std::cout << ")\n";
    }

    // Merge HF entries with existing verified pool so they pass through dedup/preprocess
    existing_verified.insert(existing_verified.end(), hf_entries.begin(), hf_entries.end());

    // Load checkpoint files (FlatBuffer format only)
    // Track which checkpoints have been processed to avoid re-processing
    std::vector<VerifiedEntry> checkpoint_entries;
    size_t checkpoints_skipped = 0;
    size_t checkpoints_loaded = 0;
    
    for (const auto& checkpoint_file : checkpoint_files) {
        // Check if this checkpoint was already processed
        if (g_stateManager && g_stateManager->hasCollectedUrl(checkpoint_file)) {
            checkpoints_skipped++;
            continue;
        }
        
        try {
            // Load FlatBuffer checkpoint
            std::ifstream file(checkpoint_file, std::ios::binary);
            if (!file.is_open()) continue;

            // Read entire file
            file.seekg(0, std::ios::end);
            size_t fileSize = file.tellg();
            file.seekg(0, std::ios::beg);

            std::vector<uint8_t> buffer(fileSize);
            file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
            file.close();

            // Parse FlatBuffer
            auto checkpoint = GRIMCheckpoint::GetCheckpoint(buffer.data());
            if (!checkpoint || !checkpoint->entries()) {
                std::cerr << "  Warning: Invalid FlatBuffer format in " << checkpoint_file << "\n";
                continue;
            }

            size_t entries_from_checkpoint = 0;
            
            // Convert to VerifiedEntry
            for (const auto* fb_entry : *checkpoint->entries()) {
                if (!fb_entry || !fb_entry->content() || !fb_entry->source_url()) continue;

                VerifiedEntry ve;
                ve.content = fb_entry->content()->str();
                ve.source_url = fb_entry->source_url()->str();
                ve.source_type = fb_entry->source_type() ? fb_entry->source_type()->str() : "web";
                ve.reliability_score = fb_entry->reliability_score();
                ve.verification_time = fb_entry->fetch_timestamp();

                if (!ve.content.empty()) {
                    checkpoint_entries.push_back(ve);
                    entries_from_checkpoint++;
                }
            }

            std::cout << "  Loaded " << entries_from_checkpoint
                      << " entries from " << fs::path(checkpoint_file).filename().string()
                      << " (" << (fileSize / 1024) << " KB FlatBuffer)" << std::endl;
            
            // Mark checkpoint as processed
            if (g_stateManager) {
                g_stateManager->markUrlCollected(checkpoint_file, "checkpoint");
            }
            checkpoints_loaded++;

        } catch (const std::exception& e) {
            std::cerr << "  Warning: Failed to read " << checkpoint_file
                      << ": " << e.what() << "\n";
            continue;
        }

        // Coarse progress during checkpoint ingestion.
        if (!checkpoint_files.empty() && (checkpoints_loaded + checkpoints_skipped) % 5 == 0) {
            const float t = static_cast<float>(checkpoints_loaded + checkpoints_skipped) /
                            static_cast<float>(checkpoint_files.size());
            reportProgress(10.0f + t * 15.0f); // 10% -> 25%
        }
    }

    reportProgress(25.0f);

    std::cout << "  Checkpoint entries: " << checkpoint_entries.size();
    if (checkpoints_skipped > 0) {
        std::cout << " (skipped " << checkpoints_skipped << " already processed checkpoints)";
    }
    std::cout << "\n\n";

    // Load previously merged cache entries so they are preserved across runs.
    // Without this, the final trunc-write would discard all data from prior merges.
    std::vector<VerifiedEntry> previously_merged;
    {
        fs::path cache_path = fs::path(resolved_output_dir) / "merged_verified_cache.jsonl";
        if (fs::exists(cache_path)) {
            std::ifstream prev(cache_path);
            std::string line;
            while (std::getline(prev, line)) {
                if (line.empty()) continue;
                try {
                    nlohmann::json j = nlohmann::json::parse(line);
                    VerifiedEntry ve;
                    ve.content = j["content"].get<std::string>();
                    ve.source_url = j.value("source_url", std::string("merged_cache"));
                    ve.source_type = j.value("source_type", std::string("cached"));
                    ve.reliability_score = j.value("reliability_score", 0.9f);
                    ve.verification_time = j.value("verification_time", (time_t)0);
                    previously_merged.push_back(std::move(ve));
                } catch (...) {
                    continue;
                }
            }
            std::cout << "  Previously merged cache: " << previously_merged.size() << " entries\n";
        }
    }

    // Combine all data
    std::vector<VerifiedEntry> all_verified = existing_verified;
    all_verified.insert(all_verified.end(), checkpoint_entries.begin(), checkpoint_entries.end());

    // Deduplicate using persistent state manager
    currentPhase = "Deduplicating";
    std::cout << "[2/7] Deduplicating...\n";
    if (g_force_rebuild) {
        std::cout << "  Force rebuild enabled - ignoring previous processing state\n";
    }
    std::vector<VerifiedEntry> deduplicated;
    size_t skipped_duplicates = 0;
    std::unordered_set<uint64_t> batch_hashes;  // In-batch dedup

    for (const auto& entry : all_verified) {
        // Check if this content was already merged into training data in a previous run
        // (NOT hasSeenContent — that's the collector's dedup, which would reject everything)
        if (!g_force_rebuild && g_stateManager && g_stateManager->hasMergedContent(entry.content)) {
            skipped_duplicates++;
            continue;
        }
        
        // Deduplicate within this batch (verified + checkpoint entries may overlap)
        uint64_t h = GRIM::DataCollection::computeContentHash(entry.content);
        if (!batch_hashes.insert(h).second) {
            skipped_duplicates++;
            continue;
        }
        
        // New unique content - add and mark as merged
        deduplicated.push_back(entry);
        if (g_stateManager) {
            g_stateManager->markContentMerged(entry.content);
        }

        // Coarse progress during dedup.
        if (!all_verified.empty() && (deduplicated.size() % 500) == 0) {
            // Not exact (deduplicated != processed), but good enough for UI feedback.
            const size_t denom = all_verified.size() == 0 ? 1 : all_verified.size();
            const float t = static_cast<float>(deduplicated.size()) /
                            static_cast<float>(denom);
            reportProgress(25.0f + t * 10.0f); // 25% -> 35%
        }
    }

    reportProgress(35.0f);

    std::cout << "\xE2\x9C\x93 Deduplicated: " << deduplicated.size() << " unique entries";
    if (skipped_duplicates > 0) {
        std::cout << " (" << skipped_duplicates << " duplicates removed)";
    }
    std::cout << "\n\n";

    // Safety check: Ensure we have data to process
    if (deduplicated.empty()) {
        // Even with no new entries, if structuring is enabled and there are previously
        // merged entries, we should continue to re-inject and structure them incrementally.
        bool has_structuring_work = g_structuring_config.enabled && !g_skip_structuring
            && g_structuring_config.mode != "raw" && !previously_merged.empty();

        if (!has_structuring_work) {
            std::cout << "WARNING: No data remaining after deduplication. Nothing to merge.\n";
            std::cout << "This usually means all data has already been processed.\n";
            std::cout << "TIP: Use 'Force Rebuild' button or --force-rebuild flag to reprocess all data.\n";
            std::cout << "=== MERGE COMPLETE (NO NEW DATA) ===\n";
            return 0;  // Not an error, just nothing to do
        }
        std::cout << "  No new entries, but " << previously_merged.size()
                  << " cached entries may need structuring. Continuing...\n\n";
    }

    // Verify deduplicated data
    currentPhase = "Verifying";
    std::cout << "[3/7] Verifying quality...\n";
    
    // Convert to UnverifiedEntry for verification
    std::vector<UnverifiedEntry> to_verify;
    to_verify.reserve(deduplicated.size());
    for (const auto& entry : deduplicated) {
        UnverifiedEntry ue;
        ue.content = entry.content;
        ue.source_url = entry.source_url;
        ue.source_type = entry.source_type;
        to_verify.push_back(std::move(ue));
    }
    
    // Configure and run verifier
    Config verifier_config;
    verifier_config.input_dir = resolved_verified_dir;
    verifier_config.output_dir = resolved_verified_dir;
    verifier_config.verbose_logging = false;  // Less noise during merge
    verifier_config.save_rejected = false;
    verifier_config.progressive_filtering = true;
    
    Verifier verifier(verifier_config);
    
    std::cout << "  Verifying " << to_verify.size() << " entries...\n";
    auto verified_entries = verifier.verify_entries(to_verify,
        [&reportProgress](float progress, size_t processed, size_t total) {
            // Map verification progress to 35% -> 55%
            reportProgress(35.0f + progress * 0.20f);
            if (processed % 1000 == 0 || processed == total) {
                std::cout << "\r  Verification: " << processed << "/" << total
                          << " (" << std::fixed << std::setprecision(1) << progress << "%)" << std::flush;
            }
        });
    std::cout << "\n";
    
    auto vstats = verifier.get_stats();
    std::cout << "\xE2\x9C\x93 Verified: " << verified_entries.size() << " entries passed\n";
    std::cout << "  High quality: " << vstats.high_quality_count
              << ", Medium: " << vstats.medium_quality_count
              << ", Low: " << vstats.low_quality_count
              << ", Rejected: " << vstats.failed_verification << "\n\n";
    
    reportProgress(55.0f);
    
    // Safety check after verification
    if (verified_entries.empty() && previously_merged.empty()) {
        std::cout << "WARNING: No data passed verification. Nothing to output.\n";
        std::cout << "=== MERGE COMPLETE (NO VERIFIED DATA) ===\n";
        return 0;
    }
    
    // Preprocess verified data
    currentPhase = "Preprocessing";
    std::cout << "[4/7] Preprocessing...\n";
    
    // Load max_seq_len from the single config snapshot.
    auto snapshot = GRIM::Config::loadAiConfigSnapshot();
    if (!snapshot || !snapshot->has_training) {
        throw std::runtime_error("runMerge: loadAiConfigSnapshot did not produce training hyperparameters");
    }
    const int max_seq_len = snapshot->hyperparameters.architecture.max_seq_len;
    std::cout << "  Using max_seq_len=" << max_seq_len << " from ai_config.json\n";
    
    // Calculate max chars based on max_seq_len (approx 4 chars per token)
    const int chars_per_token = 4;
    int max_chars = max_seq_len * chars_per_token;
    std::cout << "  Max sample length: " << max_chars << " chars (~" << max_seq_len << " tokens)\n";
    
    std::cout << "  Configuring preprocessor...\n";
    PreprocessorConfig prep_config;
    prep_config.remove_html = true;     // Remove HTML tags for clean text
    prep_config.min_length = 75;        // Accept shorter high-quality content
    prep_config.min_words = 15;         // Allow concise but valuable snippets
    prep_config.min_alpha_ratio = 0.5f; // Allow technical/structured content
    prep_config.max_length = 100000;    // Allow long-form content (up to 100K chars)
    prep_config.max_repetition_ratio = 0.5f;
    prep_config.dedup_threshold = 0.9f;
    
    // Use max_seq_len from config for chunking limit
    prep_config.max_token_estimate_chars = max_chars;
    
    std::cout << "  Creating preprocessor instance...\n";
    DataPreprocessor preprocessor(prep_config);
    std::cout << "  ✓ Preprocessor ready\n";

    std::vector<std::string> cleaned_texts;
    size_t chunks_created = 0;
    
    std::cout << "  Processing " << verified_entries.size() << " entries...\n";

    size_t processed_entries = 0;
    size_t last_reported_entries = 0;
    auto last_report_time = std::chrono::steady_clock::now();
    size_t pruned_count = 0;
    
    for (const auto& entry : verified_entries) {
        // IMPORTANT: Nothing here should invoke the model tokenizer.
        // We only do text cleanup + lightweight word-splitting for heuristics.

        // Chunk the raw text FIRST to avoid running expensive cleanup on huge blobs.
        auto raw_chunks = preprocessor.chunkLongText(entry.content);

        for (const auto& raw_chunk : raw_chunks) {
            // Preprocess per-chunk
            std::string clean = preprocessor.preprocess(raw_chunk);

            // Cleaned text may still exceed limit (rare), so re-chunk if needed.
            auto clean_chunks = preprocessor.chunkLongText(clean);
            for (const auto& chunk : clean_chunks) {
                // Prune corrupted/low-quality chunks
                if (!preprocessor.passesQualityFilter(chunk)) {
                    pruned_count++;
                    continue;
                }

                if (preprocessor.isDuplicate(chunk)) continue;

                cleaned_texts.push_back(chunk);
                if (raw_chunks.size() > 1 || clean_chunks.size() > 1) chunks_created++;
            }
        }

        processed_entries++;
        // Periodic progress update during preprocessing (keeps UI/logs alive)
        if (!verified_entries.empty()) {
            const auto now = std::chrono::steady_clock::now();
            const bool time_to_report = (now - last_report_time) >= std::chrono::milliseconds(750);
            const bool count_to_report = (processed_entries - last_reported_entries) >= 25;
            const bool final_report = processed_entries == verified_entries.size();

            if (time_to_report || count_to_report || final_report) {
                const float t = static_cast<float>(processed_entries) /
                                static_cast<float>(verified_entries.size());

                // Console line (works well even without the UI)
                const int pct = static_cast<int>(t * 100.0f);
                std::cout << "  Preprocess progress: " << processed_entries
                          << "/" << verified_entries.size()
                          << " (" << pct << "%)\n";

                reportProgress(55.0f + t * 30.0f); // 55% -> 85%

                last_reported_entries = processed_entries;
                last_report_time = now;
            }
        }
    }

    reportProgress(85.0f);

    if (chunks_created > 0) {
        std::cout << "  Split " << chunks_created << " long texts into multiple chunks\n";
    }
    if (pruned_count > 0) {
        std::cout << "  Pruned " << pruned_count << " corrupted/low-quality chunks\n";
    }
    std::cout << "\xE2\x9C\x93 Cleaned: " << cleaned_texts.size() << " entries\n\n";

    // Re-inject previously merged entries that aren't duplicates of new data.
    // These already passed cleaning + verification in prior runs.
    // Injected BEFORE structuring so old prose entries get structured incrementally.
    if (!previously_merged.empty()) {
        size_t reinjected = 0;
        for (const auto& pm : previously_merged) {
            if (pm.content.empty()) continue;
            if (preprocessor.isDuplicate(pm.content)) continue;
            cleaned_texts.push_back(pm.content);
            reinjected++;
        }
        std::cout << "  Re-injected " << reinjected << " entries from previous cache ("
                  << (previously_merged.size() - reinjected) << " were duplicates of new data)\n\n";
    }

    // ========== [5/7] Structure Data (LLM Q/A generation) ==========
    if (g_structuring_config.enabled && !g_skip_structuring && g_structuring_config.mode != "raw") {
        currentPhase = "Structuring";
        std::cout << "[5/7] Structuring data via LLM (" << g_structuring_config.mode << " mode)...\n";

        if (g_structuring_config.ollama_model.empty()) {
            throw std::runtime_error("data_structuring.ollama_model is empty — MUST specify a model in ai_config.json");
        }

        // Load ollama_url from parent config if not set in structuring config
        if (g_structuring_config.ollama_url.empty()) {
            g_structuring_config.ollama_url = "http://127.0.0.1:11434";
        }

        GRIM::DataCollection::DataStructurer structurer(g_structuring_config);

        // Health check Ollama
        std::cout << "  Checking Ollama at " << g_structuring_config.ollama_url
                  << " (model: " << g_structuring_config.ollama_model << ")...\n";
        if (!structurer.checkOllamaHealth()) {
            std::cerr << "  WARNING: Ollama is not reachable. Skipping structuring step.\n";
            std::cerr << "  All " << cleaned_texts.size() << " entries will remain as raw prose.\n\n";
        } else {
            std::cout << "  \xE2\x9C\x93 Ollama is online\n";

            // Filter entries that need structuring (skip already structured + already processed)
            std::vector<size_t> indices_to_structure;
            size_t already_structured_count = 0;
            size_t already_processed_count = 0;

            for (size_t i = 0; i < cleaned_texts.size(); ++i) {
                if (g_structuring_config.skip_already_structured &&
                    GRIM::DataCollection::DataStructurer::isAlreadyStructured(cleaned_texts[i])) {
                    already_structured_count++;
                    continue;
                }
                if (g_stateManager && g_stateManager->hasStructuredContent(cleaned_texts[i])) {
                    already_processed_count++;
                    continue;
                }
                indices_to_structure.push_back(i);
            }

            // Apply max_entries_per_run limit
            if (g_structuring_config.max_entries_per_run > 0 &&
                static_cast<int>(indices_to_structure.size()) > g_structuring_config.max_entries_per_run) {
                std::cout << "  Limiting to " << g_structuring_config.max_entries_per_run
                          << " entries this run (of " << indices_to_structure.size() << " pending)\n";
                indices_to_structure.resize(g_structuring_config.max_entries_per_run);
            }

            std::cout << "  To structure: " << indices_to_structure.size();
            if (already_structured_count > 0)
                std::cout << " (skipped " << already_structured_count << " already structured)";
            if (already_processed_count > 0)
                std::cout << " (skipped " << already_processed_count << " previously processed)";
            std::cout << "\n";

            if (!indices_to_structure.empty()) {
                // Gather texts for batch processing
                std::vector<std::string> batch_texts;
                batch_texts.reserve(indices_to_structure.size());
                for (size_t idx : indices_to_structure) {
                    batch_texts.push_back(cleaned_texts[idx]);
                }

                auto last_progress_time = std::chrono::steady_clock::now();

                auto batch_result = structurer.structureBatch(batch_texts,
                    [&](size_t done, size_t total) {
                        auto now = std::chrono::steady_clock::now();
                        if ((now - last_progress_time) >= std::chrono::seconds(2) || done == total) {
                            float t = static_cast<float>(done) / static_cast<float>(total);
                            int pct = static_cast<int>(t * 100.0f);
                            std::cout << "  Structuring: " << done << "/" << total
                                      << " (" << pct << "%)\n";
                            reportProgress(85.0f + t * 7.0f); // 85% -> 92%
                            last_progress_time = now;
                        }
                    });

                // Apply results: replace original text with structured version(s)
                std::vector<std::string> new_cleaned;
                new_cleaned.reserve(cleaned_texts.size() + batch_result.succeeded);

                size_t next_structure_pos = 0;

                for (size_t i = 0; i < cleaned_texts.size(); ++i) {
                    if (next_structure_pos < indices_to_structure.size() &&
                        i == indices_to_structure[next_structure_pos]) {
                        // This entry was sent for structuring
                        auto& structured = batch_result.structured[next_structure_pos];
                        for (auto& s : structured) {
                            new_cleaned.push_back(std::move(s));
                        }
                        // Mark original content as structured for future runs
                        if (g_stateManager && structured.size() > 0 &&
                            structured[0] != cleaned_texts[i]) {
                            g_stateManager->markContentStructured(cleaned_texts[i]);
                        }
                        next_structure_pos++;
                    } else {
                        // Keep as-is (already structured, already processed, or not selected)
                        new_cleaned.push_back(std::move(cleaned_texts[i]));
                    }
                }

                cleaned_texts = std::move(new_cleaned);

                std::cout << "\xE2\x9C\x93 Structured: " << batch_result.succeeded << " entries succeeded, "
                          << batch_result.failed << " failed (kept original), "
                          << batch_result.skipped << " skipped\n";
                std::cout << "  Total entries after structuring: " << cleaned_texts.size() << "\n\n";
            } else {
                std::cout << "  No new entries to structure.\n\n";
            }
        }
    } else if (g_skip_structuring) {
        std::cout << "[5/7] Structuring skipped (--skip-structuring)\n\n";
    } else if (!g_structuring_config.enabled) {
        std::cout << "[5/7] Structuring disabled in config\n\n";
    }

    reportProgress(92.0f);

    // ========== [6/7] External Q/A ingestion ==========
    // Ingest external Q/A and conversational JSONL files into cleaned_texts
    if (!g_qa_jsonl_paths.empty()) {
        std::cout << "[6/7] Ingesting " << g_qa_jsonl_paths.size() << " external Q/A JSONL file(s)...\n";
        size_t qa_ingested = 0, qa_bad = 0, qa_skipped = 0;
        ingestQaJsonlFiles(g_qa_jsonl_paths, cleaned_texts, qa_ingested, qa_bad, qa_skipped);
        std::cout << "\xE2\x9C\x93 Q/A ingestion: " << qa_ingested << " entries added";
        if (qa_bad > 0) std::cout << " (" << qa_bad << " bad lines)";
        if (qa_skipped > 0) std::cout << " (" << qa_skipped << " files already processed)";
        std::cout << "\n\n";
    }

    // Instead of splitting/tokenizing here, write a merged cleaned cache for training/tokenizer pipeline.
    currentPhase = "Writing";
    std::cout << "[7/7] Writing merged verified cache...\n";

    fs::create_directories(resolved_output_dir);
    fs::path cache_path = fs::path(resolved_output_dir) / "merged_verified_cache.jsonl";

    std::ofstream cache_file(cache_path, std::ios::out | std::ios::trunc);
    if (!cache_file.is_open()) {
        std::cerr << "ERROR: Failed to open merged cache file for writing: "
                  << cache_path.string() << "\n";
        return 1;
    }

    for (size_t i = 0; i < cleaned_texts.size(); ++i) {
        nlohmann::json j;
        j["content"] = cleaned_texts[i];
        cache_file << j.dump() << "\n";

        // Coarse progress during write.
        if (cleaned_texts.size() > 2000 && (i % 1000) == 0) {
            const float t = static_cast<float>(i + 1) /
                            static_cast<float>(cleaned_texts.size());
            reportProgress(93.0f + t * 6.0f); // 93% -> 99%
        }
    }

    cache_file.close();
    
    // Save state manager to persist deduplication data for future runs
    if (g_stateManager) {
        g_stateManager->saveState();
        std::cout << "[State] Saved " << g_stateManager->getTotalUniqueUrls() << " URLs, "
                  << g_stateManager->getTotalUniqueContent() << " content hashes\n";
    }

    std::cout << "\xE2\x9C\x93 Wrote merged verified cache: " << cache_path.string() << "\n";
    std::cout << "=== MERGE COMPLETE (CACHE-ONLY) ===\n";
    std::cout << "Total cached entries: " << cleaned_texts.size() << "\n\n";
    std::cout << "Ready for tokenizer/training pipeline to consume cache.\n";

    reportProgress(100.0f);

    return 0;
    
    } catch (const std::exception& e) {
        std::cerr << "\n[MERGE ERROR] Fatal exception: " << e.what() << "\n";
        std::cerr << "[MERGE ERROR] Merge operation failed. Check paths and data integrity.\n";
        return 1;
    } catch (...) {
        std::cerr << "\n[MERGE ERROR] Unknown fatal exception during merge.\n";
        return 1;
    }
}

int StartDataCollection(int argc, char** argv, std::function<void(float)> progressCallback) {
    std::cout << "[StartDataCollection] Entry point called with argc=" << argc << "\n";
    std::cout.flush();
    
    if (argc < 2) {
        printUsage();
        return 1;
    }

    std::string mode = argv[1];
    std::cout << "[StartDataCollection] Mode: " << mode << "\n";
    std::cout.flush();

    if (mode == "--help" || mode == "-h") {
        printUsage();
        return 0;
    }

    // Parse options - defaults will be overridden by ai_config.json
    std::string config_path = "";  // Will be loaded from ai_config.json
    std::string checkpoint_dir;
    std::string raw_dir;
    std::string verified_dir;
    std::string output_dir = "data";
    bool skip_verification = false;
    bool retrain_vocab = false;
    int vocab_size = 50000;  // Default vocab size

    // Load paths from ai_config.json (centralized source of truth)
    std::cout << "[DataPipeline] Loading paths from ai_config.json..." << std::endl;
    auto snapshot = GRIM::Config::loadAiConfigSnapshot();
    if (snapshot && snapshot->has_grim_paths) {
        // Use source_config path from config (CRITICAL FIX)
        if (!snapshot->grim_text_source_config.empty()) {
            config_path = snapshot->grim_text_source_config;
            std::cout << "[DataPipeline] ✓ Using source config from ai_config.json: " << config_path << std::endl;
        } else {
            std::cerr << "[DataPipeline] WARNING: source_config not set in ai_config.json" << std::endl;
            config_path = "DataCollection/source_data.json";  // Fallback
        }
        
        // Use checkpoints path from config if available
        if (!snapshot->grim_text_checkpoints.empty()) {
            checkpoint_dir = snapshot->grim_text_checkpoints;
            std::cout << "[DataPipeline] ✓ Using checkpoints path from ai_config.json: " << checkpoint_dir << std::endl;
        }
        
        // Use collected directory from config
        if (!snapshot->grim_text_collected.empty()) {
            raw_dir = snapshot->grim_text_collected;
            std::cout << "[DataPipeline] ✓ Using collected dir from ai_config.json: " << raw_dir << std::endl;
        }
        
        // Use verified directory from config
        if (!snapshot->grim_text_verified.empty()) {
            verified_dir = snapshot->grim_text_verified;
            std::cout << "[DataPipeline] ✓ Using verified dir from ai_config.json: " << verified_dir << std::endl;
        }
        
        // Use training_data path's parent directory as output_dir
        if (!snapshot->grim_text_training_data.empty()) {
            fs::path trainingDataPath(snapshot->grim_text_training_data);
            output_dir = trainingDataPath.parent_path().string();
            std::cout << "[DataPipeline] ✓ Using output directory from ai_config.json: " << output_dir << std::endl;
        }
    } else {
        std::cout << "[DataPipeline] WARNING: Could not load paths from ai_config.json, using defaults" << std::endl;
    }

    // Load Q/A JSONL paths from ai_config.json data_collection section
    try {
        std::ifstream configFile("ai_config.json");
        if (configFile.is_open()) {
            nlohmann::json config = nlohmann::json::parse(configFile);
            if (config.contains("data_collection") && config["data_collection"].contains("qa_jsonl_paths")) {
                auto& qa_paths = config["data_collection"]["qa_jsonl_paths"];
                if (qa_paths.is_array()) {
                    for (const auto& p : qa_paths) {
                        if (p.is_string()) {
                            g_qa_jsonl_paths.push_back(p.get<std::string>());
                        }
                    }
                    if (!g_qa_jsonl_paths.empty()) {
                        std::cout << "[DataPipeline] ✓ Loaded " << g_qa_jsonl_paths.size()
                                  << " Q/A JSONL path(s) from ai_config.json\n";
                    }
                }
            }
        }
    } catch (...) {}

    // Load data_structuring config from ai_config.json
    try {
        std::ifstream configFile2("ai_config.json");
        if (configFile2.is_open()) {
            nlohmann::json config = nlohmann::json::parse(configFile2);
            if (config.contains("data_collection") && config["data_collection"].contains("data_structuring")) {
                g_structuring_config = GRIM::DataCollection::DataStructuringConfig::fromJson(
                    config["data_collection"]["data_structuring"]);
                // Load ollama_url from top-level config if not in structuring block
                if (g_structuring_config.ollama_url.empty() && config.contains("ollama_url")) {
                    g_structuring_config.ollama_url = config["ollama_url"].get<std::string>();
                }
                std::cout << "[DataPipeline] \xE2\x9C\x93 Data structuring config loaded (mode: "
                          << g_structuring_config.mode << ", enabled: "
                          << (g_structuring_config.enabled ? "yes" : "no") << ")\n";
            }
        }
    } catch (...) {}

    // Command-line arguments override ai_config.json
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        }
        else if (arg == "--checkpoint-dir" && i + 1 < argc) {
            checkpoint_dir = argv[++i];
            std::cout << "[DataPipeline] Checkpoint dir overridden by command line: " << checkpoint_dir << std::endl;
        }
        else if (arg == "--raw-dir" && i + 1 < argc) {
            raw_dir = argv[++i];
        }
        else if (arg == "--verified-dir" && i + 1 < argc) {
            verified_dir = argv[++i];
        }
        else if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
            std::cout << "[DataPipeline] Output dir overridden by command line: " << output_dir << std::endl;
        }
        else if (arg == "--skip-verification") {
            skip_verification = true;
        }
        else if (arg == "--force-rebuild") {
            g_force_rebuild = true;
            std::cout << "[DataPipeline] Force rebuild enabled - ignoring deduplication state\n";
        }
        else if (arg == "--retrain-vocab") {
            retrain_vocab = true;
            std::cout << "[DataPipeline] Will retrain tokenizer (ignoring existing vocab)\n";
        }
        else if (arg == "--vocab-size" && i + 1 < argc) {
            vocab_size = std::stoi(argv[++i]);
            std::cout << "[DataPipeline] Vocab size set to: " << vocab_size << "\n";
        }
        else if (arg == "--qa-jsonl" && i + 1 < argc) {
            g_qa_jsonl_paths.push_back(argv[++i]);
            std::cout << "[DataPipeline] Added Q/A JSONL: " << g_qa_jsonl_paths.back() << "\n";
        }
        else if (arg == "--skip-structuring") {
            g_skip_structuring = true;
            std::cout << "[DataPipeline] Skipping LLM data structuring step\n";
        }
    }

    // Persist paths for helper routines that rely on global context
    g_checkpoint_dir = checkpoint_dir;
    g_verified_dir = verified_dir;
    g_output_dir = output_dir;
    g_raw_dir = raw_dir;
    
    // Run requested mode
    if (mode == "collect") {
        if (config_path.empty()) {
            std::cerr << "ERROR: Source config path not specified and not found in ai_config.json\n";
            return 1;
        }
        return runCollect(config_path, progressCallback);
    }
    else if (mode == "verify") {
        return runVerify(raw_dir, verified_dir);
    }
    else if (mode == "merge") {
        return runMerge(checkpoint_dir, verified_dir, output_dir, skip_verification, retrain_vocab, vocab_size, progressCallback);
    }
    else if (mode == "merge-rebuild") {
        // Force rebuild mode - ignores deduplication state
        g_force_rebuild = true;
        std::cout << "[DataPipeline] Force rebuild mode enabled\n";
        return runMerge(checkpoint_dir, verified_dir, output_dir, skip_verification, retrain_vocab, vocab_size, progressCallback);
    }
    else if (mode == "full") {
        if (config_path.empty()) {
            std::cerr << "ERROR: Source config path not specified and not found in ai_config.json\n";
            return 1;
        }
        
        int result = runCollect(config_path, progressCallback);
        if (result != 0) return result;
        
        result = runVerify(raw_dir, verified_dir);
        if (result != 0) return result;
        
        return runMerge(checkpoint_dir, verified_dir, output_dir, skip_verification, retrain_vocab, vocab_size, progressCallback);
    }
    else {
        std::cerr << "ERROR: Unknown mode '" << mode << "'\n\n";
        printUsage();
        return 1;
    }
}

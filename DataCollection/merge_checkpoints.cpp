//======================================================//
//  GRIM Checkpoint Merger
//  Merges checkpoint files with existing verified data
//  and continues the data processing pipeline
//======================================================//

#include <iostream>
#include <filesystem>
#include <vector>
#include "web_collector.hpp"
#include "verifier.hpp"
#include "data_preprocessor.hpp"
#include "data_splitter.hpp"
#include "../resources/models/GRIM-text/Shared/UnigramByte/UniByte.hpp"
#include "../resources/models/GRIM-text/Shared/HyperParameters/HyperParameters_GPU.hpp"

using namespace GRIM::Training;
namespace fs = std::filesystem;

static void printUsage() {
    std::cout << "Usage: merge_checkpoints [OPTIONS]\n";
    std::cout << "Options:\n";
    std::cout << "  --checkpoints <path1,path2,...>  Comma-separated checkpoint files\n";
    std::cout << "  --checkpoint-dir <dir>           Load all checkpoint_*.json from directory\n";
    std::cout << "  --verified-dir <dir>             Directory with existing verified data (default: data/verified)\n";
    std::cout << "  --output-dir <dir>               Output directory (default: data)\n";
    std::cout << "  --skip-verification              Skip re-verification of checkpoint data\n";
    std::cout << "\nExample:\n";
    std::cout << "  merge_checkpoints --checkpoint-dir ../../../data --verified-dir data/verified\n";
}

int StartMergeCheckpoints(int argc, char** argv) {
    std::vector<std::string> checkpoint_files;
    std::string checkpoint_dir;
    std::string verified_dir = "data/verified";
    std::string output_dir = "data";
    bool skip_verification = false;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            printUsage();
            return 0;
        }
        else if (arg == "--checkpoints" && i + 1 < argc) {
            std::string paths = argv[++i];
            size_t pos = 0;
            while ((pos = paths.find(',')) != std::string::npos) {
                checkpoint_files.push_back(paths.substr(0, pos));
                paths.erase(0, pos + 1);
            }
            if (!paths.empty()) checkpoint_files.push_back(paths);
        }
        else if (arg == "--checkpoint-dir" && i + 1 < argc) {
            checkpoint_dir = argv[++i];
        }
        else if (arg == "--verified-dir" && i + 1 < argc) {
            verified_dir = argv[++i];
        }
        else if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        }
        else if (arg == "--skip-verification") {
            skip_verification = true;
        }
    }
    
    std::cout << "=== GRIM CHECKPOINT MERGER & PIPELINE ===\n\n";
    
    //======================================================//
    // STEP 1: Load checkpoint files
    //======================================================//
    std::cout << "[1/7] Loading checkpoint files...\n";
    
    // If checkpoint_dir specified, find all checkpoint_*.json files
    if (!checkpoint_dir.empty()) {
        for (const auto& entry : fs::directory_iterator(checkpoint_dir)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                // C++17 compatible check
                if (filename.substr(0, 11) == "checkpoint_" && 
                    filename.size() > 5 && filename.substr(filename.size() - 5) == ".json") {
                    checkpoint_files.push_back(entry.path().string());
                }
            }
        }
    }
    
    if (checkpoint_files.empty()) {
        std::cerr << "ERROR: No checkpoint files specified!\n";
        printUsage();
        return 1;
    }
    
    std::cout << "Found " << checkpoint_files.size() << " checkpoint files:\n";
    for (const auto& file : checkpoint_files) {
        std::cout << "  - " << file << "\n";
    }
    
    // Load checkpoints into collector
    WebDataCollector collector;
    if (!collector.mergeCheckpoints(checkpoint_files)) {
        std::cerr << "WARNING: Some checkpoint files failed to load\n";
    }
    
    const auto& checkpoint_data = collector.getCollectedData();
    std::cout << "✓ Loaded " << checkpoint_data.size() << " entries from checkpoints\n\n";
    
    //======================================================//
    // STEP 2: Load existing verified data
    //======================================================//
    std::cout << "[2/7] Loading existing verified data...\n";
    
    std::vector<VerifiedEntry> existing_verified;
    
    if (fs::exists(verified_dir)) {
        for (const auto& entry : fs::directory_iterator(verified_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".jsonl") {
                // Load JSONL verified entries
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
    
    std::cout << "✓ Loaded " << existing_verified.size() << " existing verified entries\n\n";
    
    //======================================================//
    // STEP 3: Verify checkpoint data (optional)
    //======================================================//
    std::cout << "[3/7] Processing checkpoint data...\n";
    
    std::vector<VerifiedEntry> new_verified;
    
    if (!skip_verification) {
        // Convert checkpoint data to unverified entries
        std::vector<UnverifiedEntry> unverified_entries;
        for (const auto& raw : checkpoint_data) {
            UnverifiedEntry entry;
            entry.content = raw.content;
            entry.source_url = raw.source_url;
            entry.source_type = raw.source_name;
            entry.author = raw.author;
            entry.metadata = raw.metadata_json;
            unverified_entries.push_back(entry);
        }
        
        // Run verification
        Config verifier_config;
        verifier_config.input_dir = output_dir + "/collected";
        verifier_config.output_dir = output_dir + "/verified";
        verifier_config.min_length = 100;
        verifier_config.max_length = 50000;
        
        Verifier verifier(verifier_config);
        new_verified = verifier.verify_entries(unverified_entries);
        auto stats = verifier.get_stats();
        
        std::cout << "✓ Verified checkpoint data: " << stats.passed_verification 
                  << "/" << unverified_entries.size() << " passed\n\n";
    } else {
        // Skip verification - convert directly to verified
        for (const auto& raw : checkpoint_data) {
            VerifiedEntry ve;
            ve.content = raw.content;
            ve.source_url = raw.source_url;
            ve.source_type = sourceTypeToString(raw.source_type);
            ve.reliability_score = 0.7f;  // Default score
            ve.verification_time = static_cast<time_t>(raw.fetch_date);
            new_verified.push_back(ve);
        }
        std::cout << "✓ Skipped verification (as requested)\n\n";
    }
    
    //======================================================//
    // STEP 4: Merge and deduplicate
    //======================================================//
    std::cout << "[4/7] Merging and deduplicating...\n";
    
    // Combine existing + new verified entries
    std::vector<VerifiedEntry> all_verified = existing_verified;
    all_verified.insert(all_verified.end(), new_verified.begin(), new_verified.end());
    
    // Simple deduplication by content hash
    std::unordered_set<std::string> seen_hashes;
    std::vector<VerifiedEntry> deduplicated;
    
    for (const auto& entry : all_verified) {
        std::string hash = std::to_string(std::hash<std::string>{}(entry.content));
        if (seen_hashes.find(hash) == seen_hashes.end()) {
            seen_hashes.insert(hash);
            deduplicated.push_back(entry);
        }
    }
    
    std::cout << "✓ Total entries: " << all_verified.size() << "\n";
    std::cout << "✓ After dedup: " << deduplicated.size() << " (removed " 
              << (all_verified.size() - deduplicated.size()) << " duplicates)\n\n";
    
    //======================================================//
    // STEP 5: Continue normal pipeline (preprocessing, etc.)
    //======================================================//
    std::cout << "[5/7] Preprocessing and cleaning...\n";
    
    PreprocessorConfig prep_config;
    // CRITICAL: Limit text length to fit within model's max_seq_len after tokenization
    // max_seq_len=900 tokens, ~4 chars/token average = 3500 chars max
    prep_config.max_token_estimate_chars = 3500;
    
    DataPreprocessor preprocessor(prep_config);
    
    std::vector<std::string> cleaned_texts;
    size_t chunks_created = 0;
    
    for (const auto& entry : deduplicated) {
        std::string clean = preprocessor.preprocess(entry.content);
        
        // Split long texts into chunks instead of discarding them
        auto chunks = preprocessor.chunkLongText(clean);
        
        for (const auto& chunk : chunks) {
            if (!preprocessor.passesQualityFilter(chunk)) continue;
            if (preprocessor.isDuplicate(chunk)) continue;
            
            cleaned_texts.push_back(chunk);
            
            if (chunks.size() > 1) chunks_created++;
        }
    }
    
    if (chunks_created > 0) {
        std::cout << "  Split " << chunks_created << " long texts into multiple chunks\n";
    }
    std::cout << "✓ Cleaned: " << cleaned_texts.size() << "/" << deduplicated.size() 
              << " passed quality filters\n\n";
    
    //======================================================//
    // STEP 6: Split, Tokenize, Save
    //======================================================//
    std::cout << "[6/7] Splitting data (train/val/test)...\n";
    
    SplitConfig split_config;
    split_config.train_ratio = 0.8f;
    split_config.val_ratio = 0.1f;
    split_config.test_ratio = 0.1f;
    split_config.random_seed = 42;
    
    DataSplitter<std::string> splitter(split_config);
    auto split = splitter.split(cleaned_texts);
    
    fs::create_directories(output_dir + "/splits");
    splitter.saveSplits(split, output_dir + "/splits");
    
    std::cout << "✓ Train: " << split.train.size() << " samples\n";
    std::cout << "✓ Val:   " << split.validation.size() << " samples\n";
    std::cout << "✓ Test:  " << split.test.size() << " samples\n\n";
    
    std::cout << "[7/7] Tokenizing...\n";
    
    // Load vocab path from ai_config.json
    std::string vocab_path = "models/tokenizer.bin"; // default fallback
    GRIM::Config::GrimTextPaths paths;
    if (GRIM::Config::loadGrimTextPaths(paths) && !paths.vocab.empty()) {
        vocab_path = paths.vocab;
        std::cout << "  Using vocab from config: " << vocab_path << "\n";
    } else {
        std::cout << "  WARNING: Could not load vocab path from config, using default: " << vocab_path << "\n";
    }
    
    // Load or create tokenizer
    GRIM::TokenizerConfig tok_config;
    tok_config.vocab_size = 50000;
    tok_config.max_length = 2048;
    tok_config.special_tokens = {"<pad>", "<unk>", "<s>", "</s>"};
    
    GRIM::GrimTokenizer tokenizer(tok_config);
    
    // Try to load existing tokenizer
    bool has_tokenizer = false;
    if (fs::exists(vocab_path)) {
        if (tokenizer.load(vocab_path)) {
            std::cout << "✓ Loaded existing tokenizer\n";
            has_tokenizer = true;
        }
    }
    
    if (!has_tokenizer) {
        std::cout << "  Training new tokenizer...\n";
        tokenizer.trainFromCorpus(split.train, 10000);
        
        // Ensure directory exists
        fs::path vocab_file(vocab_path);
        if (vocab_file.has_parent_path()) {
            fs::create_directories(vocab_file.parent_path());
        }
        
        tokenizer.save(vocab_path);
        std::cout << "✓ Trained and saved new tokenizer to: " << vocab_path << "\n";
    }
    
    // Tokenize all splits
    auto train_tokens = tokenizer.encodeBatch(split.train);
    auto val_tokens = tokenizer.encodeBatch(split.validation);
    auto test_tokens = tokenizer.encodeBatch(split.test);
    
    // Save tokenized data
    fs::create_directories(output_dir + "/tokenized");
    
    auto save_tokens = [](const std::string& path, const std::vector<std::vector<int>>& data) {
        std::ofstream file(path, std::ios::binary);
        uint32_t num_samples = data.size();
        file.write(reinterpret_cast<const char*>(&num_samples), sizeof(num_samples));
        
        for (const auto& sample : data) {
            uint32_t len = sample.size();
            file.write(reinterpret_cast<const char*>(&len), sizeof(len));
            file.write(reinterpret_cast<const char*>(sample.data()), len * sizeof(int));
        }
        
        return file.good();
    };
    
    save_tokens(output_dir + "/tokenized/train.bin", train_tokens);
    save_tokens(output_dir + "/tokenized/val.bin", val_tokens);
    save_tokens(output_dir + "/tokenized/test.bin", test_tokens);
    
    std::cout << "✓ Tokenized and saved to " << output_dir << "/tokenized/*.bin\n\n";
    
    //======================================================//
    // Summary
    //======================================================//
    std::cout << "=== MERGE & PIPELINE COMPLETE! ===\n";
    std::cout << "✓ Loaded checkpoints: " << checkpoint_data.size() << " entries\n";
    std::cout << "✓ Existing verified: " << existing_verified.size() << " entries\n";
    std::cout << "✓ New verified: " << new_verified.size() << " entries\n";
    std::cout << "✓ Total deduplicated: " << deduplicated.size() << " entries\n";
    std::cout << "✓ After preprocessing: " << cleaned_texts.size() << " entries\n";
    std::cout << "✓ Train/val/test split: " << split.totalSize() << " samples\n";
    std::cout << "✓ Tokenizer vocab: " << tokenizer.vocabSize() << "\n\n";
    
    std::cout << "Ready for training! Use:\n";
    std::cout << "  train_gpu.exe --data " << output_dir << "/tokenized/train.bin ...\n";
    
    return 0;
}

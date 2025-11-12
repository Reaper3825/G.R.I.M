//======================================================//
//  GRIM Data Collection - COMPLETE PIPELINE
//  Collect → Verify → Merge → Tokenize
//======================================================//

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#endif


#include <iostream>
#include <filesystem>
#include <regex>
#include <fstream>
#include <chrono>
#include "web_collector.hpp"
#include "verifier.hpp"
#include "data_preprocessor.hpp"
#include "data_splitter.hpp"
#include "../resources/models/GRIM-text/GRIM/grim_tokenizer.hpp"
#include "../../../../control/training_control_generated.h"
#include <flatbuffers/flatbuffers.h>
#include <system_error>
#include "training_paths.hpp"


using namespace GRIM::Training;
namespace fs = std::filesystem;

// Simple tagging utilities
namespace TagUtils {
    std::vector<std::string> extractTags(const std::string& text) {
        std::vector<std::string> tags;
        if (text.find("```") != std::string::npos) tags.push_back("CODE");
        if (text.find("http") != std::string::npos) tags.push_back("HAS_URLS");
        
        std::regex tech_pattern(R"(\b(python|machine learning|neural|AI|API|function|class|algorithm)\b)", std::regex::icase);
        if (std::regex_search(text, tech_pattern)) tags.push_back("TECHNICAL");
        
        std::regex question_pattern(R"(\b(what|how|why|when|where|who)\b)", std::regex::icase);
        if (std::regex_search(text, question_pattern)) tags.push_back("QUESTION");
        
        if (text.length() > 5000) tags.push_back("LONG_FORM");
        else if (text.length() < 500) tags.push_back("SHORT_FORM");
        
        return tags;
    }
}

// Helper function to update collection progress in status file
void updateCollectionProgress(float progress) {
    try {
        flatbuffers::FlatBufferBuilder builder(1024);
        
        // Build minimal stats with collection progress
        auto phase_str = builder.CreateString("Collecting");
        auto error_str = builder.CreateString("");
        
        auto stats = GRIMText::Control::CreateTrainingStats(builder,
            0, 0,  // epochs
            0, 0,  // batches
            0.0f, 0.0f,  // loss
            0.0f, 0.0f,  // perplexity, tokens/sec
            0.0f, 0.0f,  // gpu memory
            0.0f,        // training progress
            progress,    // collection progress (0-100)
            phase_str, error_str,
            0, 0  // timestamps
        );
        
        auto config = GRIMText::Control::CreateTrainingConfig(builder);
        
        int64_t timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        auto status_response = GRIMText::Control::CreateStatusResponse(builder,
            GRIMText::Control::TrainingState_Collecting,
            stats, config, true, timestamp
        );
        
        builder.Finish(status_response);
        
        // Write to status file
    auto statusPath = GRIM::Training::getTrainingStatusFilePath();
    std::error_code mkdirErr;
    fs::create_directories(statusPath.parent_path(), mkdirErr);

    std::ofstream statusFile(statusPath, std::ios::binary);
        if (statusFile.is_open()) {
            statusFile.write(
                reinterpret_cast<const char*>(builder.GetBufferPointer()),
                builder.GetSize()
            );
            statusFile.close();
        }
    } catch (const std::exception& e) {
        // Silent fail - don't interrupt collection
    }
}

int StartMainDataCollection(int argc, char** argv) {
    std::string config_path = "source_data.json";
    std::string checkpoint_dir;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--checkpoint-dir" && i + 1 < argc) {
            checkpoint_dir = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else {
            // First positional argument is config path
            config_path = arg;
        }
    }
    
    std::cout << "=== GRIM COMPLETE TRAINING DATA PIPELINE ===\n\n";
    
    //======================================================//
    // STEP 0: Load existing checkpoints (if any)
    //======================================================//
    std::cout << "[0/7] Checking for existing checkpoints...\n";
    WebDataCollector collector;
    collector.setVerbose(true);
    
    // Look for checkpoint files in common locations
    std::vector<std::string> checkpoint_paths;
    std::vector<std::string> search_dirs;
    
    // If checkpoint_dir specified, use only that
    if (!checkpoint_dir.empty()) {
        search_dirs.push_back(checkpoint_dir);
        std::cout << "  Using checkpoint directory: " << checkpoint_dir << "\n";
    } else {
        // Default search locations
        search_dirs = {
            "../../../../data",           // D:\G.R.I.M\data
            "../../../../../data",        // Alternative path
            "data/checkpoints",           // Local checkpoints dir
            "."                           // Current directory
        };
    }
    
    for (const auto& dir : search_dirs) {
        if (fs::exists(dir)) {
            for (const auto& entry : fs::directory_iterator(dir)) {
                if (entry.is_regular_file()) {
                    std::string filename = entry.path().filename().string();
                    // Check for checkpoint_*.json pattern
                    if (filename.substr(0, 11) == "checkpoint_" && 
                        filename.size() > 5 && 
                        filename.substr(filename.size() - 5) == ".json") {
                        checkpoint_paths.push_back(entry.path().string());
                    }
                }
            }
        }
    }
    
    if (!checkpoint_paths.empty()) {
        std::cout << "Found " << checkpoint_paths.size() << " checkpoint file(s)\n";
        collector.mergeCheckpoints(checkpoint_paths);
        size_t checkpoint_entries = collector.getCollectedData().size();
        std::cout << "✓ Loaded " << checkpoint_entries << " entries from checkpoints\n\n";
    } else {
        std::cout << "  No checkpoint files found - starting fresh\n\n";
    }
    
    //======================================================//
    // STEP 1: Collect Data
    //======================================================//
    std::cout << "[1/7] Collecting data from web sources...\n";
    
    if (!collector.loadConfigFromJson(config_path)) {
        std::cerr << "ERROR: Failed to load config\n";
        return 1;
    }
    
    // Set progress callback to update status file
    collector.setProgressCallback([](float progress) {
        updateCollectionProgress(progress);
    });
    
    size_t collected = collector.collectData();
    size_t total_entries = collector.getCollectedData().size();
    std::cout << "✓ Collected " << collected << " new entries";
    if (total_entries > collected) {
        std::cout << " (total with checkpoints: " << total_entries << ")";
    }
    std::cout << "\n\n";
    
    //======================================================//
    // STEP 2: Verify collected data
    //======================================================//
    std::cout << "[2/7] Verifying collected data...\n";
    
    // Get raw data from collector
    const auto& raw_data = collector.getCollectedData();
    
    // Convert RawDataEntry to UnverifiedEntry for verification
    std::vector<UnverifiedEntry> unverified_entries;
    for (const auto& raw : raw_data) {
        UnverifiedEntry entry;
        entry.content = raw.content;
        entry.source_url = raw.source_url;
        entry.source_type = raw.source_name;
        entry.author = raw.author;
        entry.metadata = raw.metadata_json;
        unverified_entries.push_back(entry);
    }
    
    Config verifier_config;
    verifier_config.input_dir = "data/collected";
    verifier_config.output_dir = "data/verified";
    verifier_config.min_length = 100;
    verifier_config.max_length = 50000;
    
    Verifier verifier(verifier_config);
    auto verified_entries = verifier.verify_entries(unverified_entries);
    verifier.save_verified_entries(verified_entries);
    auto verify_stats = verifier.get_stats();
    
    // Write stats to log file and JSON for UI
    verify_stats.writeSummaryToLog("logs/data_collection.log");
    
    std::cout << "✓ Verified: " << verify_stats.passed_verification << "/" << unverified_entries.size() 
              << " entries passed\n\n";
    
    //======================================================//
    // STEP 3: Preprocess & Clean Data
    //======================================================//
    std::cout << "[3/7] Preprocessing and cleaning data...\n";
    PreprocessorConfig prep_config;
    prep_config.min_length = 100;       // More forgiving - accept shorter HTML extractions
    prep_config.min_words = 15;         // More forgiving - accept brief but quality content
    prep_config.min_alpha_ratio = 0.5f; // More forgiving - accept technical/structured content
    DataPreprocessor preprocessor(prep_config);
    
    std::cout << "  Quality thresholds:\n";
    std::cout << "    min_length: " << prep_config.min_length << " chars\n";
    std::cout << "    min_words: " << prep_config.min_words << " words\n";
    std::cout << "    min_alpha_ratio: " << prep_config.min_alpha_ratio << "\n\n";
    
    std::vector<std::string> cleaned_texts;
    int rejected_length = 0, rejected_words = 0, rejected_alpha = 0, rejected_duplicate = 0;
    
    // Process only verified entries
    for (const auto& verified : verified_entries) {
        // Clean text
        std::string clean = preprocessor.preprocess(verified.content);
        
        // Quality filter with detailed tracking
        if (!preprocessor.passesQualityFilter(clean)) {
            // Track rejection reasons (simplified - would need preprocessor API changes for exact reason)
            if (clean.length() < static_cast<size_t>(prep_config.min_length)) {
                rejected_length++;
            } else {
                rejected_words++; // Approximate - could be words, alpha ratio, or repetition
            }
            continue;
        }
        
        // Deduplication
        if (preprocessor.isDuplicate(clean)) {
            rejected_duplicate++;
            continue;
        }
        
        // Add special tokens
        clean = preprocessor.addSpecialTokens(clean);
        
        cleaned_texts.push_back(clean);
    }
    
    std::cout << "✓ Cleaned: " << cleaned_texts.size() << "/" << verified_entries.size() 
              << " passed quality filters\n";
    if (rejected_length > 0 || rejected_words > 0 || rejected_duplicate > 0) {
        std::cout << "  Rejection breakdown:\n";
        std::cout << "    Too short: " << rejected_length << "\n";
        std::cout << "    Quality/words: " << rejected_words << "\n";
        std::cout << "    Duplicates: " << rejected_duplicate << "\n";
    }
    std::cout << "\n";
    
    //======================================================//
    // STEP 4: Train/Val/Test Split
    //======================================================//
    std::cout << "[4/7] Splitting data (train/val/test)...\n";
    SplitConfig split_config;
    split_config.train_ratio = 0.8f;
    split_config.val_ratio = 0.1f;
    split_config.test_ratio = 0.1f;
    split_config.random_seed = 42;
    
    DataSplitter<std::string> splitter(split_config);
    auto split = splitter.split(cleaned_texts);
    
    fs::create_directories("data/splits");
    splitter.saveSplits(split, "data/splits");
    
    std::cout << "✓ Train: " << split.train.size() << " samples\n";
    std::cout << "✓ Val:   " << split.validation.size() << " samples\n";
    std::cout << "✓ Test:  " << split.test.size() << " samples\n\n";
    
    //======================================================//
    // STEP 5: Train Tokenizer on Clean Data
    //======================================================//
    std::cout << "[5/7] Training BPE tokenizer on data...\n";
    
    GRIM::TokenizerConfig tok_config;
    tok_config.vocab_size = 50000;
    tok_config.max_length = 2048;
    tok_config.special_tokens = {"<pad>", "<unk>", "<s>", "</s>"};
    tok_config.unk_token = "<unk>";
    tok_config.pad_token = "<pad>";
    tok_config.bos_token = "<s>";
    tok_config.eos_token = "</s>";
    tok_config.add_bos = true;
    tok_config.add_eos = true;
    
    GRIM::GrimTokenizer tokenizer(tok_config);
    
    // Train on training split only
    std::cout << "  Training BPE with " << split.train.size() << " samples...\n";
    tokenizer.trainFromCorpus(split.train, 10000);  // 10K merges
    
    // Save tokenizer
    fs::create_directories("models");
    if (!tokenizer.save("models/tokenizer.bin")) {
        std::cerr << "ERROR: Failed to save tokenizer\n";
        return 1;
    }
    
    std::cout << "✓ Tokenizer trained with vocab size: " << tokenizer.vocabSize() << "\n";
    std::cout << "✓ Saved to models/tokenizer.bin\n";
    std::cout << "  Special tokens: PAD=" << tokenizer.padId() 
              << " UNK=" << tokenizer.unkId()
              << " BOS=" << tokenizer.bosId()
              << " EOS=" << tokenizer.eosId() << "\n\n";
    
    //======================================================//
    // STEP 5: Tokenize All Data
    //======================================================//
    std::cout << "[5/7] Tokenizing all splits...\n";
    
    auto train_tokens = tokenizer.encodeBatch(split.train);
    auto val_tokens = tokenizer.encodeBatch(split.validation);
    auto test_tokens = tokenizer.encodeBatch(split.test);
    
    std::cout << "✓ Tokenized train: " << train_tokens.size() << " samples\n";
    std::cout << "✓ Tokenized val: " << val_tokens.size() << " samples\n";
    std::cout << "✓ Tokenized test: " << test_tokens.size() << " samples\n";
    
    // Save tokenized data (binary format for fast loading)
    fs::create_directories("data/tokenized");
    
    // Simple binary format: [num_samples][sample1_len][tokens...][sample2_len][tokens...]...
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
    
    if (!save_tokens("data/tokenized/train.bin", train_tokens)) {
        std::cerr << "ERROR: Failed to save train tokens\n";
        return 1;
    }
    if (!save_tokens("data/tokenized/val.bin", val_tokens)) {
        std::cerr << "ERROR: Failed to save val tokens\n";
        return 1;
    }
    if (!save_tokens("data/tokenized/test.bin", test_tokens)) {
        std::cerr << "ERROR: Failed to save test tokens\n";
        return 1;
    }
    
    std::cout << "✓ Saved to data/tokenized/*.bin\n\n";
    
    //======================================================//
    // STEP 6: Done - Use separate training pipeline
    //======================================================//
    std::cout << "[6/7] Training skipped in collection tool\n";
    std::cout << "  → Use main_training_pipeline or train_model for GPU training\n";
    std::cout << "  → Tokenized data ready at: data/tokenized/*.bin\n\n";
    
    //======================================================//
    // Summary
    //======================================================//
    std::cout << "=== Pipeline Complete! ===\n";
    std::cout << "✓ Data collection: " << collected << " entries\n";
    std::cout << "✓ Verification: " << verify_stats.passed_verification << " passed\n";
    std::cout << "✓ Preprocessing: " << cleaned_texts.size() << " passed filters\n";
    std::cout << "✓ Train/val/test split: " << split.totalSize() << " total\n";
    std::cout << "✓ Tokenizer: vocab_size=" << tokenizer.vocabSize() << "\n";
    std::cout << "✓ Tokenization: Complete\n\n";
    
    std::cout << "Next steps:\n";
    std::cout << "  1. Use main_training_pipeline for GPU training\n";
    std::cout << "  2. Or use train_model with tokenized data at data/tokenized/*.bin\n";
    std::cout << "  3. Tokenizer saved at: models/tokenizer.bin\n";
    
    return 0;
}

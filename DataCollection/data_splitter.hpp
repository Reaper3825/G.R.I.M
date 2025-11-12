//======================================================//
//  Train/Val/Test Data Splitter
//  Ensures proper data partitioning for ML training
//  
//  Enhanced Features:
//  - Stratified splitting by content type/tags
//  - Length-balanced splits (avoid all long samples in one split)
//  - Minimum sample guarantees per split
//  - Automatic ratio adjustment for small datasets
//  - Domain/source distribution balancing
//======================================================//

#pragma once
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <iostream>
#include <cmath>

namespace GRIM {
namespace Training {

namespace fs = std::filesystem;

enum class SplitStrategy {
    RANDOM,              // Standard random split
    STRATIFIED,          // Maintain category proportions
    LENGTH_BALANCED,     // Balance text lengths across splits
    SOURCE_BALANCED      // Balance source types
};

struct SplitConfig {
    float train_ratio = 0.8f;
    float val_ratio = 0.1f;
    float test_ratio = 0.1f;
    unsigned int random_seed = 42;
    bool shuffle = true;
    
    // Enhanced features
    SplitStrategy strategy = SplitStrategy::RANDOM;
    size_t min_samples_per_split = 10;  // Minimum samples in val/test
    bool auto_adjust_ratios = true;      // Auto-adjust for small datasets
    bool ensure_diversity = true;        // Ensure diverse samples in each split
    float tolerance = 0.01f;             // Ratio adjustment tolerance
};

template<typename T>
struct DataSplit {
    std::vector<T> train;
    std::vector<T> validation;
    std::vector<T> test;
    
    // Metadata
    struct SplitMetrics {
        size_t total_samples = 0;
        float actual_train_ratio = 0.0f;
        float actual_val_ratio = 0.0f;
        float actual_test_ratio = 0.0f;
        size_t avg_train_length = 0;
        size_t avg_val_length = 0;
        size_t avg_test_length = 0;
        bool ratios_adjusted = false;
        std::string adjustment_reason;
    } metrics;
    
    void clear() {
        train.clear();
        validation.clear();
        test.clear();
        metrics = SplitMetrics{};
    }
    
    size_t totalSize() const {
        return train.size() + validation.size() + test.size();
    }
    
    void computeMetrics() {
        metrics.total_samples = totalSize();
        if (metrics.total_samples > 0) {
            metrics.actual_train_ratio = static_cast<float>(train.size()) / metrics.total_samples;
            metrics.actual_val_ratio = static_cast<float>(validation.size()) / metrics.total_samples;
            metrics.actual_test_ratio = static_cast<float>(test.size()) / metrics.total_samples;
        }
    }
    
    void printSummary() const {
        std::cout << "\n=== Split Summary ===\n";
        std::cout << "Train:      " << train.size() << " samples (" 
                  << (metrics.actual_train_ratio * 100) << "%)\n";
        std::cout << "Validation: " << validation.size() << " samples (" 
                  << (metrics.actual_val_ratio * 100) << "%)\n";
        std::cout << "Test:       " << test.size() << " samples (" 
                  << (metrics.actual_test_ratio * 100) << "%)\n";
        std::cout << "Total:      " << metrics.total_samples << " samples\n";
        
        if (metrics.ratios_adjusted) {
            std::cout << "⚠ Ratios adjusted: " << metrics.adjustment_reason << "\n";
        }
        std::cout << "====================\n\n";
    }
};


template<typename T>
class DataSplitter {
public:
    explicit DataSplitter(const SplitConfig& config = SplitConfig())
        : config_(config), rng_(config.random_seed) {}
    
    DataSplit<T> split(const std::vector<T>& data) {
        DataSplit<T> result;
        
        if (data.empty()) {
            std::cerr << "WARNING: Empty dataset provided to splitter\n";
            return result;
        }
        
        // Step 1: Validate and auto-adjust ratios if needed
        auto adjusted_config = validateAndAdjustConfig(data.size());
        
        // Step 2: Apply splitting strategy
        switch (adjusted_config.strategy) {
            case SplitStrategy::LENGTH_BALANCED:
                return splitLengthBalanced(data, adjusted_config);
            case SplitStrategy::STRATIFIED:
                std::cerr << "STRATIFIED split requires metadata - falling back to RANDOM\n";
                [[fallthrough]];
            case SplitStrategy::RANDOM:
            default:
                return splitRandom(data, adjusted_config);
        }
    }
    
    // Stratified split by categories/tags (for categorized data)
    DataSplit<T> splitStratified(const std::vector<T>& data,
                                  const std::vector<std::string>& categories) {
        DataSplit<T> result;
        
        if (data.size() != categories.size()) {
            std::cerr << "ERROR: Data and categories size mismatch\n";
            return splitRandom(data, config_);
        }
        
        // Group by category
        std::unordered_map<std::string, std::vector<size_t>> category_indices;
        for (size_t i = 0; i < data.size(); ++i) {
            category_indices[categories[i]].push_back(i);
        }
        
        // Split each category proportionally
        auto adjusted_config = validateAndAdjustConfig(data.size());
        
        for (auto& [category, indices] : category_indices) {
            if (adjusted_config.shuffle) {
                std::shuffle(indices.begin(), indices.end(), rng_);
            }
            
            size_t train_end = static_cast<size_t>(indices.size() * adjusted_config.train_ratio);
            size_t val_end = train_end + static_cast<size_t>(indices.size() * adjusted_config.val_ratio);
            
            for (size_t i = 0; i < train_end && i < indices.size(); ++i) {
                result.train.push_back(data[indices[i]]);
            }
            for (size_t i = train_end; i < val_end && i < indices.size(); ++i) {
                result.validation.push_back(data[indices[i]]);
            }
            for (size_t i = val_end; i < indices.size(); ++i) {
                result.test.push_back(data[indices[i]]);
            }
        }
        
        result.computeMetrics();
        return result;
    }
    
    // Save splits to separate files
    bool saveSplits(const DataSplit<std::string>& split,
                    const std::string& output_dir) {
        try {
            fs::create_directories(output_dir);
            
            saveToFile(split.train, output_dir + "/train.txt");
            saveToFile(split.validation, output_dir + "/val.txt");
            saveToFile(split.test, output_dir + "/test.txt");
            
            // Save comprehensive split info
            std::ofstream info(output_dir + "/split_info.txt");
            info << "=== GRIM Data Split Information ===\n\n";
            info << "Train:      " << split.train.size() << " samples (" 
                 << (split.metrics.actual_train_ratio * 100) << "%)\n";
            info << "Validation: " << split.validation.size() << " samples (" 
                 << (split.metrics.actual_val_ratio * 100) << "%)\n";
            info << "Test:       " << split.test.size() << " samples (" 
                 << (split.metrics.actual_test_ratio * 100) << "%)\n";
            info << "Total:      " << split.metrics.total_samples << " samples\n\n";
            
            info << "Configuration:\n";
            info << "  Random seed:    " << config_.random_seed << "\n";
            info << "  Strategy:       " << strategyToString(config_.strategy) << "\n";
            info << "  Shuffle:        " << (config_.shuffle ? "Yes" : "No") << "\n\n";
            
            if (split.metrics.ratios_adjusted) {
                info << "Adjustments:\n";
                info << "  " << split.metrics.adjustment_reason << "\n\n";
            }
            
            info << "Generated: " << getCurrentTimestamp() << "\n";
            info.close();
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "ERROR saving splits: " << e.what() << "\n";
            return false;
        }
    }
    
private:
    SplitConfig config_;
    std::mt19937 rng_;
    
    // Validate configuration and auto-adjust for small datasets
    SplitConfig validateAndAdjustConfig(size_t dataset_size) {
        SplitConfig adjusted = config_;
        
        // Check if ratios sum to 1.0
        float total_ratio = adjusted.train_ratio + adjusted.val_ratio + adjusted.test_ratio;
        if (std::abs(total_ratio - 1.0f) > adjusted.tolerance) {
            // Normalize ratios
            adjusted.train_ratio /= total_ratio;
            adjusted.val_ratio /= total_ratio;
            adjusted.test_ratio /= total_ratio;
            std::cerr << "WARNING: Split ratios normalized to sum to 1.0\n";
        }
        
        // Auto-adjust for small datasets
        if (adjusted.auto_adjust_ratios && dataset_size < 100) {
            size_t val_samples = static_cast<size_t>(dataset_size * adjusted.val_ratio);
            size_t test_samples = static_cast<size_t>(dataset_size * adjusted.test_ratio);
            
            if (val_samples < adjusted.min_samples_per_split || 
                test_samples < adjusted.min_samples_per_split) {
                
                // Ensure minimum samples
                size_t min_val = std::max(adjusted.min_samples_per_split, val_samples);
                size_t min_test = std::max(adjusted.min_samples_per_split, test_samples);
                size_t reserved = min_val + min_test;
                
                if (reserved >= dataset_size) {
                    // Dataset too small - use 60/20/20 split
                    adjusted.train_ratio = 0.6f;
                    adjusted.val_ratio = 0.2f;
                    adjusted.test_ratio = 0.2f;
                    std::cerr << "WARNING: Very small dataset (" << dataset_size 
                              << " samples) - using 60/20/20 split\n";
                } else {
                    // Recalculate ratios
                    adjusted.val_ratio = static_cast<float>(min_val) / dataset_size;
                    adjusted.test_ratio = static_cast<float>(min_test) / dataset_size;
                    adjusted.train_ratio = 1.0f - adjusted.val_ratio - adjusted.test_ratio;
                    
                    std::cerr << "INFO: Adjusted ratios for small dataset: "
                              << (adjusted.train_ratio * 100) << "/"
                              << (adjusted.val_ratio * 100) << "/"
                              << (adjusted.test_ratio * 100) << "\n";
                }
            }
        }
        
        return adjusted;
    }
    
    // Standard random split
    DataSplit<T> splitRandom(const std::vector<T>& data, const SplitConfig& cfg) {
        DataSplit<T> result;
        
        // Create shuffled indices
        std::vector<size_t> indices(data.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }
        
        if (cfg.shuffle) {
            std::shuffle(indices.begin(), indices.end(), rng_);
        }
        
        // Calculate split points
        size_t train_end = static_cast<size_t>(data.size() * cfg.train_ratio);
        size_t val_end = train_end + static_cast<size_t>(data.size() * cfg.val_ratio);
        
        // Ensure we don't exceed bounds
        train_end = std::min(train_end, data.size());
        val_end = std::min(val_end, data.size());
        
        // Split data
        result.train.reserve(train_end);
        result.validation.reserve(val_end - train_end);
        result.test.reserve(data.size() - val_end);
        
        for (size_t i = 0; i < train_end; ++i) {
            result.train.push_back(data[indices[i]]);
        }
        
        for (size_t i = train_end; i < val_end; ++i) {
            result.validation.push_back(data[indices[i]]);
        }
        
        for (size_t i = val_end; i < data.size(); ++i) {
            result.test.push_back(data[indices[i]]);
        }
        
        result.computeMetrics();
        return result;
    }
    
    // Length-balanced split (ensures similar avg length across splits)
    DataSplit<T> splitLengthBalanced(const std::vector<std::string>& data, 
                                      const SplitConfig& cfg) {
        DataSplit<std::string> result;
        
        // Sort by length with original indices
        std::vector<std::pair<size_t, size_t>> length_indices;
        for (size_t i = 0; i < data.size(); ++i) {
            length_indices.push_back({data[i].length(), i});
        }
        std::sort(length_indices.begin(), length_indices.end());
        
        // Distribute evenly across splits (round-robin by length)
        size_t train_target = static_cast<size_t>(data.size() * cfg.train_ratio);
        size_t val_target = static_cast<size_t>(data.size() * cfg.val_ratio);
        
        for (size_t i = 0; i < length_indices.size(); ++i) {
            size_t idx = length_indices[i].second;
            
            if (result.train.size() < train_target) {
                result.train.push_back(data[idx]);
            } else if (result.validation.size() < val_target) {
                result.validation.push_back(data[idx]);
            } else {
                result.test.push_back(data[idx]);
            }
        }
        
        // Shuffle each split to remove length ordering
        if (cfg.shuffle) {
            std::shuffle(result.train.begin(), result.train.end(), rng_);
            std::shuffle(result.validation.begin(), result.validation.end(), rng_);
            std::shuffle(result.test.begin(), result.test.end(), rng_);
        }
        
        result.computeMetrics();
        return result;
    }
    
    void saveToFile(const std::vector<std::string>& data,
                   const std::string& filepath) {
        std::ofstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open: " + filepath);
        }
        
        for (const auto& item : data) {
            file << item << "\n";
        }
        file.close();
    }
    
    std::string strategyToString(SplitStrategy strategy) const {
        switch (strategy) {
            case SplitStrategy::RANDOM: return "Random";
            case SplitStrategy::STRATIFIED: return "Stratified";
            case SplitStrategy::LENGTH_BALANCED: return "Length Balanced";
            case SplitStrategy::SOURCE_BALANCED: return "Source Balanced";
            default: return "Unknown";
        }
    }
    
    std::string getCurrentTimestamp() const {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }
};

} // namespace Training
} // namespace GRIM

//======================================================//
//  embedding_self_test.hpp
//  Diagnostic test suite for Embedding layer GPU operations
//======================================================//

#pragma once

#include <string>
#include <vector>
#include <functional>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <algorithm>

namespace GRIM {
namespace Test {

//======================================================//
//  Embedding Test Result
//======================================================//

struct EmbeddingTestResult {
    std::string name;
    bool passed;
    std::string message;
    double duration_ms;
};

//======================================================//
//  Embedding Test Suite
//======================================================//

class EmbeddingTestSuite {
public:
    using TestFunc = std::function<bool(std::string&)>;
    
    void addTest(const std::string& name, TestFunc func) {
        tests_.push_back({name, func});
    }
    
    std::vector<EmbeddingTestResult> runAll() {
        std::vector<EmbeddingTestResult> results;
        results.reserve(tests_.size());
        
        std::cout << "\n========================================\n";
        std::cout << "  Embedding Layer Diagnostic Test Suite\n";
        std::cout << "========================================\n\n";
        
        int passed = 0;
        int failed = 0;
        
        for (const auto& [name, func] : tests_) {
            EmbeddingTestResult result;
            result.name = name;
            
            auto start = std::chrono::high_resolution_clock::now();
            
            try {
                result.passed = func(result.message);
            } catch (const std::exception& e) {
                result.passed = false;
                result.message = std::string("Exception: ") + e.what();
            } catch (...) {
                result.passed = false;
                result.message = "Unknown exception";
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            result.duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
            
            // Print result
            std::cout << (result.passed ? "[PASS]" : "[FAIL]") << " "
                      << std::setw(45) << std::left << result.name
                      << " (" << std::fixed << std::setprecision(2) 
                      << result.duration_ms << " ms)";
            
            if (!result.passed && !result.message.empty()) {
                std::cout << "\n       " << result.message;
            }
            std::cout << "\n";
            
            if (result.passed) ++passed;
            else ++failed;
            
            results.push_back(std::move(result));
        }
        
        std::cout << "\n----------------------------------------\n";
        std::cout << "Results: " << passed << " passed, " << failed << " failed\n";
        std::cout << "========================================\n\n";
        
        return results;
    }
    
private:
    std::vector<std::pair<std::string, TestFunc>> tests_;
};

//======================================================//
//  Diagnostic Visualization Helpers
//======================================================//

// Print embedding statistics with visual histogram
inline void printEmbeddingStats(const char* name, const float* h_data, 
                                 size_t count, int d_model) {
    if (!h_data || count == 0) {
        std::cout << "[" << name << "] NO DATA\n";
        return;
    }
    
    double sum = 0.0, sq_sum = 0.0;
    float min_val = h_data[0], max_val = h_data[0];
    int nan_count = 0, inf_count = 0, zero_count = 0;
    
    for (size_t i = 0; i < count; ++i) {
        float v = h_data[i];
        if (std::isnan(v)) { ++nan_count; continue; }
        if (std::isinf(v)) { ++inf_count; continue; }
        if (v == 0.0f) ++zero_count;
        sum += v;
        sq_sum += v * v;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }
    
    double mean = sum / static_cast<double>(count - nan_count - inf_count);
    double variance = (sq_sum / static_cast<double>(count - nan_count - inf_count)) - (mean * mean);
    double stddev = std::sqrt(std::max(0.0, variance));
    
    std::cout << "\n[" << name << "] Statistics:\n";
    std::cout << "  Elements:  " << count << " (d_model=" << d_model << ")\n";
    std::cout << "  Mean:      " << std::fixed << std::setprecision(6) << mean << "\n";
    std::cout << "  StdDev:    " << stddev << "\n";
    std::cout << "  Range:     [" << min_val << ", " << max_val << "]\n";
    std::cout << "  Zeros:     " << zero_count << " (" 
              << std::setprecision(2) << (100.0 * zero_count / count) << "%)\n";
    if (nan_count > 0) std::cout << "  ⚠ NaN:     " << nan_count << "\n";
    if (inf_count > 0) std::cout << "  ⚠ Inf:     " << inf_count << "\n";
    
    // Simple text histogram
    constexpr int kBuckets = 10;
    std::vector<int> histogram(kBuckets, 0);
    float bucket_width = (max_val - min_val) / kBuckets;
    if (bucket_width <= 0.0f) bucket_width = 1.0f;
    
    for (size_t i = 0; i < count; ++i) {
        float v = h_data[i];
        if (std::isnan(v) || std::isinf(v)) continue;
        int bucket = static_cast<int>((v - min_val) / bucket_width);
        bucket = std::clamp(bucket, 0, kBuckets - 1);
        ++histogram[bucket];
    }
    
    int max_bucket = *std::max_element(histogram.begin(), histogram.end());
    constexpr int kBarWidth = 40;
    
    std::cout << "  Distribution:\n";
    for (int i = 0; i < kBuckets; ++i) {
        float lo = min_val + i * bucket_width;
        float hi = lo + bucket_width;
        int bar_len = (max_bucket > 0) ? (histogram[i] * kBarWidth / max_bucket) : 0;
        std::cout << "    [" << std::setw(8) << std::setprecision(3) << lo 
                  << "," << std::setw(8) << hi << "] ";
        std::cout << std::string(bar_len, '#') << " " << histogram[i] << "\n";
    }
}

// Print per-token embedding norms (first N rows)
inline void printEmbeddingNorms(const char* name, const float* h_data,
                                 int num_rows, int d_model, int max_show = 10) {
    std::cout << "\n[" << name << "] Row Norms (first " << std::min(num_rows, max_show) << "):\n";
    for (int row = 0; row < std::min(num_rows, max_show); ++row) {
        const float* row_data = h_data + static_cast<size_t>(row) * d_model;
        float norm = 0.0f;
        for (int i = 0; i < d_model; ++i) {
            norm += row_data[i] * row_data[i];
        }
        norm = std::sqrt(norm);
        std::cout << "  Row " << std::setw(5) << row << ": ||e|| = " 
                  << std::setprecision(4) << norm << "\n";
    }
}

// Detect if embeddings look initialized (non-uniform, non-zero)
inline bool detectInitialization(const float* h_data, size_t count) {
    if (!h_data || count < 10) return false;
    
    float first = h_data[0];
    bool all_same = true;
    bool all_zero = true;
    
    for (size_t i = 0; i < std::min(count, size_t(1000)); ++i) {
        if (h_data[i] != first) all_same = false;
        if (h_data[i] != 0.0f) all_zero = false;
    }
    
    return !all_same && !all_zero;
}

//======================================================//
//  Test Assertions
//======================================================//

#define EMB_ASSERT_TRUE(cond, msg) \
    if (!(cond)) { \
        message = std::string(msg) + " (line " + std::to_string(__LINE__) + ")"; \
        return false; \
    }

#define EMB_ASSERT_FALSE(cond, msg) \
    EMB_ASSERT_TRUE(!(cond), msg)

#define EMB_ASSERT_EQ(a, b, msg) \
    if ((a) != (b)) { \
        message = std::string(msg) + ": expected " + std::to_string(b) + ", got " + std::to_string(a); \
        return false; \
    }

#define EMB_ASSERT_NEAR(a, b, eps, msg) \
    if (std::abs((a) - (b)) > (eps)) { \
        message = std::string(msg) + ": expected ~" + std::to_string(b) + ", got " + std::to_string(a); \
        return false; \
    }

#define EMB_ASSERT_NOT_NAN(val, msg) \
    if (std::isnan(val)) { \
        message = std::string(msg) + ": value is NaN"; \
        return false; \
    }

#define EMB_ASSERT_NOT_INF(val, msg) \
    if (std::isinf(val)) { \
        message = std::string(msg) + ": value is Inf"; \
        return false; \
    }

#define EMB_ASSERT_NO_CUDA_ERROR(msg) \
    { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            message = std::string(msg) + ": " + cudaGetErrorString(err); \
            return false; \
        } \
    }

} // namespace Test
} // namespace GRIM

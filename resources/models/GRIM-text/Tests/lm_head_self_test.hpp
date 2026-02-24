//======================================================//
//  lm_head_self_test.hpp
//  Comprehensive diagnostic test suite for LM Head GPU layer
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
#include <fstream>
#include <sstream>
#include <map>

namespace GRIM {
namespace Test {

//======================================================//
//  LM Head Test Result
//======================================================//

struct LMHeadTestResult {
    std::string name;
    std::string category;
    bool passed;
    std::string message;
    double duration_ms;
    
    // Diagnostic data for visualization
    std::map<std::string, double> metrics;
};

//======================================================//
//  Test Statistics for Summary
//======================================================//

struct LMHeadTestStatistics {
    int total_tests = 0;
    int passed = 0;
    int failed = 0;
    double total_duration_ms = 0.0;
    std::map<std::string, int> category_passed;
    std::map<std::string, int> category_failed;
    std::vector<std::string> failed_tests;
    std::vector<LMHeadTestResult> all_results;
};

//======================================================//
//  Visualization Helpers
//======================================================//

inline void printProgressBar(const char* label, double value, double max_val, 
                             int width = 40, char fill = '#', char empty = '-') {
    int filled = static_cast<int>((value / max_val) * width);
    filled = std::clamp(filled, 0, width);
    std::cout << "  " << std::setw(25) << std::left << label << " [";
    for (int i = 0; i < width; ++i) {
        std::cout << (i < filled ? fill : empty);
    }
    std::cout << "] " << std::fixed << std::setprecision(2) << value;
    if (max_val > 0) {
        std::cout << " / " << max_val << " (" << (100.0 * value / max_val) << "%)";
    }
    std::cout << "\n";
}

inline void printGradientBar(const char* label, double value, double range = 10.0) {
    // Normalize to [-range, +range]
    double normalized = std::clamp(value / range, -1.0, 1.0);
    int center = 25;
    int width = 50;
    
    std::cout << "  " << std::setw(25) << std::left << label << " [";
    for (int i = 0; i < width; ++i) {
        if (i == center) {
            std::cout << '|';
        } else if (normalized > 0 && i > center && i <= center + static_cast<int>(normalized * 25)) {
            std::cout << '+';
        } else if (normalized < 0 && i < center && i >= center + static_cast<int>(normalized * 25)) {
            std::cout << '-';
        } else {
            std::cout << ' ';
        }
    }
    std::cout << "] " << std::scientific << std::setprecision(4) << value << "\n";
}

inline void printLogitDistribution(const char* name, const float* logits, 
                                    int vocab_size, int max_buckets = 20) {
    if (!logits || vocab_size <= 0) {
        std::cout << "[" << name << "] NO DATA\n";
        return;
    }
    
    // Find min/max
    float min_val = logits[0], max_val = logits[0];
    double sum = 0.0;
    int nan_count = 0, inf_count = 0;
    
    for (int i = 0; i < vocab_size; ++i) {
        float v = logits[i];
        if (std::isnan(v)) { ++nan_count; continue; }
        if (std::isinf(v)) { ++inf_count; continue; }
        sum += v;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }
    
    double mean = sum / static_cast<double>(vocab_size - nan_count - inf_count);
    
    std::cout << "\n[" << name << "] Logit Distribution:\n";
    std::cout << "  Vocab size:  " << vocab_size << "\n";
    std::cout << "  Range:       [" << min_val << ", " << max_val << "]\n";
    std::cout << "  Mean:        " << mean << "\n";
    if (nan_count > 0) std::cout << "  ⚠ NaN:       " << nan_count << "\n";
    if (inf_count > 0) std::cout << "  ⚠ Inf:       " << inf_count << "\n";
    
    // Histogram
    std::vector<int> histogram(max_buckets, 0);
    float bucket_width = (max_val - min_val) / max_buckets;
    if (bucket_width <= 0.0f) bucket_width = 1.0f;
    
    for (int i = 0; i < vocab_size; ++i) {
        float v = logits[i];
        if (std::isnan(v) || std::isinf(v)) continue;
        int bucket = static_cast<int>((v - min_val) / bucket_width);
        bucket = std::clamp(bucket, 0, max_buckets - 1);
        ++histogram[bucket];
    }
    
    int max_bucket = *std::max_element(histogram.begin(), histogram.end());
    constexpr int kBarWidth = 40;
    
    std::cout << "  Distribution:\n";
    for (int i = 0; i < max_buckets; ++i) {
        float lo = min_val + i * bucket_width;
        float hi = lo + bucket_width;
        int bar_len = (max_bucket > 0) ? (histogram[i] * kBarWidth / max_bucket) : 0;
        std::cout << "    [" << std::setw(9) << std::setprecision(3) << lo 
                  << "," << std::setw(9) << hi << "] ";
        std::cout << std::string(bar_len, '#') << " " << histogram[i] << "\n";
    }
}

inline void printWeightStatistics(const char* name, const float* weights,
                                   int rows, int cols) {
    if (!weights || rows <= 0 || cols <= 0) {
        std::cout << "[" << name << "] NO DATA\n";
        return;
    }
    
    size_t total = static_cast<size_t>(rows) * cols;
    double sum = 0.0, sq_sum = 0.0;
    float min_val = weights[0], max_val = weights[0];
    int nan_count = 0, inf_count = 0, zero_count = 0;
    
    for (size_t i = 0; i < total; ++i) {
        float v = weights[i];
        if (std::isnan(v)) { ++nan_count; continue; }
        if (std::isinf(v)) { ++inf_count; continue; }
        if (v == 0.0f) ++zero_count;
        sum += v;
        sq_sum += v * v;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }
    
    size_t valid = total - nan_count - inf_count;
    double mean = sum / static_cast<double>(valid);
    double variance = (sq_sum / static_cast<double>(valid)) - (mean * mean);
    double stddev = std::sqrt(std::max(0.0, variance));
    
    std::cout << "\n[" << name << "] Weight Statistics:\n";
    std::cout << "  Shape:       [" << rows << " x " << cols << "] = " << total << " params\n";
    std::cout << "  Mean:        " << std::fixed << std::setprecision(6) << mean << "\n";
    std::cout << "  StdDev:      " << stddev << "\n";
    std::cout << "  Range:       [" << min_val << ", " << max_val << "]\n";
    std::cout << "  Zeros:       " << zero_count << " (" 
              << std::setprecision(2) << (100.0 * zero_count / total) << "%)\n";
    std::cout << "  RMS:         " << std::sqrt(sq_sum / static_cast<double>(total)) << "\n";
    if (nan_count > 0) std::cout << "  ⚠ NaN:       " << nan_count << "\n";
    if (inf_count > 0) std::cout << "  ⚠ Inf:       " << inf_count << "\n";
}

inline void printGradientComponents(const char* name,
                                     const std::vector<std::pair<std::string, double>>& components) {
    std::cout << "\n[" << name << "] Gradient Components:\n";
    double total_sq = 0.0;
    int total_count = 0;
    for (const auto& [component_name, rms] : components) {
        total_sq += rms * rms;
        total_count++;
    }
    double total_rms = (total_count > 0) ? std::sqrt(total_sq / total_count) : 0.0;
    
    for (const auto& [component_name, rms] : components) {
        double pct = (total_rms > 0) ? (100.0 * rms / total_rms) : 0.0;
        std::cout << "  " << std::setw(20) << std::left << component_name 
                  << " g_rms = " << std::scientific << std::setprecision(4) << rms
                  << " (" << std::fixed << std::setprecision(1) << pct << "%)\n";
    }
    std::cout << "  " << std::setw(20) << std::left << "TOTAL"
              << " g_rms = " << std::scientific << std::setprecision(4) << total_rms << "\n";
}

//======================================================//
//  ASCII Art Visualizations
//======================================================//

inline void printMatrixHeatmap(const char* name, const float* matrix,
                                int rows, int cols, int max_display = 20) {
    if (!matrix || rows <= 0 || cols <= 0) return;
    
    int display_rows = std::min(rows, max_display);
    int display_cols = std::min(cols, max_display);
    
    // Find range for normalization
    float min_val = matrix[0], max_val = matrix[0];
    for (int r = 0; r < display_rows; ++r) {
        for (int c = 0; c < display_cols; ++c) {
            float v = matrix[r * cols + c];
            if (!std::isnan(v) && !std::isinf(v)) {
                if (v < min_val) min_val = v;
                if (v > max_val) max_val = v;
            }
        }
    }
    
    const char* intensity = " .-+*#@";
    int levels = 7;
    float range = max_val - min_val;
    if (range <= 0) range = 1.0f;
    
    std::cout << "\n[" << name << "] Matrix Heatmap (" << display_rows << "x" << display_cols << "):\n";
    std::cout << "    ";
    for (int c = 0; c < display_cols; ++c) {
        std::cout << (c % 10);
    }
    std::cout << "\n    ";
    for (int c = 0; c < display_cols; ++c) std::cout << "-";
    std::cout << "\n";
    
    for (int r = 0; r < display_rows; ++r) {
        std::cout << std::setw(3) << r << "|";
        for (int c = 0; c < display_cols; ++c) {
            float v = matrix[r * cols + c];
            if (std::isnan(v)) {
                std::cout << 'N';
            } else if (std::isinf(v)) {
                std::cout << 'I';
            } else {
                int level = static_cast<int>((v - min_val) / range * (levels - 1));
                level = std::clamp(level, 0, levels - 1);
                std::cout << intensity[level];
            }
        }
        std::cout << "|\n";
    }
    std::cout << "    ";
    for (int c = 0; c < display_cols; ++c) std::cout << "-";
    std::cout << "\n";
    std::cout << "  Legend: ' '=min(" << min_val << ") -> '@'=max(" << max_val << ")\n";
}

//======================================================//
//  Summary Report Generation
//======================================================//

inline void generateSummaryReport(const LMHeadTestStatistics& stats,
                                   const std::string& output_path) {
    std::ofstream out(output_path);
    if (!out.is_open()) {
        std::cerr << "Failed to write summary to: " << output_path << "\n";
        return;
    }
    
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    out << "═══════════════════════════════════════════════════════════════════\n";
    out << "               LM HEAD DIAGNOSTIC TEST SUMMARY REPORT\n";
    out << "═══════════════════════════════════════════════════════════════════\n\n";
    out << "Generated: " << std::ctime(&time_t);
    out << "Duration:  " << std::fixed << std::setprecision(2) 
        << stats.total_duration_ms << " ms\n\n";
    
    // Overall Results
    out << "┌─────────────────────────────────────────────────────────────────┐\n";
    out << "│                        OVERALL RESULTS                         │\n";
    out << "├─────────────────────────────────────────────────────────────────┤\n";
    double pass_rate = (stats.total_tests > 0) 
                       ? (100.0 * stats.passed / stats.total_tests) : 0.0;
    out << "│  Total Tests:    " << std::setw(4) << stats.total_tests << "                                          │\n";
    out << "│  Passed:         " << std::setw(4) << stats.passed 
        << " (" << std::setw(6) << std::setprecision(2) << pass_rate << "%)                               │\n";
    out << "│  Failed:         " << std::setw(4) << stats.failed << "                                          │\n";
    out << "└─────────────────────────────────────────────────────────────────┘\n\n";
    
    // ASCII Progress Bar
    out << "Pass Rate: [";
    int bar_width = 50;
    int filled = static_cast<int>(pass_rate / 100.0 * bar_width);
    for (int i = 0; i < bar_width; ++i) {
        out << (i < filled ? '█' : '░');
    }
    out << "] " << std::setprecision(1) << pass_rate << "%\n\n";
    
    // Category Breakdown
    out << "┌─────────────────────────────────────────────────────────────────┐\n";
    out << "│                      CATEGORY BREAKDOWN                        │\n";
    out << "├───────────────────────────┬─────────┬─────────┬────────────────┤\n";
    out << "│ Category                  │ Passed  │ Failed  │ Pass Rate      │\n";
    out << "├───────────────────────────┼─────────┼─────────┼────────────────┤\n";
    
    std::set<std::string> categories;
    for (const auto& [cat, _] : stats.category_passed) categories.insert(cat);
    for (const auto& [cat, _] : stats.category_failed) categories.insert(cat);
    
    for (const auto& cat : categories) {
        int p = 0, f = 0;
        auto it_p = stats.category_passed.find(cat);
        auto it_f = stats.category_failed.find(cat);
        if (it_p != stats.category_passed.end()) p = it_p->second;
        if (it_f != stats.category_failed.end()) f = it_f->second;
        double rate = (p + f > 0) ? (100.0 * p / (p + f)) : 0.0;
        
        out << "│ " << std::setw(25) << std::left << cat 
            << " │ " << std::setw(7) << p
            << " │ " << std::setw(7) << f
            << " │ " << std::setw(6) << std::setprecision(1) << rate << "%        │\n";
    }
    out << "└───────────────────────────┴─────────┴─────────┴────────────────┘\n\n";
    
    // Failed Tests Detail
    if (!stats.failed_tests.empty()) {
        out << "┌─────────────────────────────────────────────────────────────────┐\n";
        out << "│                       FAILED TESTS                             │\n";
        out << "├─────────────────────────────────────────────────────────────────┤\n";
        for (const auto& test : stats.failed_tests) {
            out << "│ ✗ " << std::setw(62) << std::left << test << "│\n";
        }
        out << "└─────────────────────────────────────────────────────────────────┘\n\n";
    }
    
    // All Tests Detail
    out << "┌─────────────────────────────────────────────────────────────────┐\n";
    out << "│                       ALL TEST RESULTS                         │\n";
    out << "├─────────────────────────────────────────────────────────────────┤\n";
    for (const auto& result : stats.all_results) {
        out << "│ " << (result.passed ? "✓" : "✗") << " "
            << std::setw(45) << std::left << result.name
            << " " << std::setw(8) << std::right << std::setprecision(2) 
            << result.duration_ms << "ms │\n";
        if (!result.passed && !result.message.empty()) {
            out << "│     └─ " << std::setw(56) << std::left 
                << result.message.substr(0, 56) << " │\n";
        }
    }
    out << "└─────────────────────────────────────────────────────────────────┘\n\n";
    
    // Metrics Summary
    out << "┌─────────────────────────────────────────────────────────────────┐\n";
    out << "│                    KEY METRICS COLLECTED                       │\n";
    out << "├─────────────────────────────────────────────────────────────────┤\n";
    for (const auto& result : stats.all_results) {
        if (!result.metrics.empty()) {
            out << "│ [" << result.name << "]\n";
            for (const auto& [metric_name, value] : result.metrics) {
                out << "│   " << std::setw(25) << std::left << metric_name
                    << " = " << std::scientific << std::setprecision(4) << value << "\n";
            }
        }
    }
    out << "└─────────────────────────────────────────────────────────────────┘\n\n";
    
    out << "═══════════════════════════════════════════════════════════════════\n";
    out << "                         END OF REPORT\n";
    out << "═══════════════════════════════════════════════════════════════════\n";
    
    out.close();
    std::cout << "\n[INFO] Summary report written to: " << output_path << "\n";
}

//======================================================//
//  LM Head Test Suite
//======================================================//

class LMHeadTestSuite {
public:
    using TestFunc = std::function<bool(std::string&, std::map<std::string, double>&)>;
    
    void addTest(const std::string& name, const std::string& category, TestFunc func) {
        tests_.push_back({name, category, func});
    }
    
    LMHeadTestStatistics runAll() {
        LMHeadTestStatistics stats;
        stats.total_tests = static_cast<int>(tests_.size());
        
        std::cout << "\n";
        std::cout << "╔═══════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║           LM HEAD GPU DIAGNOSTIC TEST SUITE                      ║\n";
        std::cout << "║  Comprehensive tests for forward projection, backward gradients, ║\n";
        std::cout << "║  weight tying, numerical stability, and cuBLAS integration       ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════════╝\n\n";
        
        std::string current_category;
        
        for (const auto& [name, category, func] : tests_) {
            // Print category header if changed
            if (category != current_category) {
                current_category = category;
                std::cout << "\n────────────────────────────────────────────────────────────────────\n";
                std::cout << "  CATEGORY: " << category << "\n";
                std::cout << "────────────────────────────────────────────────────────────────────\n";
            }
            
            LMHeadTestResult result;
            result.name = name;
            result.category = category;
            
            auto start = std::chrono::high_resolution_clock::now();
            
            try {
                result.passed = func(result.message, result.metrics);
            } catch (const std::exception& e) {
                result.passed = false;
                result.message = std::string("Exception: ") + e.what();
            } catch (...) {
                result.passed = false;
                result.message = "Unknown exception";
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            result.duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
            stats.total_duration_ms += result.duration_ms;
            
            // Print result with visual indicators
            std::cout << (result.passed ? "  ✓ " : "  ✗ ")
                      << std::setw(50) << std::left << result.name
                      << " [" << std::fixed << std::setprecision(2) 
                      << std::setw(8) << std::right << result.duration_ms << " ms]";
            
            if (!result.passed && !result.message.empty()) {
                std::cout << "\n      └─ " << result.message;
            }
            std::cout << "\n";
            
            // Update statistics
            if (result.passed) {
                ++stats.passed;
                ++stats.category_passed[category];
            } else {
                ++stats.failed;
                ++stats.category_failed[category];
                stats.failed_tests.push_back(name);
            }
            
            stats.all_results.push_back(std::move(result));
        }
        
        // Print summary
        std::cout << "\n";
        std::cout << "╔═══════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                         TEST SUMMARY                             ║\n";
        std::cout << "╠═══════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  Total:    " << std::setw(4) << stats.total_tests 
                  << "                                                    ║\n";
        std::cout << "║  Passed:   " << std::setw(4) << stats.passed 
                  << " (" << std::setw(6) << std::setprecision(2) 
                  << (100.0 * stats.passed / std::max(1, stats.total_tests)) << "%)";
        std::cout << "                                       ║\n";
        std::cout << "║  Failed:   " << std::setw(4) << stats.failed << "                                                    ║\n";
        std::cout << "║  Duration: " << std::setw(8) << std::setprecision(2) 
                  << stats.total_duration_ms << " ms                                        ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════════╝\n\n";
        
        return stats;
    }
    
private:
    std::vector<std::tuple<std::string, std::string, TestFunc>> tests_;
};

//======================================================//
//  Test Assertions
//======================================================//

#define LMH_ASSERT_TRUE(cond, msg) \
    if (!(cond)) { \
        message = std::string(msg) + " (line " + std::to_string(__LINE__) + ")"; \
        return false; \
    }

#define LMH_ASSERT_FALSE(cond, msg) \
    LMH_ASSERT_TRUE(!(cond), msg)

#define LMH_ASSERT_EQ(a, b, msg) \
    if ((a) != (b)) { \
        message = std::string(msg) + ": expected " + std::to_string(b) + ", got " + std::to_string(a); \
        return false; \
    }

#define LMH_ASSERT_NEAR(a, b, eps, msg) \
    if (std::abs((a) - (b)) > (eps)) { \
        message = std::string(msg) + ": expected ~" + std::to_string(b) + ", got " + std::to_string(a); \
        return false; \
    }

#define LMH_ASSERT_NOT_NAN(val, msg) \
    if (std::isnan(val)) { \
        message = std::string(msg) + ": value is NaN"; \
        return false; \
    }

#define LMH_ASSERT_NOT_INF(val, msg) \
    if (std::isinf(val)) { \
        message = std::string(msg) + ": value is Inf"; \
        return false; \
    }

#define LMH_ASSERT_NO_CUDA_ERROR(msg) \
    { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            message = std::string(msg) + ": " + cudaGetErrorString(err); \
            return false; \
        } \
    }

#define LMH_RECORD_METRIC(metrics, name, value) \
    metrics[name] = static_cast<double>(value)

} // namespace Test
} // namespace GRIM

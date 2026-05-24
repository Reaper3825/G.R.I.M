#include "verifier.hpp"
#include <iostream>
#include <fstream>

int main(int argc, char** argv) {
    Config config;
    config.input_dir = GRIM::Config::getRequiredGrimTextPath("collected");
    config.output_dir = GRIM::Config::getRequiredGrimTextPath("verified");
    config.reliability_threshold = 0.3f;  // Lower threshold for basic validation
    config.min_length = 50;
    config.max_length = 100000;
    config.enable_semantic_model = true;
    config.semantic_use_gpu = true;
    config.semantic_hard_filter = true;
    config.semantic_min_score = 0.55f;
    config.semantic_quality_weight = 0.4f;
    config.semantic_model_path = "resources/models/GRIM-text/quality/deberta-v3-base-mnli.onnx";
    config.semantic_tokenizer_path = "resources/models/GRIM-text/quality/deberta-v3-base-mnli.spm";

    // Set default weights including "unknown" source type
    config.source_type_weights = {
        {"academic", 0.9f},
        {"wikipedia", 0.7f},
        {"github", 0.8f},
        {"technical", 0.75f},
        {"unknown", 0.6f}  // Allow unknown sources with moderate reliability
    };

    if (argc > 1) {
        config.input_dir = argv[1];
    }
    if (argc > 2) {
        config.output_dir = argv[2];
    }

    std::cout << "=== GRIM Data Verifier ===" << std::endl;
    std::cout << "Input: " << config.input_dir << std::endl;
    std::cout << "Output: " << config.output_dir << std::endl;
    std::cout << "Reliability threshold: " << config.reliability_threshold << std::endl;
    std::cout << std::endl;

    try {
        Verifier verifier(config);
        
        std::cout << "Loading unverified entries..." << std::endl;
        auto entries = verifier.load_unverified_entries();
        std::cout << "Loaded " << entries.size() << " entries" << std::endl;
        std::cout << std::endl;
        
        std::cout << "Verifying entries..." << std::endl;
        auto verified = verifier.verify_entries(entries);
        std::cout << "Verified " << verified.size() << " entries" << std::endl;
        std::cout << std::endl;
        
        std::cout << "Saving verified entries..." << std::endl;
        if (verifier.save_verified_entries(verified)) {
            std::cout << "Success!" << std::endl;
        } else {
            std::cerr << "Failed to save verified entries" << std::endl;
            return 1;
        }
        std::cout << std::endl;
        
        auto stats = verifier.get_stats();
        std::cout << "=== Verification Statistics ===" << std::endl;
        std::cout << "Total processed: " << stats.total_processed << std::endl;
        std::cout << "Passed: " << stats.passed_verification << std::endl;
        std::cout << "Failed: " << stats.failed_verification << std::endl;
        std::cout << "Domain rejected: " << stats.domain_rejected << std::endl;
        std::cout << "Quality rejected: " << stats.quality_rejected << std::endl;
        std::cout << "Duplicate rejected: " << stats.duplicate_rejected << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

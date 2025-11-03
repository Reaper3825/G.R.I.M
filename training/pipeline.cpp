#include "pipeline.hpp"
#include <fstream>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace grim {
namespace training {

class Pipeline::Impl {
public:
    PipelineConfig config;
    PipelineStats stats;
    
    std::unique_ptr<Collector> collector;
    std::unique_ptr<Verifier> verifier;
    std::unique_ptr<Parser> parser;
    std::unique_ptr<Trainer> trainer;
    
    StageCallback stage_callback;
    
    std::string get_timestamp() const {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }
};

Pipeline::Pipeline() : pImpl(std::make_unique<Impl>()) {
    pImpl->config = PipelineConfig{};
    initialize_stages();
}

Pipeline::Pipeline(const PipelineConfig& config) : pImpl(std::make_unique<Impl>()) {
    pImpl->config = config;
    initialize_stages();
}

Pipeline::~Pipeline() = default;

bool Pipeline::run() {
    auto start_time = std::chrono::steady_clock::now();
    
    log_event("=== Starting GRIM Self-Training Pipeline ===");
    
    bool success = true;
    
    // Stage 1: Collect
    log_event("Stage 1: Collecting online data...");
    if (!run_stage(Stage::COLLECT)) {
        log_event("ERROR: Collection stage failed");
        if (pImpl->config.stop_on_error) return false;
        success = false;
    }
    
    // Stage 2: Verify
    log_event("Stage 2: Verifying sources...");
    if (!run_stage(Stage::VERIFY)) {
        log_event("ERROR: Verification stage failed");
        if (pImpl->config.stop_on_error) return false;
        success = false;
    }
    
    // Stage 3: Parse
    log_event("Stage 3: Parsing verified data...");
    if (!run_stage(Stage::PARSE)) {
        log_event("ERROR: Parsing stage failed");
        if (pImpl->config.stop_on_error) return false;
        success = false;
    }
    
    // Stage 4: Train
    log_event("Stage 4: Training field adapter...");
    if (!run_stage(Stage::TRAIN)) {
        log_event("ERROR: Training stage failed");
        if (pImpl->config.stop_on_error) return false;
        success = false;
    }
    
    // Stage 5: Deploy (only if auto_deploy enabled and training succeeded)
    if (pImpl->config.auto_deploy && success) {
        log_event("Stage 5: Deploying adapter...");
        if (!run_stage(Stage::DEPLOY)) {
            log_event("ERROR: Deployment stage failed");
            if (pImpl->config.stop_on_error) return false;
            success = false;
        }
    } else if (!pImpl->config.auto_deploy) {
        log_event("Auto-deploy disabled. Skipping deployment stage.");
        log_event("Run deploy_adapter() manually to deploy the trained model.");
    }
    
    auto end_time = std::chrono::steady_clock::now();
    pImpl->stats.total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    log_event("=== Pipeline Complete ===");
    log_event("Total time: " + std::to_string(pImpl->stats.total_time.count()) + "ms");
    log_event("Data collected: " + std::to_string(pImpl->stats.data_collected));
    log_event("Data verified: " + std::to_string(pImpl->stats.data_verified));
    log_event("Examples parsed: " + std::to_string(pImpl->stats.examples_parsed));
    log_event("Examples trained: " + std::to_string(pImpl->stats.examples_trained));
    log_event("Deployment: " + std::string(pImpl->stats.deployment_successful ? "SUCCESS" : "SKIPPED/FAILED"));
    
    return success;
}

bool Pipeline::run_stage(Stage stage) {
    bool success = false;
    
    switch (stage) {
        case Stage::COLLECT:
            success = execute_stage(stage, [this]() {
                pImpl->stats.data_collected = pImpl->collector->fetch_online_data();
                pImpl->stats.collector_stats = pImpl->collector->get_stats();
                return pImpl->stats.data_collected > 0;
            });
            break;
            
        case Stage::VERIFY:
            success = execute_stage(stage, [this]() {
                pImpl->stats.data_verified = pImpl->verifier->verify_sources();
                pImpl->stats.verifier_stats = pImpl->verifier->get_stats();
                return pImpl->stats.data_verified > 0;
            });
            break;
            
        case Stage::PARSE:
            success = execute_stage(stage, [this]() {
                pImpl->stats.examples_parsed = pImpl->parser->parse_verified_data();
                pImpl->stats.parser_stats = pImpl->parser->get_stats();
                return pImpl->stats.examples_parsed > 0;
            });
            break;
            
        case Stage::TRAIN:
            success = execute_stage(stage, [this]() {
                bool trained = pImpl->trainer->train_field_adapter();
                if (trained) {
                    pImpl->stats.training_stats = pImpl->trainer->get_stats();
                    pImpl->stats.examples_trained = pImpl->stats.training_stats.train_examples;
                }
                return trained;
            });
            break;
            
        case Stage::DEPLOY:
            success = execute_stage(stage, [this]() {
                bool deployed = pImpl->trainer->deploy_adapter(pImpl->config.deployment);
                pImpl->stats.deployment_successful = deployed;
                return deployed;
            });
            break;
    }
    
    // Call stage callback if set
    if (pImpl->stage_callback) {
        pImpl->stage_callback(stage, success);
    }
    
    return success;
}

bool Pipeline::run_partial(Stage start_stage, Stage end_stage) {
    log_event("Running partial pipeline from " + stage_to_string(start_stage) + 
              " to " + stage_to_string(end_stage));
    
    int start = static_cast<int>(start_stage);
    int end = static_cast<int>(end_stage);
    
    bool success = true;
    
    for (int i = start; i <= end; ++i) {
        Stage current_stage = static_cast<Stage>(i);
        
        if (!run_stage(current_stage)) {
            log_event("ERROR: Stage " + stage_to_string(current_stage) + " failed");
            if (pImpl->config.stop_on_error) {
                return false;
            }
            success = false;
        }
    }
    
    return success;
}

bool Pipeline::load_config(const std::string& config_path) {
    try {
        std::ifstream file(config_path);
        if (!file.is_open()) return false;
        
        json j;
        file >> j;
        
        // Load collector config
        if (j.contains("collector")) {
            auto& c = j["collector"];
            pImpl->config.collector.output_dir = c.value("output_dir", "data/raw");
            pImpl->config.collector.max_entries_per_source = c.value("max_entries", 100);
            pImpl->config.collector.timeout_seconds = c.value("timeout", 30);
            pImpl->config.collector.save_as_jsonl = c.value("save_as_jsonl", true);
        }
        
        // Load verifier config
        if (j.contains("verifier")) {
            auto& v = j["verifier"];
            pImpl->config.verifier.input_dir = v.value("input_dir", "data/raw");
            pImpl->config.verifier.output_dir = v.value("output_dir", "data/verified");
            pImpl->config.verifier.min_reliability_threshold = v.value("min_reliability", 0.8);
            pImpl->config.verifier.require_cross_check = v.value("cross_check", true);
            pImpl->config.verifier.min_cross_references = v.value("min_cross_refs", 2);
        }
        
        // Load parser config
        if (j.contains("parser")) {
            auto& p = j["parser"];
            pImpl->config.parser.input_dir = p.value("input_dir", "data/verified");
            pImpl->config.parser.output_dir = p.value("output_dir", "data/parsed");
            pImpl->config.parser.max_token_length = p.value("max_tokens", 2048);
            pImpl->config.parser.extract_entities = p.value("extract_entities", true);
            pImpl->config.parser.extract_keywords = p.value("extract_keywords", true);
        }
        
        // Load trainer config
        if (j.contains("trainer")) {
            auto& t = j["trainer"];
            pImpl->config.trainer.input_dir = t.value("input_dir", "data/parsed");
            pImpl->config.trainer.model_dir = t.value("model_dir", "models");
            pImpl->config.trainer.learning_rate = t.value("learning_rate", 1e-5);
            pImpl->config.trainer.num_epochs = t.value("num_epochs", 2);
            pImpl->config.trainer.batch_size = t.value("batch_size", 4);
        }
        
        // Load deployment config
        if (j.contains("deployment")) {
            auto& d = j["deployment"];
            pImpl->config.deployment.min_accuracy_threshold = d.value("min_accuracy", 0.85);
            pImpl->config.deployment.run_regression_tests = d.value("run_tests", true);
            pImpl->config.deployment.cleanup_temp_data = d.value("cleanup", true);
        }
        
        // Load pipeline config
        pImpl->config.auto_deploy = j.value("auto_deploy", true);
        pImpl->config.stop_on_error = j.value("stop_on_error", true);
        pImpl->config.log_file = j.value("log_file", "logs/pipeline.log");
        
        // Reinitialize stages with new config
        initialize_stages();
        
        return true;
        
    } catch (...) {
        return false;
    }
}

bool Pipeline::save_config(const std::string& config_path) const {
    try {
        json j;
        
        // Save collector config
        j["collector"] = {
            {"output_dir", pImpl->config.collector.output_dir},
            {"max_entries", pImpl->config.collector.max_entries_per_source},
            {"timeout", pImpl->config.collector.timeout_seconds},
            {"save_as_jsonl", pImpl->config.collector.save_as_jsonl}
        };
        
        // Save verifier config
        j["verifier"] = {
            {"input_dir", pImpl->config.verifier.input_dir},
            {"output_dir", pImpl->config.verifier.output_dir},
            {"min_reliability", pImpl->config.verifier.min_reliability_threshold},
            {"cross_check", pImpl->config.verifier.require_cross_check},
            {"min_cross_refs", pImpl->config.verifier.min_cross_references}
        };
        
        // Save parser config
        j["parser"] = {
            {"input_dir", pImpl->config.parser.input_dir},
            {"output_dir", pImpl->config.parser.output_dir},
            {"max_tokens", pImpl->config.parser.max_token_length},
            {"extract_entities", pImpl->config.parser.extract_entities},
            {"extract_keywords", pImpl->config.parser.extract_keywords}
        };
        
        // Save trainer config
        j["trainer"] = {
            {"input_dir", pImpl->config.trainer.input_dir},
            {"model_dir", pImpl->config.trainer.model_dir},
            {"learning_rate", pImpl->config.trainer.learning_rate},
            {"num_epochs", pImpl->config.trainer.num_epochs},
            {"batch_size", pImpl->config.trainer.batch_size}
        };
        
        // Save deployment config
        j["deployment"] = {
            {"min_accuracy", pImpl->config.deployment.min_accuracy_threshold},
            {"run_tests", pImpl->config.deployment.run_regression_tests},
            {"cleanup", pImpl->config.deployment.cleanup_temp_data}
        };
        
        // Save pipeline config
        j["auto_deploy"] = pImpl->config.auto_deploy;
        j["stop_on_error"] = pImpl->config.stop_on_error;
        j["log_file"] = pImpl->config.log_file;
        
        std::ofstream file(config_path);
        file << j.dump(2);  // Pretty print with 2-space indent
        
        return true;
        
    } catch (...) {
        return false;
    }
}

Pipeline::PipelineStats Pipeline::get_stats() const {
    return pImpl->stats;
}

void Pipeline::set_stage_callback(StageCallback callback) {
    pImpl->stage_callback = callback;
}

void Pipeline::reset() {
    pImpl->stats = PipelineStats{};
    
    if (pImpl->collector) pImpl->collector->reset_stats();
    if (pImpl->verifier) pImpl->verifier->reset_stats();
    if (pImpl->parser) pImpl->parser->reset_stats();
    if (pImpl->trainer) pImpl->trainer->reset();
}

void Pipeline::initialize_stages() {
    pImpl->collector = std::make_unique<Collector>(pImpl->config.collector);
    pImpl->verifier = std::make_unique<Verifier>(pImpl->config.verifier);
    pImpl->parser = std::make_unique<Parser>(pImpl->config.parser);
    pImpl->trainer = std::make_unique<Trainer>(pImpl->config.trainer);
}

void Pipeline::log_event(const std::string& message) {
    std::string log_msg = "[" + pImpl->get_timestamp() + "] " + message;
    
    // Console output
    std::cout << log_msg << std::endl;
    
    // File output
    std::ofstream logfile(pImpl->config.log_file, std::ios::app);
    if (logfile.is_open()) {
        logfile << log_msg << "\n";
        logfile.close();
    }
}

bool Pipeline::execute_stage(Stage stage, const std::function<bool()>& stage_func) {
    try {
        auto start = std::chrono::steady_clock::now();
        
        bool result = stage_func();
        
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        log_event("Stage " + stage_to_string(stage) + " " +
                 (result ? "completed" : "failed") + " in " +
                 std::to_string(duration.count()) + "ms");
        
        return result;
        
    } catch (const std::exception& e) {
        log_event("Exception in stage " + stage_to_string(stage) + ": " + e.what());
        return false;
    }
}

std::string stage_to_string(Pipeline::Stage stage) {
    switch (stage) {
        case Pipeline::Stage::COLLECT: return "COLLECT";
        case Pipeline::Stage::VERIFY: return "VERIFY";
        case Pipeline::Stage::PARSE: return "PARSE";
        case Pipeline::Stage::TRAIN: return "TRAIN";
        case Pipeline::Stage::DEPLOY: return "DEPLOY";
        default: return "UNKNOWN";
    }
}

} // namespace training
} // namespace grim

#include "pipeline.hpp"
#include <iostream>
#include <memory>

using namespace grim::training;

/**
 * Example usage of the GRIM Self-Training Pipeline
 * 
 * This demonstrates how to run the complete 5-stage training process:
 * 1. Collect - Fetch fresh online data
 * 2. Verify - Filter and validate sources
 * 3. Parse - Structure data for training
 * 4. Train - Fine-tune field adapter
 * 5. Deploy - Integrate new adapter safely
 */

int main(int argc, char* argv[]) {
    std::cout << "GRIM Self-Training Pipeline Example\n" << std::endl;
    
    // Option 1: Run with default configuration
    if (argc == 1) {
        std::cout << "Running with default configuration...\n" << std::endl;
        
        auto pipeline = std::make_unique<Pipeline>();
        
        // Set up stage callback for progress updates
        pipeline->set_stage_callback([](Pipeline::Stage stage, bool success) {
            std::cout << ">>> Stage " << stage_to_string(stage) 
                     << " " << (success ? "✓" : "✗") << std::endl;
        });
        
        // Run complete pipeline
        bool success = pipeline->run();
        
        if (success) {
            auto stats = pipeline->get_stats();
            std::cout << "\n=== Pipeline Statistics ===" << std::endl;
            std::cout << "Data collected: " << stats.data_collected << std::endl;
            std::cout << "Data verified: " << stats.data_verified << std::endl;
            std::cout << "Examples parsed: " << stats.examples_parsed << std::endl;
            std::cout << "Examples trained: " << stats.examples_trained << std::endl;
            std::cout << "Best validation accuracy: " 
                     << stats.training_stats.best_val_accuracy << std::endl;
            std::cout << "Deployment: " 
                     << (stats.deployment_successful ? "SUCCESS" : "SKIPPED/FAILED") 
                     << std::endl;
            
            return 0;
        } else {
            std::cerr << "\nPipeline failed!" << std::endl;
            return 1;
        }
    }
    
    // Option 2: Run with custom configuration file
    else if (argc == 2) {
        std::string config_file = argv[1];
        std::cout << "Loading configuration from: " << config_file << "\n" << std::endl;
        
        auto pipeline = std::make_unique<Pipeline>();
        
        if (!pipeline->load_config(config_file)) {
            std::cerr << "Failed to load configuration file!" << std::endl;
            return 1;
        }
        
        bool success = pipeline->run();
        return success ? 0 : 1;
    }
    
    // Option 3: Run individual stages
    else if (argc == 3 && std::string(argv[1]) == "--stage") {
        std::string stage_name = argv[2];
        std::cout << "Running individual stage: " << stage_name << "\n" << std::endl;
        
        auto pipeline = std::make_unique<Pipeline>();
        
        Pipeline::Stage stage;
        if (stage_name == "collect") {
            stage = Pipeline::Stage::COLLECT;
        } else if (stage_name == "verify") {
            stage = Pipeline::Stage::VERIFY;
        } else if (stage_name == "parse") {
            stage = Pipeline::Stage::PARSE;
        } else if (stage_name == "train") {
            stage = Pipeline::Stage::TRAIN;
        } else if (stage_name == "deploy") {
            stage = Pipeline::Stage::DEPLOY;
        } else {
            std::cerr << "Unknown stage: " << stage_name << std::endl;
            std::cerr << "Valid stages: collect, verify, parse, train, deploy" << std::endl;
            return 1;
        }
        
        bool success = pipeline->run_stage(stage);
        return success ? 0 : 1;
    }
    
    // Option 4: Advanced custom configuration
    else if (argc == 2 && std::string(argv[1]) == "--custom") {
        std::cout << "Running with custom configuration...\n" << std::endl;
        
        // Create custom configuration
        PipelineConfig config;
        
        // Configure collector
        config.collector.output_dir = "data/raw";
        config.collector.max_entries_per_source = 200;
        config.collector.save_as_jsonl = true;
        
        // Configure verifier
        config.verifier.input_dir = "data/raw";
        config.verifier.output_dir = "data/verified";
        config.verifier.min_reliability_threshold = 0.85;
        config.verifier.require_cross_check = true;
        config.verifier.min_cross_references = 3;
        
        // Configure parser
        config.parser.input_dir = "data/verified";
        config.parser.output_dir = "data/parsed";
        config.parser.max_token_length = 2048;
        config.parser.strategy = ParserConfig::ParseStrategy::INSTRUCTION;
        
        // Configure trainer
        config.trainer.learning_rate = 5e-6;  // More conservative learning rate
        config.trainer.num_epochs = 3;
        config.trainer.batch_size = 8;
        config.trainer.use_lora = true;
        config.trainer.lora_rank = 16;
        
        // Configure deployment
        config.deployment.min_accuracy_threshold = 0.90;
        config.deployment.run_regression_tests = true;
        config.deployment.cleanup_temp_data = true;
        
        // Pipeline settings
        config.auto_deploy = false;  // Manual deployment for safety
        config.stop_on_error = true;
        
        auto pipeline = std::make_unique<Pipeline>(config);
        
        // Run pipeline
        bool success = pipeline->run();
        
        if (success) {
            std::cout << "\nTraining complete. Model ready for deployment." << std::endl;
            std::cout << "Run deployment manually: pipeline.run_stage(Stage::DEPLOY)" << std::endl;
        }
        
        return success ? 0 : 1;
    }
    
    // Show usage
    else {
        std::cout << "Usage:" << std::endl;
        std::cout << "  " << argv[0] << "                    # Run with defaults" << std::endl;
        std::cout << "  " << argv[0] << " <config.json>     # Run with config file" << std::endl;
        std::cout << "  " << argv[0] << " --stage <name>    # Run single stage" << std::endl;
        std::cout << "  " << argv[0] << " --custom          # Run with custom config" << std::endl;
        return 1;
    }
}

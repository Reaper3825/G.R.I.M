#pragma once

#include "collector.hpp"
#include "verifier.hpp"
#include "parser.hpp"
#include "trainer.hpp"
#include <memory>
#include <string>
#include <functional>

namespace grim {
namespace training {

/**
 * @brief Configuration for the entire self-training pipeline
 */
struct PipelineConfig {
    CollectorConfig collector;
    VerifierConfig verifier;
    ParserConfig parser;
    TrainerConfig trainer;
    DeploymentConfig deployment;
    
    bool auto_deploy = true;  // Automatically deploy if training successful
    bool stop_on_error = true;  // Stop pipeline if any stage fails
    std::string log_file = "logs/pipeline.log";
};

/**
 * @brief GRIM Self-Training Pipeline Orchestrator
 * 
 * Coordinates all 5 stages of the self-training process:
 * 1. Collect - Fetch fresh online data
 * 2. Verify - Filter and validate sources
 * 3. Parse - Structure data for training
 * 4. Train - Fine-tune field adapter
 * 5. Deploy - Integrate new adapter safely
 */
class Pipeline {
public:
    Pipeline();
    explicit Pipeline(const PipelineConfig& config);
    ~Pipeline();

    /**
     * @brief Run the complete end-to-end pipeline
     * 
     * Executes all 5 stages in sequence:
     * fetch_online_data() → verify_sources() → parse_verified_data() 
     * → train_field_adapter() → deploy_adapter()
     * 
     * @return True if all stages completed successfully
     */
    bool run();

    /**
     * @brief Run individual pipeline stage
     */
    enum class Stage {
        COLLECT,
        VERIFY,
        PARSE,
        TRAIN,
        DEPLOY
    };
    bool run_stage(Stage stage);

    /**
     * @brief Run partial pipeline from start_stage to end_stage
     */
    bool run_partial(Stage start_stage, Stage end_stage);

    /**
     * @brief Load pipeline configuration from file
     */
    bool load_config(const std::string& config_path);

    /**
     * @brief Save pipeline configuration to file
     */
    bool save_config(const std::string& config_path) const;

    /**
     * @brief Get overall pipeline statistics
     */
    struct PipelineStats {
        size_t data_collected = 0;
        size_t data_verified = 0;
        size_t examples_parsed = 0;
        size_t examples_trained = 0;
        bool deployment_successful = false;
        
        Collector::Stats collector_stats;
        Verifier::Stats verifier_stats;
        Parser::Stats parser_stats;
        Trainer::TrainingStats training_stats;
        
        std::chrono::milliseconds total_time{0};
    };
    PipelineStats get_stats() const;

    /**
     * @brief Set callback for stage completion notifications
     */
    using StageCallback = std::function<void(Stage, bool success)>;
    void set_stage_callback(StageCallback callback);

    /**
     * @brief Reset pipeline state
     */
    void reset();

private:
    /**
     * @brief Initialize all stage components
     */
    void initialize_stages();

    /**
     * @brief Log pipeline event
     */
    void log_event(const std::string& message);

    /**
     * @brief Stage execution wrapper with error handling
     */
    bool execute_stage(Stage stage, const std::function<bool()>& stage_func);

    class Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * @brief Helper function to convert stage enum to string
 */
std::string stage_to_string(Pipeline::Stage stage);

} // namespace training
} // namespace grim

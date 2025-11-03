#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace grim {
namespace training {

/**
 * @brief Training example for fine-tuning
 */
struct TrainingExample {
    std::string input_text;
    std::string target_text;
    double weight;  // Sample weight based on reliability
};

/**
 * @brief Configuration for model training
 */
struct TrainerConfig {
    std::string input_dir = "data/parsed";
    std::string model_dir = "models";
    std::string base_model_path = "models/grim_user_current.gguf";
    std::string output_adapter_path = "models/grim_user_temp.pt";
    
    // Training hyperparameters
    double learning_rate = 1e-5;
    int num_epochs = 2;
    int batch_size = 4;
    int max_steps = -1;  // -1 for full dataset
    double warmup_ratio = 0.1;
    
    // LoRA / Adapter settings
    bool use_lora = true;
    int lora_rank = 8;
    double lora_alpha = 16.0;
    std::vector<std::string> target_modules = {"q_proj", "v_proj"};
    
    // Validation
    double validation_split = 0.1;
    int eval_steps = 100;
    
    // Logging
    std::string log_dir = "logs/training";
    bool save_checkpoints = true;
    int checkpoint_steps = 500;
};

/**
 * @brief Configuration for deployment
 */
struct DeploymentConfig {
    std::string temp_adapter_path = "models/grim_user_temp.pt";
    std::string current_model_path = "models/grim_user_current.gguf";
    std::string archive_dir = "models/archive";
    std::string test_prompts_path = "data/test_prompts.json";
    
    double min_accuracy_threshold = 0.85;
    bool run_regression_tests = true;
    bool cleanup_temp_data = true;
};

/**
 * @brief Stage 4: Model Trainer
 * 
 * Fine-tunes GRIM's field/user model with new parsed data.
 * Trains adapter weights and validates performance.
 */
class Trainer {
public:
    Trainer();
    explicit Trainer(const TrainerConfig& config);
    ~Trainer();

    /**
     * @brief Main entry point: Train field adapter
     * 
     * Loads base model weights (grim_user_current.gguf), fine-tunes with
     * parsed dataset using low learning rate and 1-2 epochs, saves adapter
     * as grim_user_temp.pt, and logs training stats and validation accuracy.
     * 
     * @return True if training completed successfully
     */
    bool train_field_adapter();

    /**
     * @brief Load parsed training data
     */
    size_t load_training_data();

    /**
     * @brief Load base model for fine-tuning
     */
    bool load_base_model(const std::string& model_path);

    /**
     * @brief Run one training epoch
     */
    double train_epoch(int epoch_num);

    /**
     * @brief Evaluate model on validation set
     */
    double evaluate();

    /**
     * @brief Save adapter weights
     */
    bool save_adapter(const std::string& output_path);

    /**
     * @brief Get training statistics
     */
    struct TrainingStats {
        int total_examples = 0;
        int train_examples = 0;
        int val_examples = 0;
        double final_train_loss = 0.0;
        double final_val_loss = 0.0;
        double best_val_accuracy = 0.0;
        int total_steps = 0;
        std::vector<double> loss_history;
    };
    TrainingStats get_stats() const;

    /**
     * @brief Reset training state
     */
    void reset();

    /**
     * @brief Stage 5: Deploy trained adapter
     * 
     * Safely integrates the new adapter into the active GRIM instance.
     * Runs evaluation on test prompts, checks for regression, and if passed,
     * replaces current field model and archives previous version.
     * 
     * @param deployment_config Configuration for deployment process
     * @return True if deployment successful
     */
    bool deploy_adapter(const DeploymentConfig& deployment_config);

private:
    /**
     * @brief Load training examples from parsed data
     */
    std::vector<TrainingExample> load_examples() const;

    /**
     * @brief Split data into train/validation sets
     */
    void split_train_val();

    /**
     * @brief Run evaluation tests
     */
    struct EvalResults {
        double accuracy = 0.0;
        double perplexity = 0.0;
        std::vector<std::string> failed_prompts;
    };
    EvalResults run_evaluation_tests(const std::string& test_prompts_path) const;

    /**
     * @brief Archive current model
     */
    bool archive_current_model(const std::string& archive_dir);

    /**
     * @brief Replace current model with new adapter
     */
    bool replace_current_model(const std::string& new_model_path,
                               const std::string& current_path);

    /**
     * @brief Clean up temporary training data
     */
    void cleanup_temp_data();

    /**
     * @brief Log training progress
     */
    void log_progress(int step, double loss, double learning_rate);

    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace training
} // namespace grim

#include "trainer.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <random>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace grim {
namespace training {

class Trainer::Impl {
public:
    TrainerConfig config;
    TrainingStats stats;
    
    std::vector<TrainingExample> train_set;
    std::vector<TrainingExample> val_set;
    
    // Simulated model state (in real implementation, would be actual model)
    std::vector<std::vector<float>> adapter_weights;
    bool model_loaded = false;
    
    std::mt19937 rng{std::random_device{}()};
    
    // Simple loss calculation (cross-entropy simulation)
    double calculate_loss(const TrainingExample& example) {
        // Simulated loss - in real implementation would use actual model
        std::uniform_real_distribution<double> dist(0.1, 2.0);
        double base_loss = dist(rng);
        
        // Weight by reliability
        return base_loss / (example.weight + 0.1);
    }
    
    // Simulate gradient descent step
    void update_weights(double learning_rate, double loss) {
        // In real implementation, this would update actual model weights
        // For now, just simulate the process
        if (adapter_weights.empty()) {
            // Initialize some dummy weights
            adapter_weights.resize(10, std::vector<float>(128, 0.01f));
        }
        
        for (auto& layer : adapter_weights) {
            for (auto& weight : layer) {
                weight -= static_cast<float>(learning_rate * loss * 0.01);
            }
        }
    }
    
    std::string get_timestamp() const {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
        return ss.str();
    }
};

Trainer::Trainer() : pImpl(std::make_unique<Impl>()) {
    pImpl->config = TrainerConfig{};
}

Trainer::Trainer(const TrainerConfig& config) : pImpl(std::make_unique<Impl>()) {
    pImpl->config = config;
}

Trainer::~Trainer() = default;

bool Trainer::train_field_adapter() {
    // Create necessary directories
    fs::create_directories(pImpl->config.model_dir);
    fs::create_directories(pImpl->config.log_dir);
    
    // Step 1: Load training data
    size_t num_examples = load_training_data();
    if (num_examples == 0) {
        return false;
    }
    
    // Step 2: Load base model
    if (!load_base_model(pImpl->config.base_model_path)) {
        // If base model doesn't exist, initialize from scratch
        pImpl->model_loaded = true;
    }
    
    // Step 3: Train for specified epochs
    for (int epoch = 0; epoch < pImpl->config.num_epochs; ++epoch) {
        double epoch_loss = train_epoch(epoch);
        pImpl->stats.final_train_loss = epoch_loss;
        
        // Evaluate on validation set
        double val_accuracy = evaluate();
        pImpl->stats.final_val_loss = val_accuracy;
        
        if (val_accuracy > pImpl->stats.best_val_accuracy) {
            pImpl->stats.best_val_accuracy = val_accuracy;
            
            // Save checkpoint
            if (pImpl->config.save_checkpoints) {
                std::string checkpoint_path = pImpl->config.model_dir + 
                    "/checkpoint_epoch_" + std::to_string(epoch) + ".pt";
                save_adapter(checkpoint_path);
            }
        }
        
        // Log progress
        std::cout << "Epoch " << epoch + 1 << "/" << pImpl->config.num_epochs 
                  << " - Loss: " << epoch_loss 
                  << " - Val Accuracy: " << val_accuracy << std::endl;
    }
    
    // Step 4: Save final adapter
    bool saved = save_adapter(pImpl->config.output_adapter_path);
    
    if (saved) {
        std::cout << "Training complete. Adapter saved to: " 
                  << pImpl->config.output_adapter_path << std::endl;
        std::cout << "Best validation accuracy: " 
                  << pImpl->stats.best_val_accuracy << std::endl;
    }
    
    return saved;
}

size_t Trainer::load_training_data() {
    auto examples = load_examples();
    
    if (examples.empty()) {
        return 0;
    }
    
    pImpl->stats.total_examples = examples.size();
    
    // Shuffle examples
    std::shuffle(examples.begin(), examples.end(), pImpl->rng);
    
    // Split into train/val
    size_t val_size = static_cast<size_t>(
        examples.size() * pImpl->config.validation_split);
    
    pImpl->val_set.assign(examples.begin(), examples.begin() + val_size);
    pImpl->train_set.assign(examples.begin() + val_size, examples.end());
    
    pImpl->stats.train_examples = pImpl->train_set.size();
    pImpl->stats.val_examples = pImpl->val_set.size();
    
    return pImpl->stats.total_examples;
}

bool Trainer::load_base_model(const std::string& model_path) {
    if (!fs::exists(model_path)) {
        std::cout << "Base model not found at: " << model_path << std::endl;
        std::cout << "Initializing new model from scratch..." << std::endl;
        return false;
    }
    
    // In real implementation, load GGUF model here
    // For now, simulate loading
    pImpl->model_loaded = true;
    
    std::cout << "Base model loaded from: " << model_path << std::endl;
    
    return true;
}

double Trainer::train_epoch(int epoch_num) {
    double total_loss = 0.0;
    int step = 0;
    
    // Calculate steps per epoch
    int steps_per_epoch = (pImpl->train_set.size() + pImpl->config.batch_size - 1) 
                         / pImpl->config.batch_size;
    
    for (size_t i = 0; i < pImpl->train_set.size(); i += pImpl->config.batch_size) {
        double batch_loss = 0.0;
        int batch_samples = 0;
        
        // Process batch
        for (size_t j = i; j < std::min(i + pImpl->config.batch_size, 
                                        pImpl->train_set.size()); ++j) {
            double loss = pImpl->calculate_loss(pImpl->train_set[j]);
            batch_loss += loss;
            batch_samples++;
        }
        
        batch_loss /= batch_samples;
        total_loss += batch_loss;
        
        // Calculate learning rate with warmup
        int total_step = epoch_num * steps_per_epoch + step;
        int warmup_steps = static_cast<int>(
            steps_per_epoch * pImpl->config.num_epochs * pImpl->config.warmup_ratio);
        
        double lr = pImpl->config.learning_rate;
        if (total_step < warmup_steps) {
            lr = pImpl->config.learning_rate * 
                 (static_cast<double>(total_step) / warmup_steps);
        }
        
        // Update weights
        pImpl->update_weights(lr, batch_loss);
        
        // Log progress
        if (step % 10 == 0) {
            log_progress(total_step, batch_loss, lr);
        }
        
        // Track loss history
        pImpl->stats.loss_history.push_back(batch_loss);
        pImpl->stats.total_steps++;
        
        step++;
        
        // Early stopping if max_steps specified
        if (pImpl->config.max_steps > 0 && total_step >= pImpl->config.max_steps) {
            break;
        }
    }
    
    return total_loss / step;
}

double Trainer::evaluate() {
    if (pImpl->val_set.empty()) {
        return 0.0;
    }
    
    double total_accuracy = 0.0;
    
    for (const auto& example : pImpl->val_set) {
        // Simulate evaluation - in real implementation, run inference
        std::uniform_real_distribution<double> dist(0.7, 1.0);
        double accuracy = dist(pImpl->rng) * example.weight;
        total_accuracy += accuracy;
    }
    
    return total_accuracy / pImpl->val_set.size();
}

bool Trainer::save_adapter(const std::string& output_path) {
    try {
        fs::create_directories(fs::path(output_path).parent_path());
        
        // Save adapter weights
        std::ofstream outfile(output_path, std::ios::binary);
        if (!outfile.is_open()) return false;
        
        // Write metadata
        json metadata;
        metadata["config"] = {
            {"learning_rate", pImpl->config.learning_rate},
            {"num_epochs", pImpl->config.num_epochs},
            {"batch_size", pImpl->config.batch_size},
            {"lora_rank", pImpl->config.lora_rank},
            {"lora_alpha", pImpl->config.lora_alpha}
        };
        metadata["stats"] = {
            {"total_examples", pImpl->stats.total_examples},
            {"train_examples", pImpl->stats.train_examples},
            {"val_examples", pImpl->stats.val_examples},
            {"final_train_loss", pImpl->stats.final_train_loss},
            {"best_val_accuracy", pImpl->stats.best_val_accuracy}
        };
        
        std::string metadata_str = metadata.dump();
        size_t metadata_size = metadata_str.size();
        
        outfile.write(reinterpret_cast<const char*>(&metadata_size), sizeof(metadata_size));
        outfile.write(metadata_str.c_str(), metadata_size);
        
        // Write weights (simulated)
        for (const auto& layer : pImpl->adapter_weights) {
            size_t layer_size = layer.size();
            outfile.write(reinterpret_cast<const char*>(&layer_size), sizeof(layer_size));
            outfile.write(reinterpret_cast<const char*>(layer.data()), 
                         layer_size * sizeof(float));
        }
        
        outfile.close();
        return true;
        
    } catch (...) {
        return false;
    }
}

Trainer::TrainingStats Trainer::get_stats() const {
    return pImpl->stats;
}

void Trainer::reset() {
    pImpl->stats = TrainingStats{};
    pImpl->train_set.clear();
    pImpl->val_set.clear();
    pImpl->adapter_weights.clear();
    pImpl->model_loaded = false;
}

bool Trainer::deploy_adapter(const DeploymentConfig& deployment_config) {
    std::cout << "\n=== Starting Deployment Process ===" << std::endl;
    
    // Step 1: Verify adapter exists
    if (!fs::exists(deployment_config.temp_adapter_path)) {
        std::cerr << "Error: Adapter not found at " 
                  << deployment_config.temp_adapter_path << std::endl;
        return false;
    }
    
    // Step 2: Run evaluation tests
    if (deployment_config.run_regression_tests) {
        std::cout << "Running regression tests..." << std::endl;
        
        auto eval_results = run_evaluation_tests(deployment_config.test_prompts_path);
        
        std::cout << "Evaluation results:" << std::endl;
        std::cout << "  Accuracy: " << eval_results.accuracy << std::endl;
        std::cout << "  Perplexity: " << eval_results.perplexity << std::endl;
        
        if (eval_results.accuracy < deployment_config.min_accuracy_threshold) {
            std::cerr << "Error: Accuracy " << eval_results.accuracy 
                      << " below threshold " << deployment_config.min_accuracy_threshold 
                      << std::endl;
            
            if (!eval_results.failed_prompts.empty()) {
                std::cerr << "Failed prompts:" << std::endl;
                for (const auto& prompt : eval_results.failed_prompts) {
                    std::cerr << "  - " << prompt << std::endl;
                }
            }
            
            return false;
        }
        
        std::cout << "✓ Regression tests passed" << std::endl;
    }
    
    // Step 3: Archive current model
    std::cout << "Archiving current model..." << std::endl;
    
    if (fs::exists(deployment_config.current_model_path)) {
        if (!archive_current_model(deployment_config.archive_dir)) {
            std::cerr << "Warning: Failed to archive current model" << std::endl;
        } else {
            std::cout << "✓ Current model archived" << std::endl;
        }
    }
    
    // Step 4: Replace with new adapter
    std::cout << "Deploying new adapter..." << std::endl;
    
    if (!replace_current_model(deployment_config.temp_adapter_path,
                               deployment_config.current_model_path)) {
        std::cerr << "Error: Failed to deploy new adapter" << std::endl;
        return false;
    }
    
    std::cout << "✓ New adapter deployed successfully" << std::endl;
    
    // Step 5: Cleanup temporary data
    if (deployment_config.cleanup_temp_data) {
        std::cout << "Cleaning up temporary data..." << std::endl;
        cleanup_temp_data();
        std::cout << "✓ Cleanup complete" << std::endl;
    }
    
    std::cout << "\n=== Deployment Complete ===" << std::endl;
    std::cout << "New model active at: " << deployment_config.current_model_path << std::endl;
    
    return true;
}

std::vector<TrainingExample> Trainer::load_examples() const {
    std::vector<TrainingExample> examples;
    
    std::string filepath = pImpl->config.input_dir + "/parsed.jsonl";
    
    if (!fs::exists(filepath)) {
        return examples;
    }
    
    std::ifstream file(filepath);
    if (!file.is_open()) return examples;
    
    std::string line;
    while (std::getline(file, line)) {
        try {
            json j = json::parse(line);
            
            TrainingExample example;
            example.input_text = j.value("input_text", "");
            example.target_text = j.value("target_text", "");
            example.weight = j.value("reliability_score", 1.0);
            
            if (!example.input_text.empty() && !example.target_text.empty()) {
                examples.push_back(example);
            }
            
        } catch (const json::parse_error&) {
            continue;
        }
    }
    
    return examples;
}

void Trainer::split_train_val() {
    // Already handled in load_training_data()
}

Trainer::EvalResults Trainer::run_evaluation_tests(const std::string& test_prompts_path) const {
    EvalResults results;
    
    // Load test prompts
    std::vector<std::pair<std::string, std::string>> test_cases;
    
    if (fs::exists(test_prompts_path)) {
        std::ifstream file(test_prompts_path);
        json j;
        file >> j;
        
        if (j.contains("test_cases") && j["test_cases"].is_array()) {
            for (const auto& test_case : j["test_cases"]) {
                std::string prompt = test_case.value("prompt", "");
                std::string expected = test_case.value("expected", "");
                test_cases.push_back({prompt, expected});
            }
        }
    }
    
    if (test_cases.empty()) {
        // Create default test cases
        test_cases = {
            {"What is GRIM?", "GRIM is an AI assistant"},
            {"How can I help you?", "I am here to assist you"},
            {"Test prompt", "Test response"}
        };
    }
    
    // Run inference on test cases (simulated)
    int passed = 0;
    for (const auto& [prompt, expected] : test_cases) {
        // Simulate inference
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double score = dist(pImpl->rng);
        
        if (score > 0.7) {
            passed++;
        } else {
            results.failed_prompts.push_back(prompt);
        }
    }
    
    results.accuracy = static_cast<double>(passed) / test_cases.size();
    results.perplexity = 10.0 / (results.accuracy + 0.1);  // Simulated
    
    return results;
}

bool Trainer::archive_current_model(const std::string& archive_dir) {
    try {
        fs::create_directories(archive_dir);
        
        std::string timestamp = pImpl->get_timestamp();
        std::string archive_path = archive_dir + "/grim_user_" + timestamp + ".gguf";
        
        fs::copy_file(pImpl->config.base_model_path, archive_path,
                     fs::copy_options::overwrite_existing);
        
        return true;
        
    } catch (const fs::filesystem_error&) {
        return false;
    }
}

bool Trainer::replace_current_model(const std::string& new_model_path,
                                    const std::string& current_path) {
    try {
        fs::create_directories(fs::path(current_path).parent_path());
        
        fs::copy_file(new_model_path, current_path,
                     fs::copy_options::overwrite_existing);
        
        return true;
        
    } catch (const fs::filesystem_error&) {
        return false;
    }
}

void Trainer::cleanup_temp_data() {
    // Remove temporary directories
    std::vector<std::string> temp_dirs = {
        "data/raw",
        "data/verified",
        "data/parsed"
    };
    
    for (const auto& dir : temp_dirs) {
        try {
            if (fs::exists(dir)) {
                fs::remove_all(dir);
            }
        } catch (...) {
            // Continue cleanup even if one fails
        }
    }
}

void Trainer::log_progress(int step, double loss, double learning_rate) {
    // Console logging
    if (step % 100 == 0) {
        std::cout << "Step " << step 
                  << " - Loss: " << std::fixed << std::setprecision(4) << loss
                  << " - LR: " << std::scientific << learning_rate << std::endl;
    }
    
    // File logging
    std::string log_file = pImpl->config.log_dir + "/training.log";
    std::ofstream logfile(log_file, std::ios::app);
    
    if (logfile.is_open()) {
        logfile << pImpl->get_timestamp() << " - Step: " << step 
                << " - Loss: " << loss 
                << " - LR: " << learning_rate << "\n";
    }
}

} // namespace training
} // namespace grim

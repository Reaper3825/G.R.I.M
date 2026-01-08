/**
 * @file GradAccumulationController_Integration.hpp
 * @brief Integration with LanguageModel training loop
 *
 * This file provides ModelGradAccumulationController which binds
 * to a LanguageModel and registers all gradient buffers automatically.
 *
 * USAGE:
 * ─────────────────────────────────────────────────────────────────
 *   LanguageModel model;
 *   model.initialize(config);
 *
 *   ModelGradAccumulationController controller;
 *   controller.bindToModel(model);
 *   controller.configure(accum_steps, stream);
 *
 *   for (batch : batches) {
 *       controller.beginAccumulationWindow();
 *
 *       for (int micro = 0; micro < accum_steps; ++micro) {
 *           float scale = controller.getScaleFactor();
 *           
 *           controller.beginBackward();
 *           float scaled_loss = loss * scale;
 *           model.backward(scaled_loss);
 *           controller.endBackward();
 *       }
 *
 *       controller.beginOptimizerStep();
 *       clipGradNorm(model);
 *       optimizer.step();
 *       controller.endOptimizerStep();
 *   }
 * ─────────────────────────────────────────────────────────────────
 */

#pragma once

#include "GradAccumulationController_GPU.hpp"
#include "../LogRecorder/LogRecorder.hpp"
#include <iostream>
#include <sstream>
#include <string>

// Forward declarations - avoid pulling in heavy headers
namespace GRIM {
    class LanguageModel;
    class GPUGrimEncoder;
    class EncodingLayer;
}

namespace GRIM {

//======================================================//
//  ModelGradAccumulationController
//======================================================//

class ModelGradAccumulationController {
public:
    ModelGradAccumulationController() = default;
    
    /**
     * @brief Bind to a LanguageModel and register all gradient buffers
     * 
     * This function automatically discovers and registers all gradient
     * buffers from the model's TrainingState. Must be called after
     * model initialization but before training.
     *
     * @param model Initialized LanguageModel with allocated gradients
     */
    void bindToModel(LanguageModel& model);
    
    // ========== Configuration ==========
    
    /// Configure accumulation steps and stream
    void configure(int accum_steps, cudaStream_t stream = nullptr) {
        controller_.configure(accum_steps, stream);
    }
    
    /// Configure with full config
    void configure(const GradAccumulationConfig& config) {
        controller_.configure(config);
    }
    
    /// Enable verbose logging for debugging
    void setVerbose(bool verbose) {
        GradAccumulationConfig cfg = controller_.config();
        cfg.verbose = verbose;
        controller_.configure(cfg);
    }
    
    /// Enable gradient monitoring (expensive, for debugging)
    void setMonitorGradients(bool monitor, float explosion_threshold = 1e6f) {
        GradAccumulationConfig cfg = controller_.config();
        cfg.monitor_gradients = monitor;
        cfg.gradient_explosion_threshold = explosion_threshold;
        controller_.configure(cfg);
    }
    
    // ========== Traffic Light API ==========
    
    /// Get loss scale factor (1.0 / accum_steps)
    float getScaleFactor() const { return controller_.getScaleFactor(); }
    
    /// Start accumulation window (zeros gradients, resets micro-step)
    bool beginAccumulationWindow() { return controller_.beginAccumulationWindow(); }
    
    /// Called before backward pass
    bool beginBackward() { return controller_.beginBackward(); }
    
    /// Called after backward pass
    bool endBackward() { return controller_.endBackward(); }
    
    /// Called before optimizer.step()
    bool beginOptimizerStep() { return controller_.beginOptimizerStep(); }
    
    /// Called after optimizer.step()
    bool endOptimizerStep() { return controller_.endOptimizerStep(); }
    
    // ========== State Queries ==========
    
    GradControllerState state() const { return controller_.state(); }
    const char* stateString() const { return controller_.stateString(); }
    int currentMicroStep() const { return controller_.currentMicroStep(); }
    int accumSteps() const { return controller_.accumSteps(); }
    bool isAccumulationComplete() const { return controller_.isAccumulationComplete(); }
    bool isAccumulating() const { return controller_.isAccumulating(); }
    bool isReadyForStep() const { return controller_.isReadyForStep(); }
    
    // ========== Manual Control ==========
    
    void zeroAllGradients() { controller_.zeroAllGradients(); }
    void forceReset() { controller_.forceReset(); }
    
    // ========== Gradient Monitoring ==========
    
    /// Compute RMS of all gradient buffers (expensive)
    std::vector<std::pair<std::string, float>> computeGradientRMS() const {
        return controller_.computeGradientRMS();
    }
    
    /// Check for gradient explosion
    bool checkGradientExplosion() const { return controller_.checkGradientExplosion(); }
    
    /// Get buffer with highest RMS
    std::string getHottestBuffer() const { return controller_.getHottestBuffer(); }
    
    /// Print gradient RMS for all buffers (debugging)
    void printGradientRMS() const {
        auto rms = computeGradientRMS();
        std::ostringstream oss;
        oss << "\n[GradController] Gradient RMS:\n"
            << "─────────────────────────────────────────────────────\n";
        for (const auto& [name, value] : rms) {
            oss << "  " << name << ": " << value << "\n";
        }
        oss << "─────────────────────────────────────────────────────";
        GRIM::Logging::EmitModuleInfo(GRIM::Logging::ModuleId::Optimizer, oss.str());
    }
    
    // ========== Statistics & Debugging ==========
    
    const GradAccumulationStats& stats() const { return controller_.stats(); }
    void resetStats() { controller_.resetStats(); }
    void printState() const { controller_.printState(); }
    void printBuffers() const { controller_.printBuffers(); }
    bool validate() const { return controller_.validate(); }
    const std::string& lastError() const { return controller_.lastError(); }
    
    // ========== Access Underlying Controller ==========
    
    GradAccumulationController& controller() { return controller_; }
    const GradAccumulationController& controller() const { return controller_; }
    
private:
    GradAccumulationController controller_;
    LanguageModel* model_ = nullptr;
};

} // namespace GRIM

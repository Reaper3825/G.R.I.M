#pragma once
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <chrono>

namespace GRIM {

// Task step represents a single action in a multi-step task
struct TaskStep {
    enum class Type {
        Navigate,      // Open app/URL
        Click,         // Click on element
        Type,          // Type text
        Key,           // Press key/combo
        Wait,          // Wait for condition
        Verify,        // Check if something exists
        Search,        // Search for information
        Think          // AI reasoning step
    };
    
    Type type;
    std::string target;           // What to interact with
    std::string value;            // Text to type, URL, etc.
    std::string condition;        // Wait/verify condition
    float confidence = 1.0f;      // AI confidence (0-1)
    bool completed = false;
    std::string result;           // Execution result
};

// Multi-step task with execution state
struct Task {
    std::string goal;                           // Original user request
    std::string context;                        // Additional context
    std::vector<TaskStep> steps;                // Execution plan
    size_t currentStep = 0;                     // Current execution index
    bool completed = false;
    bool userConfirmed = false;                 // User approved plan
    std::chrono::system_clock::time_point startTime;
    
    float getProgress() const {
        if (steps.empty()) return 0.0f;
        return static_cast<float>(currentStep) / steps.size();
    }
    
    TaskStep* getCurrentStep() {
        if (currentStep >= steps.size()) return nullptr;
        return &steps[currentStep];
    }
};

// Intelligent task planning and execution
class TaskPlanner {
public:
    // Initialize task planner
    static void init();
    static void shutdown();
    
    // Check if input requires task decomposition (vs simple action)
    static bool isComplexTask(const std::string& input);
    
    // Create execution plan from user goal
    static std::shared_ptr<Task> planTask(const std::string& userGoal, const std::string& context = "");
    
    // Execute task step-by-step with perception feedback
    static bool executeTask(std::shared_ptr<Task> task,
                           std::function<void(const std::string&)> progressCallback = nullptr);
    
    // Execute single step
    static bool executeStep(TaskStep& step);
    
    // Adapt plan based on current screen state
    static void adaptPlan(std::shared_ptr<Task> task, const std::string& observation);
    
    // Get current active task (if any)
    static std::shared_ptr<Task> getCurrentTask();
    
private:
    static std::shared_ptr<Task> s_currentTask;
    
    // AI-powered task decomposition
    static std::vector<TaskStep> decomposeTask(const std::string& goal, const std::string& context);
    
    // Generate AI prompt for task planning
    static std::string buildPlanningPrompt(const std::string& goal, const std::string& context);
    
    // Parse AI response into task steps
    static std::vector<TaskStep> parseAIPlan(const std::string& aiResponse);
    
    // Verify step completion using perception
    static bool verifyStepCompletion(const TaskStep& step);
};

} // namespace GRIM

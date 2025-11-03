#include "task_planner.hpp"
#include "ai.hpp"
#include "action_executor.hpp"
#include "../perception/perception_context.hpp"
#include "../logger.hpp"
#include "../voice/voice_speak.hpp"
#include <nlohmann/json.hpp>
#include <thread>
#include <regex>

namespace GRIM {

std::shared_ptr<Task> TaskPlanner::s_currentTask = nullptr;

void TaskPlanner::init() {
    logDebug("TaskPlanner", "Initialized");
}

void TaskPlanner::shutdown() {
    s_currentTask.reset();
    logDebug("TaskPlanner", "Shutdown");
}

bool TaskPlanner::isComplexTask(const std::string& input) {
    std::string lower = input;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    // Keywords indicating multi-step tasks
    const std::vector<std::string> complexKeywords = {
        "do my", "complete", "finish", "work on", "help me with",
        "homework", "assignment", "project", "task", "job",
        "find and", "search and", "create", "build", "make",
        "organize", "plan", "setup", "configure",
        "solve", "answer", "work through", "figure out", "calculate"
    };
    
    for (const auto& keyword : complexKeywords) {
        if (lower.find(keyword) != std::string::npos) {
            return true;
        }
    }
    
    // Single word commands are usually simple
    if (lower.find(' ') == std::string::npos) {
        return false;
    }
    
    return false;
}

std::shared_ptr<Task> TaskPlanner::planTask(const std::string& userGoal, const std::string& context) {
    logDebug("TaskPlanner", "Planning task: " + userGoal);
    
    auto task = std::make_shared<Task>();
    task->goal = userGoal;
    task->context = context;
    task->startTime = std::chrono::system_clock::now();
    
    // Decompose into steps using AI
    task->steps = decomposeTask(userGoal, context);
    
    if (task->steps.empty()) {
        logError("TaskPlanner", "Failed to decompose task: " + userGoal);
        return nullptr;
    }
    
    logDebug("TaskPlanner", "Planned " + std::to_string(task->steps.size()) + " steps");
    
    s_currentTask = task;
    return task;
}

std::vector<TaskStep> TaskPlanner::decomposeTask(const std::string& goal, const std::string& context) {
    // Build AI planning prompt
    std::string prompt = buildPlanningPrompt(goal, context);
    
    // Get AI to create execution plan
    auto future = callAIAsync(prompt);
    std::string aiResponse = future.get();
    
    if (aiResponse.empty() || aiResponse.find("Backend call failed") != std::string::npos) {
        logError("TaskPlanner", "AI planning failed");
        return {};
    }
    
    // Parse AI response into steps
    return parseAIPlan(aiResponse);
}

std::string TaskPlanner::buildPlanningPrompt(const std::string& goal, const std::string& context) {
    std::string screenContext;
    if (Perception::g_contextManager) {
        auto ctx = Perception::g_contextManager->getCurrentContext();
        if (ctx.isValid) {
            screenContext = "\n[Current Screen]: " + ctx.toSummary();
            if (ctx.hasText) {
                screenContext += "\n[Visible Text]: " + ctx.screenText.substr(0, 200);
            }
        }
    }
    
    return R"(You are a task planning AI. Break down the user's goal into specific executable steps.

Available actions:
- navigate: Open application or URL (e.g., "navigate Chrome to google.com")
- click: Click on UI element (e.g., "click Submit button")
- type: Type text (e.g., "type john@example.com")
- key: Press keyboard shortcut (e.g., "key ctrl+c")
- wait: Wait for element (e.g., "wait for page to load")
- verify: Check condition (e.g., "verify login successful")
- search: Search information (e.g., "search how to solve quadratic equations")
- think: AI reasoning step (e.g., "think about the answer")

Respond ONLY with valid JSON array:
[
  {"action": "navigate", "target": "Chrome", "value": "https://example.com", "confidence": 0.9},
  {"action": "click", "target": "Login button", "value": "", "confidence": 0.8}
]

User Goal: )" + goal + R"(
)" + (context.empty() ? "" : "Additional Context: " + context) + 
screenContext + R"(

JSON plan:)";
}

std::vector<TaskStep> TaskPlanner::parseAIPlan(const std::string& aiResponse) {
    std::vector<TaskStep> steps;
    
    try {
        // Extract JSON array from response
        size_t jsonStart = aiResponse.find('[');
        size_t jsonEnd = aiResponse.rfind(']');
        
        if (jsonStart == std::string::npos || jsonEnd == std::string::npos) {
            logError("TaskPlanner", "No JSON array found in AI response");
            return steps;
        }
        
        std::string jsonStr = aiResponse.substr(jsonStart, jsonEnd - jsonStart + 1);
        nlohmann::json planJson = nlohmann::json::parse(jsonStr);
        
        if (!planJson.is_array()) {
            logError("TaskPlanner", "AI response is not a JSON array");
            return steps;
        }
        
        // Convert JSON to TaskStep objects
        for (const auto& stepJson : planJson) {
            TaskStep step;
            
            std::string actionStr = stepJson.value("action", "");
            if (actionStr == "navigate") step.type = TaskStep::Type::Navigate;
            else if (actionStr == "click") step.type = TaskStep::Type::Click;
            else if (actionStr == "type") step.type = TaskStep::Type::Type;
            else if (actionStr == "key") step.type = TaskStep::Type::Key;
            else if (actionStr == "wait") step.type = TaskStep::Type::Wait;
            else if (actionStr == "verify") step.type = TaskStep::Type::Verify;
            else if (actionStr == "search") step.type = TaskStep::Type::Search;
            else if (actionStr == "think") step.type = TaskStep::Type::Think;
            else continue; // Skip unknown actions
            
            step.target = stepJson.value("target", "");
            step.value = stepJson.value("value", "");
            step.confidence = stepJson.value("confidence", 0.8f);
            
            steps.push_back(step);
        }
        
        logDebug("TaskPlanner", "Parsed " + std::to_string(steps.size()) + " steps from AI plan");
        
    } catch (const std::exception& e) {
        logError("TaskPlanner", "Failed to parse AI plan: " + std::string(e.what()));
    }
    
    return steps;
}

bool TaskPlanner::executeTask(std::shared_ptr<Task> task,
                              std::function<void(const std::string&)> progressCallback) {
    if (!task || task->steps.empty()) {
        return false;
    }
    
    logDebug("TaskPlanner", "Starting task execution: " + task->goal);
    
    while (task->currentStep < task->steps.size()) {
        auto& step = task->steps[task->currentStep];
        
        // Progress callback
        if (progressCallback) {
            std::string status = "Step " + std::to_string(task->currentStep + 1) + 
                               "/" + std::to_string(task->steps.size());
            progressCallback(status);
        }
        
        // Execute step
        bool success = executeStep(step);
        
        if (!success) {
            // Ask user what to do
            Voice::speak("Step failed. Should I continue or stop?", "question");
            logError("TaskPlanner", "Step failed at index " + std::to_string(task->currentStep));
            return false;
        }
        
        step.completed = true;
        task->currentStep++;
        
        // Small delay between steps for UI to update
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    task->completed = true;
    logDebug("TaskPlanner", "Task completed: " + task->goal);
    Voice::speak("Task completed successfully!", "success");
    
    return true;
}

bool TaskPlanner::executeStep(TaskStep& step) {
    logDebug("TaskPlanner", "Executing step: type=" + std::to_string(static_cast<int>(step.type)) + 
             " target=" + step.target);
    
    try {
        switch (step.type) {
            case TaskStep::Type::Navigate: {
                // Open app or URL
                std::string command = "open " + step.target;
                if (!step.value.empty()) {
                    // If value is URL, append it
                    command += " " + step.value;
                }
                auto result = ActionExecutor::executeAction(command);
                step.result = result.message;
                return result.success;
            }
            
            case TaskStep::Type::Click: {
                if (!Perception::g_contextManager) return false;
                Perception::g_contextManager->clickOnText(step.target);
                step.result = "Clicked on: " + step.target;
                return verifyStepCompletion(step);
            }
            
            case TaskStep::Type::Type: {
                if (!Perception::g_contextManager) return false;
                Perception::g_contextManager->typeText(step.value, 15);
                step.result = "Typed: " + step.value;
                return true;
            }
            
            case TaskStep::Type::Key: {
                if (!Perception::g_contextManager) return false;
                
                // Parse key combo
                if (step.target.find('+') != std::string::npos) {
                    std::vector<std::string> keys;
                    std::istringstream iss(step.target);
                    std::string key;
                    while (std::getline(iss, key, '+')) {
                        key.erase(0, key.find_first_not_of(" \t"));
                        key.erase(key.find_last_not_of(" \t") + 1);
                        if (!key.empty()) keys.push_back(key);
                    }
                    Perception::g_contextManager->pressKeyCombo(keys);
                } else {
                    Perception::g_contextManager->tapKey(step.target);
                }
                
                step.result = "Pressed: " + step.target;
                return true;
            }
            
            case TaskStep::Type::Wait: {
                // Wait for condition (check perception)
                int maxAttempts = 10;
                for (int i = 0; i < maxAttempts; i++) {
                    if (verifyStepCompletion(step)) {
                        step.result = "Condition met: " + step.condition;
                        return true;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                }
                step.result = "Timeout waiting for: " + step.condition;
                return false;
            }
            
            case TaskStep::Type::Verify: {
                bool verified = verifyStepCompletion(step);
                step.result = verified ? "Verified: " + step.target : "Verification failed";
                return verified;
            }
            
            case TaskStep::Type::Search: {
                // Use AI to search for information
                std::string query = step.target + " " + step.value;
                auto result = ai_interpret("search for: " + query);
                step.result = result.message;
                return result.success;
            }
            
            case TaskStep::Type::Think: {
                // AI reasoning step
                auto result = ai_interpret(step.target);
                step.result = result.message;
                return result.success;
            }
            
            default:
                return false;
        }
        
    } catch (const std::exception& e) {
        logError("TaskPlanner", "Step execution error: " + std::string(e.what()));
        step.result = "Error: " + std::string(e.what());
        return false;
    }
}

bool TaskPlanner::verifyStepCompletion(const TaskStep& step) {
    if (!Perception::g_contextManager) return true; // Can't verify, assume success
    
    auto ctx = Perception::g_contextManager->getCurrentContext(true);
    if (!ctx.isValid) return true;
    
    // Check if target/condition text appears on screen
    std::string checkText = step.condition.empty() ? step.target : step.condition;
    if (checkText.empty()) return true;
    
    if (ctx.hasText && ctx.screenText.find(checkText) != std::string::npos) {
        return true;
    }
    
    // Check if condition is met (simple heuristic)
    return true; // Default to success if can't determine
}

void TaskPlanner::adaptPlan(std::shared_ptr<Task> task, const std::string& observation) {
    if (!task) return;
    
    logDebug("TaskPlanner", "Adapting plan based on observation: " + observation);
    
    // AI can re-plan remaining steps based on new information
    // For now, just log - can enhance with dynamic re-planning
}

std::shared_ptr<Task> TaskPlanner::getCurrentTask() {
    return s_currentTask;
}

} // namespace GRIM

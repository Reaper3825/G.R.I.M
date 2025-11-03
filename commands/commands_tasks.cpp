#include "commands_tasks.hpp"
#include "ai/task_planner.hpp"
#include "voice/voice_speak.hpp"
#include "logger.hpp"

CommandResult cmdExecuteTask(const std::string& arg) {
    logDebug("Commands", "cmdExecuteTask: " + arg);
    
    auto task = GRIM::TaskPlanner::getCurrentTask();
    
    if (!task) {
        return {false, "[Task] No task planned. Plan a task first.", "ERR_NO_TASK",
                "error", "No task to execute", Colors::Red};
    }
    
    if (task->completed) {
        return {false, "[Task] Current task already completed.", "ERR_TASK_COMPLETE",
                "error", "Task already done", Colors::Yellow};
    }
    
    // Execute with progress callback
    Voice::speak("Starting task execution...", "routine");
    
    bool success = GRIM::TaskPlanner::executeTask(task, [](const std::string& status) {
        logDebug("TaskExecution", status);
    });
    
    if (success) {
        return {true, "[Task] Task completed successfully!", "ERR_NONE",
                "success", "Task completed", Colors::Green};
    } else {
        return {false, "[Task] Task execution failed or incomplete.", "ERR_TASK_FAILED",
                "error", "Task failed", Colors::Red};
    }
}

CommandResult cmdTaskStatus(const std::string& /*arg*/) {
    auto task = GRIM::TaskPlanner::getCurrentTask();
    
    if (!task) {
        return {true, "[Task] No active task.", "ERR_NONE",
                "routine", "No active task", Colors::Cyan};
    }
    
    std::string status = "[Task] \"" + task->goal + "\" - " +
                        std::to_string(task->currentStep) + "/" +
                        std::to_string(task->steps.size()) + " steps (" +
                        std::to_string(static_cast<int>(task->getProgress() * 100)) + "%)";
    
    return {true, status, "ERR_NONE",
            "routine", "Task status", Colors::Cyan};
}

CommandResult cmdTaskCancel(const std::string& /*arg*/) {
    auto task = GRIM::TaskPlanner::getCurrentTask();
    
    if (!task) {
        return {false, "[Task] No active task to cancel.", "ERR_NO_TASK",
                "error", "Nothing to cancel", Colors::Red};
    }
    
    // Reset current task
    task.reset();
    
    return {true, "[Task] Task cancelled.", "ERR_NONE",
            "routine", "Task cancelled", Colors::Yellow};
}

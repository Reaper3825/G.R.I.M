#pragma once

// Export macro for GRIM host symbols that plugins need
#if defined(_WIN32)
  #if defined(GRIM_BUILD_HOST)
    #define GRIM_EXPORT_SYMBOL __declspec(dllexport)
  #else
    #define GRIM_EXPORT_SYMBOL __declspec(dllimport)
  #endif
#else
  #define GRIM_EXPORT_SYMBOL __attribute__((visibility("default")))
#endif

// Forward declarations with exports
class NLP;
namespace GRIM {
    class UnifiedMemoryStorage;
}
struct LocationInfo;
struct Intent;
struct Timer;
class ConsoleHistory;
namespace GRIM::MMO { struct HardwareInventory; }

namespace nlohmann {
    template<typename... Args>
    class basic_json;
    using json = basic_json<>;
}

// Exported global variables
extern "C" {
    GRIM_EXPORT_SYMBOL NLP g_nlp;
    GRIM_EXPORT_SYMBOL GRIM::UnifiedMemoryStorage g_memoryStorage;
    GRIM_EXPORT_SYMBOL GRIM::MMO::HardwareInventory g_hardwareInventory;
    GRIM_EXPORT_SYMBOL LocationInfo g_location;
    GRIM_EXPORT_SYMBOL Intent g_lastIntent;
    GRIM_EXPORT_SYMBOL std::filesystem::path g_currentDir;
    GRIM_EXPORT_SYMBOL std::vector<Timer> timers;
    GRIM_EXPORT_SYMBOL nlohmann::json aiConfig;
    GRIM_EXPORT_SYMBOL nlohmann::json longTermMemory;
    GRIM_EXPORT_SYMBOL ConsoleHistory history;
    GRIM_EXPORT_SYMBOL bool g_wifiConnected;
}

// Exported functions
extern "C" {
    GRIM_EXPORT_SYMBOL std::string getResourcePath();
    GRIM_EXPORT_SYMBOL CommandResult reloadNlpRules();
    GRIM_EXPORT_SYMBOL CommandResult ai_process(const std::string& input);
    GRIM_EXPORT_SYMBOL void logPhaseInternal(const std::string& file, const std::string& phase, bool success);
}

// Namespace exports
namespace aliases {
    GRIM_EXPORT_SYMBOL CommandResult refreshNow();
    GRIM_EXPORT_SYMBOL std::unordered_map<std::string, std::string> getAll();
    GRIM_EXPORT_SYMBOL std::string info(const std::string& name);
}

namespace Voice {
    GRIM_EXPORT_SYMBOL std::string runVoiceDemo(nlohmann::json& aiConfig, nlohmann::json& longTermMemory);
    GRIM_EXPORT_SYMBOL void playAudio(const std::string& file);
    GRIM_EXPORT_SYMBOL bool isSpeaking();
    GRIM_EXPORT_SYMBOL std::string coquiSpeak(const std::string& text, const std::string& model, double speed, double pitch = 1.0);
    
    struct State;
    extern GRIM_EXPORT_SYMBOL State g_state;
}

namespace VoiceStream {
    struct whisper_context;
    GRIM_EXPORT_SYMBOL bool start(whisper_context* ctx, ConsoleHistory* history, 
                                  std::vector<Timer>& timers, nlohmann::json& cfg, NLP& nlp);
}

namespace Audio {
    GRIM_EXPORT_SYMBOL bool playWav(const std::string& file);
}

class ErrorManager {
public:
    static GRIM_EXPORT_SYMBOL std::string getUserMessage(const std::string& code);
    static GRIM_EXPORT_SYMBOL CommandResult report(const std::string& code);
};

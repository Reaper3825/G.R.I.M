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
    GRIM_EXPORT_SYMBOL std::filesystem::path g_currentDir;
    GRIM_EXPORT_SYMBOL nlohmann::json aiConfig;
    GRIM_EXPORT_SYMBOL nlohmann::json longTermMemory;
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







class ErrorManager {
public:
    static GRIM_EXPORT_SYMBOL std::string getUserMessage(const std::string& code);
    static GRIM_EXPORT_SYMBOL CommandResult report(const std::string& code);
};

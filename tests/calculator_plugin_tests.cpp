#include "core/plugin_api.hpp"
#include "MMO/Core/Registries/ToolRegistry.hpp"
#include "commands/command_result.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

using Json = nlohmann::json;
using namespace GRIM::MMO;
static GrimToolDispatch dispatch;
static void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

static bool publishCalculator(const std::filesystem::path& path) {
    std::ifstream input(path);
    check(static_cast<bool>(input), "Missing calculator manifest");
    const auto entries = Json::parse(input).at("tools");
    check(entries.size() == 1, "Expected one calculator tool");
    const auto& entry = entries[0];
    check(entry.value("usage_type", 1) == 1 && entry.at("is_informational") == true &&
          entry.at("needs_confirmation") == false, "Invalid tool policy");
    check(entry.at("parameters").size() == 1 && entry.at("parameters")[0].at("required") == true &&
          entry.at("parameters")[0].at("name") == "expression", "Missing expression parameter");
    ToolDescriptor descriptor;
    descriptor.tool_id = entry.at("tool_id").get<std::string>();
    descriptor.usage = entry.at("usage").get<std::string>();
    descriptor.bindings = entry.at("bindings").get<decltype(descriptor.bindings)>();
    return ProviderRegistry::instance().publish({ToolProviderType::Plugin, "Calculator Plugin", "calculator-test", {}}, {{descriptor,
        [](const std::string& arguments) {
            struct Capture { CommandResult result; int completions = 0; } capture;
            GrimToolResultWriter writer;
            writer.host_context = &capture;
            writer.complete = [](GrimToolResultWriter* w, const GrimToolResult* r) {
                auto& c = *static_cast<Capture*>(w->host_context);
                ++c.completions;
                c.result.success = r->success;
                c.result.message = r->message;
                c.result.errorCode = r->error_code;
                c.result.structuredOutputJson = r->structured_output_json;
                return GRIM_OK;
            };
            GrimToolCall call;
            call.arguments = arguments.data();
            call.arguments_size = arguments.size();
            check(dispatch("calculate", &call, &writer) == GRIM_OK, "Handler failed at ABI boundary");
            check(capture.completions == 1, "Handler must complete exactly once");
            return capture.result;
        }}});
}

static Json invoke(const std::string& expression, bool success) {
    auto& registry = ToolRegistry::instance();
    const auto result = registry.invoke("calculate", expression, {.expected_registry_version = registry.version(), .require_canonical_model = true});
    check(result.success == success, expression.c_str());
    auto output = Json::parse(result.structuredOutputJson);
    check(output.at("success") == success, "Structured success mismatch");
    if (success) check(output.at("answer") == result.message, "Answer mismatch");
    else check(output.at("error") == result.errorCode &&
               output.at("error_message") == result.message, "Error mismatch");
    return output;
}

int main(int argc, char** argv) {
    try {
        check(argc == 2, "Pass the built calculator plugin path");
#ifdef _WIN32
        const auto library = LoadLibraryA(argv[1]);
        check(library != nullptr, "Could not load calculator plugin DLL");
        const auto symbol = [library](const char* name) { return GetProcAddress(library, name); };
#else
        const auto library = dlopen(argv[1], RTLD_NOW);
        check(library != nullptr, "Could not load calculator plugin library");
        const auto symbol = [library](const char* name) { return dlsym(library, name); };
#endif
        const auto pluginInit = reinterpret_cast<GrimResult (*)(const GrimPluginAPI*, GrimPluginHandle)>(symbol("grim_plugin_init"));
        const auto pluginShutdown = reinterpret_cast<void (*)()>(symbol("grim_plugin_shutdown"));
        check(pluginInit && pluginShutdown, "Missing native plugin exports");
        check(pluginInit(nullptr, 0) == GRIM_ERROR_API_VERSION_MISMATCH, "Null API accepted");
        GrimPluginAPI api{};
        api.api_version = GRIM_API_VERSION - 0x100;
        check(pluginInit(&api, 1) == GRIM_ERROR_API_VERSION_MISMATCH, "Old API accepted");
        api.api_version = GRIM_API_VERSION;
        dispatch = reinterpret_cast<GrimToolDispatch>(symbol("grim_plugin_dispatch_tool"));
        check(dispatch != nullptr, "Missing dispatcher export");
        check(pluginInit(&api, 1) == GRIM_OK, "Plugin initialization failed");
        check(publishCalculator(std::filesystem::path(argv[1]).replace_extension(".tools.json")),
              "Calculator manifest did not publish");
        check(ToolRegistry::instance().getModelTools().size() == 1, "Tool missing from model discovery");

        for (const auto& entry : std::initializer_list<std::pair<const char*, const char*>>{
            {"(12 - 2)", "10"}, {"2 + 3 * 4", "14"}, {"(2 + 3) * 4", "20"},
            {"2^3^2", "512"}, {"-2^2", "-4"}, {"2^-3", "0.125"},
            {"0.1 + 0.2", "0.3"}, {"sqrt(81) + max(3, 7)", "16"},
            {"sin(pi / 2)", "1"}, {"1e3 / 4", "250"}, {"10 % 4", "2"},
            {"-0", "0"}, {"abs(-4)+floor(2.9)+ceil(1.1)+round(1.5)", "10"},
            {"log(100)+ln(e)+min(3,4)+pow(2,3)", "14"}}) {
            check(invoke(entry.first, true).at("answer") == entry.second, entry.first);
        }
        for (const auto& entry : std::initializer_list<std::pair<std::string, const char*>>{
            {"", "EMPTY_EXPRESSION"}, {" \t\n", "EMPTY_EXPRESSION"},
            {"12 / 0", "DIVISION_BY_ZERO"}, {"3 % 0", "DIVISION_BY_ZERO"},
            {"sqrt(-1)", "DOMAIN_ERROR"}, {"(-1)^0.5", "DOMAIN_ERROR"},
            {"2 +", "UNEXPECTED_TOKEN"}, {"(2 + 3", "MISSING_CLOSING_PARENTHESIS"},
            {"unknown(2)", "UNKNOWN_IDENTIFIER"}, {"min(1)", "INVALID_ARGUMENT_COUNT"},
            {"2 3", "UNEXPECTED_TOKEN"}, {"1e999", "NON_FINITE_RESULT"},
            {"exp(1000)", "NON_FINITE_RESULT"}, {std::string(4097, '1'), "EXPRESSION_TOO_LONG"},
            {std::string(100, '(') + "1" + std::string(100, ')'), "NESTING_LIMIT_EXCEEDED"},
            {std::string(4095, '-') + "1", "NESTING_LIMIT_EXCEEDED"},
            {std::string("1^") + std::string(100, '-') + "1", "NESTING_LIMIT_EXCEEDED"},
            {std::string("2\0+3", 4), "UNEXPECTED_TOKEN"}, {"1\"", "UNEXPECTED_TOKEN"}}) {
            check(invoke(entry.first, false).at("error") == std::string("ERR_CALC_") + entry.second,
                  entry.first.c_str());
        }
        check(invoke("12 / 0", false).at("error_offset") == 3, "Incorrect error offset");
        std::string powers = "1";
        for (int i = 0; i < 100; ++i) powers += "^1";
        check(invoke(powers, false).at("error") == "ERR_CALC_NESTING_LIMIT_EXCEEDED",
              "Exponent recursion must be bounded");
        check(invoke(std::string(4095, ' ') + "1", true).at("value") == 1, "Input boundary failed");
        GrimToolCall bad;
        GrimToolResultWriter writer;
        check(dispatch("calculate", nullptr, &writer) == GRIM_ERROR_INVALID_PARAM, "Null call accepted");
        check(dispatch("calculate", &bad, nullptr) == GRIM_ERROR_INVALID_PARAM, "Null writer accepted");
        check(dispatch("calculate", &bad, &writer) == GRIM_ERROR_INVALID_PARAM, "Null completion accepted");
        pluginShutdown();
        ProviderRegistry::instance().retire("Calculator Plugin", "calculator-test");
        check(!ToolRegistry::instance().isRegistered("calculate"), "Provider removal failed");
#ifdef _WIN32
        FreeLibrary(library);
#else
        dlclose(library);
#endif
        std::cout << "calculator plugin/registry tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

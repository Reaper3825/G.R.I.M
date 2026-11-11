#include "core/plugin_api.hpp"
#include "MMO/Core/ToolRegistry.hpp"
#include "commands/command_result.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

using Json = nlohmann::json;
using namespace GRIM::MMO;
static GrimToolHandler handler;
static void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

static GrimResult registerTools(const GrimPluginAPI*, const GrimToolRegistration* tools, size_t count) {
    check(count == 1, "Expected one calculator tool");
    const auto& tool = tools[0];
    const auto& d = tool.descriptor;
    check(tool.usage_type == GRIM_TOOL_USAGE_MODEL && d.is_informational &&
          !d.needs_confirmation && !d.required_permissions, "Invalid tool policy");
    check(d.parameter_count == 1 && d.parameters[0].required &&
          std::string(d.parameters[0].name) == "expression", "Missing expression parameter");
    check(tool.binding_count == 1, "Missing tool documentation");
    handler = tool.handler;
    ToolDescriptor descriptor;
    descriptor.tool_id = d.tool_id;
    descriptor.provider_name = "Calculator Plugin";
    descriptor.provider_type = ToolProviderType::Plugin;
    descriptor.display_name = d.display_name;
    descriptor.description = d.description;
    descriptor.short_description = tool.short_description;
    descriptor.usage = d.usage;
    descriptor.category = d.category;
    descriptor.argument_format = d.argument_format;
    descriptor.is_informational = d.is_informational;
    descriptor.needs_confirmation = d.needs_confirmation;
    descriptor.usage_type = ToolUsage::Model;
    descriptor.bindings.emplace(tool.bindings[0].placeholder, tool.bindings[0].type);
    return ToolRegistry::instance().registerBatch("Calculator Plugin", {{descriptor,
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
            check(handler(&call, &writer, nullptr) == GRIM_OK, "Handler failed at ABI boundary");
            check(capture.completions == 1, "Handler must complete exactly once");
            return capture.result;
        }}}) ? GRIM_OK : GRIM_ERROR_ALREADY_EXISTS;
}

static Json invoke(const std::string& expression, bool success) {
    auto& registry = ToolRegistry::instance();
    const auto result = registry.invokeModel("calculate", expression, registry.version());
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
        const auto pluginInfo = reinterpret_cast<const GrimPluginInfo* (*)()>(symbol("grim_plugin_get_info"));
        const auto pluginInit = reinterpret_cast<GrimResult (*)(const GrimPluginAPI*)>(symbol("grim_plugin_init"));
        const auto pluginShutdown = reinterpret_cast<void (*)()>(symbol("grim_plugin_shutdown"));
        check(pluginInfo && pluginInit && pluginShutdown, "Missing native plugin exports");
        check(pluginInfo()->required_permissions == 0, "Unexpected plugin permissions");
        check(pluginInit(nullptr) == GRIM_ERROR_API_VERSION_MISMATCH, "Null API accepted");
        GrimPluginAPI api{};
        api.api_version = GRIM_API_VERSION - 0x100;
        check(pluginInit(&api) == GRIM_ERROR_API_VERSION_MISMATCH, "Old API accepted");
        api.api_version = GRIM_API_VERSION;
        check(pluginInit(&api) == GRIM_ERROR_INVALID_PARAM, "Missing registration callback accepted");
        api.register_tools = registerTools;
        check(pluginInit(&api) == GRIM_OK, "Plugin registration failed");
        check(pluginInit(&api) == GRIM_ERROR_ALREADY_EXISTS, "Registration failure not propagated");
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
        check(handler(nullptr, &writer, nullptr) == GRIM_ERROR_INVALID_PARAM, "Null call accepted");
        check(handler(&bad, nullptr, nullptr) == GRIM_ERROR_INVALID_PARAM, "Null writer accepted");
        check(handler(&bad, &writer, nullptr) == GRIM_ERROR_INVALID_PARAM, "Null completion accepted");
        pluginShutdown();
        ToolRegistry::instance().unregisterByProvider("Calculator Plugin");
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

#include "core/plugin_api.hpp"
#include <memory>
#include "plugins/input_plugin/InputController.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>
#include <string>
#include <iostream>
extern "C" GrimResult grim_plugin_init(const GrimPluginAPI*, GrimPluginHandle);
extern "C" void grim_plugin_shutdown();
extern "C" GrimToolReply grim_plugin_dispatch_tool(const char*, const GrimToolCall*);
namespace {
constexpr const char* tools[] = {"move_mouse", "click", "type", "key"};
 std::vector<std::string> effects;
void check(bool value,const char* message){if(!value) throw std::runtime_error(message);}
bool success = false;
GrimResult invoke(size_t index, const std::string& text) {
    const GrimToolCall call{sizeof(GrimToolCall), text.data(), text.size()};
    const auto reply = grim_plugin_dispatch_tool(tools[index], &call);
    if (reply.delivery != GRIM_OK) return reply.delivery;
    check(reply.json && reply.release, "Handler returned no JSON");
    const std::unique_ptr<const char, decltype(reply.release)> bytes(reply.json, reply.release);
    success = nlohmann::json::parse(bytes.get(), bytes.get() + reply.size).at("status") == "completed";
    return GRIM_OK;
}
}
namespace GRIM {
void InputController::moveMouse(int x,int y){effects.push_back(std::to_string(x)+","+std::to_string(y));}
void InputController::click(MouseButton){effects.push_back("click");}
void InputController::typeText(const std::string& text,int delay){check(delay==15,"Typing delay changed");effects.push_back(text);}
void InputController::keyEvent(WORD,KeyAction){effects.push_back("key");}
void InputController::combo(const std::vector<WORD>& keys){effects.push_back("combo:"+std::to_string(keys.size()));}
}
int main(){
 GrimPluginAPI api{};api.api_version=GRIM_API_VERSION;
 auto old=api;old.api_version=10<<16;check(grim_plugin_init(&old, 1)==GRIM_ERROR_API_VERSION_MISMATCH,"Old ABI accepted");
 check(grim_plugin_init(&api, 1)==GRIM_OK,"Initialization failed");check(effects.empty(),"Initialization injected input");
#ifndef _WIN32
 invoke(0,"-1200,300 trailing");check(!success && effects.empty(),"Unsupported platform accepted native input");
 grim_plugin_shutdown();return 0;
#endif
 invoke(0,"-1200,300 trailing");check(success && effects.back()=="-1200,300","Coordinate parsing changed");
 invoke(1,"");check(success && effects.back()=="click","Default click failed");
 invoke(2,std::string("a\0b",3));check(success && effects.back()==std::string("a\0b",3),"Raw text truncated");
 invoke(3,"ctrl + c");check(success && effects.back()=="combo:2","Combo failed");
 auto count=effects.size();
 invoke(0,"bad");check(!success && effects.size()==count,"Invalid move caused effect");
 invoke(2,"");check(!success && effects.size()==count,"Empty text caused effect");
 for(auto entry:{std::pair<size_t,const char*>{1,"bogus"},{3,"bogus"},{3,"ctrl+bogus"},{3,"+"}}){bool rejected=false;try{invoke(entry.first,entry.second);}catch(const std::invalid_argument&){rejected=true;}check(rejected && effects.size()==count,"Unknown native input accepted");}
 invoke(2,std::string(65537,'x'));check(!success && effects.size()==count,"Size bound lost");
 grim_plugin_shutdown();grim_plugin_shutdown();check(invoke(1,"")==GRIM_ERROR_PLUGIN_NOT_LOADED,"Retired provider injected input");
 check(grim_plugin_init(&api, 1)==GRIM_OK,"Reload failed");grim_plugin_shutdown();
 std::cout<<"Input provider parsing, native mapping, rejection and lifecycle passed\n";
}

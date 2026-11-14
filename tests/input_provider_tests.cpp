#include "core/plugin_api.hpp"
#include "plugins/input_plugin/InputController.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>
#include <string>
#include <iostream>
extern "C" GrimResult grim_plugin_init(const GrimPluginAPI*);
extern "C" void grim_plugin_shutdown();
namespace {
std::vector<GrimToolRegistration> tools; bool permitted=true,current=true; std::vector<std::string> effects;
void check(bool value,const char* message){if(!value) throw std::runtime_error(message);}
GrimToolResult result; std::string message,error;
GrimResult complete(GrimToolResultWriter*,const GrimToolResult* r){result=*r;message=r->message?r->message:"";error=r->error_code?r->error_code:"";return GRIM_OK;}
GrimResult invoke(size_t index,const std::string& text){GrimToolCall call{};call.arguments=text.data();call.arguments_size=text.size();GrimToolResultWriter writer{};writer.complete=complete;return tools[index].handler(&call,&writer,tools[index].user_data);}
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
 api.has_permission=[](const GrimPluginAPI*,GrimPermission){return permitted;};
 api.is_provider_current=[](const GrimPluginAPI*){return current;};
 api.register_tools=[](const GrimPluginAPI*,const GrimToolRegistration* t,size_t n){tools.assign(t,t+n);return GRIM_OK;};
 auto old=api;old.api_version=10<<16;check(grim_plugin_init(&old)==GRIM_ERROR_API_VERSION_MISMATCH,"Old ABI accepted");
 check(grim_plugin_init(&api)==GRIM_OK && tools.size()==4,"Registration failed");check(effects.empty(),"Initialization injected input");
 permitted=false;check(invoke(1,"")==GRIM_ERROR_PERMISSION_DENIED && effects.empty(),"Permission bypass");permitted=true;
 current=false;check(invoke(1,"")==GRIM_ERROR_PERMISSION_DENIED && effects.empty(),"Inactive provider injected input");current=true;
#ifndef _WIN32
 invoke(0,"-1200,300 trailing");check(!result.success && effects.empty(),"Unsupported platform accepted native input");
 grim_plugin_shutdown();return 0;
#endif
 invoke(0,"-1200,300 trailing");check(result.success && effects.back()=="-1200,300","Coordinate parsing changed");
 invoke(1,"");check(result.success && effects.back()=="click","Default click failed");
 invoke(2,std::string("a\0b",3));check(result.success && effects.back()==std::string("a\0b",3),"Raw text truncated");
 invoke(3,"ctrl + c");check(result.success && effects.back()=="combo:2","Combo failed");
 auto count=effects.size();
 invoke(0,"bad");check(!result.success && effects.size()==count,"Invalid move caused effect");
 invoke(2,"");check(!result.success && effects.size()==count,"Empty text caused effect");
 for(auto entry:{std::pair<size_t,const char*>{1,"bogus"},{3,"bogus"},{3,"ctrl+bogus"},{3,"+"}}){bool rejected=false;try{invoke(entry.first,entry.second);}catch(const std::invalid_argument&){rejected=true;}check(rejected && effects.size()==count,"Unknown native input accepted");}
 invoke(2,std::string(65537,'x'));check(!result.success && effects.size()==count,"Size bound lost");
 grim_plugin_shutdown();grim_plugin_shutdown();check(invoke(1,"")==GRIM_ERROR_PERMISSION_DENIED,"Retired provider injected input");
 check(grim_plugin_init(&api)==GRIM_OK,"Reload failed");grim_plugin_shutdown();
 std::cout<<"Input provider permissions, parsing, native mapping, rejection and lifecycle passed\n";
}

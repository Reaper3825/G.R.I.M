#include "core/plugin_api.hpp"
#include "plugins/osint_plugin/provider_paths.hpp"
#include "plugins/osint_plugin/process.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <thread>
#include <stdexcept>
extern "C" GrimResult grim_plugin_init(const GrimPluginAPI*);
extern "C" void grim_plugin_shutdown();
namespace {
std::vector<GrimToolRegistration> tools; std::atomic<int> runs{0}; bool permitted=true,current=true;
std::atomic<bool> block{false},entered{false}; bool success;std::string output,message;
void check(bool value,const char* text){if(!value)throw std::runtime_error(text);}
GrimResult invoke(size_t index,const std::string& args){
 GrimToolCall call{};call.arguments=args.data();call.arguments_size=args.size();GrimToolResultWriter writer{};
 writer.complete=[](GrimToolResultWriter*,const GrimToolResult* result){success=result->success;output=result->structured_output_json?result->structured_output_json:"";message=result->message?result->message:"";return GRIM_OK;};
 return tools[index].handler(&call,&writer,tools[index].user_data);
}
}
int osintProvider::runProcess(const std::vector<std::string>& args,int){
 ++runs;entered=true;
 while(block && !cancelled)std::this_thread::sleep_for(std::chrono::milliseconds(5));
 if(cancelled)return -1;
 check(args.size()==6,"Bridge argument contract changed");
 std::ofstream(args[3])<<R"({"Example":{"url":"https://example.invalid/alice","status":"Claimed","http_status":"200"}})";
 return 0;
}
int main(){
 namespace fs=std::filesystem;
 auto original=fs::current_path(); auto temp=fs::temp_directory_path()/("grim-osint-fixture-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
 fs::create_directories(temp);fs::current_path(temp);
 GrimPluginAPI api{};api.api_version=GRIM_API_VERSION;
 api.has_permission=[](const GrimPluginAPI*,GrimPermission){return permitted;};api.is_provider_current=[](const GrimPluginAPI*){return current;};
 api.log=[](const GrimPluginAPI*,GrimLogLevel,const char*){};
 api.register_tools=[](const GrimPluginAPI*,const GrimToolRegistration* t,size_t n){tools.assign(t,t+n);return GRIM_OK;};
 try {
  auto old=api;old.api_version=10<<16;check(grim_plugin_init(&old)==GRIM_ERROR_API_VERSION_MISMATCH,"Old ABI accepted");
  check(grim_plugin_init(&api)==GRIM_OK && tools.size()==7,"OSINT registration failed");
  check(!fs::exists(temp/"cache") && runs==0,"OSINT load had side effects");
  fs::create_directories(osintProvider::scripts());std::ofstream(osintProvider::scripts()/"osit_bridge.py")<<"# fixture";
  invoke(2,"alice");check(success && runs==1,"Report did not use provider process");
  invoke(2,"alice");check(success && runs==1,"Cache was not reused");
  fs::current_path(original);invoke(2,"alice");check(success && runs==1,"Provider root changed with host cwd");fs::current_path(temp);
  for(const auto* name:{"../escape","x/y","x\\y","--flag","CON.json"}) {bool rejected=false;try{invoke(2,name);}catch(const std::invalid_argument&){rejected=true;}check(rejected && runs==1,"Invalid target reached runtime");}
  bool invalidCacheTarget=false;try{invoke(5,"alice ../../escape");}catch(const std::invalid_argument&){invalidCacheTarget=true;}
  check(invalidCacheTarget && runs==1,"Full cache target bypassed validation");
  permitted=false;check(invoke(2,"alice")==GRIM_ERROR_PERMISSION_DENIED,"Permission bypass");permitted=true;
  current=false;check(invoke(2,"alice")==GRIM_ERROR_PERMISSION_DENIED,"Inactive provider accepted work");current=true;
  std::ofstream(osintProvider::cache()/"sensitive_alice.jsonl")<<R"({"domain":"example.invalid","source_url":"https://example.invalid","findings":[{"tag":"fixture","severity":9,"match":"redacted"},{"severity":2}]})"<<"\n";
  invoke(6,"alice --severity high");check(success,"Findings read failed");auto snapshot=nlohmann::json::parse(output);
  check(snapshot.at("entries").at(0).at("findings").size()==1,"Copied findings ignored severity filter");
  block=true;entered=false;invoke(1,"bob --async");
  auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
  while(!entered && std::chrono::steady_clock::now()<deadline)std::this_thread::sleep_for(std::chrono::milliseconds(5));
  check(entered,"Async process never started");grim_plugin_shutdown();check(osintProvider::cancelled,"Unload failed to cancel children");grim_plugin_shutdown();
  block=false;check(grim_plugin_init(&api)==GRIM_OK,"OSINT reload failed");invoke(3,"");check(message.find("No active scans")!=std::string::npos,"Jobs survived reload");grim_plugin_shutdown();
  fs::current_path(original);fs::remove_all(temp);
  std::cout<<"OSINT cache, copied findings, permissions, cancellation and reload passed\n";
 }catch(...){osintProvider::cancelled=true;grim_plugin_shutdown();fs::current_path(original);fs::remove_all(temp);throw;}
}

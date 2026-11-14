#include "plugins/application_plugin/user_aliases.hpp"
#include "plugins/application_plugin/application_catalog.hpp"
#include "plugins/application_plugin/aliases_generated.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <chrono>
#include <iostream>

int main() {
 namespace fs=std::filesystem;
 using namespace GRIM::Applications;
 auto check=[](bool value,const char* text){if(!value)throw std::runtime_error(text);};
 const auto root=fs::temp_directory_path()/("grim-user-aliases-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
 fs::create_directories(root);
 struct Cleanup {fs::path root;~Cleanup(){std::error_code error;fs::remove_all(root,error);}} cleanup{root};
 UserAliases store(root);
 check(store.reload().empty() && fs::is_empty(root),"Empty configuration caused a write");
 flatbuffers::FlatBufferBuilder builder;
 auto manual=AliasesSchema::CreateAliasEntryDirect(builder,"Editor","C:/My Editor/editor.exe");
 auto automatic=AliasesSchema::CreateAliasEntryDirect(builder,"AutoApp","C:/old.exe");
 auto users=builder.CreateVector(std::vector{manual});auto autos=builder.CreateVector(std::vector{automatic});
 auto document=AliasesSchema::CreateAliases(builder,users,autos,0);builder.Finish(document);
 {std::ofstream file(root/"app_aliases.fb",std::ios::binary);file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),builder.GetSize());}
 auto originalTime=fs::last_write_time(root/"app_aliases.fb");
 check(store.reload().size()==1,"Legacy automatic aliases were imported");
 check(store.resolve(" EDITOR ")=="C:/My Editor/editor.exe","User alias normalization failed");
 check(store.resolve("AutoApp")=="AutoApp","Automatic alias still resolves");
 check(store.resolve("  unmatched target  ")=="  unmatched target  ","Unmatched raw target changed");
 check(fs::last_write_time(root/"app_aliases.fb")==originalTime,"Legacy database was rewritten");
 auto write=[&](const std::string& text){std::ofstream(root/"application_aliases.json")<<text;};
 write(R"({"version":1,"aliases":{"work browser":"https://example.invalid","Editor":"D:/new.exe"}})");
 check(store.reload().size()==2 && store.resolve("editor")=="D:/new.exe","JSON override failed");
 for(auto text:{"broken",R"({"version":1,"aliases":{"Editor":"a"," editor ":"b"}})",R"({"version":1,"aliases":{"bad":""}})"}) {
  write(text);bool failed=false;try{store.reload();}catch(const std::exception&){failed=true;}
  check(failed && store.resolve("editor")=="D:/new.exe","Invalid reload discarded last-good aliases");
 }
 write(R"({"version":1,"aliases":{}})");check(store.reload().empty(),"Explicit empty aliases resurrected legacy records");
 fs::remove(root/"application_aliases.json");check(store.reload().size()==1,"Legacy user fallback failed");
 {std::ofstream(root/"app_aliases.fb",std::ios::binary)<<"invalid";}
 bool corrupt=false;try{store.reload();}catch(const std::exception&){corrupt=true;}
 check(corrupt && store.resolve("editor")=="C:/My Editor/editor.exe","Corrupt legacy file erased aliases");
 auto programs=root/"programs";fs::create_directories(programs/"Portable"/"deep");
 std::ofstream(programs/"direct.exe");std::ofstream(programs/"Portable"/"app.exe");
 std::ofstream(programs/"Portable"/"script.cmd");std::ofstream(programs/"Portable"/"notes.txt");
 std::ofstream(programs/"Portable"/"deep"/"excluded.exe");
 auto discovered=scanPortableDirectories({programs,root/"missing"});
 check(discovered.size()==3,"Portable discovery lost legacy coverage or exceeded depth");
 std::cout<<"User-only legacy import, JSON precedence, reload retention and portable discovery passed\n";
}

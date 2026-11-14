#include "plugins/osint_plugin/process.hpp"
#include "plugins/osint_plugin/provider_paths.hpp"
#include <future>
#include <thread>
#include <iostream>
int main(int argc,char** argv) {
 if(argc>1 && std::string(argv[1])=="--arguments")
  return argc==5 && std::string(argv[2])=="space value" && std::string(argv[3])=="quote\"and\\slash\\" && std::string(argv[4])=="&literal" ? 0 : 2;
 if(argc>1 && std::string(argv[1])=="--sleep") {std::this_thread::sleep_for(std::chrono::seconds(10));return 0;}
 osintProvider::initializePaths();
 const auto executable=std::filesystem::absolute(argv[0]).string();
 auto check=[](bool value){if(!value)throw std::runtime_error("Native OSINT process lifecycle test failed");};
 check(osintProvider::runProcess({executable,"--arguments","space value","quote\"and\\slash\\","&literal"},5)==0);
 check(osintProvider::runProcess({executable,"--sleep"},0)!=0);
 auto running=std::async(std::launch::async,[&]{return osintProvider::runProcess({executable,"--sleep"},20);});
 std::this_thread::sleep_for(std::chrono::milliseconds(100));osintProvider::cancelled=true;
 check(running.wait_for(std::chrono::seconds(3))==std::future_status::ready);check(running.get()!=0);
 check(osintProvider::runProcess({executable,"--arguments"},5)!=0);
 osintProvider::cancelled=false;
 check(osintProvider::runProcess({executable,"--arguments","space value","quote\"and\\slash\\","&literal"},5)==0);
 std::cout<<"Shell-free process arguments, timeout, cancellation and restart passed\n";
}

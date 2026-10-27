#include "ui/world_lab/environment_registry.hpp"
#include <iostream>
int main(int argc,char** argv) {
    try {
        if(argc!=2)throw std::runtime_error("Expected world path");
        const auto file=std::filesystem::absolute(argv[1]);
        const auto raw=GRIM::WorldLab::loadJsonFile(file);
        const auto world=GRIM::WorldLab::Document::parse(raw);
        if(world.json()!=raw)throw std::runtime_error("Document roundtrip changed world content");
        GRIM::WorldLab::EnvironmentRegistry registry(file.parent_path());
        registry.checkDestination(world,file);
        registry.registerFile(file);
        const auto loaded=registry.load(world.environmentId);
        if(loaded.json()!=raw)throw std::runtime_error("Registered world differs from saved world");
        std::cout<<"Native document roundtrip and registry resolution passed: "<<world.title<<"\n";
        for(const auto& room:loaded.paths)std::cout<<loaded.roomPath(room.id)<<"\n";
    }catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}
}

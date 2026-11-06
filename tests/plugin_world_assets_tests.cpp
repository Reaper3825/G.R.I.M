#include "ui/world_lab/world_asset.hpp"
#include <iostream>
#include <stdexcept>

using namespace GRIM::WorldLab;
int main(int argc,char** argv) {
    try {
        if(argc!=2)throw std::runtime_error("Expected asset library directory");
        std::size_t rooms=0,objects=0;
        for(const auto& entry:listAssets(argv[1])) {
            if(!entry.path.filename().string().starts_with("plugin_"))continue;
            if(!entry.valid)throw std::runtime_error("Invalid plugin asset: "+entry.path.string());
            const auto input=loadJsonFile(entry.path);
            const auto asset=Asset::parse(input);
            if(asset.json()!=input)throw std::runtime_error("Asset roundtrip changed authored metadata");
            auto world=Document::initial();
            const auto first=asset.insert(world,world.spawn,400,80);
            const auto second=asset.insert(world,world.spawn,800,80);
            if(first==second)throw std::runtime_error("Asset instances reused world identity");
            const auto saved=world.json();
            if(Document::parse(saved).json()!=saved)throw std::runtime_error("World roundtrip lost metadata");
            const auto captured=Asset::capture(world,first,asset.name);
            if(captured.entities.size()!=asset.entities.size())throw std::runtime_error("Capture lost resources");
            const auto before=world.entities.size();
            for(const auto& entity:captured.entities) {
                const auto original=std::find_if(asset.entities.begin(),asset.entities.end(),[&](const auto& e){
                    return e.assetSource->assetId==entity.assetSource->assetId && e.assetSource->entityId==entity.assetSource->entityId;
                });
                if(original==asset.entities.end())throw std::runtime_error("Capture changed provenance");
                auto expected=nlohmann::json(*original),actual=nlohmann::json(entity);
                for(const char* field:{"id","name","parent"}){expected.erase(field);actual.erase(field);}
                if(actual!=expected)throw std::runtime_error("Insertion changed resource or room metadata");
            }
            if(world.entities.size()!=before)throw std::runtime_error("Capture changed world");
            if(asset.preview().find(asset.root)->kind=="room")++rooms;else ++objects;
        }
        if(!rooms || !objects)throw std::runtime_error("Plugin fixtures missing");
        std::cout<<"Validated "<<rooms<<" plugin rooms and "<<objects<<" resource assets with native codec and insertion\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}

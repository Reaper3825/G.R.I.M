#include "resources/models/GRIM-text/Shared/UnigramByte/ExactPieceMatcher.hpp"
#include <iostream>
#include <stdexcept>

using namespace GRIM::Tokenizer;
void check(bool condition) {
    if(!condition)throw std::runtime_error("Exact-piece matcher growth regression failed");
}
int main() {
    ExactPieceMatcher matcher;
    std::vector<ExactPieceDefinition> definitions;
    // Thousands of branches force vector growth while constructing parent edges.
    for(int i=0;i<4096;++i)definitions.push_back({"piece/"+std::to_string(i)+"/end",i});
    definitions.push_back({"piece/",5000});
    matcher.rebuild(definitions);
    for(int i=0;i<4096;++i) {
        const auto matches=matcher.findMatches(definitions[i].text);
        check(matches.size()==1 && matches[0].token_id==i &&
              matches[0].start==0 && matches[0].end==definitions[i].text.size());
    }
    matcher.rebuild({{"replacement",42}});
    check(matcher.findMatches("piece/100/end").empty());
    check(matcher.findMatches("replacement").front().token_id==42);
    matcher.rebuild({});check(matcher.empty());
    std::cout<<"Exact-piece matcher growth regression passed\n";
}

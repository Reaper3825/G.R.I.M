#include <SFML/Graphics.hpp>
#include <iostream>
#include <deque>
#include <string>
#include <sstream>
#include <vector>
#include <filesystem>
#include <system_error>
#include <algorithm>
#include "NLP.hpp"
#include "ai.hpp"
#include "aliases.hpp"
#include "synonyms.hpp"
#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>


static std::string findOnPath(const std::string& app) {
    char buffer[MAX_PATH];
    DWORD result = SearchPathA(
        NULL,
        app.c_str(),
        ".exe",            // try with .exe extension automatically
        MAX_PATH,
        buffer,
        NULL
    );
    if (result > 0 && result < MAX_PATH) {
        return std::string(buffer);
    }
    return app;
}
#endif



namespace fs = std::filesystem;

// ---------------- Tunables ----------------
static float    kTitleBarH      = 48.f;
static float    kInputBarH      = 44.f;
static float    kSidePad        = 12.f;
static float    kTopPad         = 6.f;
static float    kBottomPad      = 6.f;
static float    kLineSpacing    = 1.25f;
static unsigned kFontSize       = 18;
static unsigned kTitleFontSize  = 22;
static size_t   kMaxHistory     = 1000;
// -------------------------------------------

// -------- Font discovery helper ------------
static std::string findAnyFontInResources(int argc, char** argv) {
    std::error_code ec;
    auto has_ttf = [&](const fs::path& dir) -> std::string {
        if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) return {};
        for (auto& e : fs::directory_iterator(dir, ec)) {
            if (!e.is_regular_file(ec)) continue;
            auto ext = e.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".ttf") return e.path().string();
        }
        return {};
    };

    fs::path exeDir = (argc > 0) ? fs::path(argv[0]).parent_path() : fs::current_path(ec);
    if (auto s = has_ttf(exeDir / "resources"); !s.empty()) return s;
    if (auto s = has_ttf(fs::current_path(ec) / "resources"); !s.empty()) return s;

#ifdef _WIN32
    for (const char* f : {
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeui.ttf"
    }) if (fs::exists(f, ec)) return std::string(f);
#endif
    return {};
}
// -------------------------------------------

struct WrappedLine {
    std::string text;
    sf::Color color{sf::Color::White};
};

class ConsoleHistory {
public:
    void push(const std::string& line, sf::Color c = sf::Color::White) {
        if (raw_.size() >= kMaxHistory) raw_.pop_front();
        raw_.push_back({line, c});
        dirty_ = true;
    }

    void ensureWrapped(float maxWidth, sf::Text& meas) {
        if (!dirty_ && lastWrapWidth_ == maxWidth && lastFontSize_ == meas.getCharacterSize()) return;
        wrapped_.clear();
        for (auto& ln : raw_) wrapLine(ln, maxWidth, meas, wrapped_);
        dirty_ = false;
        lastWrapWidth_ = maxWidth;
        lastFontSize_  = meas.getCharacterSize();
    }

    const std::vector<WrappedLine>& wrapped() const { return wrapped_; }
    size_t wrappedCount() const { return wrapped_.size(); }

private:
    static void wrapLine(const WrappedLine& ln, float maxW, sf::Text& meas, std::vector<WrappedLine>& out) {
        if (ln.text.empty()) { out.push_back({"", ln.color}); return; }
        std::string word, current; std::istringstream iss(ln.text);
        auto flush = [&](bool f=false){ if(f||!current.empty()){ out.push_back({current,ln.color}); current.clear(); } };

        while (iss >> word) {
            std::string test = current.empty()?word:current+" "+word;
            meas.setString(test);
            if (meas.getLocalBounds().width <= maxW) {
                current=test;
            } else {
                if (current.empty()) {
                    std::string accum;
                    for (char c: word) {
                        meas.setString(accum+c);
                        if (meas.getLocalBounds().width <= maxW) accum+=c;
                        else {
                            if(!accum.empty()) out.push_back({accum,ln.color});
                            accum=std::string(1,c);
                        }
                    }
                    if (!accum.empty()) current=accum;
                } else {
                    out.push_back({current,ln.color});
                    current=word;
                }
            }
        }
        flush(true);
    }

    bool dirty_=true;
    float lastWrapWidth_=-1.f;
    unsigned lastFontSize_=0;
    std::deque<WrappedLine> raw_;
    std::vector<WrappedLine> wrapped_;
};

// -------------- helpers -------------------
static std::string trim(const std::string& s) {
    size_t b=s.find_first_not_of(" \t\r\n"); if(b==std::string::npos) return "";
    size_t e=s.find_last_not_of(" \t\r\n"); return s.substr(b,e-b+1);
}

static std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> out;
    std::istringstream iss(line);
    std::string t;
    while(iss>>t) out.push_back(t);
    return out;
}

static fs::path resolvePath(const fs::path& cur,const std::string& u){
    std::error_code ec;
    fs::path p(u);
    return p.is_absolute()?fs::weakly_canonical(p,ec):fs::weakly_canonical(cur/p,ec);
}
// -------------------------------------------

int main(int argc, char** argv) {
    // NLP load
    NLP nlp;
    {
        std::error_code ec;
        fs::path exeDir=(argc>0)?fs::path(argv[0]).parent_path():fs::current_path(ec);
        fs::path rulesExe=exeDir/"nlp_rules.json";
        fs::path rulesCwd="nlp_rules.json";
        std::string err;
        bool ok=fs::exists(rulesExe,ec)?nlp.load_rules(rulesExe.string(),&err):nlp.load_rules(rulesCwd.string(),&err);
        if(!ok) std::cerr<<"[NLP] Failed to load rules: "<<err<<"\n";
        else {
            std::cerr<<"[NLP] Loaded rules. Intents:";
            for(auto&s:nlp.list_intents()) std::cerr<<" "<<s;
            std::cerr<<"\n";
        }
    }

    // Load synonyms
    loadSynonyms("synonyms.json");
    // Load aliases
    loadAliases("app_aliases.json");


    sf::RenderWindow window(sf::VideoMode(500,800),"GRIM");
    window.setVerticalSyncEnabled(true);

    sf::Font font;
    std::string fontPath=findAnyFontInResources(argc,argv);
    if(fontPath.empty()||!font.loadFromFile(fontPath))
        std::cerr<<"[WARN] Could not load a font. Put a .ttf in resources/.\n";
    else
        std::cerr<<"[INFO] Using font: "<<fontPath<<"\n";

    sf::RectangleShape titleBar; titleBar.setFillColor(sf::Color(26,26,30));
    sf::Text titleText;
    if(font.getInfo().family!=""){
        titleText.setFont(font);
        titleText.setCharacterSize(kTitleFontSize);
        titleText.setFillColor(sf::Color(220,220,235));
        titleText.setString("G R I M");
    }

    sf::RectangleShape inputBar; inputBar.setFillColor(sf::Color(30,30,35));
    sf::Text inputText,lineText;
    if(font.getInfo().family!=""){
        inputText.setFont(font);
        inputText.setCharacterSize(kFontSize);
        inputText.setFillColor(sf::Color::White);
        lineText.setFont(font);
        lineText.setCharacterSize(kFontSize);
        lineText.setFillColor(sf::Color::White);
    }

    ConsoleHistory history;
    auto addHistory=[&](const std::string&s,sf::Color c=sf::Color::White){
        history.push(s,c);
        std::cout<<s<<"\n";
    };

    std::string buffer;
    fs::path currentDir=fs::current_path();
    addHistory("GRIM is ready. Type 'help' for commands. Type 'quit' to exit.",sf::Color(160,200,255));

    sf::Clock caretClock;
    bool caretVisible=true;
    float scrollOffsetLines=0.f;
    auto clampScroll=[&](float m){
        if(scrollOffsetLines<0)scrollOffsetLines=0;
        if(scrollOffsetLines>m)scrollOffsetLines=m;
    };

    while(window.isOpen()){
        sf::Event e;
        while(window.pollEvent(e)){
            if(e.type==sf::Event::Closed) window.close();
            if(e.type==sf::Event::KeyPressed){
                if(e.key.code==sf::Keyboard::Escape) window.close();
                if(e.key.code==sf::Keyboard::PageUp) scrollOffsetLines+=10.f;
                if(e.key.code==sf::Keyboard::PageDown){
                    scrollOffsetLines-=10.f;
                    if(scrollOffsetLines<0)scrollOffsetLines=0.f;
                }
                if(e.key.code==sf::Keyboard::Home) scrollOffsetLines=1e6f;
                if(e.key.code==sf::Keyboard::End) scrollOffsetLines=0.f;
            }
            if(e.type==sf::Event::MouseWheelScrolled && e.mouseWheelScroll.wheel==sf::Mouse::VerticalWheel){
                scrollOffsetLines+=(e.mouseWheelScroll.delta>0?3.f:-3.f);
                if(scrollOffsetLines<0)scrollOffsetLines=0.f;
            }

            if(e.type==sf::Event::TextEntered){
                if(e.text.unicode==8){ // backspace
                    if(!buffer.empty()) buffer.pop_back();
                }
                else if(e.text.unicode==13||e.text.unicode==10){ // enter
                    std::string line=trim(buffer);
                    buffer.clear();
                    if(line=="quit"||line=="exit"){ window.close(); break; }
                    if(!line.empty()) addHistory("> "+line,sf::Color(150,255,150));
                    else { addHistory("> "); continue; }

                    auto args = split(line);
                    std::string cmd = args.empty() ? "" : args[0];

                    // Normalize synonyms
                    cmd = normalizeWord(cmd);
                    if (!args.empty()) {
                        args[0] = cmd;
                        std::ostringstream oss;
                        for (size_t i = 0; i < args.size(); ++i) {
                            if (i > 0) oss << " ";
                            oss << args[i];
                        }
                        line = oss.str();
                    }

                    // NLP fallback
Intent intent = nlp.parse(line);
if(!intent.matched) {
    addHistory("[NLP] No intent matched.", sf::Color(255,200,140));
} else {
    std::ostringstream oss;
    oss << "[NLP] intent=" << intent.name << " score=" << intent.score;
    addHistory(oss.str(), sf::Color(180,255,180));
    for(auto& kv : intent.slots) addHistory("  " + kv.first + " = " + kv.second);

    if(intent.name=="open_app") {
        auto it=intent.slots.find("app");
        if(it!=intent.slots.end()) {
            std::string app = it->second;
            addHistory("Opening app: " + app);
#ifdef _WIN32
            HINSTANCE result=ShellExecuteA(NULL,"open",app.c_str(),NULL,NULL,SW_SHOWNORMAL);
            if((INT_PTR)result<=32)
                addHistory("Failed to open app: " + app, sf::Color(255,140,140));
#else
            int ret=system(app.c_str());
            if(ret!=0) addHistory("Failed to open app: " + app, sf::Color(255,140,140));
#endif
        }
    }
    else if(intent.name=="search_web") {
        auto it=intent.slots.find("query");
        if(it!=intent.slots.end()) {
            addHistory("[Web] Searching for: " + it->second);
            // TODO: hook into your web search logic
        }
    }
    else if(intent.name=="set_timer") {
        auto it=intent.slots.find("minutes");
        if(it!=intent.slots.end()) {
            addHistory("[Timer] Setting timer for " + it->second + " minutes.");
            // TODO: hook into your timer logic
        }
    }
    else if(intent.name=="clean") {
        addHistory("[Clean] Removing files in: " + currentDir.string());
        // TODO: delete files
    }
    else if(intent.name=="show_help") {
        addHistory("Available commands: help, pwd, cd <dir>, list, mkdir <name>, rm <target>, reloadnlp, grim <query>, open <app>, search <q>, timer <min>");
    }
    else if(intent.name=="show_pwd") {
        addHistory(currentDir.string());
    }
    else if(intent.name=="change_dir") {
        auto it=intent.slots.find("path");
        if(it!=intent.slots.end()) {
            fs::path newPath = resolvePath(currentDir, it->second);
            std::error_code ec;
            if(fs::exists(newPath,ec) && fs::is_directory(newPath,ec)) {
                currentDir = newPath;
                fs::current_path(currentDir, ec);
                addHistory("[cd] Changed directory to " + currentDir.string());
            } else {
                addHistory("[cd] Directory not found: " + it->second, sf::Color(255,140,140));
            }
        }
    }
    else if(intent.name=="list_dir") {
        std::error_code ec;
        addHistory("[ls] Listing contents of: " + currentDir.string());
        for(auto& e : fs::directory_iterator(currentDir, ec)) {
            addHistory("  " + e.path().filename().string());
        }
    }
    else if(intent.name=="make_dir") {
        auto it=intent.slots.find("dirname");
        if(it!=intent.slots.end()) {
            std::error_code ec;
            fs::path newDir = currentDir / it->second;
            if(fs::create_directory(newDir, ec)) {
                addHistory("[mkdir] Created: " + newDir.string());
            } else {
                addHistory("[mkdir] Failed to create: " + newDir.string(), sf::Color(255,140,140));
            }
        }
    }
    else if(intent.name=="remove_file") {
        auto it=intent.slots.find("target");
        if(it!=intent.slots.end()) {
            std::error_code ec;
            fs::path target = currentDir / it->second;
            if(fs::remove_all(target, ec) > 0) {
                addHistory("[rm] Removed: " + target.string());
            } else {
                addHistory("[rm] Failed to remove: " + target.string(), sf::Color(255,140,140));
            }
        }
    }
    else if(intent.name=="reload_nlp") {
        std::string err;
        bool ok = nlp.load_rules("nlp_rules.json", &err);
        if(ok) addHistory("[NLP] Reloaded rules.");
        else   addHistory("[NLP] Reload failed: " + err, sf::Color(255,140,140));
    }
    else if(intent.name=="grim_ai") {
        auto it=intent.slots.find("query");
        if(it!=intent.slots.end()) {
            try {
                std::string reply = callAI(it->second);
                addHistory("[AI] " + reply, sf::Color(180,200,255));
            } catch(const std::exception& e) {
                addHistory(std::string("[AI] Error: ") + e.what(), sf::Color(255,140,140));
            }
        }
    }
}

                else if(e.text.unicode>=32 && e.text.unicode<127) buffer.push_back((char)e.text.unicode);
            }
        }

        if(caretClock.getElapsedTime().asSeconds()>0.5f){
            caretVisible=!caretVisible;
            caretClock.restart();
        }

        // Layout...
        sf::Vector2u ws=window.getSize();
        float winW=ws.x, winH=ws.y;
        titleBar.setSize({winW,kTitleBarH}); titleBar.setPosition(0.f,0.f);
        inputBar.setSize({winW,kInputBarH}); inputBar.setPosition(0.f,winH-kInputBarH);

        if(font.getInfo().family!=""){
            sf::FloatRect tb=titleText.getLocalBounds();
            titleText.setPosition((winW-tb.width)*0.5f,(kTitleBarH-tb.height)*0.5f-6.f);
        }

        if(font.getInfo().family!=""){
            std::string toShow=buffer;
            if(caretVisible) toShow.push_back('_');
            inputText.setString(toShow);
            inputText.setPosition(kSidePad,winH-kInputBarH+(kInputBarH-(float)kFontSize)*0.5f-2.f);
        }

        if(font.getInfo().family!=""){
            lineText.setCharacterSize(kFontSize);
            float lineH=kLineSpacing*(float)kFontSize;
            float histTop=kTitleBarH+kTopPad;
            float histBottom=winH-kInputBarH-kBottomPad;
            float histH=std::max(0.f,histBottom-histTop);
            float wrapW=std::max(10.f,winW-2.f*kSidePad);

            history.ensureWrapped(wrapW,lineText);
            float viewLines=std::max(1.f,histH/lineH);
            size_t wrapCount=history.wrappedCount();
            float maxScroll=(wrapCount>(size_t)viewLines)?(wrapCount-(size_t)viewLines):0.f;
            clampScroll(maxScroll);

            window.clear(sf::Color(18,18,22));
            window.draw(titleBar);
            window.draw(titleText);

            long start=std::max(0L,(long)wrapCount-(long)std::ceil(viewLines)-(long)std::floor(scrollOffsetLines));
            long end=std::min<long>(wrapCount,start+(long)std::ceil(viewLines)+1);
            float y=histTop;

            for(long i=start;i<end;++i){
                if(i<0||i>=(long)history.wrapped().size()) continue;
                auto& wl=history.wrapped()[i];
                lineText.setString(wl.text);
                lineText.setFillColor(wl.color);
                lineText.setPosition(kSidePad,y);
                window.draw(lineText);
                y+=lineH;
                if(y>histBottom) break;
            }

            if(wrapCount>(size_t)viewLines){
                float trackTop=histTop, trackH=histH;
                float thumbH=std::max(20.f,trackH*(viewLines/(float)wrapCount));
                float t=(maxScroll<=0.f)?0.f:(scrollOffsetLines/maxScroll);
                float thumbTop=trackTop+(trackH-thumbH)*t;

                sf::RectangleShape track({4.f,trackH});
                track.setFillColor(sf::Color(50,50,58));
                track.setPosition(winW-6.f,trackTop);

                sf::RectangleShape thumb({4.f,thumbH});
                thumb.setFillColor(sf::Color(120,120,135));
                thumb.setPosition(winW-6.f,thumbTop);

                window.draw(track);
                window.draw(thumb);
            }

            window.draw(inputBar);
            window.draw(inputText);
        } else {
            window.clear(sf::Color(18,18,22));
            window.draw(titleBar);
            window.draw(inputBar);
        }

        window.display();
    } // <-- closes while(window.isOpen())
} // <-- closes int main()

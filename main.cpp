// main.cpp
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

namespace fs = std::filesystem;

// ----------------------- Tunables -----------------------
static float    kTitleBarH      = 48.f;   // banner at the top
static float    kInputBarH      = 44.f;   // chat input height
static float    kSidePad        = 12.f;   // left/right padding
static float    kTopPad         = 6.f;    // top padding under title
static float    kBottomPad      = 6.f;    // bottom padding above input
static float    kLineSpacing    = 1.25f;  // line spacing multiplier
static unsigned kFontSize       = 18;     // history & input font size
static unsigned kTitleFontSize  = 22;     // title font size
static size_t   kMaxHistory     = 1000;   // max raw lines to store
// --------------------------------------------------------

struct WrappedLine {
    std::string text;
    sf::Color color{sf::Color::White};
};

class ConsoleHistory {
public:
    void push(const std::string& line, sf::Color color = sf::Color::White) {
        if (raw_.size() >= kMaxHistory) raw_.pop_front();
        raw_.push_back({line, color});
        dirty_ = true; // needs re-wrap on next draw
    }
    void clear() { raw_.clear(); dirty_ = true; }

    void ensureWrapped(float maxWidth, sf::Text& measurer) {
        if (!dirty_ && lastWrapWidth_ == maxWidth && lastFontSize_ == measurer.getCharacterSize())
            return;

        wrapped_.clear();
        for (const auto& ln : raw_) {
            wrapLine(ln, maxWidth, measurer, wrapped_);
        }
        dirty_ = false;
        lastWrapWidth_ = maxWidth;
        lastFontSize_  = measurer.getCharacterSize();
    }

    const std::vector<WrappedLine>& wrapped() const { return wrapped_; }
    size_t wrappedCount() const { return wrapped_.size(); }

private:
    static void wrapLine(const WrappedLine& ln, float maxW, sf::Text& meas, std::vector<WrappedLine>& out) {
        if (ln.text.empty()) { out.push_back({"", ln.color}); return; }

        std::string word, current;
        std::istringstream iss(ln.text);

        auto flush = [&](bool force=false){
            if (force || !current.empty()) {
                out.push_back({current, ln.color});
                current.clear();
            }
        };

        while (iss >> word) {
            std::string test = current.empty() ? word : current + " " + word;
            meas.setString(test);
            float w = meas.getLocalBounds().width;
            if (w <= maxW) {
                current = std::move(test);
            } else {
                if (current.empty()) {
                    // very long single word: hard-wrap by chars
                    std::string accum;
                    for (char c : word) {
                        std::string test2 = accum + c;
                        meas.setString(test2);
                        if (meas.getLocalBounds().width <= maxW) {
                            accum.push_back(c);
                        } else {
                            if (!accum.empty()) out.push_back({accum, ln.color});
                            accum = std::string(1, c);
                        }
                    }
                    if (!accum.empty()) current = accum;
                } else {
                    out.push_back({current, ln.color});
                    current = word;
                }
            }
        }
        flush(true);
    }

    bool dirty_ = true;
    float lastWrapWidth_ = -1.f;
    unsigned lastFontSize_ = 0;
    std::deque<WrappedLine> raw_;
    std::vector<WrappedLine> wrapped_;
};

static std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}
static std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> out; std::istringstream iss(line); std::string tok;
    while (iss >> tok) out.push_back(tok);
    return out;
}
static fs::path resolvePath(const fs::path& currentDir, const std::string& userPath) {
    fs::path p(userPath);
    std::error_code ec;
    if (p.is_absolute()) return fs::weakly_canonical(p, ec);
    return fs::weakly_canonical(currentDir / p, ec);
}
static std::string findFontPath(int argc, char** argv) {
    std::error_code ec;
    fs::path exeDir = (argc > 0) ? fs::path(argv[0]).parent_path() : fs::current_path(ec);
    std::vector<fs::path> candidates = {
        exeDir / "DejaVuSans.ttf",
        exeDir / "arial.ttf",
        "DejaVuSans.ttf",
        "arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeui.ttf"
    };
    for (auto& p : candidates) {
        if (fs::exists(p, ec) && !fs::is_directory(p, ec) && fs::file_size(p, ec) > 1024) return p.string();
    }
    return {};
}

int main(int argc, char** argv) {
    // --- NLP load ---
    NLP nlp;
    {
        std::error_code ec;
        fs::path exeDir = (argc > 0) ? fs::path(argv[0]).parent_path() : fs::current_path(ec);
        fs::path rulesPathExe = exeDir / "nlp_rules.json";
        fs::path rulesPathCwd = "nlp_rules.json";
        std::string err;
        bool ok = fs::exists(rulesPathExe, ec) ? nlp.load_rules(rulesPathExe.string(), &err)
                                               : nlp.load_rules(rulesPathCwd.string(), &err);
        if (!ok) std::cerr << "[NLP] Failed to load rules: " << err << "\n";
        else {
            std::cerr << "[NLP] Loaded rules. Intents:";
            for (auto& s : nlp.list_intents()) std::cerr << " " << s;
            std::cerr << "\n";
        }
    }

    // --- Window ---
    sf::RenderWindow window(sf::VideoMode(500, 800), "GRIM");
    window.setVerticalSyncEnabled(true);

    // --- Font ---
    sf::Font font;
    const std::string fontPath = findFontPath(argc, argv);
    if (fontPath.empty() || !font.loadFromFile(fontPath)) {
        std::cerr << "[WARN] Could not load a font. Place a TTF (e.g., arial.ttf) next to the exe.\n";
    } else {
        std::cerr << "[INFO] Using font: " << fontPath << "\n";
    }

    // --- Geometry / UI elements (sizes/positions updated each frame) ---
    sf::RectangleShape titleBar;
    titleBar.setFillColor(sf::Color(26, 26, 30));

    sf::Text titleText;
    if (font.getInfo().family != "") {
        titleText.setFont(font);
        titleText.setCharacterSize(kTitleFontSize);
        titleText.setFillColor(sf::Color(220, 220, 235));
        titleText.setString("G R I M");
    }

    sf::RectangleShape inputBar;
    inputBar.setFillColor(sf::Color(30, 30, 35));

    sf::Text inputText, lineText;
    if (font.getInfo().family != "") {
        inputText.setFont(font);
        inputText.setCharacterSize(kFontSize);
        inputText.setFillColor(sf::Color::White);

        lineText.setFont(font);
        lineText.setCharacterSize(kFontSize);
        lineText.setFillColor(sf::Color::White);
    }

    // History model
    ConsoleHistory history;
    auto addHistory = [&](const std::string& s, sf::Color c = sf::Color::White){
        history.push(s, c);
        std::cout << s << "\n";
    };

    std::string buffer;
    fs::path currentDir = fs::current_path();
    addHistory("GRIM is ready. Type 'help' for commands. Type 'quit' to exit.", sf::Color(160, 200, 255));

    // caret + scrolling
    sf::Clock caretClock; bool caretVisible = true;
    float scrollOffsetLines = 0.f; // 0 = bottom (latest). Positive = scrolled up.

    auto clampScroll = [&](float maxLines){
        if (scrollOffsetLines < 0) scrollOffsetLines = 0;
        if (scrollOffsetLines > maxLines) scrollOffsetLines = maxLines;
    };

    while (window.isOpen()) {
        int winx = window.getSize().x;
        int winy = window.getSize().y;
        // Poll events (no manual size tracking needed)
        sf::Event e;
        while (window.pollEvent(e)) {
            if (e.type == sf::Event::Closed) window.close();

            if (e.type == sf::Event::KeyPressed) {
                if (e.key.code == sf::Keyboard::Escape) window.close();
                if (e.key.code == sf::Keyboard::PageUp)   { scrollOffsetLines += 10.f; }
                if (e.key.code == sf::Keyboard::PageDown) { scrollOffsetLines -= 10.f; if (scrollOffsetLines < 0.f) scrollOffsetLines = 0.f; }
                if (e.key.code == sf::Keyboard::Home)     { scrollOffsetLines = 1e6f; } // clamp later
                if (e.key.code == sf::Keyboard::End)      { scrollOffsetLines = 0.f; }
            }

            if (e.type == sf::Event::MouseWheelScrolled) {
                if (e.mouseWheelScroll.wheel == sf::Mouse::VerticalWheel) {
                    scrollOffsetLines += (e.mouseWheelScroll.delta > 0 ? 3.f : -3.f);
                    if (scrollOffsetLines < 0.f) scrollOffsetLines = 0.f;
                }
            }

            if (e.type == sf::Event::TextEntered) {
                if (e.text.unicode == 8) { // backspace
                    if (!buffer.empty()) buffer.pop_back();
                } else if (e.text.unicode == 13 || e.text.unicode == 10) { // Enter
                    std::string line = trim(buffer); buffer.clear();

                    if (line == "quit" || line == "exit") { window.close(); break; }
                    if (!line.empty()) addHistory("> " + line, sf::Color(150, 255, 150));
                    else { addHistory("> "); continue; }

                    // Commands
                    auto args = split(line);
                    const std::string cmd = args.empty() ? "" : args[0];

                    if (cmd == "help") {
                        addHistory("Commands:", sf::Color(255, 210, 120));
                        addHistory("  help                      - show this help");
                        addHistory("  pwd                       - print current directory");
                        addHistory("  cd <path>                 - change directory");
                        addHistory("  list                      - list items in current directory");
                        addHistory("  mkdir <path>              - create directory");
                        addHistory("  rm <path>                 - remove file or empty directory");
                        addHistory("  reloadnlp                 - reload nlp_rules.json");
                        addHistory("  grim <prompt>               - ask local model via Ollama");
                        addHistory("Try: 'open notepad', 'search cats', 'timer for 3 min'");
                        scrollOffsetLines = 0.f;
                        continue;
                    }
                    if (cmd == "pwd") { addHistory(currentDir.string()); scrollOffsetLines = 0.f; continue; }
                    if (cmd == "cd") {
                        if (args.size() < 2) { addHistory("Error: cd requires a path."); scrollOffsetLines = 0.f; continue; }
                        fs::path target = resolvePath(currentDir, args[1]);
                        std::error_code ec;
                        if (!fs::exists(target, ec)) { addHistory("Error: path does not exist."); scrollOffsetLines = 0.f; continue; }
                        if (!fs::is_directory(target, ec)) { addHistory("Error: not a directory."); scrollOffsetLines = 0.f; continue; }
                        currentDir = target;
                        addHistory("Directory changed to: " + currentDir.string());
                        scrollOffsetLines = 0.f;
                        continue;
                    }
                    if (cmd == "list") {
                        std::error_code ec;
                        if (!fs::exists(currentDir, ec) || !fs::is_directory(currentDir, ec)) {
                            addHistory("Error: current directory invalid."); scrollOffsetLines = 0.f; continue;
                        }
                        for (const auto& entry : fs::directory_iterator(currentDir, ec)) {
                            if (ec) break;
                            bool isDir = entry.is_directory(ec);
                            addHistory(std::string(isDir ? "[D] " : "    ") + entry.path().filename().string());
                        }
                        scrollOffsetLines = 0.f;
                        continue;
                    }
                    if (cmd == "mkdir") {
                        if (args.size() < 2) { addHistory("Error: mkdir requires <path>."); scrollOffsetLines = 0.f; continue; }
                        fs::path p = resolvePath(currentDir, args[1]);
                        std::error_code ec; fs::create_directories(p, ec);
                        if (ec) addHistory("Error creating directory: " + ec.message());
                        else addHistory("Directory created.");
                        scrollOffsetLines = 0.f;
                        continue;
                    }
                    if (cmd == "rm") {
                        if (args.size() < 2) { addHistory("Error: rm requires <path>."); scrollOffsetLines = 0.f; continue; }
                        fs::path p = resolvePath(currentDir, args[1]);
                        std::error_code ec;
                        if (!fs::exists(p, ec)) { addHistory("Error: path does not exist."); scrollOffsetLines = 0.f; continue; }
                        bool ok = fs::remove(p, ec);
                        if (ec) addHistory("Error removing: " + ec.message());
                        else if (!ok) addHistory("Error: could not remove (directory may not be empty).");
                        else addHistory("Removed.");
                        scrollOffsetLines = 0.f;
                        continue;
                    }
                    if (cmd == "reloadnlp") {
                        std::error_code ec;
                        fs::path exeDir = (argc > 0) ? fs::path(argv[0]).parent_path() : fs::current_path(ec);
                        fs::path rulesPathExe = exeDir / "nlp_rules.json";
                        fs::path rulesPathCwd = "nlp_rules.json";
                        std::string err;
                        bool ok = fs::exists(rulesPathExe, ec) ? nlp.load_rules(rulesPathExe.string(), &err)
                                                               : nlp.load_rules(rulesPathCwd.string(), &err);
                        if (!ok) addHistory("Failed to reload nlp_rules.json: " + err, sf::Color(255,140,140));
                        else addHistory("NLP rules reloaded.", sf::Color(140,255,180));
                        scrollOffsetLines = 0.f;
                        continue;
                    }
                    if (cmd == "grim") {
                        std::string query = (line.size() > 3) ? trim(line.substr(3)) : "";
                        if (query.empty()) { addHistory("Usage: grim <your question>"); scrollOffsetLines = 0.f; continue; }
                        try {
                            std::string answer = callAI(query);
                            if (answer.empty()) answer = "[grim] (empty response)";
                            std::istringstream iss(answer);
                            std::string l;
                            while (std::getline(iss, l)) addHistory(l, sf::Color(200, 220, 255));
                        } catch (const std::exception& ex) {
                            addHistory(std::string("[grim] Error: ") + ex.what(), sf::Color(255,140,140));
                        }
                        scrollOffsetLines = 0.f;
                        continue;
                    }

                    // NLP fallback
                    {
                        Intent intent = nlp.parse(line);
                        if (!intent.matched) {
                            addHistory("[NLP] No intent matched.", sf::Color(255, 200, 140));
                        } else {
                            std::ostringstream oss;
                            oss << "[NLP] intent=" << intent.name << " score=" << intent.score;
                            addHistory(oss.str(), sf::Color(180, 255, 180));
                            for (const auto& kv : intent.slots) addHistory("  " + kv.first + " = " + kv.second);

                            if (intent.name == "open_app") {
                                auto it = intent.slots.find("app");
                                if (it != intent.slots.end()) addHistory("Would open app: " + it->second);
                            } else if (intent.name == "search_web") {
                                auto it = intent.slots.find("query");
                                if (it != intent.slots.end()) addHistory("Would search web for: " + it->second);
                            } else if (intent.name == "set_timer") {
                                auto it = intent.slots.find("minutes");
                                if (it != intent.slots.end()) addHistory("Would set timer for " + it->second + " minute(s).");
                            }
                        }
                        scrollOffsetLines = 0.f;
                    }
                } else {
                    if (e.text.unicode >= 32 && e.text.unicode < 127) {
                        buffer.push_back(static_cast<char>(e.text.unicode));
                    }
                }
            }
        }

        // caret blink
        if (caretClock.getElapsedTime().asSeconds() > 0.5f) {
            caretVisible = !caretVisible;
            caretClock.restart();
        }

        // --------- Layout from live window size ----------
        sf::Vector2u ws = window.getSize();
        float winW = getwindowsize().x;
        float winH = getwindowsize().y;

        // Bars sizes/positions
        titleBar.setSize({winx, kTitleBarH});
        titleBar.setPosition(0.f, 0.f);

        inputBar.setSize({winy, kInputBarH});
        inputBar.setPosition(0.f, winH - kInputBarH);

        // Title centered
        if (font.getInfo().family != "") {
            sf::FloatRect tb = titleText.getLocalBounds();
            titleText.setPosition((winW - tb.width) * 0.5f, (kTitleBarH - tb.height) * 0.5f - 6.f);
        }

        // Input text
        if (font.getInfo().family != "") {
            std::string toShow = buffer;
            if (caretVisible) toShow.push_back('_');
            inputText.setString(toShow);
            inputText.setPosition(kSidePad, winH - kInputBarH + (kInputBarH - (float)kFontSize) * 0.5f - 2.f);
        }

        // Wrapping & scrolling
        if (font.getInfo().family != "") {
            lineText.setCharacterSize(kFontSize);

            float lineHeightPx = kLineSpacing * (float)kFontSize;
            float histTop = kTitleBarH + kTopPad;
            float histBottom = winH - kInputBarH - kBottomPad;
            float histHeight = std::max(0.f, histBottom - histTop);
            float wrapWidth = std::max(10.f, winW - 2.f * kSidePad);

            history.ensureWrapped(wrapWidth, lineText);

            float viewLinesF = std::max(1.f, histHeight / lineHeightPx);
            size_t wrappedCount = history.wrappedCount();

            float maxScroll = (wrappedCount > (size_t)viewLinesF) ? (wrappedCount - (size_t)viewLinesF) : 0.f;
            clampScroll(maxScroll);

            // Render
            window.clear(sf::Color(18, 18, 22));
            window.draw(titleBar);
            window.draw(titleText);

            // draw wrapped lines within viewport from the bottom (latest)
            long start = std::max(0L, (long)wrappedCount - (long)std::ceil(viewLinesF) - (long)std::floor(scrollOffsetLines));
            long end   = std::min<long>(wrappedCount, start + (long)std::ceil(viewLinesF) + 1);

            float y = histTop;
            for (long i = start; i < end; ++i) {
                if (i < 0 || i >= (long)history.wrapped().size()) continue;
                const auto& wl = history.wrapped()[i];
                lineText.setString(wl.text);
                lineText.setFillColor(wl.color);
                lineText.setPosition(kSidePad, y);
                window.draw(lineText);
                y += lineHeightPx;
                if (y > histBottom) break;
            }

            // minimal scrollbar
            if (wrappedCount > (size_t)viewLinesF) {
                float trackTop = histTop;
                float trackH   = histHeight;
                float thumbH   = std::max(20.f, trackH * (viewLinesF / (float)wrappedCount));
                float t = (maxScroll <= 0.f) ? 0.f : (scrollOffsetLines / maxScroll); // 0..1
                float thumbTop = trackTop + (trackH - thumbH) * t;

                sf::RectangleShape track({4.f, trackH});
                track.setFillColor(sf::Color(50,50,58));
                track.setPosition(winW - 6.f, trackTop);

                sf::RectangleShape thumb({4.f, thumbH});
                thumb.setFillColor(sf::Color(120,120,135));
                thumb.setPosition(winW - 6.f, thumbTop);

                window.draw(track);
                window.draw(thumb);
            }

            // input bar + text
            window.draw(inputBar);
            window.draw(inputText);
        } else {
            // No font: still draw bars to indicate layout
            window.clear(sf::Color(18, 18, 22));
            window.draw(titleBar);
            window.draw(inputBar);
        }

        window.display();
    }

    return 0;
}

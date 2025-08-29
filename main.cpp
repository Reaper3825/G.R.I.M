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

// ---------- NEW: Find any font in resources ----------
static std::string findAnyFontInResources(int argc, char** argv) {
    std::error_code ec;

    auto has_ttf = [&](const fs::path& dir) -> std::string {
        if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) return {};
        for (auto& e : fs::directory_iterator(dir, ec)) {
            if (ec) break;
            if (!e.is_regular_file(ec)) continue;
            auto p = e.path();
            auto ext = p.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".ttf") {
                return p.string(); // first font we find
            }
        }
        return {};
    };

    // 1) resources next to the executable
    fs::path exeDir = (argc > 0) ? fs::path(argv[0]).parent_path() : fs::current_path(ec);
    if (auto s = has_ttf(exeDir / "resources"); !s.empty()) return s;

    // 2) resources relative to current working directory
    if (auto s = has_ttf(fs::current_path(ec) / "resources"); !s.empty()) return s;

#ifdef _WIN32
    // 3) Windows fallbacks
    for (const char* f : {
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeui.ttf"
    }) {
        if (fs::exists(f, ec)) return std::string(f);
    }
#endif

    return {};
}
// --------------------------------------------------------

class ConsoleHistory {
public:
    void push(const std::string& line, sf::Color color = sf::Color::White) {
        if (raw_.size() >= kMaxHistory) raw_.pop_front();
        raw_.push_back({line, color});
        dirty_ = true;
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
    std::string fontPath = findAnyFontInResources(argc, argv);
    if (fontPath.empty() || !font.loadFromFile(fontPath)) {
        std::cerr << "[WARN] Could not load a font. Put a .ttf inside resources/.\n";
    } else {
        std::cerr << "[INFO] Using font: " << fontPath << "\n";
    }

    // --- Geometry / UI elements ---
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

    ConsoleHistory history;
    auto addHistory = [&](const std::string& s, sf::Color c = sf::Color::White){
        history.push(s, c);
        std::cout << s << "\n";
    };

    std::string buffer;
    fs::path currentDir = fs::current_path();
    addHistory("GRIM is ready. Type 'help' for commands. Type 'quit' to exit.", sf::Color(160, 200, 255));

    sf::Clock caretClock; bool caretVisible = true;
    float scrollOffsetLines = 0.f;

    auto clampScroll = [&](float maxLines){
        if (scrollOffsetLines < 0) scrollOffsetLines = 0;
        if (scrollOffsetLines > maxLines) scrollOffsetLines = maxLines;
    };

    // ---------------- MAIN LOOP -----------------
    while (window.isOpen()) {
        sf::Event e;
        while (window.pollEvent(e)) {
            if (e.type == sf::Event::Closed) window.close();

            if (e.type == sf::Event::KeyPressed) {
                if (e.key.code == sf::Keyboard::Escape) window.close();
                if (e.key.code == sf::Keyboard::PageUp)   { scrollOffsetLines += 10.f; }
                if (e.key.code == sf::Keyboard::PageDown) { scrollOffsetLines -= 10.f; if (scrollOffsetLines < 0.f) scrollOffsetLines = 0.f; }
                if (e.key.code == sf::Keyboard::Home)     { scrollOffsetLines = 1e6f; }
                if (e.key.code == sf::Keyboard::End)      { scrollOffsetLines = 0.f; }
            }

            if (e.type == sf::Event::MouseWheelScrolled) {
                if (e.mouseWheelScroll.wheel == sf::Mouse::VerticalWheel) {
                    scrollOffsetLines += (e.mouseWheelScroll.delta > 0 ? 3.f : -3.f);
                    if (scrollOffsetLines < 0.f) scrollOffsetLines = 0.f;
                }
            }

            if (e.type == sf::Event::TextEntered) {
                if (e.text.unicode == 8) {
                    if (!buffer.empty()) buffer.pop_back();
                } else if (e.text.unicode == 13 || e.text.unicode == 10) {
                    std::string line = trim(buffer); buffer.clear();
                    if (line == "quit" || line == "exit") { window.close(); break; }
                    if (!line.empty()) addHistory("> " + line, sf::Color(150, 255, 150));
                    else { addHistory("> "); continue; }

                    // (your existing commands + NLP handling stay the same)
                } else {
                    if (e.text.unicode >= 32 && e.text.unicode < 127) {
                        buffer.push_back(static_cast<char>(e.text.unicode));
                    }
                }
            }
        }

        if (caretClock.getElapsedTime().asSeconds() > 0.5f) {
            caretVisible = !caretVisible;
            caretClock.restart();
        }

        // -------- Layout --------
        sf::Vector2u ws = window.getSize();
        float winW = ws.x;
        float winH = ws.y;

        titleBar.setSize({winW, kTitleBarH});
        titleBar.setPosition(0.f, 0.f);

        inputBar.setSize({winW, kInputBarH});
        inputBar.setPosition(0.f, winH - kInputBarH);

        if (font.getInfo().family != "") {
            sf::FloatRect tb = titleText.getLocalBounds();
            titleText.setPosition((winW - tb.width) * 0.5f, (kTitleBarH - tb.height) * 0.5f - 6.f);
        }

        if (font.getInfo().family != "") {
            std::string toShow = buffer;
            if (caretVisible) toShow.push_back('_');
            inputText.setString(toShow);
            inputText.setPosition(kSidePad, winH - kInputBarH + (kInputBarH - (float)kFontSize) * 0.5f - 2.f);
        }

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

            window.clear(sf::Color(18, 18, 22));
            window.draw(titleBar);
            window.draw(titleText);

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

            if (wrappedCount > (size_t)viewLinesF) {
                float trackTop = histTop;
                float trackH   = histHeight;
                float thumbH   = std::max(20.f, trackH * (viewLinesF / (float)wrappedCount));
                float t = (maxScroll <= 0.f) ? 0.f : (scrollOffsetLines / maxScroll);
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

            window.draw(inputBar);
            window.draw(inputText);
        } else {
            window.clear(sf::Color(18, 18, 22));
            window.draw(titleBar);
            window.draw(inputBar);
        }

        window.display();
    }

    return 0;
}

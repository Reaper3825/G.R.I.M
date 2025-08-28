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
static float    kTitleBarH      = 48.f;
static float    kInputBarH      = 44.f;
static float    kSidePad        = 12.f;
static float    kTopPad         = 6.f;
static float    kBottomPad      = 6.f;
static float    kLineSpacing    = 1.25f;
static unsigned kFontSize       = 18;
static unsigned kTitleFontSize  = 22;
static size_t   kMaxHistory     = 1000;
// --------------------------------------------------------

struct WrappedLine {
    std::string text;
    sf::Color color{sf::Color::White};
};

// ---------- NEW: Cross-platform resource path ----------
static std::string resourcePath(const std::string& file, int argc, char** argv) {
    std::error_code ec;
    fs::path exeDir = (argc > 0) ? fs::path(argv[0]).parent_path() : fs::current_path(ec);

    // look inside resources/ first
    fs::path candidate = exeDir / "resources" / file;
    if (fs::exists(candidate, ec) && !fs::is_directory(candidate, ec)) {
        return candidate.string();
    }

    // fallback: relative "resources/file"
    candidate = fs::current_path(ec) / "resources" / file;
    if (fs::exists(candidate, ec) && !fs::is_directory(candidate, ec)) {
        return candidate.string();
    }

#ifdef _WIN32
    // last resort: Windows Fonts
    std::vector<std::string> winFonts = {
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeui.ttf"
    };
    for (auto& f : winFonts) {
        if (fs::exists(f, ec)) return f;
    }
#endif

    return {}; // nothing found
}
// -------------------------------------------------------

class ConsoleHistory {
    // … unchanged …
    // (keep all your existing ConsoleHistory code here)
};

// … trim(), split(), resolvePath() remain unchanged …
// remove your old findFontPath() because we now use resourcePath()

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
    std::string fontPath = resourcePath("Arial.ttf", argc, argv); // pick your actual .ttf inside resources/
    if (fontPath.empty() || !font.loadFromFile(fontPath)) {
        std::cerr << "[WARN] Could not load a font. Place a TTF inside resources/.\n";
    } else {
        std::cerr << "[INFO] Using font: " << fontPath << "\n";
    }

    // … everything else in your main() stays the same …
}

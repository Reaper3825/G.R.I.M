#include "pch.hpp"
#include "input_parser.hpp"
#include "commands_core.hpp"     // for CommandResult / CommandFunc / commandMap
#include "synonyms.hpp"
#include "logger.hpp"
#include "ui/ui_root.hpp"        // ✅ NEW: For checking UI visibility
#include "popup_ui/popup_ui.hpp" // ✅ NEW: For checking popup visibility
#include "platform_input.hpp"    // ✅ NEW: Cross-platform input abstraction
#include "platform_clipboard.hpp" // ✅ NEW: Cross-platform clipboard abstraction

#include <algorithm>
#include <cctype>
#include <vector>
#include <sstream>
#include <unordered_map>
#include "grim_platform.h"

namespace {

const char* boolText(bool value)
{
    if (value) return "true";
    return "false";
}

bool shouldLogKeyDebugVK(int vk)
{
    switch (vk) {
        case VK_LSHIFT:
        case VK_RSHIFT:
        case VK_LCONTROL:
        case VK_RCONTROL:
        case VK_LMENU:
        case VK_RMENU:
        case VK_OEM_3:
            return true;
        default:
            return false;
    }
}

std::string virtualKeyDebugName(int vk)
{
    switch (vk) {
        case VK_LSHIFT: return "VK_LSHIFT";
        case VK_RSHIFT: return "VK_RSHIFT";
        case VK_LCONTROL: return "VK_LCONTROL";
        case VK_RCONTROL: return "VK_RCONTROL";
        case VK_LMENU: return "VK_LMENU";
        case VK_RMENU: return "VK_RMENU";
        case VK_OEM_3: return "VK_OEM_3";
        default:
            return std::string("VK(") + std::to_string(vk) + ")";
    }
}

void logInputTransition(const char* source,
                        int vk,
                        bool down,
                        bool pressed,
                        bool released,
                        bool ctrl,
                        bool shift,
                        bool alt)
{
    if (!shouldLogKeyDebugVK(vk)) {
        return;
    }

    LOG_DEBUG("InputState",
        std::string("source=") + source +
        " vk=" + std::to_string(vk) +
        " (" + virtualKeyDebugName(vk) + ")" +
        " down=" + boolText(down) +
        " pressed=" + boolText(pressed) +
        " released=" + boolText(released) +
        " ctrl=" + boolText(ctrl) +
        " shift=" + boolText(shift) +
        " alt=" + boolText(alt));
}

} // namespace

// ====================================================
// Check if mouse input should be processed by UI
// ====================================================
bool InputState::shouldProcessMouseInput()
{
    // ✅ ALWAYS RETURN TRUE - Input should always be captured
    // UI widgets will check their own bounds to decide if they respond
    return true;
}

// ====================================================
// InputState implementation
// ====================================================
InputState InputState::capture()
{
    InputState state{};
    state.mouseInputEnabled = true;

    static std::array<bool, 256> prevKeyDown{};

    int x = 0, y = 0;
    PlatformInput::getCursorPos(x, y);
    state.mousePos.x = static_cast<float>(x);
    state.mousePos.y = static_cast<float>(y);

    for (int i = 0; i < 3; ++i)
    {
        state.mouseDown[i] = PlatformInput::isMouseButtonDown(i);
    }

    state.ctrl = PlatformInput::isKeyDown(static_cast<int>(PlatformInput::Key::Control))
                 || PlatformInput::isCommandDown();
    state.shift = PlatformInput::isKeyDown(static_cast<int>(PlatformInput::Key::Shift));
    state.alt = PlatformInput::isKeyDown(static_cast<int>(PlatformInput::Key::Alt));

    for (int i = 0; i < 256; ++i)
    {
        bool down = PlatformInput::isKeyDown(i);
        state.keysDown[i] = down;

        if (down && !prevKeyDown[i])
        {
            state.keyPressed[i] = true;
        }
        else if (!down && prevKeyDown[i])
        {
            state.keyReleased[i] = true;
        }

        prevKeyDown[i] = down;
    }

    for (int i = 0; i < 256; ++i)
    {
        if (state.keyPressed[i] || state.keyReleased[i])
        {
            logInputTransition("capture",
                               i,
                               state.keysDown[i],
                               state.keyPressed[i],
                               state.keyReleased[i],
                               state.ctrl,
                               state.shift,
                               state.alt);
        }
    }

    return state;
}

void InputState::captureFromHWND(HWND hwnd)
{
    // Static variables to persist across frames
    static bool prevMouseDown[3] = {false, false, false};
    static std::array<bool, 256> prevKeyDown{};
    static Vec2 prevMousePos{0.0f, 0.0f};
    
    mouseInputEnabled = true;
    mouseWheelDelta = 0.0f;
    
    // Reset per-frame data
    textInput.clear();
    pastedText.clear();
    copyRequested = false;
    pasteRequested = false;
    cutRequested = false;
    
    for (int i = 0; i < 3; ++i)
    {
        mousePressed[i] = false;
        mouseReleased[i] = false;
    }
    keyPressed.fill(false);
    keyReleased.fill(false);
    
    int relX = 0;
    int relY = 0;
    if (hwnd)
    {
        PlatformInput::getCursorPosRelative(reinterpret_cast<void*>(hwnd), relX, relY);
    }
    else
    {
        PlatformInput::getCursorPos(relX, relY);
    }

    mousePos.x = static_cast<float>(relX);
    mousePos.y = static_cast<float>(relY);
    mouseDelta.x = mousePos.x - prevMousePos.x;
    mouseDelta.y = mousePos.y - prevMousePos.y;
    prevMousePos = mousePos;

    // Keyboard state transitions
    // On Windows, use GetKeyboardState (1 kernel call) instead of
    // 256 × GetAsyncKeyState (256 kernel calls). Also read mouse
    // buttons and modifiers from the same state snapshot.
#ifdef _WIN32
    BYTE keyboardState[256]{};
    GetKeyboardState(keyboardState);

    // Mouse buttons from same snapshot (avoids 3 extra GetAsyncKeyState calls)
    {
        bool lDown = (keyboardState[VK_LBUTTON] & 0x80) != 0;
        bool rDown = (keyboardState[VK_RBUTTON] & 0x80) != 0;
        bool mDown = (keyboardState[VK_MBUTTON] & 0x80) != 0;
        bool btns[3] = {lDown, rDown, mDown};
        for (int i = 0; i < 3; ++i) {
            if (btns[i] && !prevMouseDown[i]) mousePressed[i] = true;
            if (!btns[i] && prevMouseDown[i]) mouseReleased[i] = true;
            mouseDown[i] = btns[i];
            prevMouseDown[i] = btns[i];
        }
    }

    for (int i = 0; i < 256; ++i)
    {
        bool down = (keyboardState[i] & 0x80) != 0;

        if (down && !prevKeyDown[i]) keyPressed[i] = true;
        if (!down && prevKeyDown[i]) keyReleased[i] = true;

        keysDown[i] = down;
        prevKeyDown[i] = down;
    }

    // Modifiers from same snapshot
    ctrl = (keyboardState[VK_CONTROL] & 0x80) != 0;
    shift = (keyboardState[VK_SHIFT] & 0x80) != 0;
    alt = (keyboardState[VK_MENU] & 0x80) != 0;
#else
    for (int i = 0; i < 3; ++i)
    {
        bool down = PlatformInput::isMouseButtonDown(i);
        if (down && !prevMouseDown[i]) mousePressed[i] = true;
        if (!down && prevMouseDown[i]) mouseReleased[i] = true;
        mouseDown[i] = down;
        prevMouseDown[i] = down;
    }

    for (int i = 0; i < 256; ++i)
    {
        bool down = PlatformInput::isKeyDown(i);

        if (down && !prevKeyDown[i]) keyPressed[i] = true;
        if (!down && prevKeyDown[i]) keyReleased[i] = true;

        keysDown[i] = down;
        prevKeyDown[i] = down;
    }

    ctrl = PlatformInput::isKeyDown(static_cast<int>(PlatformInput::Key::Control))
           || PlatformInput::isCommandDown();
    shift = PlatformInput::isKeyDown(static_cast<int>(PlatformInput::Key::Shift));
    alt = PlatformInput::isKeyDown(static_cast<int>(PlatformInput::Key::Alt));
#endif

    for (int i = 0; i < 256; ++i)
    {
        if (keyPressed[i] || keyReleased[i])
        {
            logInputTransition("captureFromHWND",
                               i,
                               keysDown[i],
                               keyPressed[i],
                               keyReleased[i],
                               ctrl,
                               shift,
                               alt);
        }
    }
    
    // Detect clipboard shortcuts (Ctrl+C / Cmd+C, Ctrl+V / Cmd+V, Ctrl+X / Cmd+X)
    if (ctrl && !shift && !alt) {
        // Ctrl+C - Copy
        if (keyPressed['C']) {
            copyRequested = true;
        }
        // Ctrl+V - Paste
        else if (keyPressed['V']) {
            pasteRequested = true;
            pastedText = PlatformClipboard::getText();
        }
        // Ctrl+X - Cut
        else if (keyPressed['X']) {
            cutRequested = true;
        }
    }
}

void InputState::resetFrameState()
{
    textInput.clear();
    pastedText.clear();
    copyRequested = false;
    pasteRequested = false;
    cutRequested = false;
    
    for (int i = 0; i < 3; ++i)
    {
        mousePressed[i] = false;
        mouseReleased[i] = false;
    }
    
    keyPressed.fill(false);
    keyReleased.fill(false);
}

namespace GRIMInput
{
    // ====================================================
    // Helper: Levenshtein distance (edit distance)
    // ====================================================
    static int levenshteinDistance(const std::string& s1, const std::string& s2)
    {
        const size_t m = s1.size(), n = s2.size();
        std::vector<int> prev(n + 1), curr(n + 1);
        for (size_t j = 0; j <= n; ++j)
            prev[j] = static_cast<int>(j);

        for (size_t i = 1; i <= m; ++i)
        {
            curr[0] = static_cast<int>(i);
            for (size_t j = 1; j <= n; ++j)
            {
                int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
                curr[j] = (std::min)({ prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost });
            }
            prev.swap(curr);
        }
        return prev[n];
    }

    // ====================================================
    // Helper: Fuzzy match to known commands
    // ====================================================
    static std::string fuzzyMatch(const std::string& input)
    {
        std::string best = input;
        int bestDist = 2; // tolerate minor typos

        for (const auto& [key, _] : commandMap)
        {
            int dist = levenshteinDistance(input, key);
            if (dist < bestDist)
            {
                bestDist = dist;
                best = key;
            }
        }
        return best;
    }

    // ====================================================
    // Split a user line into command + argument
    // ====================================================
    std::pair<std::string, std::string> parseInput(const std::string& input)
    {
        std::string line = input;
        if (line.empty())
            return { "", "" };

        // trim both ends
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        line.erase(line.find_last_not_of(" \t\n\r") + 1);

        // ✅ NEW: Normalize common multi-word commands to single words
        // This handles cases like "never mind" -> "nevermind"
        std::string lowerLine = line;
        std::transform(lowerLine.begin(), lowerLine.end(), lowerLine.begin(), ::tolower);
        
        const std::unordered_map<std::string, std::string> multiWordCommands = {
            {"never mind", "nevermind"},
            {"no problem", "okay"},
            {"no worries", "okay"},
            {"all right", "okay"},
            {"all good", "okay"}
        };
        
        for (const auto& [multiWord, singleWord] : multiWordCommands) {
            // Check if the line starts with this multi-word command
            if (lowerLine == multiWord || lowerLine.find(multiWord + " ") == 0) {
                // Replace the multi-word command with the single-word version
                line = singleWord + line.substr(multiWord.length());
                break;
            }
        }

        auto pos = line.find(' ');
        if (pos == std::string::npos)
            return { line, "" };
        return { line.substr(0, pos), line.substr(pos + 1) };
    }

    // ====================================================
    // Split input by commas for multi-command support
    // ====================================================
    std::vector<std::string> splitCommands(const std::string& input)
    {
        std::vector<std::string> commands;
        std::string current;
        
        for (char c : input)
        {
            if (c == ',')
            {
                // Trim and add non-empty commands
                current.erase(0, current.find_first_not_of(" \t\n\r"));
                current.erase(current.find_last_not_of(" \t\n\r") + 1);
                
                if (!current.empty())
                    commands.push_back(current);
                
                current.clear();
            }
            else
            {
                current.push_back(c);
            }
        }
        
        // Don't forget the last command
        current.erase(0, current.find_first_not_of(" \t\n\r"));
        current.erase(current.find_last_not_of(" \t\n\r") + 1);
        
        if (!current.empty())
            commands.push_back(current);
        
        return commands;
    }

    // ====================================================
    // Normalize a command: lowercase → synonym → fuzzy
    // ====================================================
    std::string normalizeCommand(const std::string& input)
    {
        if (input.empty())
            return input;

        std::string out = input;
        std::transform(out.begin(), out.end(), out.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        out = normalizeWord(out);   // synonym normalization
        out = fuzzyMatch(out);      // fuzzy correction
        return out;
    }

    // ====================================================
    // Clean argument text (strip symbols, lowercase)
    // ====================================================
    std::string cleanArg(const std::string& arg)
    {
        if (arg.empty())
            return "";

        std::string out;
        out.reserve(arg.size());

        for (char c : arg)
        {
            if (std::isalnum(static_cast<unsigned char>(c)) ||
                std::isspace(static_cast<unsigned char>(c)))
            {
                out.push_back(static_cast<char>(std::tolower(c)));
            }
        }

        // trim
        if (!out.empty())
        {
            out.erase(0, out.find_first_not_of(" \n\r\t"));
            out.erase(out.find_last_not_of(" \n\r\t") + 1);
        }
        return out;
    }

    // ====================================================
    // Normalize an entire line for NLP intent parsing
    // ====================================================
    std::string normalizeLine(const std::string& line)
    {
        std::istringstream iss(line);
        std::ostringstream oss;
        std::string token;

        while (iss >> token)
            oss << normalizeWord(token) << ' ';

        std::string result = oss.str();
        if (!result.empty() && result.back() == ' ')
            result.pop_back();
        return result;
    }
}

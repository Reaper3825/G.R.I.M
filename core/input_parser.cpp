#include "pch.hpp"
#include "input_parser.hpp"
#include "commands_core.hpp"     // for CommandResult / CommandFunc / commandMap
#include "synonyms.hpp"
#include "logger.hpp"
#include "ui/ui_root.hpp"        // ✅ NEW: For checking UI visibility
#include "popup_ui/popup_ui.hpp" // ✅ NEW: For checking popup visibility

#include <algorithm>
#include <cctype>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <windows.h>

// ====================================================
// ✅ NEW: Check if mouse input should be processed
// ====================================================
bool InputState::shouldProcessMouseInput()
{
    // Mouse input is only active when UI panels or popup are visible
    bool popupVisible = isPopupVisible();
    bool uiPanelVisible = false;
    
    // Check if any UI panel is visible
    auto& uiRoot = UIRoot::get();
    POINT cursorPos;
    GetCursorPos(&cursorPos);
    uiPanelVisible = uiRoot.shouldReceiveInputAt(
        static_cast<float>(cursorPos.x), 
        static_cast<float>(cursorPos.y)
    );
    
    // Allow mouse input only if popup or a UI panel is visible
    return popupVisible || uiPanelVisible;
}

// ====================================================
// InputState implementation
// ====================================================
InputState InputState::capture()
{
    InputState state{};
    
    // ✅ NEW: Check if mouse input should be enabled
    state.mouseInputEnabled = shouldProcessMouseInput();
    
    // Capture mouse position
    POINT p{};
    GetCursorPos(&p);
    state.mousePos.x = static_cast<float>(p.x);
    state.mousePos.y = static_cast<float>(p.y);
    
    // ✅ MODIFIED: Only capture mouse buttons if input is enabled
    if (state.mouseInputEnabled)
    {
        state.mouseDown[0] = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
        state.mouseDown[1] = (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0;
        state.mouseDown[2] = (GetAsyncKeyState(VK_MBUTTON) & 0x8000) != 0;
    }
    else
    {
        // Mouse input disabled - report all buttons as up
        state.mouseDown[0] = false;
        state.mouseDown[1] = false;
        state.mouseDown[2] = false;
    }
    
    // Capture modifier keys (always enabled for keyboard shortcuts)
    state.ctrl = (GetAsyncKeyState(VK_CONTROL) & 0x8000) != 0;
    state.shift = (GetAsyncKeyState(VK_SHIFT) & 0x8000) != 0;
    state.alt = (GetAsyncKeyState(VK_MENU) & 0x8000) != 0;
    
    // Capture all keyboard keys
    for (int i = 0; i < 256; ++i)
    {
        bool isDown = (GetAsyncKeyState(i) & 0x8000) != 0;
        state.keysDown[i] = isDown;
    }
    
    return state;
}

void InputState::captureFromHWND(HWND hwnd)
{
    // Static variables to persist across frames
    static bool prevMouseDown[3] = {false, false, false};
    static bool prevMouseInputEnabled = true;
    
    // ✅ Check if mouse input should be enabled
    mouseInputEnabled = shouldProcessMouseInput();
    
    // ✅ FIX: DON'T clear Mouse class state immediately when filtering changes
    // Let it persist for one frame to allow button callbacks to complete
    // Only log the state change
    if (mouseInputEnabled != prevMouseInputEnabled) {
        if (!mouseInputEnabled) {
            LOG_DEBUG("InputState", "Mouse input disabled - will clear on next frame");
        }
        else {
            LOG_DEBUG("InputState", "Mouse input enabled");
        }
        prevMouseInputEnabled = mouseInputEnabled;
    }
    
    // Reset per-frame data
    textInput.clear();
    for (int i = 0; i < 3; ++i)
    {
        mousePressed[i] = false;
        mouseReleased[i] = false;
    }
    keyPressed.clear();
    keyReleased.clear();
    
    // Mouse position relative to window
    POINT p{};
    GetCursorPos(&p);
    ScreenToClient(hwnd, &p);
    
    static POINT lastPos{};
    mouseDelta.x = static_cast<float>(p.x - lastPos.x);
    mouseDelta.y = static_cast<float>(p.y - lastPos.y);
    lastPos = p;
    
    mousePos.x = static_cast<float>(p.x);
    mousePos.y = static_cast<float>(p.y);
    
    // ✅ MODIFIED: Only process mouse buttons if input is enabled
    if (mouseInputEnabled)
    {
        // Mouse buttons - use static prevMouseDown to track transitions
        for (int i = 0; i < 3; ++i)
        {
            int vk = (i == 0) ? VK_LBUTTON : (i == 1) ? VK_RBUTTON : VK_MBUTTON;
            bool down = (GetAsyncKeyState(vk) & 0x8000) != 0;
            
            if (down && !prevMouseDown[i]) mousePressed[i] = true;  // Transition: up -> down
            if (!down && prevMouseDown[i]) mouseReleased[i] = true; // Transition: down -> up
            
            mouseDown[i] = down;
            prevMouseDown[i] = down;
        }
    }
    else
    {
        // ✅ FIX: When input disabled, just copy current mouse state
        // Don't forcefully clear prevMouseDown - let it naturally transition
        for (int i = 0; i < 3; ++i)
        {
            int vk = (i == 0) ? VK_LBUTTON : (i == 1) ? VK_RBUTTON : VK_MBUTTON;
            bool down = (GetAsyncKeyState(vk) & 0x8000) != 0;
            
            // Report current state but don't generate press/release events
            mouseDown[i] = down;
            
            // ✅ FIX: Update prevMouseDown to prevent false events when re-enabling
            if (!down) {
                prevMouseDown[i] = false;
            }
        }
    }
    
    // Keyboard (always enabled)
    for (int i = 0; i < 256; ++i)
    {
        bool down = (GetAsyncKeyState(i) & 0x8000) != 0;
        bool wasDown = keysDown.count(i) && keysDown[i];
        
        if (down && !wasDown) keyPressed[i] = true;
        if (!down && wasDown) keyReleased[i] = true;
        keysDown[i] = down;
    }
    
    ctrl = (GetAsyncKeyState(VK_CONTROL) & 0x8000) != 0;
    shift = (GetAsyncKeyState(VK_SHIFT) & 0x8000) != 0;
    alt = (GetAsyncKeyState(VK_MENU) & 0x8000) != 0;
}

void InputState::resetFrameState()
{
    textInput.clear();
    
    for (int i = 0; i < 3; ++i)
    {
        mousePressed[i] = false;
        mouseReleased[i] = false;
    }
    
    keyPressed.clear();
    keyReleased.clear();
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

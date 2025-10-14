#include "ui_events.hpp"
#include "console_history.hpp"
#include "commands/commands_core.hpp"
#include "logger.hpp"

#include <windows.h>
#include <string>
#include <vector>
#include <filesystem>
#include <nlohmann/json.hpp>

// ============================================================
// Process Win32 events (replaces SFML pollEvent loop)
// ============================================================
bool processEvents(HWND hwnd,
                   std::string& buffer,
                   std::filesystem::path& currentDir,
                   std::vector<Timer>& uiTimers,
                   nlohmann::json& longTermMemory,
                   ConsoleHistory& uiHistory)
{
    MSG msg{};
    while (PeekMessage(&msg, hwnd, 0, 0, PM_REMOVE))
    {
        switch (msg.message)
        {
        case WM_QUIT:
            return false;

        case WM_CLOSE:
            PostQuitMessage(0);
            return false;

        case WM_CHAR:
        {
            if (msg.wParam == VK_RETURN)
            {
                std::string input = buffer;
                buffer.clear();
                if (!input.empty())
                {
                    handleCommand(input);
                    uiHistory.push(input, sf::Color::Green);
                }
                return true;
            }
            else if (msg.wParam == VK_BACK)
            {
                if (!buffer.empty())
                    buffer.pop_back();
                return true;
            }
            else if (msg.wParam == VK_ESCAPE)
            {
                PostQuitMessage(0);
                return false;
            }
            else if (msg.wParam >= 32 && msg.wParam <= 126)
            {
                buffer.push_back(static_cast<char>(msg.wParam));
                return true;
            }
            break;
        }

        case WM_LBUTTONDOWN:
        {
            LOG_DEBUG("UI", "Popup clicked — showing GRIM console");
            std::thread([]() {
                system("start cmd /k \"D:\\G.R.I.M\\out\\build\\Debug\\GRIM.exe\"");
            }).detach();
            break;
        }

        default:
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            break;
        }
    }

    return true;
}

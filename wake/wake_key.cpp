#include "wake_key.hpp"
#include "helpers/key.hpp"
#include "popup_ui/popup_ui.hpp"
#include "pch.hpp"

namespace WakeKey
{
    static std::thread keyThread;
    static std::atomic<bool> running{ false };

    static void onActivationKey(KeyCode)
    {
        LOG_DEBUG("WakeKey", "Wake key pressed — activating popup");
        notifyPopupActivity();
    }

    static void keyLoop()
    {
        Key::initialize();
        Key::onPress(KeyCode::RCtrl, onActivationKey);
        LOG_PHASE("WakeKey system active", true);

        MSG msg{};
        while (running)
        {
            while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
            {
                if (msg.message == WM_QUIT)
                {
                    LOG_DEBUG("WakeKey", "WM_QUIT received - stopping wake key thread");
                    running = false;
                    break;
                }
                // Only process keyboard-related messages, ignore others
                if (msg.message >= WM_KEYFIRST && msg.message <= WM_KEYLAST)
                {
                    TranslateMessage(&msg);
                    DispatchMessage(&msg);
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        Key::shutdown();
        LOG_PHASE("WakeKey system stopped", true);
    }

    void start()
    {
        if (running)
            return;
        running = true;
        keyThread = std::thread(keyLoop);
    }

    void stop()
    {
        if (!running)
            return;
        running = false;
        // Post WM_QUIT to the WakeKey thread, not the current thread
        PostThreadMessage(GetThreadId(keyThread.native_handle()), WM_QUIT, 0, 0);
        if (keyThread.joinable())
            keyThread.join();
    }
}

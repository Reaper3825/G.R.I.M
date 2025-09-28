#include "popup_ui.hpp"
#include "popup_window.hpp"
#include "popup_renderer.hpp"
#include "popup_anim.hpp"
#include "logger.hpp"

#include <SFML/Graphics/RenderTexture.hpp>
#include <SFML/Graphics/Image.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/System/Clock.hpp>
#include <thread>
#include <atomic>
#include <chrono>
#include <windows.h>
#include <cstdint>
#include <cmath>
#include <algorithm>
// Query whether voice playback is active
#include "../voice_speak.hpp"

// ===========================================================
// Globals
// ===========================================================
static std::atomic<bool> g_popupVisible{false};
static std::atomic<bool> g_running{true};
static std::atomic<int>  g_idleTimerMs{0};
static std::atomic<bool> g_pendingPopup{false};
static HWND g_hwnd = nullptr;
static PopupAnimState g_anim;
static sf::Clock g_idleClock; // <<< added: tracks idle timing

// ===========================================================
// Popup UI main loop
// ===========================================================
void runPopupUI(int width, int height) {
    // The overlay window is produced as a small HUD. Force a 128x128 render
    // size here so our SFML render target matches the HWND created by
    // createOverlayWindow (which positions a 128x128 window bottom-right).
    constexpr unsigned overlayW = 128u;
    constexpr unsigned overlayH = 128u;

    // Request creation of the overlay window using the overlay size so
    // any pending logic that depends on the params is consistent.
    g_hwnd = createOverlayWindow(static_cast<int>(overlayW), static_cast<int>(overlayH));
    if (!g_hwnd) return;

    ShowWindow(g_hwnd, SW_SHOW);
    LOG_DEBUG("PopupUI", "ShowWindow called at " + std::to_string(GetTickCount()));

    // Immediately push a blank transparent update so the layered window is
    // realized and visible at the correct size/position before resource
    // loading finishes. This prevents the large-texture centered snapshot
    // previously observed when the first update was delayed.
    try {
        sf::Image blank(sf::Vector2u(overlayW, overlayH), sf::Color(0,0,0,0));
        updateOverlay(g_hwnd, blank);
        LOG_DEBUG("PopupUI", "Pushed immediate blank overlay to realize window");
    } catch (...) {
        LOG_DEBUG("PopupUI", "Exception while pushing immediate blank overlay");
    }

    if (g_pendingPopup) {
        LOG_DEBUG("PopupUI", "Processing queued popup activity");
        showPopup();
        // Ensure the idle timer is set when a queued popup is processed so
        // it doesn't vanish immediately after being shown.
        g_idleTimerMs = 3000; // 3s grace
        g_idleClock.restart();

        // Push an immediate simple placeholder image so the layered window
        // becomes visibly non-empty while the heavier resources load.
        try {
            sf::Image placeholder(sf::Vector2u(overlayW, overlayH), sf::Color(0,0,0,0));
            // draw a solid white circle in the center (radius ~24px)
            const int cx = static_cast<int>(overlayW) / 2;
            const int cy = static_cast<int>(overlayH) / 2;
            const int r = 24;
            for (int y = cy - r; y <= cy + r; ++y) {
                for (int x = cx - r; x <= cx + r; ++x) {
                    int dx = x - cx;
                    int dy = y - cy;
                    if (dx*dx + dy*dy <= r*r) {
                        if (x >= 0 && y >= 0 && x < static_cast<int>(overlayW) && y < static_cast<int>(overlayH)) {
                            placeholder.setPixel({static_cast<unsigned>(x), static_cast<unsigned>(y)}, sf::Color(255,255,255,255));
                        }
                    }
                }
            }
            updateOverlay(g_hwnd, placeholder);
            LOG_DEBUG("PopupUI", "Pushed immediate placeholder overlay while resources load");
        } catch (...) {
            LOG_DEBUG("PopupUI", "Exception while pushing placeholder overlay");
        }

        g_pendingPopup = false;
    }

    // Load resources
    sf::Texture diffuse, opacity;
    sf::Shader shader;
    if (!loadResources(diffuse, opacity, shader)) return;

    // Merge opacity.r into diffuse alpha on the CPU to ensure the RenderTexture
    // contains per-pixel alpha (some shader/render paths produced alpha=0).
    sf::Image diffImg = diffuse.copyToImage();
    sf::Image opImg = opacity.copyToImage();
    if (diffImg.getSize() == opImg.getSize()) {
        sf::Vector2u sz = diffImg.getSize();
        for (unsigned y = 0; y < sz.y; ++y) {
            for (unsigned x = 0; x < sz.x; ++x) {
                sf::Color d = diffImg.getPixel({x, y});
                sf::Color o = opImg.getPixel({x, y});
                diffImg.setPixel({x, y}, sf::Color(d.r, d.g, d.b, o.r));
            }
        }
        // Upload merged image to a new texture and use that for the sprite.
    } else {
        LOG_DEBUG("PopupUI", "Diffuse and opacity sizes differ — skipping CPU merge");
    }
    sf::Texture mergedTex;
    if (!mergedTex.loadFromImage(diffImg)) {
        LOG_DEBUG("PopupUI", "Failed to create merged texture; falling back to diffuse");
        mergedTex = diffuse;
    }

    // One-shot: export the raw textures so we can inspect them directly.
    try {
        bool ok1 = diffuse.copyToImage().saveToFile("D:/G.R.I.M/resources/diffuse_debug.png");
        bool ok2 = opacity.copyToImage().saveToFile("D:/G.R.I.M/resources/opacity_debug.png");
        if (ok1 && ok2) {
            LOG_DEBUG("PopupUI", "Exported diffuse_debug.png and opacity_debug.png to resources");
        } else {
            LOG_DEBUG("PopupUI", "Failed to export one or more raw textures for debugging");
        }
    } catch (...) {
        LOG_DEBUG("PopupUI", "Exception while exporting raw textures for debugging");
    }

    // IMPORTANT: popup.frag must declare:
    // uniform float animAlpha;
    // fragColor = vec4(diffuse.rgb, opacity.r * animAlpha);

    // Use merged texture (diffuse with opacity in alpha channel) and scale
    // it to the overlay render target size so no large-desktop image is
    // pasted into the small layered window (was causing centered content).
    sf::Sprite sprite = createScaledSprite(mergedTex, { overlayW, overlayH });
    // Preserve the base scale (set by createScaledSprite) so we can apply
    // animation scale multiplicatively each frame without shifting position.
    sf::Vector2f baseScale = sprite.getScale();

    // Create once, not per-frame. Use the overlay size to match the HWND.
    sf::RenderTexture rtex({overlayW, overlayH});
    sf::Clock logClock;
    sf::Clock frameClock;
    MSG msg{};
    bool dumpedSnapshot = false;

    LOG_PHASE("Popup UI launched", true);
    uint32_t frameTraceLeft = 60; // log animAlpha for first 60 frames to trace timing
    auto startTick = GetTickCount();

    while (g_running) {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) g_running = false;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        // Compute frame delta time
        float dt = frameClock.restart().asSeconds();

        // Idle timer check — but keep visible while TTS is playing
        if (g_idleTimerMs > 0 && g_idleClock.getElapsedTime().asMilliseconds() > g_idleTimerMs) {
            if (!Voice::isPlaying()) {
                hidePopup();
                // After hiding, clear the idle timer so we don't repeatedly hide
                // (hidePopup already sets g_popupVisible=false).
                g_idleTimerMs = 0;
            } else {
                // extend the idle deadline slightly while audio is still playing
                // but don't zero the timer; restart the clock so we hide after
                // the audio finishes plus the configured grace period.
                g_idleClock.restart();
            }
        }

        // Animate (time-based)
        updateAnim(g_anim, g_popupVisible, dt, 0.08f);

        if (frameTraceLeft > 0) {
            LOG_DEBUG("PopupUI", "FrameTrace animAlpha=" + std::to_string(g_anim.alpha) +
                              " time=" + std::to_string(GetTickCount() - startTick));
            --frameTraceLeft;
        }

        // Draw
        rtex.clear(sf::Color(0, 0, 0, 0));
    // Apply animation scale on top of the base scale so the sprite still
    // fits the render target while animating.
    sprite.setScale({ baseScale.x * g_anim.scale, baseScale.y * g_anim.scale });

    // Apply per-sprite alpha modulation (0..255)
    uint8_t alphaByte = static_cast<uint8_t>(std::round(std::clamp(g_anim.alpha, 0.0f, 1.0f) * 255.0f));
    sprite.setColor(sf::Color(255, 255, 255, alphaByte));

    // We draw without the shader to avoid the earlier shader->readback alpha issue.
    // The merged texture already contains per-pixel alpha in its alpha channel.
    rtex.draw(sprite);
        rtex.display();

    // One-shot debug: dump render texture to the resources folder so we can
    // inspect alpha. Include the current animation alpha in the log so we
    // can correlate shader input with the output image.
        if (!dumpedSnapshot) {
            const std::string outPath = "D:/G.R.I.M/resources/popup_snapshot.png";
            try {
                bool ok = rtex.getTexture().copyToImage().saveToFile(outPath);
                if (ok) {
                    LOG_DEBUG("PopupUI", "Saved popup snapshot to " + outPath +
                                      " (animAlpha=" + std::to_string(g_anim.alpha) + ")");
                } else {
                    LOG_DEBUG("PopupUI", "Failed to save popup snapshot to " + outPath);
                }
            } catch (...) {
                LOG_DEBUG("PopupUI", "Exception while saving popup snapshot to resources");
            }
            dumpedSnapshot = true;
        }

        // Additional one-shot: render the sprite without the shader to check
        // whether raw sprite drawing produces visible pixels.
        static bool dumpedNoShader = false;
        if (!dumpedNoShader) {
            sf::RenderTexture rtexNoShader({(unsigned)width, (unsigned)height});
            rtexNoShader.clear(sf::Color(0,0,0,0));
            // draw sprite without shader
            rtexNoShader.draw(sprite);
            rtexNoShader.display();
            try {
                bool ok = rtexNoShader.getTexture().copyToImage().saveToFile("D:/G.R.I.M/resources/popup_snapshot_no_shader.png");
                if (ok) LOG_DEBUG("PopupUI", "Saved popup_snapshot_no_shader.png");
                else LOG_DEBUG("PopupUI", "Failed to save popup_snapshot_no_shader.png");
            } catch (...) {
                LOG_DEBUG("PopupUI", "Exception while saving popup_snapshot_no_shader.png");
            }
            dumpedNoShader = true;
        }

    // Copy once here and pass the Image to the renderer. This avoids
    // any potential cross-context copyToImage behavior inside the
    // renderer and guarantees the saved snapshot matches what's shown.
    sf::Image composedImg = rtex.getTexture().copyToImage();
    updateOverlay(g_hwnd, composedImg);

        if (logClock.getElapsedTime().asSeconds() > 5.f) {
            LOG_DEBUG("PopupUI", "Animating alpha=" + std::to_string(g_anim.alpha) +
                                 " scale=" + std::to_string(g_anim.scale));
            logClock.restart();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
}

// ===========================================================
// Popup UI controls
// ===========================================================
void showPopup() {
    if (g_hwnd) {
        ShowWindow(g_hwnd, SW_SHOW);
        g_popupVisible = true;
        LOG_PHASE("PopupUI shown", true);
    }
}

void hidePopup() {
    if (g_hwnd) {
        ShowWindow(g_hwnd, SW_HIDE);
        g_popupVisible = false;
        LOG_PHASE("PopupUI hidden", true);
        LOG_DEBUG("PopupUI", "hidePopup called at " + std::to_string(GetTickCount()) +
                          " idleTimerMs=" + std::to_string(g_idleTimerMs));
    }
}

void notifyPopupActivity() {
    if (!g_hwnd) {
        g_pendingPopup = true;
        LOG_DEBUG("PopupUI", "notifyPopupActivity called but HWND not ready - queued");
        return;
    }

    showPopup();
    g_idleTimerMs = 3000;     // 3s grace
    g_idleClock.restart();    // <<< FIX: restart clock when activity occurs
    LOG_DEBUG("PopupUI", "Activity notified, idle timer reset");
}

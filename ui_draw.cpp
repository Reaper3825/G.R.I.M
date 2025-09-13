#include "ui_draw.hpp"
#include "ui_helpers.hpp"   // g_ui_textbox + g_inputBuffer
#include <algorithm>
#include <cmath>

// Globals defined in main.cpp, declared extern in ui_helpers.hpp
extern sf::Text g_ui_textbox;       // renderable text object
extern std::string g_inputBuffer;   // raw editable buffer

void drawUI(
    sf::RenderWindow& window,
    sf::Font& font,
    ConsoleHistory& history,
    const std::string& /*unused*/,   // buffer param unused now
    bool caretVisible,
    float& scrollOffsetLines
) {
    // UI elements
    sf::RectangleShape titleBar; titleBar.setFillColor(sf::Color(26,26,30));
    sf::RectangleShape inputBar; inputBar.setFillColor(sf::Color(30,30,35));
    sf::Text titleText, lineText;

    if (font.getInfo().family != "") {
        titleText.setFont(font);
        titleText.setCharacterSize(kTitleFontSize);
        titleText.setFillColor(sf::Color(220,220,235));
        titleText.setString("G R I M");

        lineText.setFont(font);
        lineText.setCharacterSize(kFontSize);
        lineText.setFillColor(sf::Color::White);
    }

    // Window size
    float winW = window.getSize().x;
    float winH = window.getSize().y;

    // Bars
    titleBar.setSize({winW, kTitleBarH});
    titleBar.setPosition(0.f, 0.f);

    inputBar.setSize({winW, kInputBarH});
    inputBar.setPosition(0.f, winH - kInputBarH);

    // --- Draw bars ---
    window.draw(titleBar);
    window.draw(inputBar);

    if (font.getInfo().family != "") {
        // --- Title ---
        sf::FloatRect tb = titleText.getLocalBounds();
        titleText.setOrigin(tb.left + tb.width/2.f, 0.f);
        titleText.setPosition(winW/2.f, (kTitleBarH - tb.height) / 2.f - tb.top);
        window.draw(titleText);

        // --- Input text ---
        g_ui_textbox.setPosition(
            kSidePad,
            winH - kInputBarH + (kInputBarH - (float)kFontSize) * 0.5f
        );
        window.draw(g_ui_textbox);

        // --- Caret ---
        if (caretVisible) {
            sf::FloatRect bounds = g_ui_textbox.getGlobalBounds();
            sf::RectangleShape caret({2.f, (float)kFontSize});
            caret.setFillColor(sf::Color::White);
            caret.setPosition(bounds.left + bounds.width + 2.f, g_ui_textbox.getPosition().y);
            window.draw(caret);
        }

        // --- History ---
        float lineH = kLineSpacing * (float)kFontSize;
        float histTop = kTitleBarH + kTopPad;
        float histBottom = winH - kInputBarH - kBottomPad;
        float histH = std::max(0.f, histBottom - histTop);
        float wrapW = std::max(10.f, winW - 2.f * kSidePad);

        history.ensureWrapped(wrapW, lineText);
        float viewLines = std::max(1.f, histH / lineH);
        size_t wrapCount = history.wrappedCount();
        float maxScroll = (wrapCount > (size_t)viewLines) ? (wrapCount - (size_t)viewLines) : 0.f;

        if (scrollOffsetLines < 0) scrollOffsetLines = 0;
        if (scrollOffsetLines > maxScroll) scrollOffsetLines = maxScroll;

        long start = std::max(0L, (long)wrapCount - (long)std::ceil(viewLines) - (long)std::floor(scrollOffsetLines));
        long end = std::min<long>(wrapCount, start + (long)std::ceil(viewLines) + 1);
        float y = histTop;

        for (long i = start; i < end; ++i) {
            if (i < 0 || i >= (long)history.wrapped().size()) continue;
            auto& wl = history.wrapped()[i];
            lineText.setString(wl.text);
            lineText.setFillColor(wl.color);
            lineText.setPosition(kSidePad, y);
            window.draw(lineText);
            y += lineH;
            if (y > histBottom) break;
        }

        // --- Scrollbar ---
        if (wrapCount > (size_t)viewLines) {
            float trackTop = histTop, trackH = histH;
            float thumbH = std::max(30.f, trackH * (viewLines / (float)wrapCount));
            float t = (maxScroll <= 0.f) ? 0.f : (scrollOffsetLines / maxScroll);
            float thumbTop = trackTop + (trackH - thumbH) * t;

            sf::RectangleShape track({6.f, trackH});
            track.setFillColor(sf::Color(50,50,58));
            track.setPosition(winW - 8.f, trackTop);

            sf::RectangleShape thumb({6.f, thumbH});
            thumb.setFillColor(sf::Color(120,120,135));
            thumb.setPosition(winW - 8.f, thumbTop);

            window.draw(track);
            window.draw(thumb);
        }
    }
}

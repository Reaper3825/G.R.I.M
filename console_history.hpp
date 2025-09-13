#pragma once
#include <SFML/Graphics.hpp>
#include <deque>
#include <string>
#include <vector>

/// ConsoleHistory
/// Stores raw and wrapped console lines for display,
/// and automatically triggers audible speech on push().
class ConsoleHistory {
public:
    struct WrappedLine {
        std::string text;
        sf::Color color{ sf::Color::White };
    };

    /// Add a new line to history (default white color).
    /// Also triggers audible speech via speak() in voice_speak.cpp.
    void push(const std::string& line, sf::Color c = sf::Color::White);

    /// Re-wrap text for drawing into given width.
    void ensureWrapped(float maxWidth, sf::Text& meas);

    /// Clear all history lines.
    void clear();

    /// Accessors
    size_t rawCount() const;
    size_t wrappedCount() const;
    const std::vector<WrappedLine>& wrapped() const;

private:
    void wrapLine(const WrappedLine& ln,
                  float maxW,
                  sf::Text& meas,
                  std::vector<WrappedLine>& out);

    bool dirty_ = true;
    float lastWrapWidth_ = -1.f;
    unsigned lastFontSize_ = 0;

    std::deque<WrappedLine> raw_;
    std::vector<WrappedLine> wrapped_;
};

#pragma once
#include <SFML/Graphics.hpp>
#include <deque>
#include <string>
#include <sstream>
#include <vector>

// A single wrapped line with text + color
struct WrappedLine {
    std::string text;
    sf::Color color{ sf::Color::White };
};

class ConsoleHistory {
public:
    // Add a new line to history
    void push(const std::string& line, sf::Color c = sf::Color::White);

    // Wrap text for drawing into window width
    void ensureWrapped(float maxWidth, sf::Text& meas);

    // Access wrapped lines
    const std::vector<WrappedLine>& wrapped() const { return wrapped_; }
    size_t wrappedCount() const { return wrapped_.size(); }

    // Clear all history
    void clear();

private:
    static void wrapLine(const WrappedLine& ln, float maxW, sf::Text& meas, std::vector<WrappedLine>& out);

    bool dirty_ = true;
    float lastWrapWidth_ = -1.f;
    unsigned lastFontSize_ = 0;

    std::deque<WrappedLine> raw_;
    std::vector<WrappedLine> wrapped_;
};

#pragma once
#include <mutex>
#include <deque>
#include <vector>
#include <string>
#include <SFML/Graphics.hpp>




class ConsoleHistory {
public:
    struct WrappedLine { std::string text; sf::Color color; };
    static constexpr size_t kMaxHistory = 500;
    void push(const std::string& line, sf::Color c);
    void ensureWrapped(float maxWidth, sf::Text& meas);
    void clear();
    size_t rawCount() const;
    size_t wrappedCount() const;
    const std::vector<WrappedLine> wrapped() const;
private:
    void wrapLine(const WrappedLine&, float, sf::Text&, std::vector<WrappedLine>&);
    mutable std::mutex mtx_;
    std::deque<WrappedLine> raw_;
    std::vector<WrappedLine> wrapped_;
    bool dirty_ = false;
    float lastWrapWidth_ = 0.f;
    unsigned lastFontSize_ = 0;
};

// ✅ Global-safe accessor
ConsoleHistory& getConsoleHistory();

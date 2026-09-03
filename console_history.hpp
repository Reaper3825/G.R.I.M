#pragma once
#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <cstdint>

class ConsoleHistory
{
public:
    struct WrappedLine {
        std::string text;
        uint32_t color; // ABGR (BGFX-compatible)
    };

    void push(const std::string& line, uint32_t color = 0xFFFFFFFF);
    void ensureWrapped(float maxWidth);
    void clear();

    size_t rawCount() const;
    size_t wrappedCount() const;
    const std::vector<WrappedLine> wrapped() const;

    static constexpr size_t kMaxHistory = 512;

private:
    void wrapLine(const WrappedLine& ln,
                  float maxW,
                  std::vector<WrappedLine>& out);

    mutable std::mutex mtx_;
    std::deque<WrappedLine> raw_;
    std::vector<WrappedLine> wrapped_;
    bool dirty_ = true;
    float lastWrapWidth_ = 0.0f;
};


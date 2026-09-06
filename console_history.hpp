#pragma once
#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <cstdint>

class ConsoleHistory
{
public:
    enum class Alignment : uint8_t {
        Left,
        Right
    };

    struct WrappedLine {
        std::string text;
        uint32_t color; // ABGR (BGFX-compatible)
        std::string role = "assistant"; // "user" | "assistant" | "system"
        Alignment alignment = Alignment::Left;
        uint64_t message_id = 0;
    };

    // Alignment is derived from role, not chosen by the caller: "user"
    // renders right-aligned, everything else ("assistant" | "system" | ...)
    // renders left-aligned.
    static Alignment alignmentForRole(const std::string& role);

    void push(const std::string& line,
              uint32_t color = 0xFFFFFFFF,
              const std::string& role = "assistant");
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
    uint64_t next_message_id_ = 1;
};


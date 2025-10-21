#include "console_history.hpp"
#include <sstream>
#include <crtdbg.h>

#define CHECK_HEAP() _CrtCheckMemory()

// -------------------------------------------------------------
// Temporary text width approximation (fixed-width assumption)
static float measureTextWidth(const std::string& text, float charWidth = 8.0f)
{
    return static_cast<float>(text.size()) * charWidth;
}
// -------------------------------------------------------------

void ConsoleHistory::push(const std::string& line, uint32_t color)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (raw_.size() >= kMaxHistory)
        raw_.pop_front();

    raw_.push_back({ line, color });
    dirty_ = true;
    CHECK_HEAP();
}

void ConsoleHistory::ensureWrapped(float maxWidth)
{
    std::lock_guard<std::mutex> lock(mtx_);

    if (!dirty_ && lastWrapWidth_ == maxWidth)
        return;

    wrapped_.clear();
    for (const auto& ln : raw_) {
        wrapLine(ln, maxWidth, wrapped_);
    }

    dirty_ = false;
    lastWrapWidth_ = maxWidth;
    CHECK_HEAP();
}

void ConsoleHistory::clear()
{
    std::lock_guard<std::mutex> lock(mtx_);
    raw_.clear();
    wrapped_.clear();
    dirty_ = true;
    CHECK_HEAP();
}

void ConsoleHistory::wrapLine(const WrappedLine& ln,
                              float maxW,
                              std::vector<WrappedLine>& out)
{
    if (ln.text.empty()) {
        out.push_back({ "", ln.color });
        return;
    }

    std::string word, current;
    std::istringstream iss(ln.text);

    auto flush = [&](bool force = false) {
        if (force || !current.empty()) {
            out.push_back({ current, ln.color });
            current.clear();
        }
    };

    while (iss >> word) {
        std::string test = current.empty() ? word : current + " " + word;

        if (measureTextWidth(test) <= maxW) {
            current = test;
        } else {
            if (current.empty()) {
                // Word too long: split character by character
                std::string accum;
                for (char c : word) {
                    std::string temp = accum + c;
                    if (measureTextWidth(temp) <= maxW) {
                        accum = temp;
                    } else {
                        if (!accum.empty())
                            out.push_back({ accum, ln.color });
                        accum = std::string(1, c);
                    }
                }
                if (!accum.empty())
                    current = accum;
            } else {
                out.push_back({ current, ln.color });
                current = word;
            }
        }
    }
    flush(true);
}

// ---------------- Convenience ----------------

size_t ConsoleHistory::rawCount() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    return raw_.size();
}

size_t ConsoleHistory::wrappedCount() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    return wrapped_.size();
}

const std::vector<ConsoleHistory::WrappedLine> ConsoleHistory::wrapped() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    return wrapped_;
}

// Thread-safe singleton instance
ConsoleHistory& getConsoleHistory()
{
    static ConsoleHistory instance;
    return instance;
}

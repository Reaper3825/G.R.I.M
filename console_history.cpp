#include "console_history.hpp"
#include "ui_config.hpp" 

// Push a new line into history (with optional color)
void ConsoleHistory::push(const std::string& line, sf::Color c) {
    if (raw_.size() >= kMaxHistory) {
        raw_.pop_front(); // cap history size
    }
    raw_.push_back({ line, c });
    dirty_ = true;
}

// Re-wrap lines if font/width changed or marked dirty
void ConsoleHistory::ensureWrapped(float maxWidth, sf::Text& meas) {
    if (!dirty_ && lastWrapWidth_ == maxWidth && lastFontSize_ == meas.getCharacterSize()) {
        return; // nothing to do
    }

    wrapped_.clear();
    for (const auto& ln : raw_) {
        wrapLine(ln, maxWidth, meas, wrapped_);
    }

    dirty_ = false;
    lastWrapWidth_ = maxWidth;
    lastFontSize_  = meas.getCharacterSize();
}

// Clear history
void ConsoleHistory::clear() {
    raw_.clear();
    wrapped_.clear();
    dirty_ = true;
}

// Core wrapping routine
void ConsoleHistory::wrapLine(const WrappedLine& ln,
                              float maxW,
                              sf::Text& meas,
                              std::vector<WrappedLine>& out) {
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
        meas.setString(test);

        if (meas.getLocalBounds().width <= maxW) {
            current = test;
        } else {
            if (current.empty()) {
                // Word too long → split character by character
                std::string accum;
                for (char c : word) {
                    meas.setString(accum + c);
                    if (meas.getLocalBounds().width <= maxW) {
                        accum += c;
                    } else {
                        if (!accum.empty()) {
                            out.push_back({ accum, ln.color });
                        }
                        accum = std::string(1, c);
                    }
                }
                if (!accum.empty()) current = accum;
            } else {
                out.push_back({ current, ln.color });
                current = word;
            }
        }
    }
    flush(true);
}

// ---------------- Convenience ----------------

size_t ConsoleHistory::rawCount() const {
    return raw_.size();
}

size_t ConsoleHistory::wrappedCount() const {
    return wrapped_.size();
}

const std::vector<ConsoleHistory::WrappedLine>& ConsoleHistory::wrapped() const {
    return wrapped_;
}

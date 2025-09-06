#include "console_history.hpp"

void ConsoleHistory::push(const std::string& line, sf::Color c) {
    if (raw_.size() >= 1000)
        raw_.pop_front(); // cap history size
    raw_.push_back({ line, c });
    dirty_ = true;
}

void ConsoleHistory::ensureWrapped(float maxWidth, sf::Text& meas) {
    // Only re-wrap if width/font changed or marked dirty
    if (!dirty_ && lastWrapWidth_ == maxWidth && lastFontSize_ == meas.getCharacterSize())
        return;

    wrapped_.clear();
    for (const auto& ln : raw_) {
        wrapLine(ln, maxWidth, meas, wrapped_);
    }

    dirty_ = false;
    lastWrapWidth_ = maxWidth;
    lastFontSize_  = meas.getCharacterSize();
}

void ConsoleHistory::clear() {
    raw_.clear();
    wrapped_.clear();
    dirty_ = true;
}

void ConsoleHistory::wrapLine(const WrappedLine& ln, float maxW, sf::Text& meas, std::vector<WrappedLine>& out) {
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
                // word itself too long → split character by character
                std::string accum;
                for (char c : word) {
                    meas.setString(accum + c);
                    if (meas.getLocalBounds().width <= maxW) {
                        accum += c;
                    } else {
                        if (!accum.empty()) out.push_back({ accum, ln.color });
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

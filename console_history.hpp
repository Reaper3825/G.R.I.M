#pragma once
#include <SFML/Graphics.hpp>
#include <deque>
#include <string>
#include <sstream>
#include <vector>

// -------------- ConsoleHistory -----------------
struct WrappedLine {
    std::string text;
    sf::Color color{sf::Color::White};
};

class ConsoleHistory {
public:
    void push(const std::string& line, sf::Color c = sf::Color::White) {
        if (raw_.size() >= kMaxHistory) raw_.pop_front();
        raw_.push_back({line, c});
        dirty_ = true;
    }

    void ensureWrapped(float maxWidth, sf::Text& meas) {
        if (!dirty_ && lastWrapWidth_ == maxWidth && lastFontSize_ == meas.getCharacterSize()) return;
        wrapped_.clear();
        for (auto& ln : raw_) wrapLine(ln, maxWidth, meas, wrapped_);
        dirty_ = false;
        lastWrapWidth_ = maxWidth;
        lastFontSize_  = meas.getCharacterSize();
    }

    const std::vector<WrappedLine>& wrapped() const { return wrapped_; }
    size_t wrappedCount() const { return wrapped_.size(); }

private:
    static void wrapLine(const WrappedLine& ln, float maxW, sf::Text& meas, std::vector<WrappedLine>& out) {
        if (ln.text.empty()) { out.push_back({"", ln.color}); return; }
        std::string word, current; std::istringstream iss(ln.text);
        auto flush = [&](bool f=false){ if(f||!current.empty()){ out.push_back({current,ln.color}); current.clear(); } };
        while (iss >> word) {
            std::string test = current.empty()?word:current+" "+word;
            meas.setString(test);
            if (meas.getLocalBounds().width <= maxW) {
                current=test;
            } else {
                if (current.empty()) {
                    std::string accum;
                    for (char c: word) {
                        meas.setString(accum+c);
                        if (meas.getLocalBounds().width <= maxW) accum+=c;
                        else {
                            if(!accum.empty()) out.push_back({accum,ln.color});
                            accum=std::string(1,c);
                        }
                    }
                    if (!accum.empty()) current=accum;
                } else {
                    out.push_back({current,ln.color});
                    current=word;
                }
            }
        }
        flush(true);
    }

    bool dirty_=true;
    float lastWrapWidth_=-1.f;
    unsigned lastFontSize_=0;
    std::deque<WrappedLine> raw_;
    std::vector<WrappedLine> wrapped_;

    static constexpr size_t kMaxHistory = 1000;
};

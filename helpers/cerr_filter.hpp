
#pragma once
#include <iostream>
#include <streambuf>
#include <string>

namespace GRIM {

class CerrFilterBuf : public std::streambuf {
public:
    CerrFilterBuf(std::streambuf* original) : original_buf(original) {}

protected:
    int_type overflow(int_type ch) override {
        if (ch == '\n') {
            flushLine();
        } else {
            line_buf += static_cast<char>(ch);
        }
        return ch;
    }
    int sync() override {
        flushLine();
        return 0;
    }
public:
    std::streambuf* original_buf;
    std::string line_buf;
    void flushLine() {
        if (!line_buf.empty()) {
            if (line_buf[0] == '[' || line_buf[0] == '|') {
                for (char c : line_buf) {
                    original_buf->sputc(c);
                }
                original_buf->sputc('\n');
            }
            line_buf.clear();
        }
    }
};

class CerrFilter {
public:
    CerrFilter() : filter_buf(std::cerr.rdbuf()) {
        std::cerr.rdbuf(&filter_buf);
    }
    ~CerrFilter() {
        std::cerr.rdbuf(filter_buf.original_buf);
    }
private:
    CerrFilterBuf filter_buf;
};

} // namespace GRIM

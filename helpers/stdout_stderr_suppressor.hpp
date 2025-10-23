// helper to suppress all process-level stdout and stderr output (Windows only)
#pragma once
#include <iostream>
#include <fstream>
#include <cstdio>

namespace GRIM {

class StdoutStderrSuppressor {
public:
    StdoutStderrSuppressor() {
        // Save old buffers
        old_cout = std::cout.rdbuf();
        old_cerr = std::cerr.rdbuf();
        // Redirect to NUL
        null_stream.open("NUL");
        std::cout.rdbuf(null_stream.rdbuf());
        std::cerr.rdbuf(null_stream.rdbuf());
        // Also redirect C FILE* handles
        freopen("NUL", "w", stdout);
        freopen("NUL", "w", stderr);
    }
    ~StdoutStderrSuppressor() {
        // Restore C++ streams
        std::cout.rdbuf(old_cout);
        std::cerr.rdbuf(old_cerr);
        null_stream.close();
        // Restore C FILE* handles (best effort)
        freopen("CON", "w", stdout);
        freopen("CON", "w", stderr);
    }
private:
    std::ofstream null_stream;
    std::streambuf* old_cout;
    std::streambuf* old_cerr;
};

} // namespace GRIM

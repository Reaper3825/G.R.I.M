// AtomicWriter — write-temp + rename for crash-safe file persistence.
//
// All GRIM memory writes MUST use this utility so that inference never
// reads a partially-written file.  Pattern:
//   1. Write entire content to path + ".tmp"
//   2. fsync / close
//   3. std::filesystem::rename(tmp, path)  — atomic on same filesystem
//======================================================//
#pragma once

#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace GRIM {

class AtomicWriter {
public:
    // Write binary data atomically.
    static void write(const std::string& path,
                      const void* data, size_t size) {
        std::string tmp = path + ".tmp";
        auto parent = std::filesystem::path(path).parent_path();
        if (!parent.empty())
            std::filesystem::create_directories(parent);

        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (!out)
                throw std::runtime_error(
                    "AtomicWriter: cannot open tmp file: " + tmp);
            out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
            out.flush();
            if (!out)
                throw std::runtime_error(
                    "AtomicWriter: write failed: " + tmp);
        }

        replace(tmp, path);
    }

    // Write string data atomically.
    static void writeString(const std::string& path,
                            const std::string& content) {
        write(path, content.data(), content.size());
    }

    // Write via callback (for FlatBuffer builders, etc.).
    // Callback receives an open ofstream reference.
    static void writeWith(const std::string& path,
                          const std::function<void(std::ofstream&)>& writer) {
        std::string tmp = path + ".tmp";
        auto parent = std::filesystem::path(path).parent_path();
        if (!parent.empty())
            std::filesystem::create_directories(parent);

        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (!out)
                throw std::runtime_error(
                    "AtomicWriter: cannot open tmp file: " + tmp);
            writer(out);
            out.flush();
            if (!out)
                throw std::runtime_error(
                    "AtomicWriter: write failed: " + tmp);
        }

        replace(tmp, path);
    }

private:
    static void replace(const std::string& tmp, const std::string& path) {
#ifdef _WIN32
        const auto source = std::filesystem::path(tmp).wstring();
        const auto destination = std::filesystem::path(path).wstring();
        if (!::MoveFileExW(source.c_str(), destination.c_str(),
                           MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            throw std::runtime_error(
                "AtomicWriter: replace failed with Win32 error " +
                std::to_string(::GetLastError()));
        }
#else
        std::error_code ec;
        std::filesystem::rename(tmp, path, ec);
        if (ec) {
            throw std::runtime_error(
                "AtomicWriter: rename failed: " + ec.message());
        }
#endif
    }
};

} // namespace GRIM

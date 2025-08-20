#pragma once

// blank_window.h
// Header for a blank window application

#include <string>

class BlankWindow {
public:
    BlankWindow(const std::string& title, int width, int height);
    ~BlankWindow();

    void show();
    void close();

private:
    // Platform-specific window handle (opaque pointer)
    struct Impl;
    Impl* pImpl;
};
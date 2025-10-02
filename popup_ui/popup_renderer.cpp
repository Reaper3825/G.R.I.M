#include "popup_renderer.hpp"
#include "logger.hpp"
#include <bgfx/bgfx.h>
#include <bgfx/platform.h>
#include <fstream>
#include <vector>

// Helper to load a .bin shader file
static bgfx::ShaderHandle loadShader(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_ERROR("PopupRenderer", std::string("Failed to open shader: ") + path);
        return BGFX_INVALID_HANDLE;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size + 1);
    if (!file.read(buffer.data(), size)) {
        LOG_ERROR("PopupRenderer", std::string("Failed to read shader: ") + path);
        return BGFX_INVALID_HANDLE;
    }

    buffer[size] = '\0'; // null terminate
    const bgfx::Memory* mem = bgfx::copy(buffer.data(), static_cast<uint32_t>(size + 1));
    return bgfx::createShader(mem);
}

// Public: load popup program
bgfx::ProgramHandle loadPopupProgram() {
    bgfx::ShaderHandle vsh = loadShader("D:/G.R.I.M/resources/vs_popup.bin");
    bgfx::ShaderHandle fsh = loadShader("D:/G.R.I.M/resources/fs_popup.bin");

    if (!bgfx::isValid(vsh) || !bgfx::isValid(fsh)) {
        LOG_ERROR("PopupRenderer", "Shader load failed");
        return BGFX_INVALID_HANDLE;
    }

    bgfx::ProgramHandle prog = bgfx::createProgram(vsh, fsh, true);
    if (!bgfx::isValid(prog)) {
        LOG_ERROR("PopupRenderer", "Failed to create bgfx program");
    } else {
        LOG_PHASE("Popup program loaded", true);

    }
    return prog;
}

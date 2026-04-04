#include "popup_3d_mailbox.hpp"
#include <cstring>
#include <stdexcept>

// ===========================================================
// Popup 3D Mailbox — mutex-protected full-frame copy
// ===========================================================

void popupMailboxPublish(PopupFrameMailbox& mailbox,
                         const uint8_t* data, uint32_t width, uint32_t height)
{
    if (!data)
        throw std::runtime_error("popupMailboxPublish: data is NULL");
    if (width == 0 || height == 0)
        throw std::runtime_error("popupMailboxPublish: width or height is 0");

    uint32_t stride = width * 4;
    size_t totalBytes = static_cast<size_t>(stride) * height;

    std::lock_guard<std::mutex> lock(mailbox.mutex);
    mailbox.buffer.resize(totalBytes);
    std::memcpy(mailbox.buffer.data(), data, totalBytes);
    mailbox.width  = width;
    mailbox.height = height;
    mailbox.stride = stride;
    mailbox.generation++;
}

bool popupMailboxConsume(PopupFrameMailbox& mailbox,
                         std::vector<uint8_t>& outBuffer,
                         uint32_t& outWidth, uint32_t& outHeight,
                         uint64_t& lastGeneration)
{
    std::lock_guard<std::mutex> lock(mailbox.mutex);

    if (mailbox.generation == lastGeneration)
        return false;  // no new frame

    if (mailbox.buffer.empty())
        return false;

    size_t totalBytes = static_cast<size_t>(mailbox.stride) * mailbox.height;
    outBuffer.resize(totalBytes);
    std::memcpy(outBuffer.data(), mailbox.buffer.data(), totalBytes);
    outWidth  = mailbox.width;
    outHeight = mailbox.height;
    lastGeneration = mailbox.generation;
    return true;
}

#pragma once

#include "popup_3d_types.hpp"

// ===========================================================
// Popup 3D Mailbox — thread-safe frame handoff
// ===========================================================

// Publish a completed readback frame into the mailbox (submission thread).
void popupMailboxPublish(PopupFrameMailbox& mailbox,
                         const uint8_t* data, uint32_t width, uint32_t height);

// Consume the latest frame from the mailbox (presenter thread).
// Returns true if a new frame was available and copied into outBuffer.
// outBuffer is resized as needed. lastGeneration is updated.
bool popupMailboxConsume(PopupFrameMailbox& mailbox,
                         std::vector<uint8_t>& outBuffer,
                         uint32_t& outWidth, uint32_t& outHeight,
                         uint64_t& lastGeneration);

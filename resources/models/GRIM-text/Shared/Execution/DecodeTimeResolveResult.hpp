#pragma once
//======================================================//
//  DecodeTimeResolveResult — shared decode-time <NUM>
//  resolve/selection result contracts
//======================================================//

#include <cstdint>

namespace GRIM {

enum class SlotSelectionStatus : uint8_t {
    Selected,   // Exactly one legal slot resolved
    Null,       // Explicit null selection (no slot needed)
    Ambiguous   // Legal candidates exist but no unique winner
};

struct SlotSelectionResult {
    SlotSelectionStatus status = SlotSelectionStatus::Null;
    int32_t selected_slot = -1;   // Valid only when status == Selected; actual slot index in L
    float confidence = 0.0f;      // top1 - top2 margin when applicable
};

struct DecodeTimeResolveResult {
    bool valid = false;
    SlotSelectionStatus status = SlotSelectionStatus::Null;
    int32_t selected_slot = -1;
    float selected_value = 0.0f;

    void reset() {
        valid = false;
        status = SlotSelectionStatus::Null;
        selected_slot = -1;
        selected_value = 0.0f;
    }
};

} // namespace GRIM

//======================================================//
//  SlotIdentity.hpp
//
//  Strong execution-slot primitives.
//
//  SlotId is an episode-local semantic identity. It is durable across
//  compilation and may appear in authored guidance and host-side traces.
//
//  SlotIndex is a dense address valid only for one compiled execution payload
//  and its runtime materializations. It may be serialized with that compiled
//  payload and appear in batching/device ABI data, but must never be reused
//  across payloads or cross back into semantic metadata without explicit
//  resolution through CompiledSlotBinding.
//======================================================//

#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace GRIM::Execution {

class SlotId {
public:
    using Storage = std::uint64_t;

    constexpr SlotId() noexcept = default;

    static SlotId fromSerialized(Storage value) {
        if (value == 0) {
            throw std::invalid_argument("SlotId: serialized value 0 is reserved for invalid");
        }
        return SlotId(value);
    }

    static SlotId fromLocalOrdinal(std::uint64_t ordinal) {
        if (ordinal == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("SlotId: local ordinal cannot be represented");
        }
        return SlotId(ordinal + 1);
    }

    constexpr bool valid() const noexcept { return value_ != 0; }

    Storage serialized() const {
        if (!valid()) {
            throw std::logic_error("SlotId: cannot serialize an invalid identity");
        }
        return value_;
    }

    friend constexpr bool operator==(SlotId lhs, SlotId rhs) noexcept {
        return lhs.value_ == rhs.value_;
    }
    friend constexpr bool operator!=(SlotId lhs, SlotId rhs) noexcept {
        return !(lhs == rhs);
    }
    friend constexpr bool operator<(SlotId lhs, SlotId rhs) noexcept {
        return lhs.value_ < rhs.value_;
    }

private:
    explicit constexpr SlotId(Storage value) noexcept : value_(value) {}
    Storage value_ = 0;
};

class SlotIndex {
public:
    using Storage = std::int32_t;
    static constexpr Storage kInvalid = -1;

    constexpr SlotIndex() noexcept = default;

    static SlotIndex fromDense(Storage value) {
        if (value < 0) {
            throw std::invalid_argument("SlotIndex: dense index must be non-negative");
        }
        return SlotIndex(value);
    }

    constexpr bool valid() const noexcept { return value_ >= 0; }

    Storage dense() const {
        if (!valid()) {
            throw std::logic_error("SlotIndex: invalid index has no dense address");
        }
        return value_;
    }

    constexpr Storage denseOrInvalid() const noexcept { return value_; }

    friend constexpr bool operator==(SlotIndex lhs, SlotIndex rhs) noexcept {
        return lhs.value_ == rhs.value_;
    }
    friend constexpr bool operator!=(SlotIndex lhs, SlotIndex rhs) noexcept {
        return !(lhs == rhs);
    }
    friend constexpr bool operator<(SlotIndex lhs, SlotIndex rhs) noexcept {
        return lhs.value_ < rhs.value_;
    }

private:
    explicit constexpr SlotIndex(Storage value) noexcept : value_(value) {}
    Storage value_ = kInvalid;
};

inline std::string describeSlotId(SlotId id) {
    return id.valid() ? std::to_string(id.serialized()) : std::string("<invalid>");
}

inline std::string describeSlotIndex(SlotIndex index) {
    return index.valid() ? std::to_string(index.dense()) : std::string("<invalid>");
}

static_assert(sizeof(SlotId) == sizeof(SlotId::Storage),
    "SlotId must remain a compact strong scalar");
static_assert(sizeof(SlotIndex) == sizeof(SlotIndex::Storage),
    "SlotIndex must remain a compact strong scalar");

}  // namespace GRIM::Execution

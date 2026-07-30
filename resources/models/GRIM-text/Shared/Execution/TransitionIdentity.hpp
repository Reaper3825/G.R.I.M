//======================================================//
//  TransitionIdentity.hpp
//
//  Strong execution-transition primitives.
//
//  TransitionId identifies an executable transition in guidance and host-side
//  traces. The identity says nothing about the operation's implementation,
//  argument values, or model-head position.
//
//  TransitionIndex is a dense class address valid only for one compiled
//  transition payload and its runtime materializations. It may cross the
//  device ABI; it must not become semantic identity without an explicit
//  CompiledTransitionBinding lookup.
//======================================================//

#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace GRIM::Execution {

class TransitionId {
public:
    using Storage = std::uint64_t;

    constexpr TransitionId() noexcept = default;

    static TransitionId fromSerialized(Storage value) {
        if (value == 0) {
            throw std::invalid_argument(
                "TransitionId: serialized value 0 is reserved for invalid");
        }
        return TransitionId(value);
    }

    static TransitionId fromLocalOrdinal(std::uint64_t ordinal) {
        if (ordinal == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error(
                "TransitionId: local ordinal cannot be represented");
        }
        return TransitionId(ordinal + 1);
    }

    constexpr bool valid() const noexcept { return value_ != 0; }

    Storage serialized() const {
        if (!valid()) {
            throw std::logic_error(
                "TransitionId: cannot serialize an invalid identity");
        }
        return value_;
    }

    friend constexpr bool operator==(TransitionId lhs, TransitionId rhs) noexcept {
        return lhs.value_ == rhs.value_;
    }

    friend constexpr bool operator!=(TransitionId lhs, TransitionId rhs) noexcept {
        return !(lhs == rhs);
    }

    friend constexpr bool operator<(TransitionId lhs, TransitionId rhs) noexcept {
        return lhs.value_ < rhs.value_;
    }

private:
    explicit constexpr TransitionId(Storage value) noexcept : value_(value) {}
    Storage value_ = 0;
};

class TransitionIndex {
public:
    using Storage = std::int32_t;
    static constexpr Storage kInvalid = -1;

    constexpr TransitionIndex() noexcept = default;

    static TransitionIndex fromDense(Storage value) {
        if (value < 0) {
            throw std::invalid_argument(
                "TransitionIndex: dense index must be non-negative");
        }
        return TransitionIndex(value);
    }

    constexpr bool valid() const noexcept { return value_ >= 0; }

    Storage dense() const {
        if (!valid()) {
            throw std::logic_error(
                "TransitionIndex: invalid index has no dense address");
        }
        return value_;
    }

    constexpr Storage denseOrInvalid() const noexcept { return value_; }

    friend constexpr bool operator==(
        TransitionIndex lhs,
        TransitionIndex rhs) noexcept
    {
        return lhs.value_ == rhs.value_;
    }

    friend constexpr bool operator!=(
        TransitionIndex lhs,
        TransitionIndex rhs) noexcept
    {
        return !(lhs == rhs);
    }

    friend constexpr bool operator<(
        TransitionIndex lhs,
        TransitionIndex rhs) noexcept
    {
        return lhs.value_ < rhs.value_;
    }

private:
    explicit constexpr TransitionIndex(Storage value) noexcept : value_(value) {}
    Storage value_ = kInvalid;
};

inline std::string describeTransitionId(TransitionId id) {
    return id.valid()
        ? std::to_string(id.serialized())
        : std::string("<invalid>");
}

inline std::string describeTransitionIndex(TransitionIndex index) {
    return index.valid()
        ? std::to_string(index.dense())
        : std::string("<invalid>");
}

static_assert(sizeof(TransitionId) == sizeof(TransitionId::Storage),
    "TransitionId must remain a compact strong scalar");
static_assert(sizeof(TransitionIndex) == sizeof(TransitionIndex::Storage),
    "TransitionIndex must remain a compact strong scalar");

}  // namespace GRIM::Execution

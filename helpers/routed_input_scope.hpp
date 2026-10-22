#pragma once
#include "core/input_parser.hpp"

// Compatibility boundary for older panel code that still queries Mouse/Key.
// Does not mutate hardware state, dispatch callbacks, or affect global hotkeys.
inline thread_local const InputState* routedUIInput = nullptr;
class RoutedInputScope {
public:
    explicit RoutedInputScope(const InputState& input) : previous(routedUIInput) { routedUIInput = &input; }
    ~RoutedInputScope() { routedUIInput = previous; }
    RoutedInputScope(const RoutedInputScope&) = delete;
    RoutedInputScope& operator=(const RoutedInputScope&) = delete;
private:
    const InputState* previous;
};

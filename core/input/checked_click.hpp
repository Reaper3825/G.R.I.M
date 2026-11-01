#pragma once
#include <functional>

namespace GRIM::Input {
struct CheckedClickServices {
    std::function<bool(int, int)> move;
    std::function<bool(int, int)> positionMatches;
    std::function<unsigned int()> sendDownUp;
    std::function<void()> release;
};
inline bool SendCheckedClick(int x, int y, const CheckedClickServices& services) {
    if (!services.move(x, y) || !services.positionMatches(x, y)) return false;
    const auto sent = services.sendDownUp();
    if (sent == 1) services.release();
    return sent == 2;
}
}

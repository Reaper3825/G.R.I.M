#include "input_state.hpp"

InputState InputState::capture() {
    InputState s{};
    POINT p = Mouse::getPosition();
    s.mousePos = { (float)p.x, (float)p.y };

    s.mouseDown[0] = Mouse::isDown(MouseButton::Left);
    s.mouseDown[1] = Mouse::isDown(MouseButton::Right);
    s.mouseDown[2] = Mouse::isDown(MouseButton::Middle);

    for (int i = 0; i < 256; ++i) {
        KeyCode code = static_cast<KeyCode>(i);
        s.keysDown[code] = Key::isDown(code);
    }
    return s;
}

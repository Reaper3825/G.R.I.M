#version 330 core

// SFML provides these automatically when drawing sprites
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 vTexCoord;

uniform mat4 uMVP; // SFML sets this to the model-view-projection matrix

void main() {
    vTexCoord = aTexCoord;
    gl_Position = uMVP * vec4(aPos, 0.0, 1.0);
}

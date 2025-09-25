#version 330 core
layout (location = 0) in vec2 position;
layout (location = 1) in vec2 texCoord;

out vec2 vTexCoord;

uniform mat4 transform;

void main() {
    vTexCoord = texCoord;
    gl_Position = transform * vec4(position, 0.0, 1.0);
}

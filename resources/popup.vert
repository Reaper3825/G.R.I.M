#version 330 core

in vec2 position;   // provided by SFML
in vec2 texCoords;  // provided by SFML
in vec4 color;      // provided by SFML (unused but available)

out vec2 vTexCoord;

void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
    vTexCoord   = texCoords;
}

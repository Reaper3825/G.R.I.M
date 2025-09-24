#version 330 core

// Vertex attributes
layout (location = 0) in vec2 position;
layout (location = 1) in vec2 texCoord;

// Passed to fragment shader
out vec2 fragUV;

void main()
{
    fragUV = texCoord;
    gl_Position = vec4(position, 0.0, 1.0);
}

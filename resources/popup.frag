#version 330 core
in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D diffuseMap;
uniform sampler2D opacityMap;

void main() {
    vec4 diffuse = texture(diffuseMap, vTexCoord);
    vec4 opacity = texture(opacityMap, vTexCoord);

    // Force correct blue tint: keep only blue, suppress R+G
    fragColor = vec4(0.0, 0.0, diffuse.b, opacity.r);
}

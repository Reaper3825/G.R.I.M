#version 330 core
in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D diffuseMap;
uniform sampler2D opacityMap;

void main() {
    vec4 diffuse = texture(diffuseMap, vTexCoord);
    vec4 opacity = texture(opacityMap, vTexCoord);

    // Fix yellow tint: swap R/B, keep alpha from opacity map
    fragColor = vec4(diffuse.b, diffuse.g, diffuse.r, opacity.r);
}

// opacity and diffuse working pair

#version 330 core
in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D diffuseMap;
uniform sampler2D opacityMap;
uniform float animAlpha; // animated global alpha (0..1)

void main() {
    vec4 diffuse = texture(diffuseMap, vTexCoord);
    // opacityMap has no alpha channel; use red channel as opacity
    float op = texture(opacityMap, vTexCoord).r;

    // Combine per-pixel opacity with animation alpha
    float outA = clamp(op * animAlpha, 0.0, 1.0);

    // Fix yellow tint: swap R/B. Leave RGB unpremultiplied here;
    // premultiplication is performed later when creating the DIB for
    // UpdateLayeredWindow.
    fragColor = vec4(diffuse.b, diffuse.g, diffuse.r, outA);
}

// opacity and diffuse working pair with subtle voice highlight

#version 330 core
in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D diffuseMap;
uniform sampler2D opacityMap;
uniform float animAlpha;      // animated global alpha (0..1)
uniform float voicePulse;     // voice pulse intensity (0..1)
uniform float voiceIntensity; // overall voice activity (0..1)

void main() {
    vec4 diffuse = texture(diffuseMap, vTexCoord);
    // opacityMap has no alpha channel; use red channel as opacity
    float op = texture(opacityMap, vTexCoord).r;

    // Combine per-pixel opacity with animation alpha
    float outA = clamp(op * animAlpha, 0.0, 1.0);

    // Subtle white edge highlight when speaking (not full glow)
    vec3 originalRGB = vec3(diffuse.b, diffuse.g, diffuse.r); // BGR->RGB
    
    // Only brighten edges slightly when speaking
    float edgeFactor = 1.0 - op; // Edges have lower opacity
    float brighten = edgeFactor * voiceIntensity * 0.15; // Max 15% brighter on edges
    
    vec3 finalRGB = originalRGB * (1.0 + brighten);
    finalRGB = clamp(finalRGB, 0.0, 1.0);

    fragColor = vec4(finalRGB, outA);
}

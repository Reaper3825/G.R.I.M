// opacity and diffuse working pair with voice-reactive glow

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

    // Voice-reactive glow effect (add subtle cyan tint when speaking)
    vec3 glowColor = vec3(0.3, 0.7, 1.0); // Cyan glow
    float glowStrength = voicePulse * voiceIntensity * 0.3; // Max 30% glow
    
    // Mix original color with glow
    vec3 finalRGB = mix(
        vec3(diffuse.b, diffuse.g, diffuse.r), // Original (BGR->RGB)
        glowColor,
        glowStrength
    );

    fragColor = vec4(finalRGB, outA);
}

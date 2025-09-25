#version 330 core
in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D diffuseMap;
uniform sampler2D opacityMap;
uniform sampler2D normalMap;

// simple directional light
const vec3 lightDir = normalize(vec3(0.0, 0.0, 1.0));

void main() {
    vec4 diffuse = texture(diffuseMap, vTexCoord);
    vec4 opacity = texture(opacityMap, vTexCoord);

    // decode normal from normal map (0–1 → -1–1)
    vec3 normal = texture(normalMap, vTexCoord).rgb;
    normal = normalize(normal * 2.0 - 1.0);

    // simple lambert lighting
    float lighting = max(dot(normal, lightDir), 0.0);

    // keep only blue, modulated by lighting
    float blueLit = diffuse.b * lighting;

    // final color: blue only, opacity red as alpha
    fragColor = vec4(0.0, 0.0, blueLit, opacity.r);
}

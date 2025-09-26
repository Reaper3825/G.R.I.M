#version 330 core

in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D diffuseMap;
uniform sampler2D opacityMap;

// 0 = final (diffuse + opacity.r as alpha)
// 1 = diffuse only
// 2 = opacity red channel debug
uniform int debugMode;

void main()
{
    vec4 diffuse = texture(diffuseMap, vTexCoord);
    vec4 opacity = texture(opacityMap, vTexCoord);

    if (debugMode == 1) {
        fragColor = diffuse; // show diffuse only
        return;
    }

    if (debugMode == 2) {
        fragColor = vec4(opacity.r, opacity.r, opacity.r, 1.0); // grayscale mask
        return;
    }

    // Normal render: diffuse RGB, opacity red as alpha
    fragColor = vec4(diffuse.rgb, opacity.r);
}

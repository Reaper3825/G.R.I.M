$input v_texcoord0
#include "common_sprite.sh"

SAMPLER2D(s_texColor, 0);
SAMPLER2D(s_texOpacity, 1);

// Add fade uniform
uniform vec4 u_alpha; // we’ll use .w as the alpha multiplier

void main()
{
    // Sample diffuse + opacity
    vec4 color = texture2D(s_texColor, v_texcoord0);
    vec4 opacity = texture2D(s_texOpacity, v_texcoord0);

    // Combine baked alpha with runtime fade multiplier
    float finalAlpha = opacity.a * u_alpha.w;
    color.a *= finalAlpha;

    // Apply fade to entire color output
    gl_FragColor = vec4(color.rgb, color.a);
}

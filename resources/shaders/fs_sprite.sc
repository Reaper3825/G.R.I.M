$input v_texcoord0
#include "common_sprite.sh"

SAMPLER2D(s_texColor, 0);
SAMPLER2D(s_texOpacity, 1);

void main()
{
    // Diffuse and opacity RGBA
    vec4 color = texture2D(s_texColor, v_texcoord0);
    vec4 opacity = texture2D(s_texOpacity, v_texcoord0);

    // Use the Oreo texture's actual alpha channel
    color.a *= opacity.a;

    gl_FragColor = color;
}

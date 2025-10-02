$input v_texcoord0

#include "common.sh"
#include "bgfx_shader.sh"
#include "shaderlib.sh"

SAMPLER2D(s_texColor, 0);

void main()
{
    vec4 color = texture2D(s_texColor, v_texcoord0);
    gl_FragColor = color;
}

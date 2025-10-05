$input v_color0

#include "common_sprite.sh"

uniform vec4 u_alpha;

void main()
{
    vec4 color = v_color0;       // take vertex color
    color.a *= u_alpha.x;        // apply animated fade alpha
    gl_FragColor = color;
}

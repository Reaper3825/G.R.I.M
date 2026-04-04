$input a_position, a_normal, a_texcoord0
$output v_worldNormal, v_texcoord0, v_worldPos

#include <bgfx_shader.sh>

void main()
{
    // Transform position by model-view-projection
    gl_Position = mul(u_modelViewProj, vec4(a_position, 1.0));

    // Transform normal to world space (using model matrix upper-3x3)
    vec3 worldNormal;
    worldNormal.x = dot(vec3(u_model[0].xyz), a_normal);
    worldNormal.y = dot(vec3(u_model[0 + 1].xyz), a_normal);
    worldNormal.z = dot(vec3(u_model[0 + 2].xyz), a_normal);
    v_worldNormal = normalize(worldNormal);

    // Pass world position for potential specular computation
    vec4 worldPos = mul(u_model[0], vec4(a_position, 1.0));
    v_worldPos = worldPos.xyz;

    // Pass UV
    v_texcoord0 = a_texcoord0;
}

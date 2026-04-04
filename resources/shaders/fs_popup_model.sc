$input v_worldNormal, v_texcoord0, v_worldPos

#include <bgfx_shader.sh>

// Uniforms
uniform vec4 u_lightDir;      // xyz = normalized light direction
uniform vec4 u_lightParams;   // x = intensity, y = ambient
uniform vec4 u_alpha;          // x = alpha multiplier
uniform vec4 u_emissive;       // x = emissive multiplier

// Samplers (optional — test cube renders without textures)
SAMPLER2D(s_albedo, 0);

// sRGB linearization
vec3 srgbToLinear(vec3 c)
{
    return pow(c, vec3_splat(2.2));
}

// Linear to sRGB
vec3 linearToSrgb(vec3 c)
{
    return pow(c, vec3_splat(1.0 / 2.2));
}

void main()
{
    vec3 normal = normalize(v_worldNormal);
    vec3 lightDir = normalize(u_lightDir.xyz);

    // Base color: use albedo texture if available, else derive from normal
    vec4 albedo = texture2D(s_albedo, v_texcoord0);

    // If no texture is bound (test cube), use a default color derived from normal
    // bgfx sets texture to white (1,1,1,1) when no texture is bound, so detect that
    // For the test cube, we use normal-based coloring for visual verification
    vec3 baseColor = albedo.rgb;

    // If the texture appears to be the default white, use normal coloring
    // This gives each face a distinct color for easy debugging
    if (albedo.r > 0.99 && albedo.g > 0.99 && albedo.b > 0.99)
    {
        baseColor = abs(normal) * 0.5 + vec3_splat(0.3);
    }

    // Convert to linear space for lighting
    vec3 linearColor = srgbToLinear(baseColor);

    // Directional light (Lambertian diffuse)
    float NdotL = max(dot(normal, lightDir), 0.0);
    float diffuse = NdotL * u_lightParams.x;  // intensity

    // Simple specular (Blinn-Phong approximation)
    vec3 viewDir = normalize(vec3(0.0, 0.0, 2.5) - v_worldPos);
    vec3 halfVec = normalize(lightDir + viewDir);
    float NdotH = max(dot(normal, halfVec), 0.0);
    float specular = pow(NdotH, 32.0) * 0.3 * u_lightParams.x;

    // Ambient
    float ambient = u_lightParams.y;

    // Final lighting
    vec3 lit = linearColor * (ambient + diffuse) + vec3_splat(specular);

    // Emissive boost
    lit += linearColor * u_emissive.x;

    // Convert back to sRGB for output
    vec3 finalRGB = linearToSrgb(lit);

    // Alpha
    float finalAlpha = albedo.a * u_alpha.x;

    // Output straight-alpha color
    gl_FragColor = vec4(finalRGB, finalAlpha);
}

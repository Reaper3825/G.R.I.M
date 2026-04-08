$input v_worldNormal, v_texcoord0, v_worldPos

#include <bgfx_shader.sh>

// Uniforms
uniform vec4 u_lightDir;      // xyz = normalized light direction
uniform vec4 u_lightParams;   // x = intensity, y = ambient
uniform vec4 u_alpha;          // x = alpha multiplier
uniform vec4 u_emissive;       // x = emissive multiplier

// Samplers
SAMPLER2D(s_albedo, 0);        // base color (RGB + A)
SAMPLER2D(s_packed, 2);        // R=AO, G=roughness, B=metallic, A=opacity

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
    // ---- Geometry normal (from vertex) ----
    vec3 N = normalize(v_worldNormal);

    // ---- Light direction ----
    vec3 lightDir = normalize(u_lightDir.xyz);

    // ---- Albedo ----
    vec4 albedo = texture2D(s_albedo, v_texcoord0);

    // If the albedo texture is the 1x1 white default, use normal-based coloring
    // so the model remains visually distinct even without textures.
    vec3 baseColor = albedo.rgb;
    if (albedo.r > 0.99 && albedo.g > 0.99 && albedo.b > 0.99)
    {
        baseColor = abs(N) * 0.5 + vec3_splat(0.3);
    }

    // ---- Packed material ----
    vec4 matl = texture2D(s_packed, v_texcoord0);
    float ao        = matl.r;
    float roughness = matl.g;
    float metallic  = matl.b;
    float opacity   = matl.a;

    // Convert to linear space for lighting
    vec3 linearColor = srgbToLinear(baseColor);

    // ---- Schlick F0: dielectric=0.04, metal=albedo ----
    vec3 F0 = mix(vec3_splat(0.04), linearColor, metallic);

    // ---- Diffuse: Lambertian (metals have no diffuse) ----
    float NdotL = max(dot(N, lightDir), 0.0);
    vec3 diffuse = linearColor * (1.0 - metallic) * NdotL * u_lightParams.x;

    // ---- Specular: Blinn-Phong driven by roughness ----
    vec3 viewDir = normalize(vec3(0.0, 0.0, 2.5) - v_worldPos);
    vec3 halfVec = normalize(lightDir + viewDir);
    float NdotH  = max(dot(N, halfVec), 0.0);

    // Roughness → shininess: smooth = tight highlight, rough = broad
    float shininess = mix(128.0, 4.0, roughness);
    float specPower = pow(NdotH, shininess);

    // Fresnel approximation (Schlick)
    float VdotH = max(dot(viewDir, halfVec), 0.0);
    vec3 fresnel = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);

    // Scale specular intensity inversely with roughness
    float specScale = mix(0.5, 0.05, roughness) * u_lightParams.x;
    vec3 specular = fresnel * specPower * specScale;

    // ---- Ambient (modulated by AO) ----
    float ambient = u_lightParams.y * ao;
    vec3 ambientColor = linearColor * ambient;

    // ---- Combine ----
    vec3 lit = ambientColor + diffuse + specular;

    // Emissive boost
    lit += linearColor * u_emissive.x;

    // Convert back to sRGB for output
    vec3 finalRGB = linearToSrgb(lit);

    // Alpha: texture alpha * packed opacity * uniform alpha
    float finalAlpha = albedo.a * opacity * u_alpha.x;

    // Output straight-alpha color
    gl_FragColor = vec4(finalRGB, finalAlpha);
}

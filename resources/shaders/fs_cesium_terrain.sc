$input v_worldNormal, v_texcoord0, v_worldPos

#include <bgfx_shader.sh>

uniform vec4 u_lightDir;
uniform vec4 u_lightParams;
uniform vec4 u_baseColorFactor;
uniform vec4 u_overlayTransform;
uniform vec4 u_cameraPos;
uniform vec4 u_colorParams;
uniform vec4 u_surfaceParams;

SAMPLER2D(s_imagery, 0);

vec3 srgbToLinear(vec3 color)
{
    return pow(max(color, vec3_splat(0.0)), vec3_splat(2.2));
}

vec3 linearToSrgb(vec3 color)
{
    return pow(max(color, vec3_splat(0.0)), vec3_splat(1.0 / 2.2));
}

vec3 applyColorControls(vec3 color)
{
    float exposure = u_colorParams.x;
    float saturation = u_colorParams.y;
    float contrast = u_colorParams.z;

    color *= exposure;
    float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
    color = mix(vec3_splat(luminance), color, saturation);
    color = (color - vec3_splat(0.5)) * contrast + vec3_splat(0.5);
    return max(color, vec3_splat(0.0));
}

void main()
{
    vec2 imageryUv = v_texcoord0 * u_overlayTransform.zw + u_overlayTransform.xy;
    vec4 imagery = texture2D(s_imagery, imageryUv) * u_baseColorFactor;
    imagery.a = 1.0;
    vec3 baseColor = applyColorControls(srgbToLinear(imagery.rgb));

    vec3 viewDirection = normalize(u_cameraPos.xyz - v_worldPos);
    vec3 normalWorld = normalize(v_worldNormal);
    if (dot(normalWorld, viewDirection) < 0.0) {
        normalWorld = -normalWorld;
    }
    vec3 lightDirection = normalize(u_lightDir.xyz);
    vec3 halfVector = normalize(lightDirection + viewDirection);

    float diffuseTerm = max(dot(normalWorld, lightDirection), 0.0);
    float ambient = u_lightParams.y;
    float diffuse = diffuseTerm * u_lightParams.x;

    float specularStrength = u_surfaceParams.x;
    float roughness = clamp(u_surfaceParams.y, 0.04, 1.0);
    float fresnelStrength = u_surfaceParams.z;
    float nightFloor = u_surfaceParams.w;

    float shininess = mix(96.0, 8.0, roughness);
    float specular = pow(max(dot(normalWorld, halfVector), 0.0), shininess) * specularStrength * diffuseTerm;
    float fresnel = pow(1.0 - max(dot(normalWorld, viewDirection), 0.0), 5.0) * fresnelStrength;

    vec3 lit = baseColor * max(ambient + diffuse, nightFloor);
    lit += vec3_splat(specular + fresnel);

    gl_FragColor = vec4(linearToSrgb(lit), 1.0);
}
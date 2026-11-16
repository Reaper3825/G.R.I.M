$input v_worldNormal, v_texcoord0, v_worldPos

#include <bgfx_shader.sh>

uniform vec4 u_lightDir;
uniform vec4 u_lightParams;
uniform vec4 u_baseColorFactor;
uniform vec4 u_overlayTransform;
uniform vec4 u_textureParams;
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

    color *= exposure;
    float luminance = dot(color, vec3(0.30, 0.55, 0.15));
    color = mix(vec3_splat(luminance), color, saturation);
    return max(color, vec3_splat(0.0));
}

float bayer2(vec2 cell)
{
    return 3.0 * cell.y + 2.0 * cell.x - 4.0 * cell.x * cell.y;
}


float bayer4(vec2 fragCoord)
{
    vec2 p = floor(mod(fragCoord, 4.0));
    float index = 4.0 * bayer2(floor(mod(p, 2.0))) + bayer2(floor(mod(floor(p * 0.5), 2.0)));
    return (index + 0.5) / 16.0;
}

void main()
{
    vec2 baseUv = v_texcoord0 * u_overlayTransform.zw + u_overlayTransform.xy;
    vec2 sourceMin = vec2(0.0, 0.0);
    vec2 sourceMax = vec2(1.0, 1.0);
    vec2 imageryUv = mix(sourceMin, sourceMax, baseUv);
    vec2 sampleUv = imageryUv;
    bool clipOverlayUv = u_textureParams.x > 0.5;
    bool outsideOverlayUv = imageryUv.x < 0.0 || imageryUv.x > 1.0 || imageryUv.y < 0.0 || imageryUv.y > 1.0;
    if (clipOverlayUv) {
        sampleUv = clamp(imageryUv, vec2_splat(0.0), vec2_splat(1.0));
        // cesium-native generates _CESIUMOVERLAY_N tile-local UVs with
        // invertVCoordinate=false (v=0 at the tile's SOUTH edge), while the
        // downloaded raster tile image rows are north-up (row 0 = north).
        // Flip V here so south imagery lands on the south of the tile instead
        // of appearing at the north (the "Antarctica in the northern
        // hemisphere" artifact). Applied after translation/scale so it stays
        // correct for sub-tile raster mappings.
        sampleUv.y = 1.0 - sampleUv.y;
    }

    vec4 imagery = texture2D(s_imagery, sampleUv) * u_baseColorFactor;
    imagery.a *= u_textureParams.z;
    if (clipOverlayUv && outsideOverlayUv) {
        discard;
    }

    float fade = u_textureParams.y;
    if (fade < 0.999 && fade < bayer4(gl_FragCoord.xy)) {
        discard;
    }
    if (imagery.a <= 0.02) {
        discard;
    }
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

    // Wire the full material: diffuse + ambient key light, specular sun
    // highlight, and a fresnel rim, floored by nightFloor so the dark side of
    // the globe never crushes to pure black.
    vec3 lit = baseColor * (ambient + diffuse);
    lit += vec3_splat(specular);
    lit += baseColor * fresnel;
    lit = max(lit, baseColor * nightFloor);

    gl_FragColor = vec4(linearToSrgb(lit), imagery.a);
}
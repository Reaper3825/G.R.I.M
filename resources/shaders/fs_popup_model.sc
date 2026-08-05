$input v_worldNormal, v_texcoord0, v_worldPos, v_color0

#include <bgfx_shader.sh>

// Uniforms
uniform vec4 u_lightDir;      // xyz = normalized light direction
uniform vec4 u_lightParams;   // x = intensity, y = ambient
uniform vec4 u_alpha;          // x = alpha multiplier
uniform vec4 u_emissive;       // x = emissive multiplier
uniform vec4 u_cameraPos;      // xyz = world-space camera position

// Samplers
SAMPLER2D(s_albedo, 0);        // base color (RGB + A)
SAMPLER2D(s_packed, 2);        // R=AO, G=roughness, B=metallic, A=opacity
SAMPLER2D(s_materialProgram, 3); // RGBA32F VM instructions and parameters

#define MATERIAL_TEXTURE_WIDTH 512.0
#define MATERIAL_INSTRUCTION_OFFSET 1.0
#define MATERIAL_PARAMETER_OFFSET 129.0
#define MATERIAL_MAX_REGISTERS 64
#define MATERIAL_MAX_INSTRUCTIONS 64
#define MATERIAL_MAX_RAMP_ELEMENTS 32

vec4 materialTexel(float index)
{
    return texture2D(s_materialProgram,
        vec2((index + 0.5) / MATERIAL_TEXTURE_WIDTH, 0.5));
}

float materialFlag(float flags, float flag)
{
    return mod(floor(flags / flag), 2.0);
}

float hash41(vec4 value)
{
    return fract(sin(dot(value, vec4(127.1, 311.7, 74.7, 269.5))) * 43758.5453);
}

float valueNoise(vec4 samplePosition)
{
    vec4 base = floor(samplePosition);
    vec4 blend = fract(samplePosition);
    blend = blend * blend * (3.0 - 2.0 * blend);

    float z0w0 = mix(
        mix(hash41(base + vec4(0.0, 0.0, 0.0, 0.0)), hash41(base + vec4(1.0, 0.0, 0.0, 0.0)), blend.x),
        mix(hash41(base + vec4(0.0, 1.0, 0.0, 0.0)), hash41(base + vec4(1.0, 1.0, 0.0, 0.0)), blend.x), blend.y);
    float z1w0 = mix(
        mix(hash41(base + vec4(0.0, 0.0, 1.0, 0.0)), hash41(base + vec4(1.0, 0.0, 1.0, 0.0)), blend.x),
        mix(hash41(base + vec4(0.0, 1.0, 1.0, 0.0)), hash41(base + vec4(1.0, 1.0, 1.0, 0.0)), blend.x), blend.y);
    float z0w1 = mix(
        mix(hash41(base + vec4(0.0, 0.0, 0.0, 1.0)), hash41(base + vec4(1.0, 0.0, 0.0, 1.0)), blend.x),
        mix(hash41(base + vec4(0.0, 1.0, 0.0, 1.0)), hash41(base + vec4(1.0, 1.0, 0.0, 1.0)), blend.x), blend.y);
    float z1w1 = mix(
        mix(hash41(base + vec4(0.0, 0.0, 1.0, 1.0)), hash41(base + vec4(1.0, 0.0, 1.0, 1.0)), blend.x),
        mix(hash41(base + vec4(0.0, 1.0, 1.0, 1.0)), hash41(base + vec4(1.0, 1.0, 1.0, 1.0)), blend.x), blend.y);
    return mix(mix(z0w0, z1w0, blend.z), mix(z0w1, z1w1, blend.z), blend.w);
}

float fractalNoise(vec4 samplePosition, float detail, float roughness, float lacunarity)
{
    float amplitude = 1.0;
    float total = 0.0;
    float weight = 0.0;
    for (int octave = 0; octave < 8; ++octave)
    {
        float enabled = clamp(detail - float(octave), 0.0, 1.0);
        total += valueNoise(samplePosition) * amplitude * enabled;
        weight += amplitude * enabled;
        samplePosition *= max(lacunarity, 1.0);
        amplitude *= clamp(roughness, 0.0, 1.0);
    }
    return total / max(weight, 0.00001);
}

vec3 rotateEulerXYZ(vec3 value, vec3 rotation)
{
    vec3 sine = sin(rotation);
    vec3 cosine = cos(rotation);
    value = vec3(value.x, cosine.x * value.y - sine.x * value.z,
                  sine.x * value.y + cosine.x * value.z);
    value = vec3(cosine.y * value.x + sine.y * value.z, value.y,
                  -sine.y * value.x + cosine.y * value.z);
    return vec3(cosine.z * value.x - sine.z * value.y,
                sine.z * value.x + cosine.z * value.y, value.z);
}

vec4 evaluateColorRamp(float factor, float parameterOffset, vec4 header)
{
    int elementCount = int(floor(header.z + 0.5));
    vec4 firstColor = materialTexel(parameterOffset + 2.0);
    vec4 result = firstColor;
    float previousPosition = materialTexel(parameterOffset + 1.0).x;
    vec4 previousColor = firstColor;
    bool found = factor <= previousPosition;

    for (int element = 1; element < MATERIAL_MAX_RAMP_ELEMENTS; ++element)
    {
        if (element < elementCount)
        {
            float elementOffset = parameterOffset + 1.0 + float(element * 2);
            float position = materialTexel(elementOffset).x;
            vec4 color = materialTexel(elementOffset + 1.0);
            if (!found && factor <= position)
            {
                float blend = clamp((factor - previousPosition) /
                                    max(position - previousPosition, 0.00001), 0.0, 1.0);
                if (header.y < 0.5)
                    blend = 0.0;
                else if (header.y >= 1.5 && header.y < 2.5)
                    blend = blend * blend * (3.0 - 2.0 * blend);
                if (header.y < 2.5)
                {
                    result = mix(previousColor, color, blend);
                }
                else
                {
                    int previousPrevious = element - 2;
                    if (previousPrevious < 0)
                        previousPrevious = 0;
                    int nextElement = element + 1;
                    if (nextElement >= elementCount)
                        nextElement = elementCount - 1;
                    vec4 color0 = materialTexel(parameterOffset + 2.0 +
                                                float(previousPrevious * 2));
                    vec4 color3 = materialTexel(parameterOffset + 2.0 +
                                                float(nextElement * 2));
                    float blend2 = blend * blend;
                    float blend3 = blend2 * blend;
                    if (header.y < 3.5)
                    {
                        result = ((-color0 + 3.0 * previousColor - 3.0 * color + color3) * blend3 +
                                  (3.0 * color0 - 6.0 * previousColor + 3.0 * color) * blend2 +
                                  (-3.0 * color0 + 3.0 * color) * blend +
                                  color0 + 4.0 * previousColor + color) / 6.0;
                    }
                    else
                    {
                        result = 0.5 * ((2.0 * previousColor) +
                            (-color0 + color) * blend +
                            (2.0 * color0 - 5.0 * previousColor + 4.0 * color - color3) * blend2 +
                            (-color0 + 3.0 * previousColor - 3.0 * color + color3) * blend3);
                    }
                }
                found = true;
            }
            if (!found)
                result = color;
            previousPosition = position;
            previousColor = color;
        }
    }
    if (header.x >= 0.5)
        result = vec4_splat(result.a);
    return result;
}

vec4 executeMaterialProgram(vec3 materialWorldNormal, vec2 materialTexcoord,
                            vec3 materialWorldPosition, vec4 materialVertexColor)
{
    vec4 registers[MATERIAL_MAX_REGISTERS];
    for (int registerIndex = 0; registerIndex < MATERIAL_MAX_REGISTERS; ++registerIndex)
        registers[registerIndex] = vec4_splat(0.0);

    vec4 metadata = materialTexel(0.0);
    int instructionCount = int(floor(metadata.y + 0.5));
    vec4 surface = vec4(1.0, 0.25, 0.1, 1.0);

    for (int instructionIndex = 0; instructionIndex < MATERIAL_MAX_INSTRUCTIONS; ++instructionIndex)
    {
        if (instructionIndex < instructionCount)
        {
            float instructionOffset = MATERIAL_INSTRUCTION_OFFSET + float(instructionIndex * 2);
            vec4 first = materialTexel(instructionOffset);
            vec4 second = materialTexel(instructionOffset + 1.0);
            int opcode = int(floor(first.x + 0.5));
            int destination = int(floor(first.y + 0.5));
            int sourceA = int(floor(first.z + 0.5));
            int sourceB = int(floor(first.w + 0.5));
            int sourceC = int(floor(second.x + 0.5));
            float parameterOffset = MATERIAL_PARAMETER_OFFSET + second.y;
            float flags = second.w;
            vec4 result = vec4_splat(0.0);

            if (opcode == 1)
                result = materialTexel(parameterOffset);
            else if (opcode == 2)
                result = materialVertexColor;
            else if (opcode == 3)
                result = vec4(materialTexcoord, 0.0, 0.0);
            else if (opcode == 4)
                result = vec4(normalize(materialWorldNormal), 0.0);
            else if (opcode == 5)
                result = vec4(materialWorldPosition, 1.0);
            else if (opcode == 6)
                result = vec4(normalize(u_cameraPos.xyz - materialWorldPosition), 0.0);
            else if (opcode == 7)
            {
                vec3 normal = normalize(registers[sourceA].xyz);
                vec3 viewDirection = normalize(u_cameraPos.xyz - materialWorldPosition);
                float blend = clamp(materialTexel(parameterOffset).x, 0.0, 0.9999);
                float facingDot = abs(dot(normal, viewDirection));
                float exponent = max(blend * 2.0, 0.0001);
                if (blend >= 0.5)
                    exponent = 0.5 / max(1.0 - blend, 0.0001);
                float facing = 1.0 - pow(facingDot, exponent);
                float eta = 1.0 / max(1.0 - blend, 0.0001);
                float f0 = pow((eta - 1.0) / (eta + 1.0), 2.0);
                float fresnel = f0 + (1.0 - f0) * pow(1.0 - facingDot, 5.0);
                result = vec4(fresnel, facing, 0.0, 0.0);
            }
            else if (opcode == 8)
            {
                vec3 normal = normalize(registers[sourceA].xyz);
                vec3 viewDirection = normalize(u_cameraPos.xyz - materialWorldPosition);
                float eta = max(materialTexel(parameterOffset).x, 0.0001);
                float f0 = pow((eta - 1.0) / (eta + 1.0), 2.0);
                float fresnel = f0 + (1.0 - f0) *
                    pow(1.0 - abs(dot(normal, viewDirection)), 5.0);
                result = vec4_splat(fresnel);
            }
            else if (opcode == 9)
            {
                vec4 mode = materialTexel(parameterOffset);
                vec4 controlsA = materialTexel(parameterOffset + 1.0);
                vec4 controlsB = materialTexel(parameterOffset + 2.0);
                vec4 noisePosition = registers[sourceA];
                if (mode.y < 3.5)
                    noisePosition.w = controlsA.x;
                if (mode.y < 2.5)
                    noisePosition.z = 0.0;
                if (mode.y < 1.5)
                    noisePosition = vec4(controlsA.x, 0.0, 0.0, 0.0);
                noisePosition *= controlsA.y;
                float distortion = controlsB.y;
                noisePosition += distortion * (valueNoise(noisePosition + vec4(19.1, 7.7, 3.4, 11.9)) - 0.5);
                float factor = fractalNoise(noisePosition, controlsA.z, controlsA.w, controlsB.x);
                vec4 color = vec4(
                    factor,
                    fractalNoise(noisePosition + vec4(17.0, 31.0, 47.0, 59.0), controlsA.z, controlsA.w, controlsB.x),
                    fractalNoise(noisePosition + vec4(101.0, 73.0, 29.0, 13.0), controlsA.z, controlsA.w, controlsB.x),
                    1.0);
                result = color;
                if (mode.x < 0.5)
                    result = vec4_splat(factor);
            }
            else if (opcode == 10)
            {
                vec4 mode = materialTexel(parameterOffset);
                vec4 controlsA = materialTexel(parameterOffset + 1.0);
                vec4 controlsB = materialTexel(parameterOffset + 2.0);
                vec3 wavePosition = registers[sourceA].xyz * controlsA.x;
                float coordinate = wavePosition.x;
                if (mode.y < 0.5)
                {
                    if (mode.z > 0.5 && mode.z < 1.5) coordinate = wavePosition.y;
                    else if (mode.z > 1.5 && mode.z < 2.5) coordinate = wavePosition.z;
                    else if (mode.z > 2.5) coordinate = dot(wavePosition, normalize(vec3_splat(1.0)));
                }
                else
                    coordinate = length(wavePosition);
                coordinate += controlsB.y;
                coordinate += controlsA.y * (valueNoise(vec4(wavePosition * max(controlsA.w, 0.0001), 0.0)) - 0.5);
                float phase = coordinate * 6.28318530718;
                float factor = 0.5 + 0.5 * sin(phase);
                if (mode.w > 0.5 && mode.w < 1.5)
                    factor = fract(coordinate);
                else if (mode.w > 1.5)
                    factor = abs(fract(coordinate) * 2.0 - 1.0);
                result = vec4_splat(factor);
                if (mode.x < 0.5)
                    result = vec4(vec3_splat(factor), 1.0);
            }
            else if (opcode == 11)
                result = evaluateColorRamp(registers[sourceA].x, parameterOffset,
                                           materialTexel(parameterOffset));
            else if (opcode == 12)
                result = registers[sourceA] + registers[sourceB];
            else if (opcode == 13)
                result = registers[sourceA] * registers[sourceB];
            else if (opcode == 14)
                result = vec4(registers[sourceA].xyz * registers[sourceB].x, registers[sourceA].w);
            else if (opcode == 15)
            {
                vec4 factor = registers[sourceA];
                if (materialFlag(flags, 2.0) > 0.5)
                    factor = clamp(factor, 0.0, 1.0);
                result = mix(registers[sourceB], registers[sourceC], factor);
                if (materialFlag(flags, 4.0) > 0.5)
                    result = clamp(result, 0.0, 1.0);
            }
            else if (opcode == 16)
                result = vec4(registers[sourceA].rgb * registers[sourceB].x, registers[sourceA].a);
            else if (opcode == 17)
                result = vec4(registers[sourceA].rgb, 0.0);
            else if (opcode == 18)
                result = mix(registers[sourceB], registers[sourceC], clamp(registers[sourceA].x, 0.0, 1.0));
            else if (opcode == 19)
            {
                result = registers[sourceA];
                surface = result;
            }
            else if (opcode == 20)
            {
                int component = int(floor(materialTexel(parameterOffset).x + 0.5));
                float value = registers[sourceA].w;
                if (component == 0)
                    value = registers[sourceA].x;
                else if (component == 1)
                    value = registers[sourceA].y;
                else if (component == 2)
                    value = registers[sourceA].z;
                result = vec4_splat(value);
            }
            else if (opcode == 21)
            {
                vec3 location = materialTexel(parameterOffset).xyz;
                vec3 rotation = materialTexel(parameterOffset + 1.0).xyz;
                vec3 scale = materialTexel(parameterOffset + 2.0).xyz;
                result = vec4(rotateEulerXYZ(registers[sourceA].xyz * scale, rotation) + location,
                              registers[sourceA].w);
            }
            else if (opcode == 22)
            {
                vec4 baseColor = registers[sourceA];
                vec4 materialParameters = materialTexel(parameterOffset);
                vec3 emissionColor = materialTexel(parameterOffset + 1.0).rgb;
                float metallic = clamp(materialParameters.x, 0.0, 1.0);
                float roughness = clamp(materialParameters.y, 0.045, 1.0);
                float ior = max(materialParameters.z, 1.0001);
                float emissionStrength = max(materialParameters.w, 0.0);

                vec3 normal = normalize(registers[sourceC].xyz);
                vec3 lightDirection = normalize(u_lightDir.xyz);
                vec3 viewDirection = normalize(u_cameraPos.xyz - materialWorldPosition);
                vec3 halfDirection = normalize(lightDirection + viewDirection);
                float normalDotLight = max(dot(normal, lightDirection), 0.0);
                float normalDotHalf = max(dot(normal, halfDirection), 0.0);
                float viewDotHalf = max(dot(viewDirection, halfDirection), 0.0);

                vec3 linearBase = pow(clamp(baseColor.rgb, 0.0, 1.0), vec3_splat(2.2));
                float dielectricF0 = pow((ior - 1.0) / (ior + 1.0), 2.0);
                vec3 fresnelZero = mix(vec3_splat(dielectricF0), linearBase, metallic);
                vec3 fresnel = fresnelZero + (1.0 - fresnelZero) *
                    pow(1.0 - viewDotHalf, 5.0);

                vec3 diffuse = linearBase * (1.0 - metallic) * normalDotLight *
                    u_lightParams.x;
                float shininess = mix(128.0, 4.0, roughness);
                float specularPower = pow(normalDotHalf, shininess);
                float specularScale = mix(0.5, 0.05, roughness) * u_lightParams.x;
                vec3 specular = fresnel * specularPower * specularScale;
                vec3 ambient = linearBase * u_lightParams.y;
                vec3 emitted = emissionColor * emissionStrength;
                vec3 litColor = ambient + diffuse + specular + emitted;

                result = vec4(pow(max(litColor, vec3_splat(0.0)), vec3_splat(1.0 / 2.2)),
                              baseColor.a * registers[sourceB].x);
            }

            if (materialFlag(flags, 1.0) > 0.5)
                result = clamp(result, 0.0, 1.0);
            registers[destination] = result;
        }
    }
    return surface;
}

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
    vec4 metadata = materialTexel(0.0);
    if (metadata.x > 0.5)
    {
        vec4 material = executeMaterialProgram(v_worldNormal, v_texcoord0,
                               v_worldPos, v_color0);
        gl_FragColor = vec4(material.rgb * (1.0 + u_emissive.x),
                            material.a * u_alpha.x);
        return;
    }

    // ---- Geometry normal (from vertex) ----
    vec3 N = normalize(v_worldNormal);

    // ---- Light direction ----
    vec3 lightDir = normalize(u_lightDir.xyz);

    // ---- Albedo ----
    vec4 albedo = texture2D(s_albedo, v_texcoord0);
    vec3 baseColor = albedo.rgb * v_color0.rgb;

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

    // Alpha: texture alpha * vertex material alpha * packed opacity * uniform alpha
    float finalAlpha = albedo.a * v_color0.a * opacity * u_alpha.x;

    // Output straight-alpha color
    gl_FragColor = vec4(finalRGB, finalAlpha);
}

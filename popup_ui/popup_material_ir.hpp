#pragma once

#include <cstdint>
#include <type_traits>
#include <vector>

// ===========================================================
// Popup material intermediate representation
// ===========================================================
// Every material value occupies one vec4 register. Instructions have a fixed
// 32-byte serialized layout and are converted to two exact-integer float
// texels when uploaded to the renderer's RGBA32F program texture.

enum class PopupMaterialOpcode : uint32_t
{
    Invalid = 0,

    LoadConstant      = 1,
    LoadVertexColor   = 2,
    LoadTexCoord      = 3,
    LoadWorldNormal   = 4,
    LoadWorldPosition = 5,
    LoadViewDirection = 6,

    LayerWeight = 7,
    Fresnel      = 8,
    NoiseTexture = 9,
    WaveTexture  = 10,
    ColorRamp    = 11,

    Add         = 12,
    Multiply    = 13,
    VectorScale = 14,
    Mix         = 15,

    Emission     = 16,
    Transparent  = 17,
    MixShader    = 18,
    OutputSurface = 19,

    ExtractComponent = 20,
    Mapping          = 21,
    PrincipledLit    = 22,

    Count = 23
};

enum class PopupMaterialInstructionFlag : uint32_t
{
    None        = 0,
    Clamp       = 1u << 0,
    ClampFactor = 1u << 1,
    ClampResult = 1u << 2
};

struct alignas(16) PopupMaterialInstruction
{
    PopupMaterialOpcode opcode = PopupMaterialOpcode::Invalid;
    uint32_t destination       = 0;
    uint32_t sourceA           = 0;
    uint32_t sourceB           = 0;

    uint32_t sourceC           = 0;
    uint32_t parameterOffset   = 0;
    uint32_t parameterCount    = 0;
    PopupMaterialInstructionFlag flags = PopupMaterialInstructionFlag::None;
};

struct alignas(16) PopupMaterialParameter
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;
};

constexpr uint32_t POPUP_MATERIAL_MAX_REGISTERS    = 64;
constexpr uint32_t POPUP_MATERIAL_MAX_INSTRUCTIONS = 64;
constexpr uint32_t POPUP_MATERIAL_MAX_PARAMETERS   = 256;

struct PopupMaterialProgram
{
    uint32_t registerCount = 0;
    std::vector<PopupMaterialInstruction> instructions;
    std::vector<PopupMaterialParameter> parameters;

    bool empty() const { return instructions.empty(); }
};

static_assert(sizeof(PopupMaterialOpcode) == sizeof(uint32_t));
static_assert(sizeof(PopupMaterialInstructionFlag) == sizeof(uint32_t));
static_assert(sizeof(PopupMaterialInstruction) == 32);
static_assert(alignof(PopupMaterialInstruction) == 16);
static_assert(std::is_standard_layout_v<PopupMaterialInstruction>);
static_assert(std::is_trivially_copyable_v<PopupMaterialInstruction>);
static_assert(sizeof(PopupMaterialParameter) == 16);
static_assert(alignof(PopupMaterialParameter) == 16);
static_assert(std::is_standard_layout_v<PopupMaterialParameter>);
static_assert(std::is_trivially_copyable_v<PopupMaterialParameter>);

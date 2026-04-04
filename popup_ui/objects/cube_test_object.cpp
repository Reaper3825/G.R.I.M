#include "cube_test_object.hpp"

// ===========================================================
// Cube test object — 24 vertices (4 per face), 36 indices
// Unit cube from -0.5 to +0.5, CCW winding, outward normals
// Per-face vertex colors (ABGR packed for bgfx)
// ===========================================================

PopupObjectDefinition createCubeTestObject()
{
    PopupObjectDefinition def;

    // Face colors (ABGR format: A=0xFF fully opaque)
    constexpr uint32_t kRed     = 0xFF0000CC; // B=00 G=00 R=CC
    constexpr uint32_t kGreen   = 0xFF00CC00; // B=00 G=CC R=00
    constexpr uint32_t kBlue    = 0xFFCC0000; // B=CC G=00 R=00
    constexpr uint32_t kYellow  = 0xFF00CCCC; // B=00 G=CC R=CC
    constexpr uint32_t kWhite   = 0xFFCCCCCC; // B=CC G=CC R=CC
    constexpr uint32_t kGray    = 0xFF666666; // B=66 G=66 R=66

    // Each face has 4 unique vertices (for correct per-face normals)
    def.vertices = {
        // Front face (+Z toward camera) — normal (0, 0, 1) — Red
        { -0.5f, -0.5f,  0.5f,   0.0f, 0.0f, 1.0f,  kRed },
        {  0.5f, -0.5f,  0.5f,   0.0f, 0.0f, 1.0f,  kRed },
        {  0.5f,  0.5f,  0.5f,   0.0f, 0.0f, 1.0f,  kRed },
        { -0.5f,  0.5f,  0.5f,   0.0f, 0.0f, 1.0f,  kRed },

        // Back face (-Z) — normal (0, 0, -1) — Green
        {  0.5f, -0.5f, -0.5f,   0.0f, 0.0f, -1.0f,  kGreen },
        { -0.5f, -0.5f, -0.5f,   0.0f, 0.0f, -1.0f,  kGreen },
        { -0.5f,  0.5f, -0.5f,   0.0f, 0.0f, -1.0f,  kGreen },
        {  0.5f,  0.5f, -0.5f,   0.0f, 0.0f, -1.0f,  kGreen },

        // Right face (+X) — normal (1, 0, 0) — Blue
        {  0.5f, -0.5f,  0.5f,   1.0f, 0.0f, 0.0f,  kBlue },
        {  0.5f, -0.5f, -0.5f,   1.0f, 0.0f, 0.0f,  kBlue },
        {  0.5f,  0.5f, -0.5f,   1.0f, 0.0f, 0.0f,  kBlue },
        {  0.5f,  0.5f,  0.5f,   1.0f, 0.0f, 0.0f,  kBlue },

        // Left face (-X) — normal (-1, 0, 0) — Yellow
        { -0.5f, -0.5f, -0.5f,  -1.0f, 0.0f, 0.0f,  kYellow },
        { -0.5f, -0.5f,  0.5f,  -1.0f, 0.0f, 0.0f,  kYellow },
        { -0.5f,  0.5f,  0.5f,  -1.0f, 0.0f, 0.0f,  kYellow },
        { -0.5f,  0.5f, -0.5f,  -1.0f, 0.0f, 0.0f,  kYellow },

        // Top face (+Y) — normal (0, 1, 0) — White
        { -0.5f,  0.5f,  0.5f,   0.0f, 1.0f, 0.0f,  kWhite },
        {  0.5f,  0.5f,  0.5f,   0.0f, 1.0f, 0.0f,  kWhite },
        {  0.5f,  0.5f, -0.5f,   0.0f, 1.0f, 0.0f,  kWhite },
        { -0.5f,  0.5f, -0.5f,   0.0f, 1.0f, 0.0f,  kWhite },

        // Bottom face (-Y) — normal (0, -1, 0) — Gray
        { -0.5f, -0.5f, -0.5f,   0.0f, -1.0f, 0.0f,  kGray },
        {  0.5f, -0.5f, -0.5f,   0.0f, -1.0f, 0.0f,  kGray },
        {  0.5f, -0.5f,  0.5f,   0.0f, -1.0f, 0.0f,  kGray },
        { -0.5f, -0.5f,  0.5f,   0.0f, -1.0f, 0.0f,  kGray },
    };

    // CCW winding per face (2 triangles each)
    def.indices = {
        0,  1,  2,   2,  3,  0,   // front
        4,  5,  6,   6,  7,  4,   // back
        8,  9,  10,  10, 11, 8,   // right
        12, 13, 14,  14, 15, 12,  // left
        16, 17, 18,  18, 19, 16,  // top
        20, 21, 22,  22, 23, 20,  // bottom
    };

    def.defaultTransform = {};  // identity
    def.defaultLight.direction[0] = -0.5f;
    def.defaultLight.direction[1] =  0.7f;
    def.defaultLight.direction[2] =  0.5f;
    def.defaultLight.intensity    =  1.0f;
    def.defaultLight.ambient      =  0.15f;

    return def;
}

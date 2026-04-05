#include "grim_popup_object.hpp"
#include <cmath>
#include <stdexcept>

// ===========================================================
// GRIM Popup Object — UV sphere, 32 longitude × 24 latitude
// Gradient vertex colors: deep blue/purple at base → cyan glow at top
// CCW winding, outward normals, centered at origin, radius 0.5
// ===========================================================

static constexpr float kPi = 3.14159265358979323846f;

// Pack ABGR color from float RGBA [0,1]
static uint32_t packABGR(float r, float g, float b, float a)
{
    auto clamp = [](float v) -> uint8_t {
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        return static_cast<uint8_t>(v * 255.0f + 0.5f);
    };
    return (static_cast<uint32_t>(clamp(a)) << 24)
         | (static_cast<uint32_t>(clamp(b)) << 16)
         | (static_cast<uint32_t>(clamp(g)) <<  8)
         | (static_cast<uint32_t>(clamp(r)));
}

PopupObjectDefinition createGrimPopupObject()
{
    PopupObjectDefinition def;

    constexpr int kLonSegments = 32;
    constexpr int kLatSegments = 24;
    constexpr float kRadius = 0.5f;

    // -------------------------------------------------------
    // Generate vertices: (kLatSegments + 1) rings of (kLonSegments + 1) verts
    // -------------------------------------------------------
    int vertCount = (kLatSegments + 1) * (kLonSegments + 1);
    def.vertices.reserve(static_cast<size_t>(vertCount));

    for (int lat = 0; lat <= kLatSegments; ++lat)
    {
        float v = static_cast<float>(lat) / static_cast<float>(kLatSegments);
        float theta = v * kPi;  // 0 at top pole, pi at bottom pole

        float sinTheta = std::sin(theta);
        float cosTheta = std::cos(theta);

        for (int lon = 0; lon <= kLonSegments; ++lon)
        {
            float u = static_cast<float>(lon) / static_cast<float>(kLonSegments);
            float phi = u * 2.0f * kPi;

            float sinPhi = std::sin(phi);
            float cosPhi = std::cos(phi);

            // Position on unit sphere, scaled to radius
            float nx = sinTheta * cosPhi;
            float ny = cosTheta;          // +Y = up
            float nz = sinTheta * sinPhi;

            float px = nx * kRadius;
            float py = ny * kRadius;
            float pz = nz * kRadius;

            // Gradient color: t=0 at bottom, t=1 at top
            // Bottom: deep indigo (0.15, 0.05, 0.35)
            // Middle: royal blue (0.10, 0.20, 0.70)
            // Top: cyan glow (0.20, 0.70, 0.90)
            float t = (cosTheta + 1.0f) * 0.5f;  // 0=bottom pole, 1=top pole

            float cr, cg, cb;
            if (t < 0.5f)
            {
                // Bottom half: indigo → blue
                float s = t * 2.0f;
                cr = 0.15f + (0.10f - 0.15f) * s;
                cg = 0.05f + (0.20f - 0.05f) * s;
                cb = 0.35f + (0.70f - 0.35f) * s;
            }
            else
            {
                // Top half: blue → cyan
                float s = (t - 0.5f) * 2.0f;
                cr = 0.10f + (0.20f - 0.10f) * s;
                cg = 0.20f + (0.70f - 0.20f) * s;
                cb = 0.70f + (0.90f - 0.70f) * s;
            }

            PopupVertex vert;
            vert.px = px;
            vert.py = py;
            vert.pz = pz;
            vert.nx = nx;
            vert.ny = ny;
            vert.nz = nz;
            vert.abgr = packABGR(cr, cg, cb, 1.0f);
            def.vertices.push_back(vert);
        }
    }

    // -------------------------------------------------------
    // Generate indices: two triangles per quad, CCW winding
    // -------------------------------------------------------
    int indexCount = kLatSegments * kLonSegments * 6;
    def.indices.reserve(static_cast<size_t>(indexCount));

    for (int lat = 0; lat < kLatSegments; ++lat)
    {
        for (int lon = 0; lon < kLonSegments; ++lon)
        {
            int current = lat * (kLonSegments + 1) + lon;
            int next    = current + kLonSegments + 1;

            // Triangle 1 (CCW when viewed from outside)
            def.indices.push_back(static_cast<uint16_t>(current));
            def.indices.push_back(static_cast<uint16_t>(current + 1));
            def.indices.push_back(static_cast<uint16_t>(next));

            // Triangle 2
            def.indices.push_back(static_cast<uint16_t>(current + 1));
            def.indices.push_back(static_cast<uint16_t>(next + 1));
            def.indices.push_back(static_cast<uint16_t>(next));
        }
    }

    // -------------------------------------------------------
    // Default transform and lighting
    // -------------------------------------------------------
    def.defaultTransform = {};
    def.defaultTransform.scale[0] = 1.0f;
    def.defaultTransform.scale[1] = 1.0f;
    def.defaultTransform.scale[2] = 1.0f;

    // Upper-front-left light per plan convention
    def.defaultLight.direction[0] = -0.5f;
    def.defaultLight.direction[1] =  0.7f;
    def.defaultLight.direction[2] =  0.5f;
    def.defaultLight.intensity    =  1.0f;
    def.defaultLight.ambient      =  0.20f;

    return def;
}

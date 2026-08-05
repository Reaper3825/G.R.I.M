#include "popup_mesh_cache_loader.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <algorithm>

// ===========================================================
// GRIM Mesh Cache (.gmc) loader implementation
// ===========================================================

namespace
{
// Read a trivially-copyable value from the stream; throws on short read.
template <typename T>
T readPOD(std::istream& in, const std::string& path, const char* what)
{
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in)
        throw std::runtime_error("Mesh cache loader: unexpected EOF reading " +
                                 std::string(what) + " in " + path);
    return value;
}

void readBytes(std::istream& in, void* dst, size_t bytes,
               const std::string& path, const char* what)
{
    if (bytes == 0) return;
    in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    if (!in)
        throw std::runtime_error("Mesh cache loader: unexpected EOF reading " +
                                 std::string(what) + " in " + path);
}

void normalize3(float* v)
{
    float len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (len > 1e-8f) { v[0] /= len; v[1] /= len; v[2] /= len; }
}

// Area-weighted face-normal accumulation (matches the OBJ loader fallback).
void generateNormals(PopupMeshFrame& frame)
{
    for (auto& v : frame.vertices) { v.nx = v.ny = v.nz = 0.0f; }

    const auto& idx = frame.indices;
    for (size_t i = 0; i + 2 < idx.size(); i += 3)
    {
        PopupVertex& v0 = frame.vertices[idx[i + 0]];
        PopupVertex& v1 = frame.vertices[idx[i + 1]];
        PopupVertex& v2 = frame.vertices[idx[i + 2]];

        float e1x = v1.px - v0.px, e1y = v1.py - v0.py, e1z = v1.pz - v0.pz;
        float e2x = v2.px - v0.px, e2y = v2.py - v0.py, e2z = v2.pz - v0.pz;

        float fnx = e1y * e2z - e1z * e2y;
        float fny = e1z * e2x - e1x * e2z;
        float fnz = e1x * e2y - e1y * e2x;

        v0.nx += fnx; v0.ny += fny; v0.nz += fnz;
        v1.nx += fnx; v1.ny += fny; v1.nz += fnz;
        v2.nx += fnx; v2.ny += fny; v2.nz += fnz;
    }

    for (auto& v : frame.vertices)
    {
        float n[3] = { v.nx, v.ny, v.nz };
        normalize3(n);
        v.nx = n[0]; v.ny = n[1]; v.nz = n[2];
    }
}

void validateMaterialProgram(const PopupMaterialProgram& program, const std::string& path)
{
    if (program.registerCount == 0 || program.registerCount > POPUP_MATERIAL_MAX_REGISTERS)
        throw std::runtime_error("Mesh cache loader: invalid material register count in " + path);
    if (program.instructions.empty() ||
        program.instructions.size() > POPUP_MATERIAL_MAX_INSTRUCTIONS)
        throw std::runtime_error("Mesh cache loader: invalid material instruction count in " + path);
    if (program.parameters.size() > POPUP_MATERIAL_MAX_PARAMETERS)
        throw std::runtime_error("Mesh cache loader: material parameter limit exceeded in " + path);

    for (const PopupMaterialInstruction& instruction : program.instructions)
    {
        const uint32_t opcode = static_cast<uint32_t>(instruction.opcode);
        if (opcode < static_cast<uint32_t>(PopupMaterialOpcode::LoadConstant) ||
            opcode >= static_cast<uint32_t>(PopupMaterialOpcode::Count))
            throw std::runtime_error("Mesh cache loader: invalid material opcode in " + path);
        constexpr uint32_t knownInstructionFlags =
            static_cast<uint32_t>(PopupMaterialInstructionFlag::Clamp) |
            static_cast<uint32_t>(PopupMaterialInstructionFlag::ClampFactor) |
            static_cast<uint32_t>(PopupMaterialInstructionFlag::ClampResult);
        if ((static_cast<uint32_t>(instruction.flags) & ~knownInstructionFlags) != 0)
            throw std::runtime_error("Mesh cache loader: unknown material instruction flags in " + path);
        if (instruction.destination >= program.registerCount ||
            instruction.sourceA >= program.registerCount ||
            instruction.sourceB >= program.registerCount ||
            instruction.sourceC >= program.registerCount)
            throw std::runtime_error("Mesh cache loader: material register index out of range in " + path);
        if (instruction.parameterOffset > program.parameters.size() ||
            instruction.parameterCount > program.parameters.size() - instruction.parameterOffset)
            throw std::runtime_error("Mesh cache loader: material parameter span out of range in " + path);

        uint32_t expectedParameterCount = 0;
        switch (instruction.opcode)
        {
            case PopupMaterialOpcode::LoadConstant:
            case PopupMaterialOpcode::LayerWeight:
            case PopupMaterialOpcode::Fresnel:
            case PopupMaterialOpcode::ExtractComponent:
                expectedParameterCount = 1;
                break;
            case PopupMaterialOpcode::NoiseTexture:
            case PopupMaterialOpcode::WaveTexture:
            case PopupMaterialOpcode::Mapping:
                expectedParameterCount = 3;
                break;
            case PopupMaterialOpcode::PrincipledLit:
                expectedParameterCount = 2;
                break;
            case PopupMaterialOpcode::ColorRamp:
            {
                if (instruction.parameterCount < 3 || (instruction.parameterCount & 1u) == 0)
                    throw std::runtime_error("Mesh cache loader: invalid ColorRamp parameter count in " + path);
                const PopupMaterialParameter& header = program.parameters[instruction.parameterOffset];
                if (!std::isfinite(header.z) || header.z < 1.0f || header.z > 32.0f ||
                    std::floor(header.z) != header.z)
                    throw std::runtime_error("Mesh cache loader: invalid ColorRamp element count in " + path);
                const uint32_t elementCount = static_cast<uint32_t>(header.z);
                if (instruction.parameterCount != 1 + elementCount * 2)
                    throw std::runtime_error("Mesh cache loader: invalid ColorRamp element count in " + path);
                expectedParameterCount = instruction.parameterCount;
                break;
            }
            case PopupMaterialOpcode::LoadVertexColor:
            case PopupMaterialOpcode::LoadTexCoord:
            case PopupMaterialOpcode::LoadWorldNormal:
            case PopupMaterialOpcode::LoadWorldPosition:
            case PopupMaterialOpcode::LoadViewDirection:
            case PopupMaterialOpcode::Add:
            case PopupMaterialOpcode::Multiply:
            case PopupMaterialOpcode::VectorScale:
            case PopupMaterialOpcode::Mix:
            case PopupMaterialOpcode::Emission:
            case PopupMaterialOpcode::Transparent:
            case PopupMaterialOpcode::MixShader:
            case PopupMaterialOpcode::OutputSurface:
                break;
            case PopupMaterialOpcode::Invalid:
            case PopupMaterialOpcode::Count:
                throw std::runtime_error("Mesh cache loader: invalid material opcode schema in " + path);
        }
        if (instruction.parameterCount != expectedParameterCount)
            throw std::runtime_error("Mesh cache loader: invalid material opcode parameter count in " + path);
    }

    if (program.instructions.back().opcode != PopupMaterialOpcode::OutputSurface)
        throw std::runtime_error("Mesh cache loader: material program has no final OutputSurface in " + path);
}
} // namespace

PopupMeshCache loadPopupMeshCache(const std::string& gmcPath, uint32_t defaultColorABGR)
{
    std::ifstream file(gmcPath, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Mesh cache loader: cannot open file: " + gmcPath);

    char magic[8] = { 0 };
    readBytes(file, magic, sizeof(magic), gmcPath, "magic");
    if (std::memcmp(magic, "GRIMMC03", 8) != 0)
        throw std::runtime_error("Mesh cache loader: bad magic (expected GRIMMC03; rebake the cache): " + gmcPath);

    PopupMeshCache cache;
    cache.fps            = readPOD<float>(file, gmcPath, "fps");
    uint32_t frameCount  = readPOD<uint32_t>(file, gmcPath, "frameCount");
    uint32_t flags       = readPOD<uint32_t>(file, gmcPath, "flags");
    cache.maxVertices    = readPOD<uint32_t>(file, gmcPath, "maxVertices");
    cache.maxIndices     = readPOD<uint32_t>(file, gmcPath, "maxIndices");

    if (cache.fps <= 0.0f)
        throw std::runtime_error("Mesh cache loader: fps must be > 0 in " + gmcPath);
    if (frameCount == 0)
        throw std::runtime_error("Mesh cache loader: frameCount is 0 in " + gmcPath);

    constexpr uint32_t knownFlags = GMC_FLAG_HAS_NORMALS | GMC_FLAG_HAS_UV |
                                    GMC_FLAG_HAS_COLOR | GMC_FLAG_HAS_MATERIAL_PROGRAM;
    if ((flags & ~knownFlags) != 0)
        throw std::runtime_error("Mesh cache loader: unknown flags in " + gmcPath);

    const bool hasNormals = (flags & GMC_FLAG_HAS_NORMALS) != 0;
    const bool hasUV      = (flags & GMC_FLAG_HAS_UV) != 0;
    const bool hasColor   = (flags & GMC_FLAG_HAS_COLOR) != 0;
    const bool hasMaterialProgram = (flags & GMC_FLAG_HAS_MATERIAL_PROGRAM) != 0;

    if (hasMaterialProgram)
    {
        cache.materialProgram.registerCount = readPOD<uint32_t>(file, gmcPath, "material registerCount");
        const uint32_t instructionCount = readPOD<uint32_t>(file, gmcPath, "material instructionCount");
        const uint32_t parameterCount = readPOD<uint32_t>(file, gmcPath, "material parameterCount");

        if (instructionCount > POPUP_MATERIAL_MAX_INSTRUCTIONS)
            throw std::runtime_error("Mesh cache loader: material instruction limit exceeded in " + gmcPath);
        if (parameterCount > POPUP_MATERIAL_MAX_PARAMETERS)
            throw std::runtime_error("Mesh cache loader: material parameter limit exceeded in " + gmcPath);

        cache.materialProgram.instructions.resize(instructionCount);
        readBytes(file, cache.materialProgram.instructions.data(),
                  cache.materialProgram.instructions.size() * sizeof(PopupMaterialInstruction),
                  gmcPath, "material instructions");
        cache.materialProgram.parameters.resize(parameterCount);
        readBytes(file, cache.materialProgram.parameters.data(),
                  cache.materialProgram.parameters.size() * sizeof(PopupMaterialParameter),
                  gmcPath, "material parameters");
        validateMaterialProgram(cache.materialProgram, gmcPath);
    }

    cache.frames.reserve(frameCount);

    std::vector<float> posBuf, nrmBuf, uvBuf;
    std::vector<uint32_t> colorBuf;

    for (uint32_t f = 0; f < frameCount; ++f)
    {
        uint32_t vertCount  = readPOD<uint32_t>(file, gmcPath, "vertCount");
        uint32_t indexCount = readPOD<uint32_t>(file, gmcPath, "indexCount");

        if (vertCount > 65535)
            throw std::runtime_error("Mesh cache loader: frame " + std::to_string(f) +
                                     " exceeds uint16 vertex limit (65535) in " + gmcPath);

        // Empty frames are valid (e.g. the start of a build-from-nothing "load in"
        // animation). Consume any payload that is present and store an empty frame;
        // the renderer simply draws nothing for it.
        PopupMeshFrame frame;
        if (vertCount == 0)
        {
            // No vertex payload; still consume indices if the exporter wrote any.
            if (indexCount > 0)
            {
                std::vector<uint16_t> skip(indexCount);
                readBytes(file, skip.data(), skip.size() * sizeof(uint16_t), gmcPath, "indices");
            }
            cache.frames.push_back(std::move(frame));
            continue;
        }

        posBuf.resize(static_cast<size_t>(vertCount) * 3);
        readBytes(file, posBuf.data(), posBuf.size() * sizeof(float), gmcPath, "positions");

        if (hasNormals)
        {
            nrmBuf.resize(static_cast<size_t>(vertCount) * 3);
            readBytes(file, nrmBuf.data(), nrmBuf.size() * sizeof(float), gmcPath, "normals");
        }
        if (hasUV)
        {
            uvBuf.resize(static_cast<size_t>(vertCount) * 2);
            readBytes(file, uvBuf.data(), uvBuf.size() * sizeof(float), gmcPath, "uvs");
        }
        if (hasColor)
        {
            colorBuf.resize(vertCount);
            readBytes(file, colorBuf.data(), colorBuf.size() * sizeof(uint32_t), gmcPath, "material colors");
        }

        frame.vertices.resize(vertCount);
        for (uint32_t v = 0; v < vertCount; ++v)
        {
            PopupVertex& vert = frame.vertices[v];
            vert.px = posBuf[v * 3 + 0];
            vert.py = posBuf[v * 3 + 1];
            vert.pz = posBuf[v * 3 + 2];

            if (hasNormals)
            {
                vert.nx = nrmBuf[v * 3 + 0];
                vert.ny = nrmBuf[v * 3 + 1];
                vert.nz = nrmBuf[v * 3 + 2];
            }
            else { vert.nx = vert.ny = vert.nz = 0.0f; }

            if (hasUV)
            {
                vert.u = uvBuf[v * 2 + 0];
                vert.v = uvBuf[v * 2 + 1];
            }
            else { vert.u = vert.v = 0.0f; }

            // Tangents are not used without a normal map; leave a sane default.
            vert.tx = 1.0f; vert.ty = 0.0f; vert.tz = 0.0f; vert.tw = 1.0f;
            if (hasColor)
                vert.abgr = colorBuf[v];
            else
                vert.abgr = defaultColorABGR;
        }

        frame.indices.resize(indexCount);
        readBytes(file, frame.indices.data(), frame.indices.size() * sizeof(uint16_t),
                  gmcPath, "indices");

        // Validate indices and (optionally) synthesize normals.
        for (uint16_t i : frame.indices)
        {
            if (i >= vertCount)
                throw std::runtime_error("Mesh cache loader: index out of range in frame " +
                                         std::to_string(f) + " of " + gmcPath);
        }
        if (!hasNormals)
            generateNormals(frame);

        cache.frames.push_back(std::move(frame));
    }

    // Recompute maxima defensively (header values are advisory).
    uint32_t mv = 0, mi = 0;
    for (const auto& fr : cache.frames)
    {
        mv = std::max<uint32_t>(mv, static_cast<uint32_t>(fr.vertices.size()));
        mi = std::max<uint32_t>(mi, static_cast<uint32_t>(fr.indices.size()));
    }
    cache.maxVertices = std::max(cache.maxVertices, mv);
    cache.maxIndices  = std::max(cache.maxIndices,  mi);

    return cache;
}

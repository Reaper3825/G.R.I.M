#include "popup_obj_loader.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <cstdint>
#include <unordered_map>

// ===========================================================
// Lightweight Wavefront OBJ loader
// Handles: v, vn, f (triangle faces with v//vn or v/vt/vn or v)
// Auto-generates normals if none provided.
// Triangulates quads (4-vertex faces) automatically.
// ===========================================================

struct OBJFaceVertex
{
    int vi;  // vertex position index (1-based in OBJ)
    int ni;  // normal index (1-based), 0 = none
};

// Hash for deduplicating unique (position, normal) pairs
struct FaceVertexHash
{
    size_t operator()(const OBJFaceVertex& fv) const
    {
        size_t h = std::hash<int>()(fv.vi);
        h ^= std::hash<int>()(fv.ni) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct FaceVertexEqual
{
    bool operator()(const OBJFaceVertex& a, const OBJFaceVertex& b) const
    {
        return a.vi == b.vi && a.ni == b.ni;
    }
};

// Parse one face vertex token: "v", "v/vt", "v/vt/vn", "v//vn"
static OBJFaceVertex parseFaceToken(const std::string& token)
{
    OBJFaceVertex fv = { 0, 0 };

    // Find slash positions
    size_t s1 = token.find('/');
    if (s1 == std::string::npos)
    {
        // Just vertex index: "v"
        fv.vi = std::stoi(token);
        return fv;
    }

    fv.vi = std::stoi(token.substr(0, s1));

    size_t s2 = token.find('/', s1 + 1);
    if (s2 == std::string::npos)
    {
        // "v/vt" — skip texture coord
        return fv;
    }

    // "v/vt/vn" or "v//vn"
    if (s2 + 1 < token.size())
        fv.ni = std::stoi(token.substr(s2 + 1));

    return fv;
}

static void normalize3(float* v)
{
    float len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (len > 1e-8f)
    {
        v[0] /= len;
        v[1] /= len;
        v[2] /= len;
    }
}

PopupObjectDefinition loadPopupObjectFromOBJ(const std::string& objPath,
                                              uint32_t defaultColorABGR)
{
    std::ifstream file(objPath);
    if (!file.is_open())
        throw std::runtime_error("OBJ loader: cannot open file: " + objPath);

    // Raw OBJ data (1-based indexing, store at [0] unused)
    std::vector<float> positions;  // flat xyz, groups of 3
    std::vector<float> normals;    // flat xyz, groups of 3
    positions.push_back(0); positions.push_back(0); positions.push_back(0); // dummy [0]
    normals.push_back(0); normals.push_back(0); normals.push_back(0);     // dummy [0]

    // Deduplicated output vertices
    std::unordered_map<OBJFaceVertex, uint16_t, FaceVertexHash, FaceVertexEqual> vertexMap;
    std::vector<PopupVertex> vertices;
    std::vector<uint16_t> indices;

    // Temporary face vertex buffer for triangulation
    std::vector<OBJFaceVertex> faceVerts;

    std::string line;
    int lineNum = 0;
    while (std::getline(file, line))
    {
        ++lineNum;
        if (line.empty() || line[0] == '#')
            continue;

        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v")
        {
            float x, y, z;
            if (!(iss >> x >> y >> z))
                throw std::runtime_error("OBJ loader: bad vertex at line " + std::to_string(lineNum));
            positions.push_back(x);
            positions.push_back(y);
            positions.push_back(z);
        }
        else if (prefix == "vn")
        {
            float nx, ny, nz;
            if (!(iss >> nx >> ny >> nz))
                throw std::runtime_error("OBJ loader: bad normal at line " + std::to_string(lineNum));
            normals.push_back(nx);
            normals.push_back(ny);
            normals.push_back(nz);
        }
        else if (prefix == "f")
        {
            faceVerts.clear();
            std::string token;
            while (iss >> token)
                faceVerts.push_back(parseFaceToken(token));

            if (faceVerts.size() < 3)
                throw std::runtime_error("OBJ loader: face with < 3 vertices at line " + std::to_string(lineNum));

            // Triangulate: fan from vertex 0
            // Works for triangles (1 tri) and quads (2 tris) and n-gons
            for (size_t i = 1; i + 1 < faceVerts.size(); ++i)
            {
                OBJFaceVertex tri[3] = { faceVerts[0], faceVerts[i], faceVerts[i + 1] };

                for (int t = 0; t < 3; ++t)
                {
                    OBJFaceVertex& fv = tri[t];

                    // Handle negative indices (relative to end)
                    int posCount = static_cast<int>(positions.size() / 3) - 1; // -1 for dummy
                    int nrmCount = static_cast<int>(normals.size() / 3) - 1;

                    if (fv.vi < 0) fv.vi = posCount + 1 + fv.vi;
                    if (fv.ni < 0) fv.ni = nrmCount + 1 + fv.ni;

                    if (fv.vi < 1 || fv.vi > posCount)
                        throw std::runtime_error("OBJ loader: vertex index " + std::to_string(fv.vi) +
                                                 " out of range at line " + std::to_string(lineNum));

                    // Deduplicate
                    auto it = vertexMap.find(fv);
                    if (it != vertexMap.end())
                    {
                        indices.push_back(it->second);
                    }
                    else
                    {
                        if (vertices.size() >= 65535)
                            throw std::runtime_error("OBJ loader: mesh exceeds uint16 vertex limit (65535)");

                        PopupVertex vert;
                        vert.px = positions[fv.vi * 3 + 0];
                        vert.py = positions[fv.vi * 3 + 1];
                        vert.pz = positions[fv.vi * 3 + 2];

                        if (fv.ni > 0 && fv.ni <= nrmCount)
                        {
                            vert.nx = normals[fv.ni * 3 + 0];
                            vert.ny = normals[fv.ni * 3 + 1];
                            vert.nz = normals[fv.ni * 3 + 2];
                        }
                        else
                        {
                            // Will compute face normal below if no normals provided
                            vert.nx = 0.0f;
                            vert.ny = 0.0f;
                            vert.nz = 0.0f;
                        }

                        vert.abgr = defaultColorABGR;

                        uint16_t idx = static_cast<uint16_t>(vertices.size());
                        vertices.push_back(vert);
                        vertexMap[fv] = idx;
                        indices.push_back(idx);
                    }
                }
            }
        }
        // Skip: vt, mtllib, usemtl, g, o, s, etc.
    }

    if (vertices.empty() || indices.empty())
        throw std::runtime_error("OBJ loader: no geometry found in " + objPath);

    // -------------------------------------------------------
    // Auto-generate face normals for vertices that have none
    // -------------------------------------------------------
    bool hasNormals = (normals.size() > 3); // more than just the dummy
    if (!hasNormals)
    {
        // Accumulate face normals per vertex
        for (size_t i = 0; i + 2 < indices.size(); i += 3)
        {
            PopupVertex& v0 = vertices[indices[i + 0]];
            PopupVertex& v1 = vertices[indices[i + 1]];
            PopupVertex& v2 = vertices[indices[i + 2]];

            // Edge vectors
            float e1x = v1.px - v0.px, e1y = v1.py - v0.py, e1z = v1.pz - v0.pz;
            float e2x = v2.px - v0.px, e2y = v2.py - v0.py, e2z = v2.pz - v0.pz;

            // Cross product (face normal, not normalized — area-weighted)
            float fnx = e1y * e2z - e1z * e2y;
            float fny = e1z * e2x - e1x * e2z;
            float fnz = e1x * e2y - e1y * e2x;

            v0.nx += fnx; v0.ny += fny; v0.nz += fnz;
            v1.nx += fnx; v1.ny += fny; v1.nz += fnz;
            v2.nx += fnx; v2.ny += fny; v2.nz += fnz;
        }

        // Normalize accumulated normals
        for (auto& v : vertices)
        {
            float n[3] = { v.nx, v.ny, v.nz };
            normalize3(n);
            v.nx = n[0]; v.ny = n[1]; v.nz = n[2];
        }
    }

    // -------------------------------------------------------
    // Build the definition
    // -------------------------------------------------------
    PopupObjectDefinition def;
    def.vertices = std::move(vertices);
    def.indices  = std::move(indices);

    // Defaults
    def.defaultTransform = {};
    def.defaultTransform.scale[0] = 1.0f;
    def.defaultTransform.scale[1] = 1.0f;
    def.defaultTransform.scale[2] = 1.0f;

    def.defaultLight.direction[0] = -0.5f;
    def.defaultLight.direction[1] =  0.7f;
    def.defaultLight.direction[2] =  0.5f;
    def.defaultLight.intensity    =  1.0f;
    def.defaultLight.ambient      =  0.20f;

    return def;
}

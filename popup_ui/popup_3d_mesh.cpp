#include "popup_3d_mesh.hpp"
#include <bgfx/bgfx.h>
#include <stdexcept>
#include <string>

// ===========================================================
// Popup 3D Mesh — bgfx vertex/index buffer ownership
// ===========================================================

struct PopupMeshGPU
{
    // Static buffers (used when dynamic == false)
    bgfx::VertexBufferHandle vbh = BGFX_INVALID_HANDLE;
    bgfx::IndexBufferHandle  ibh = BGFX_INVALID_HANDLE;

    // Dynamic buffers (used when dynamic == true)
    bgfx::DynamicVertexBufferHandle dvbh = BGFX_INVALID_HANDLE;
    bgfx::DynamicIndexBufferHandle  dibh = BGFX_INVALID_HANDLE;

    bgfx::VertexLayout       layout;
    uint32_t                 indexCount  = 0;   // current active index count
    uint32_t                 vertexCount = 0;   // current active vertex count
    uint32_t                 maxVerts    = 0;   // capacity (dynamic only)
    uint32_t                 maxIndices  = 0;   // capacity (dynamic only)
    bool                     dynamic     = false;
};

static bgfx::VertexLayout createPopupVertexLayout()
{
    bgfx::VertexLayout layout;
    layout.begin()
        .add(bgfx::Attrib::Position,  3, bgfx::AttribType::Float)
        .add(bgfx::Attrib::Normal,    3, bgfx::AttribType::Float)
        .add(bgfx::Attrib::Tangent,   4, bgfx::AttribType::Float)
        .add(bgfx::Attrib::TexCoord0, 2, bgfx::AttribType::Float)
        .add(bgfx::Attrib::Color0,    4, bgfx::AttribType::Uint8, true)
    .end();
    return layout;
}

PopupMeshGPU* popupMeshCreate(const PopupVertex* verts, uint32_t vertCount,
                               const uint16_t* indices, uint32_t indexCount)
{
    if (!verts || vertCount == 0)
        throw std::runtime_error("popupMeshCreate: verts is NULL or vertCount is 0");
    if (!indices || indexCount == 0)
        throw std::runtime_error("popupMeshCreate: indices is NULL or indexCount is 0");

    auto* mesh = new PopupMeshGPU();
    mesh->layout     = createPopupVertexLayout();
    mesh->indexCount  = indexCount;

    const bgfx::Memory* vbMem = bgfx::copy(verts, vertCount * sizeof(PopupVertex));
    mesh->vbh = bgfx::createVertexBuffer(vbMem, mesh->layout);
    if (!bgfx::isValid(mesh->vbh))
    {
        delete mesh;
        throw std::runtime_error("popupMeshCreate: bgfx::createVertexBuffer failed");
    }

    const bgfx::Memory* ibMem = bgfx::copy(indices, indexCount * sizeof(uint16_t));
    mesh->ibh = bgfx::createIndexBuffer(ibMem);
    if (!bgfx::isValid(mesh->ibh))
    {
        bgfx::destroy(mesh->vbh);
        delete mesh;
        throw std::runtime_error("popupMeshCreate: bgfx::createIndexBuffer failed");
    }

    return mesh;
}

PopupMeshGPU* popupMeshCreateDynamic(uint32_t maxVerts, uint32_t maxIndices)
{
    if (maxVerts == 0 || maxIndices == 0)
        throw std::runtime_error("popupMeshCreateDynamic: maxVerts/maxIndices must be > 0");
    if (maxVerts > 65535)
        throw std::runtime_error("popupMeshCreateDynamic: maxVerts exceeds uint16 limit (65535)");

    auto* mesh = new PopupMeshGPU();
    mesh->layout     = createPopupVertexLayout();
    mesh->dynamic    = true;
    mesh->maxVerts   = maxVerts;
    mesh->maxIndices = maxIndices;

    // BGFX_BUFFER_ALLOW_RESIZE lets us upload smaller/larger frames safely.
    mesh->dvbh = bgfx::createDynamicVertexBuffer(maxVerts, mesh->layout,
                                                 BGFX_BUFFER_ALLOW_RESIZE);
    if (!bgfx::isValid(mesh->dvbh))
    {
        delete mesh;
        throw std::runtime_error("popupMeshCreateDynamic: createDynamicVertexBuffer failed");
    }

    mesh->dibh = bgfx::createDynamicIndexBuffer(maxIndices, BGFX_BUFFER_ALLOW_RESIZE);
    if (!bgfx::isValid(mesh->dibh))
    {
        bgfx::destroy(mesh->dvbh);
        delete mesh;
        throw std::runtime_error("popupMeshCreateDynamic: createDynamicIndexBuffer failed");
    }

    return mesh;
}

void popupMeshUpdate(PopupMeshGPU* mesh,
                     const PopupVertex* verts, uint32_t vertCount,
                     const uint16_t* indices, uint32_t indexCount)
{
    if (!mesh || !mesh->dynamic)
        return;  // static meshes are immutable
    if (!verts || vertCount == 0 || !indices || indexCount == 0)
        return;
    if (vertCount > mesh->maxVerts || indexCount > mesh->maxIndices)
        throw std::runtime_error("popupMeshUpdate: frame exceeds dynamic buffer capacity");

    const bgfx::Memory* vbMem = bgfx::copy(verts, vertCount * sizeof(PopupVertex));
    bgfx::update(mesh->dvbh, 0, vbMem);

    const bgfx::Memory* ibMem = bgfx::copy(indices, indexCount * sizeof(uint16_t));
    bgfx::update(mesh->dibh, 0, ibMem);

    mesh->vertexCount = vertCount;
    mesh->indexCount  = indexCount;
}

void popupMeshDestroy(PopupMeshGPU* mesh)
{
    if (!mesh) return;
    if (bgfx::isValid(mesh->vbh))  bgfx::destroy(mesh->vbh);
    if (bgfx::isValid(mesh->ibh))  bgfx::destroy(mesh->ibh);
    if (bgfx::isValid(mesh->dvbh)) bgfx::destroy(mesh->dvbh);
    if (bgfx::isValid(mesh->dibh)) bgfx::destroy(mesh->dibh);
    delete mesh;
}

void popupMeshBind(const PopupMeshGPU* mesh)
{
    if (!mesh)
        throw std::runtime_error("popupMeshBind: mesh is NULL");

    if (mesh->dynamic)
    {
        // Bind only the active range that was last uploaded.
        bgfx::setVertexBuffer(0, mesh->dvbh, 0, mesh->vertexCount);
        bgfx::setIndexBuffer(mesh->dibh, 0, mesh->indexCount);
    }
    else
    {
        bgfx::setVertexBuffer(0, mesh->vbh);
        bgfx::setIndexBuffer(mesh->ibh);
    }
}

uint32_t popupMeshIndexCount(const PopupMeshGPU* mesh)
{
    if (!mesh)
        throw std::runtime_error("popupMeshIndexCount: mesh is NULL");
    return mesh->indexCount;
}

#include "popup_3d_mesh.hpp"
#include <bgfx/bgfx.h>
#include <stdexcept>
#include <string>

// ===========================================================
// Popup 3D Mesh — bgfx vertex/index buffer ownership
// ===========================================================

struct PopupMeshGPU
{
    bgfx::VertexBufferHandle vbh = BGFX_INVALID_HANDLE;
    bgfx::IndexBufferHandle  ibh = BGFX_INVALID_HANDLE;
    bgfx::VertexLayout       layout;
    uint32_t                 indexCount = 0;
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

void popupMeshDestroy(PopupMeshGPU* mesh)
{
    if (!mesh) return;
    if (bgfx::isValid(mesh->vbh)) bgfx::destroy(mesh->vbh);
    if (bgfx::isValid(mesh->ibh)) bgfx::destroy(mesh->ibh);
    delete mesh;
}

void popupMeshBind(const PopupMeshGPU* mesh)
{
    if (!mesh)
        throw std::runtime_error("popupMeshBind: mesh is NULL");
    bgfx::setVertexBuffer(0, mesh->vbh);
    bgfx::setIndexBuffer(mesh->ibh);
}

uint32_t popupMeshIndexCount(const PopupMeshGPU* mesh)
{
    if (!mesh)
        throw std::runtime_error("popupMeshIndexCount: mesh is NULL");
    return mesh->indexCount;
}

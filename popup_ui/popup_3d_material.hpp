#pragma once

// ===========================================================
// Popup 3D Material — textures, samplers, shader binding
// ===========================================================

// Opaque material state (bgfx details in .cpp)
struct PopupMaterialState;

// Create material from texture file paths. Returns owned pointer.
// albedoPath: path to albedo/diffuse PNG
// packedPath: path to packed material PNG (R=occlusion, G=roughness, B=emissive, A=opacity)
//             may be nullptr if no packed texture is needed
PopupMaterialState* popupMaterialCreate(const char* albedoPath, const char* packedPath);

// Destroy material GPU resources.
void popupMaterialDestroy(PopupMaterialState* mat);

// Bind material textures for a draw call.
void popupMaterialBind(const PopupMaterialState* mat);

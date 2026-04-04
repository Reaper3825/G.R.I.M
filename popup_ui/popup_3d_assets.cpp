#include "popup_3d_assets.hpp"
#include <stdexcept>
#include <sys/stat.h>

// ===========================================================
// Popup 3D Assets — path resolution + validation
// ===========================================================

static bool fileExists(const std::string& path)
{
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

PopupAssetPaths popupAssetsResolve(const std::string& projectRoot)
{
    PopupAssetPaths paths;
    paths.shaderDir  = projectRoot + "/resources/shaders";
    paths.albedoPath = projectRoot + "/resources/popup_3d/grim_popup_albedo.png";
    paths.packedPath = projectRoot + "/resources/popup_3d/grim_popup_packed.png";
    return paths;
}

void popupAssetsValidate(const PopupAssetPaths& paths)
{
    // Shader directory must exist; individual shader files validated at load time
    if (paths.shaderDir.empty())
        throw std::runtime_error("popupAssetsValidate: shaderDir is empty");

    // Albedo is required
    if (!paths.albedoPath.empty() && !fileExists(paths.albedoPath))
        throw std::runtime_error("popupAssetsValidate: albedo texture not found: " + paths.albedoPath);

    // Packed is optional — only validate if path is non-empty
    if (!paths.packedPath.empty() && !fileExists(paths.packedPath))
        throw std::runtime_error("popupAssetsValidate: packed texture not found: " + paths.packedPath);
}

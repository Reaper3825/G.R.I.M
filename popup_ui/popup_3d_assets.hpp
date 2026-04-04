#pragma once

#include <string>

// ===========================================================
// Popup 3D Assets — cross-platform resource path resolution
// ===========================================================

struct PopupAssetPaths
{
    std::string shaderDir;     // directory containing compiled shader binaries
    std::string albedoPath;    // albedo/diffuse texture
    std::string packedPath;    // packed material texture (may be empty)
};

// Resolve asset paths relative to the executable or a known project root.
// projectRoot: path to the project root directory (e.g. from argv[0] or config)
PopupAssetPaths popupAssetsResolve(const std::string& projectRoot);

// Validate that all required assets exist. Throws on missing files.
void popupAssetsValidate(const PopupAssetPaths& paths);

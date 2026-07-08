vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO CesiumGS/cesium-native
    REF v0.62.0
    SHA512 94b6f3e5d8668c948d18ad831e86a96e980462c65330816718a851fed8f10a02f943cdcf1c93016dc0f62749c4591777f26ad80b532aa2b1b66a1225b0733479
    HEAD_REF main
    PATCHES
        availability-node-move-only.patch
)

vcpkg_replace_string(
    "${SOURCE_PATH}/CesiumGltfReader/src/decodeSpz.cpp"
[[  spz::GaussianCloud gaussians = spz::loadSpz(
      reinterpret_cast<uint8_t*>(
          buffer.cesium.data.data() + bufferView.byteOffset),
      static_cast<int32_t>(bufferView.byteLength),
      spz::UnpackOptions{spz::CoordinateSystem::UNSPECIFIED});
]]
[[  const uint8_t* pSpzData = reinterpret_cast<const uint8_t*>(
      buffer.cesium.data.data() + bufferView.byteOffset);
  const std::vector<uint8_t> spzData(
      pSpzData,
      pSpzData + static_cast<size_t>(bufferView.byteLength));
  spz::GaussianCloud gaussians = spz::loadSpz(
      spzData,
      spz::UnpackOptions{spz::CoordinateSystem::UNSPECIFIED});
]]
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DBUILD_SHARED_LIBS=OFF
        -DCESIUM_USE_EZVCPKG=OFF
        -DCESIUM_TESTS_ENABLED=OFF
        -DCESIUM_ENABLE_CLANG_TIDY=OFF
        -DCESIUM_INSTALL_HEADERS=ON
        -DCESIUM_INSTALL_STATIC_LIBS=ON
        -DCESIUM_DISABLE_CURL=OFF
        -DCESIUM_DISABLE_LIBJPEG_TURBO=OFF
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME cesium-native CONFIG_PATH share/cesium-native/cmake)
vcpkg_replace_string(
    "${CURRENT_PACKAGES_DIR}/share/cesium-native/cesium-nativeTargets.cmake"
    [=[;\$<LINK_ONLY:picosha2::picosha2>]=]
    ""
)
vcpkg_fixup_pkgconfig()

file(REMOVE_RECURSE
    "${CURRENT_PACKAGES_DIR}/debug/include"
    "${CURRENT_PACKAGES_DIR}/debug/share"
)

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
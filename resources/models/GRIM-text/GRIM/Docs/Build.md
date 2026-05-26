# Build & Run

GRIM-text is a **separate build** from the main G.R.I.M program. It MUST NOT include headers from `../../../../core/` or any G.R.I.M main-program libraries.

## Build

```powershell
cd resources/models/GRIM-text/training/TrainingLoop
cmake --preset windows-release
cmake --build --preset windows-release --target train_gpu train_tokenizer
```

TrainingLoop has its own minimal vcpkg manifest at `resources/models/GRIM-text/training/vcpkg.json`. Its presets install those dependencies into `resources/models/GRIM-text/training/vcpkg_installed` rather than the repo-root `vcpkg_installed`, because vcpkg manifest mode prunes packages that are not in the active manifest. Sharing the main GRIM install tree would let the TrainingLoop manifest remove root-project dependencies such as CPR/OpenCV/etc.

`cpp-httplib` in that pinned TrainingLoop manifest baseline installs as a header-only package without `httplibConfig.cmake`. `TrainingLoop/CMakeLists.txt` therefore resolves `httplib.h` from the manifest install root and publishes a local `httplib::httplib` alias target for the rest of the build instead of calling `find_package(httplib CONFIG REQUIRED)`.

For PSC Bridges-2, use the launcher as usual; it selects the TrainingLoop preset by GPU type:

```bash
./scripts/run_train_on_bridges2.sh --build --gpu-type h100-80
```

Manual Bridges-2 configure/build, after loading modules and setting CUDA 12 via `GRIM_CUDA_ROOT`, is:

```bash
cd resources/models/GRIM-text/training/TrainingLoop
cmake --preset bridges2-h100-release -DCUDAToolkit_ROOT="$GRIM_CUDA_ROOT"
cmake --build --preset bridges2-h100-release --target train_gpu train_tokenizer -j 100
```

## Bridges-2 launcher dependency root

`scripts/run_train_on_bridges2.sh` uses `$BRIDGES2_DIR/external/vcpkg` by default so the Bridges-2 layout matches local `TrainingLoop/CMakeLists.txt` builds. Before configuring, it verifies the checkout matches the repo's `external/vcpkg` gitlink and only runs a forced `git checkout` when that repo has drifted from the pinned gitlink or its tracked files are dirty. It still fetches the training manifest's `builtin-baseline` commit when needed so manifest mode can read `versions/baseline.json` reliably on the cluster.

If the remote `resources/models/GRIM-text/training/vcpkg_installed/x64-linux` tree already contains the three TrainingLoop dependencies (`nlohmann-json`, `flatbuffers`, and `cpp-httplib`), the launcher passes `-DVCPKG_MANIFEST_INSTALL=OFF` and reuses that installed tree instead of running `vcpkg install` again during configure. Set `GRIM_BRIDGES2_FORCE_VCPKG_INSTALL=1` only when you intentionally want to refresh that installed tree.

The launcher does not run `bootstrap-vcpkg.sh` when `$BRIDGES2_DIR/external/vcpkg/vcpkg` already exists. If the executable is missing, it bootstraps the repo-local checkout in place, which may download `vcpkg-glibc` on Linux.

The Bridges-2 launcher always prefers system `cmake` plus `ninja` when those tools are available on PATH, because vcpkg's compiler-detection subconfigure prefers the Ninja generator and the cluster's network path is less reliable than its module stack. In that common case the launcher exports `VCPKG_FORCE_SYSTEM_BINARIES=1`, even if `GRIM_ALLOW_VCPKG_TOOL_DOWNLOADS=1` is set, so the build stays on the already-loaded cluster tools. The launcher intentionally loads the cluster's default `cmake` module rather than pinning `cmake/3.30.x`, because TrainingLoop itself requires only CMake 3.20. If Bridges-2 does not provide `ninja` or `cmake`, either load them before launching or set `GRIM_ALLOW_VCPKG_TOOL_DOWNLOADS=1` / pass `--allow-vcpkg-tool-downloads` so vcpkg may download its own helper tools instead of failing during `detect_compiler`. Be aware that enabling tool downloads can still let the pinned vcpkg toolchain fetch a newer CMake helper than the project minimum when the system tools are missing.

When the launcher is interrupted during `--build` (for example with Ctrl+C while vcpkg is downloading or configuring), it now records the remote build process group, terminates that group on the next local interrupt, and only removes `$BRIDGES2_DIR/external/vcpkg/.vcpkg-root`, `TrainingLoop/build`, and `training/vcpkg_installed/x64-linux` after it verifies no live holder still owns the lock. On the next launch it also checks for orphaned vcpkg/cmake helper processes under that repo root; if no competing `run_train_on_bridges2` build state is active, it reaps those orphan holders before retrying configure. That makes repeated debug/cancel/retry loops much less sticky: interrupted runs stop leaving behind a zombie vcpkg lock plus half-written TrainingLoop build state that the next launch has to clean manually.

Use `GRIM_VCPKG_ROOT` only when intentionally pointing Bridges-2 at another pinned vcpkg checkout. The default should remain `$BRIDGES2_DIR/external/vcpkg` so local and Bridges-2 builds resolve the same toolchain path. The flash-attention sync helper owns only `external/flash-attention` and nested Cutlass; it must not deinitialize vcpkg.

If the remote `TrainingLoop/build/CMakeCache.txt` points at an older vcpkg toolchain root, the launcher deletes that build directory before reconfiguring. Changing `CMAKE_TOOLCHAIN_FILE` in-place on an existing CMake cache is unsupported and will otherwise keep routing manifest installs back to the stale vcpkg root.

## Run training

```powershell
cd resources/models/GRIM-text/training
.\TrainingLoop\build\Release\train_gpu.exe
```

## Tokenizer self-test

```powershell
cd resources/models/GRIM-text/training/build
cmake --build . --config Release --target unigrambyte_self_test
.\Release\unigrambyte_self_test.exe
```

## Server

`grim_text_server.exe` runs on port 11435 as a pure HTTP bridge. It launches `train_gpu --inference --inference-worker-port 11436` and proxies requests; `train_gpu` owns Phase1 startup, `TrainingContext`, tokenizer access, and Phase2 inference.

## CMake cache trap

When removing `.cu` files from `CMakeLists.txt`, clean the cache to remove stale device-link objects:

```powershell
Remove-Item -Recurse -Force build\CMakeFiles\grim_training_kernels.dir
# or:
cmake --build build --config Release --clean-first
```

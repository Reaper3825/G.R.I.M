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

`scripts/run_train_on_bridges2.sh` now defaults to **manual TrainingLoop deps** on Bridges-2 (`GRIM_BRIDGES2_USE_MANUAL_DEPS=1`). In that mode it seeds three source archives — `nlohmann-json`, `flatbuffers`, and `cpp-httplib` — from the local cache into the remote cache and then runs `resources/models/GRIM-text/training/scripts/prepare_manual_deps.py` **on the Linux side** to stage headers under `resources/models/GRIM-text/training/third_party`. That cross-OS split is intentional: the local machine only uploads platform-neutral source tarballs, while the remote Linux machine performs the extraction and build. No Windows-built libraries or headers are copied into the Linux build tree.

This means a Windows local workstation is fine. The archives are plain upstream source tarballs, the staged TrainingLoop dependencies are header-only for this build, and the actual compile still happens entirely on Bridges-2 with the cluster toolchain.

If you intentionally want the old manifest flow instead, set `GRIM_BRIDGES2_USE_MANUAL_DEPS=0`. In that fallback mode the launcher uses `$BRIDGES2_DIR/external/vcpkg` so the Bridges-2 layout matches local `TrainingLoop/CMakeLists.txt` vcpkg builds. Before configuring, it verifies the checkout matches the repo's `external/vcpkg` gitlink and only runs a forced `git checkout` when that repo has drifted from the pinned gitlink or its tracked files are dirty. It still fetches the training manifest's `builtin-baseline` commit when needed so manifest mode can read `versions/baseline.json` reliably on the cluster.

In vcpkg fallback mode, if the remote `resources/models/GRIM-text/training/vcpkg_installed/x64-linux` tree already contains the three TrainingLoop dependencies (`nlohmann-json`, `flatbuffers`, and `cpp-httplib`), the launcher passes `-DVCPKG_MANIFEST_INSTALL=OFF` and reuses that installed tree instead of running `vcpkg install` again during configure. Set `GRIM_BRIDGES2_FORCE_VCPKG_INSTALL=1` only when you intentionally want to refresh that installed tree.

The launcher does not run `bootstrap-vcpkg.sh` when `$BRIDGES2_DIR/external/vcpkg/vcpkg` already exists. If the executable is missing, it bootstraps the repo-local checkout in place, which may download `vcpkg-glibc` on Linux.

The Bridges-2 launcher always prefers system `cmake` plus `ninja` when those tools are available on PATH. In manual-deps mode that is usually all it needs. In vcpkg fallback mode it also checks whether the cluster `cmake` is new enough for the pinned `external/vcpkg` helper requirement. In the compatible case the launcher exports `VCPKG_FORCE_SYSTEM_BINARIES=1`, even if `GRIM_ALLOW_VCPKG_TOOL_DOWNLOADS=1` is set, so the build stays on the already-loaded cluster tools. The launcher intentionally loads the cluster's default `cmake` module rather than pinning `cmake/3.30.x`, because TrainingLoop itself requires only CMake 3.20.

In manual-deps mode, the launcher seeds the same remote cache path (`$BRIDGES2_DIR/external/vcpkg/downloads` by default, override with `GRIM_BRIDGES2_VCPKG_DOWNLOADS`) with the three source archives needed by `prepare_manual_deps.py`, then stages those headers remotely. In vcpkg fallback mode, when helper downloads are allowed (`GRIM_ALLOW_VCPKG_TOOL_DOWNLOADS=1` or `--allow-vcpkg-tool-downloads`) and the cluster `cmake` is missing or older than the pinned helper version, the launcher exports `VCPKG_DOWNLOADS` explicitly and prefers seeding that remote cache from the local repo cache (`$REPO_ROOT/external/vcpkg/downloads`, override with `GRIM_LOCAL_VCPKG_DOWNLOADS`) before the build starts. If the local cache does not already have the pinned helper archive, the launcher attempts a local download first and then uploads the cached tarball to Bridges-2. Only if that local-seed path is unavailable does it fall back to downloading from the cluster with `aria2c`, `wget`, or `curl`. That keeps the first slow GitHub download off the HPC login node whenever your local machine can fetch the archive faster. If helper downloads stay disabled, the launcher now fails early with a clear error when the cluster `cmake` is too old instead of letting vcpkg surprise you mid-configure.

When the launcher is interrupted during `--build` (for example with Ctrl+C while vcpkg is downloading or configuring), it now records the remote build process group, terminates that group on the next local interrupt, and only removes `$BRIDGES2_DIR/external/vcpkg/.vcpkg-root`, `TrainingLoop/build`, and `training/vcpkg_installed/x64-linux` after it verifies no live holder still owns the lock. On the next launch it also checks for orphaned vcpkg/cmake helper processes under that repo root; if no competing `run_train_on_bridges2` build state is active, it reaps those orphan holders before retrying configure. That makes repeated debug/cancel/retry loops much less sticky: interrupted runs stop leaving behind a zombie vcpkg lock plus half-written TrainingLoop build state that the next launch has to clean manually.

Use `GRIM_VCPKG_ROOT` only when intentionally pointing Bridges-2 at another pinned vcpkg checkout in fallback mode. The flash-attention sync helper owns only `external/flash-attention` and nested Cutlass; it must not deinitialize vcpkg or the staged manual dependency tree.

If the remote `TrainingLoop/build/CMakeCache.txt` points at an older vcpkg toolchain root **or** still reflects a prior dependency mode (`manual` vs `vcpkg`), the launcher deletes that build directory before reconfiguring. Changing `CMAKE_TOOLCHAIN_FILE` or dependency mode in-place on an existing CMake cache is unsupported and will otherwise keep routing configuration back to the stale path.

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

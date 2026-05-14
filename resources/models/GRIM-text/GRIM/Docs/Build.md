# Build & Run

GRIM-text is a **separate build** from the main G.R.I.M program. It MUST NOT include headers from `../../../../core/` or any G.R.I.M main-program libraries.

## Build

```powershell
cd resources/models/GRIM-text/training/TrainingLoop
cmake --build build --config Release --target train_gpu
cmake --build build --config Release --target grim_text_server
```
 vb
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

`grim_text_server.exe` runs on port 11435.

## CMake cache trap

When removing `.cu` files from `CMakeLists.txt`, clean the cache to remove stale device-link objects:

```powershell
Remove-Item -Recurse -Force build\CMakeFiles\grim_training_kernels.dir
# or:
cmake --build build --config Release --clean-first
```

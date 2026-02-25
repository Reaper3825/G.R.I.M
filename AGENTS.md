# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

GRIM (G.R.I.M) is a modular, C++-based personal AI assistant. The main executable is **Windows-only** (Win32 API, SAPI, HWND, bgfx). The GRIM-text subsystem (custom transformer model) requires **CUDA 12.5+** with sm_86 architecture. See `README.md` for project structure and `.github/copilot-instructions.md` for GRIM-text coding conventions.

### Platform Limitations on Linux Cloud VMs

- **Main GRIM binary**: Cannot build or run on Linux. It uses Win32 APIs (`CreateWindowExW`, `PeekMessage`, `SAPI`, `<crtdbg.h>`) and vcpkg x64-windows triplet. CMake configure also requires CMake 3.29+ for `enable_language(CUDA OPTIONAL)` (3.28 is installed).
- **GRIM-text training/inference**: Requires CUDA toolkit and NVIDIA GPU (sm_86). Not available in cloud VMs.
- **Python bridges** and **Flask server** (`grim_text_server.py`): Work on Linux. This is the primary development surface for cloud agents.

### What Works on Linux

| Component | Status | How to Run |
|-----------|--------|------------|
| Python bridges (Coqui TTS, OSINT, RL, Mistral) | Dependencies install, module imports work | `.venv/bin/python -c "import <module>"` |
| GRIM-text Flask API server | Functional (placeholder responses without C++ DLL) | `.venv/bin/python grim_text_server.py` or use Flask test client |
| Python linting (flake8) | Works | `.venv/bin/python -m flake8 --max-line-length=120 <files>` |
| C++ static analysis (cppcheck) | Works | `cppcheck --enable=style --std=c++20 --language=c++ <files>` |
| Node.js (openai package) | Works | `npm install` |

### Development Environment Setup

- **Python venv**: Located at `/workspace/.venv`. Activate with `source /workspace/.venv/bin/activate`.
- **Node.js**: Uses nvm (v22.x). `package.json` only depends on `openai`.
- **Git submodules**: `external/whisper.cpp` and `external/flash-attention` are initialized. `external/vcpkg` is not in `.gitmodules` (tracked as a regular directory, usually present only on Windows dev machines). `external/bgfx.cmake` is defined in `.gitmodules` but the directory is not present.

### Linting Commands

- **Python**: `.venv/bin/python -m flake8 --max-line-length=120 grim_text_server.py resources/python/*.py`
- **C++**: `cppcheck --enable=style --suppress=missingIncludeSystem --std=c++20 --language=c++ main.cpp logger.cpp system_detect.cpp`

### Testing the Flask API (without C++ DLL)

Use Flask's test client for development testing:
```python
from grim_text_server import app
client = app.test_client()
resp = client.post('/api/generate', json={'prompt': 'Hello', 'max_tokens': 50})
```

### Key Gotchas

- `grim_text_server.py` calls `sys.exit(1)` if the C++ model DLL is not found. Use the Flask test client to bypass this for API development.
- The `resources/python/requirements.txt` pins `numpy==1.22.0` which is incompatible with Python 3.12+. Install numpy without version pin on modern Python.
- PyTorch is installed CPU-only (`--index-url https://download.pytorch.org/whl/cpu`) since no NVIDIA GPU is available.
- `transformers` version in requirements is pinned to `4.33.0` but newer versions work fine for bridge development.

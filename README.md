# G.R.I.M — General Request and Information Manager

GRIM is a modular, C++-based personal AI assistant and automation platform. It combines natural language processing, plugin-based command handling, and extensible subsystems (voice, wakeword, UI, device control) so you can build a Jarvis-style assistant tailored to your environment.

Key goals
- **Offline-first design**: All core functions (NLP, voice, commands, AI inference) work completely offline. Only browser-based features require internet connectivity.
- Practical local-first assistant with extendable plugins
- Natural language intent handling and memory
- Cross-platform design with configuration for native dependencies

Status
- Active development. The project contains native C++ sources, CMake build scripts, optional CUDA support, and integrations for third-party libraries (e.g. whisper, porcupine).

Table of contents
- Overview
- Features (current & roadmap)
- Quick start (build & run)
- Development notes
- Project structure
- Contributing
- License & contact

## Overview

GRIM (G.R.I.M) is intended as a foundation for a personal assistant that can be extended via plugins and modules. The codebase is written in modern C++ (C++20) and organized to separate concerns across directories such as `ai/`, `commands/`, `voice/`, `nlp/`, `plugins/`, and `ui/`.

**Privacy & Offline Operation**: GRIM is designed to function entirely offline for all core features. Voice processing (Whisper.cpp, Coqui TTS), wake word detection (Porcupine), NLP, and AI inference run locally on your machine without requiring internet connectivity. Only features explicitly designed for web interaction (browser commands, external APIs) require network access.

The project can be used as a command-line assistant or integrated into a richer UI through the `ui` and `popup_ui` modules. The design favors local models and components where possible, while providing hooks for external integrations.

## Features

Current (based on repository contents)
- Intent recognition and NLP modules (see `nlp/` and `ai/`)
- Command and plugin systems with hot-reloadable plugin architecture (`commands/`, `plugins/`)
- Voice synthesis via **Coqui TTS** (XTTS model support via Python bridge)
- Speech-to-text via **Whisper.cpp** integration (local inference)
- Wakeword detection using **Porcupine** wake word engine
- OSINT capabilities via **Sherlock** username search integration
- Build configured with CMake 3.22+, C++20, optional CUDA support, and **CMakePresets.json** for Visual Studio

Planned / Roadmap items (described in repo but may be in-progress)
- Enhanced voice interaction (continuous conversation mode)
- External API hooks (calendar, email, music streaming services)
- Personal knowledge graph and expanded long-term memory
- Cross-platform polish and optional VR overlay mode (Quest headset integration)
- Biometric profiling using PIR sensors and camera-based silhouette/face recognition

Note: The README avoids promising features that are not present in the code. See the repository files and modules for exact capabilities and current implementation status.

## Quick start — Build & run (Linux / macOS / Windows with minor adjustments)

Prerequisites
- CMake 3.22 or newer
- A C++20-capable compiler (g++ / clang / MSVC)
- Optional: CUDA toolkit if you want CUDA-accelerated code paths
- Optional: vcpkg or other dependency management for third-party libs (the CMake config references a vcpkg layout)

Recommended build steps (out-of-source build):

```bash
# from repository root
mkdir -p build
cd build
cmake -S .. -B . -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release --parallel

# Run the built executable (binary name: GRIM)
./GRIM
```

**Using CMakePresets (Visual Studio / VS Code)**

The project includes `CMakePresets.json` for streamlined builds with VS Code or Visual Studio:

```bash
# Configure with preset
cmake --preset=release   # or debug

# Build with preset
cmake --build --preset=release
```

This preset approach handles vcpkg toolchain, CUDA paths, and configuration automatically.

Notes
- On Windows the build uses vcpkg paths and copies DLLs into the runtime folder. If you use vcpkg, set `VCPKG_INSTALLED_DIR` or use the included vcpkg manifest approach.
- Dependencies such as **Whisper.cpp**, **Porcupine**, and graphics/audio libraries are automatically linked; ensure their build artifacts are in the expected paths or adjust CMake variables.
- **Coqui TTS** and **Sherlock OSINT** are accessed via Python bridges in `resources/python/`. Install requirements via `pip install -r resources/python/requirements.txt`.

VS Code
- The workspace contains a VS Code task for compiling a single C++ file. For the full project prefer the CMake flow above or use CMake Tools extension and the provided `CMakePresets.json`.

## Development notes

- **Code style**: modern C++ (C++20), precompiled headers are used (`pch.hpp`).
- **Primary CMake target**: `GRIM` (created by `add_executable(GRIM ...)`).
- **Hot-reloadable plugins**: the `plugins/` folder supports dynamic plugin loading at runtime. Example: `osint_plugin` (Sherlock integration). The `core_plugin` is compiled directly into the host executable.
- **Third-party integrations**:
  - **Whisper.cpp** — local speech-to-text (see `external/whisper.cpp`)
  - **Porcupine** — wake word detection engine
  - **Coqui TTS** — text-to-speech via Python bridge (`resources/python/coqui_bridge.py`)
  - **Sherlock** — OSINT username search via Python bridge (`resources/python/osit_bridge.py`)
- **Python bridges**: located in `resources/python/`, these allow C++ to call Python-based AI/OSINT tools. Install dependencies with `pip install -r resources/python/requirements.txt`.
- **CMakePresets.json**: pre-configured build presets for Debug/Release with CUDA 12.5 support and vcpkg integration.

Testing & running small pieces
- For fast iteration you can build individual modules or create small test harnesses that link against the core target. See `commands/` and `ai/` for examples of components and how they are wired into the main target.

## Project structure (high level)

- `ai/` — AI-related modules and intent handling
- `commands/` — command implementations and command routing
- `plugins/` — plugin implementations (hot-reloadable plugins and core plugin)
- `voice/`, `wake/` — audio, wakeword, and voice features
- `nlp/` — natural language processing helpers and rules
- `ui/`, `popup_ui/` — user-facing interface components
- `cmake/` — CMake helper modules used by the build
- `external/` — third-party code and integrations (may require separate builds)

Explore the repository to find more specific files and examples for each subsystem.

## Contributing

Contributions are welcome! A few guidelines:

**Core Principles:**
- **Maintain offline-first design**: All core features must work completely offline. Only features explicitly meant for web interaction (browser commands, external API integrations) should require internet connectivity.
- Open an issue to discuss larger changes before starting work.
- Keep changes small and focused; prefer documentation and tests for new behavior.
- Follow the existing C++ style (C++20) and include unit tests where practical.

**Development Guidelines:**
- Use local models and inference wherever possible (Whisper.cpp, Coqui TTS, local LLMs).
- Avoid introducing cloud dependencies for core functionality.
- If adding features that require licenses (models, third-party SDKs, or proprietary binaries), document those requirements in `docs/` or the relevant module folder.
- Hot-reloadable plugins should be self-contained and minimize dependencies on the host executable.

We value contributions that enhance privacy, performance, and extensibility while keeping GRIM a true local-first AI assistant.

## License

No license file is included in this repository by default. If you intend to publish or share this project, add a license (e.g., MIT, Apache-2.0) and include it as `LICENSE` in the repo root.

## Contact & credits

Author / maintainer: see repository metadata (commits). For questions, open an issue or create a pull request.

## Next steps / suggested improvements

- Add a concise `CONTRIBUTING.md` and `LICENSE` file.
- Add example `config` files and a minimal demo scenario showing a common workflow (e.g., open an app, query an intent, run a command).
- Add CI that builds the project on Linux/macOS/Windows and runs basic smoke tests.

---

If you'd like, I can also:
- add badges (build / license / coverage),
- include a short runnable example, or
- generate a `CONTRIBUTING.md` and `LICENSE` for the repo.

Thanks for building GRIM — tell me which follow-ups you'd like and I'll implement them.

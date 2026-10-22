# TextWorld dependency setup

TextWorld is an optional standalone environment dependency. Inference feedback,
tool dispatch, services, and training integration are deferred.

## Dependency ownership

- C++ libraries remain managed by the root `vcpkg.json` manifest and existing
  toolchain. TextWorld is a Python/Linux application, not a Windows C++ port;
  this setup does not add an overlay port or run vcpkg/CMake.
- Source follows the existing Git submodule convention: `external/textworld`,
  upstream release `1.7.0`, commit
  `9fce9ee107fa042ef2656e41e0b362450a35ecd8`.
- Docker installs the official TextWorld 1.7.0 Linux x86_64 wheel using the URL
  and SHA-256 in `scripts/textworld/requirements.txt`. The wheel bundles Inform
  compiler assets, avoiding the source installer’s legacy HTTP download.
- Python dependencies are isolated in the image, separate from the existing
  TTS/RL environments and `resources/python/requirements.txt`.
- TextWorld itself is pinned. The Python base-image tag, Debian packages, and
  transitive Python dependencies are not fully locked. The resolved Python
  package list is saved at `/opt/grim-textworld/installed-packages.txt` in the
  image. This is not a claim of bit-for-bit reproducible builds.

## Windows setup

Install and start Docker Desktop using Linux containers. Hardware virtualization
and Docker's chosen Linux backend must be available. Docker installation is
managed by the user; the setup script does not install it or change OS features.

From the repository root in PowerShell:

```powershell
git submodule update --init external/textworld
./scripts/setup_textworld.ps1
```

This builds only `grim-textworld:1.7.0` using the small `scripts/textworld`
build context, then runs the standalone smoke check. It does not send the GRIM
repository, models, or configuration to Docker. The Linux amd64 platform is
explicit because the pinned wheel targets x86_64.

To rerun verification without building:

```powershell
./scripts/setup_textworld.ps1 -CheckOnly
```

The check generates a deterministic small `.z8` game, compiles it, resets the
environment, executes `look`, plays the generated walkthrough to a win, resets
again, and closes the environment. Walkthrough access is only for installation
verification. No model is involved. Games are temporary and removed afterwards.
The test container has networking disabled and no workspace mounts.

Successful output includes `"status": "passed"`. Import success alone does not
establish that game compilation and execution work.

## Standalone use

```powershell
# Interactive shell in the installed environment; nothing persists on exit.
docker run --rm -it --network none grim-textworld:1.7.0 bash

# Inspect the resolved installed package versions.
docker run --rm --network none grim-textworld:1.7.0 cat /opt/grim-textworld/installed-packages.txt
```

No ports are exposed and no background service is started. Persistent games,
worker transport, and GRIM session integration can be added separately.

## Validation status

Source checkout and setup files were prepared on Windows without Docker installed.
The Docker image build and executable smoke check remain pending until Docker is
available. The PowerShell script reports that prerequisite explicitly rather than
reporting an installed runtime.

Upstream: https://github.com/microsoft/TextWorld/tree/1.7.0

TextWorld is MIT-licensed; its bundled/dependent components have their own
licenses (including Inform and Jericho). See the upstream README and license
files when distributing the image.

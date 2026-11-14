# Input provider — API v11

The provider owns raw argument parsing, key/button interpretation and native
Windows execution in `InputController` and `input_service`. It imports no host
input implementation and has no perception runtime dependency. The public ABI
has no input forwarding slot. Load/unload never injects input.

Four Debug tools retain raw-text arguments: `move_mouse`, `click`, `type`, `key`.
INPUT permission and a current provider are checked before execution. Calls are
serialized; the host registry drains active tool calls before unloading the DLL.
Unknown buttons/keys and empty combinations fail. Windows send/move failures
fail explicitly; success does not verify the focused application's response.
Non-Windows native input is unavailable. Typing retains the legacy byte-oriented
`VkKeyScanA` behavior and skips unmappable characters; Unicode typing is not added.

Pure observation targeting and checked-click helpers stay under `core/input`.
They do not register `click_on`. Host UI keybindings, cursor presentation and
platform window event handling are separate and remain host-owned.

## Validation (no deployment)

```powershell
cmake -S plugins/input_plugin -B .codex-build/input-owned -G "Visual Studio 17 2022" -A x64 -DGRIM_DEPLOY_PLUGINS=OFF -DGRIM_INPUT_PROVIDER_TESTS=ON
cmake --build .codex-build/input-owned --config Release --target input_plugin input_provider_tests --parallel 2
ctest --test-dir .codex-build/input-owned -C Release --output-on-failure
powershell -File tests/run_plugin_shutdown.ps1 -InputPluginDirectory .codex-build/input-owned/plugins
powershell -File tests/run_click_target.ps1
```

Provider tests substitute native device calls. DLL integration only invokes
invalid arguments. No test injects input or launches GRIM. Rebuild the host and
all providers against v11 before deploying; existing v10 DLLs are incompatible.

# Application provider: user aliases — API v12

The application provider owns user alias loading/resolution and native discovery.
There is no host alias store, scanner, detached refresh worker, or alias-specific
ABI service. ToolRegistry command aliases are unrelated and unchanged.

## Configuration and compatibility

The provider reads `application_aliases.json` in its configuration directory:

```json
{
  "version": 1,
  "aliases": {
    "editor": "C:/Tools/Editor/editor.exe",
    "work browser": "https://example.com/work"
  }
}
```

The directory is selected by absolute `GRIM_APPLICATION_CONFIG_DIR`, otherwise
the build's repository `resources` directory when available, then the host
executable's `resources` directory on Windows, then current-directory `resources`.
It is selected once per provider load. No directories or files are created by
loading/refreshing aliases. To edit aliases, edit this JSON and call
`applications.refresh`; there are no new alias-management tools.

If the JSON file does not exist, the provider reads **only the user vector** from
the existing `app_aliases.fb`. Automatic records are ignored. The original file
is never rewritten, deleted, or silently repaired. The compatibility schema and
generated reader are private provider sources. A present JSON file takes full
precedence, including an empty `aliases` object; it does not merge stale records.

Names are trimmed and matched case-insensitively for ASCII names. Targets retain
their exact text. Duplicate normalized names, invalid UTF-8, empty/oversized
values, malformed files and corrupt legacy buffers fail that backend's refresh;
the last valid aliases and resource snapshot remain available. Errors appear in
the existing catalog `errors` array. If no prior valid aliases exist, none are
loaded. Missing both files means an empty alias set. Each file is limited to
16 MiB, with at most 10,000 entries, 256-byte names and 8,192-byte targets.

On Windows `open editor` resolves the explicit alias before calling the native
opener. Unmatched raw targets retain their original untrimmed behavior. Targets
are passed to the OS opener, not interpreted as shell command lines. There is
no fuzzy application-alias fallback. Existing non-Windows raw `open` stubs are
unchanged; native `applications.launch` remains available through each platform's
existing discovery backend.

Existing absolute Windows executable/shortcut targets (`.exe`, `.lnk`, `.bat`,
`.cmd`) also appear as `user.alias` catalog records. Two user names for the same
target retain separate identities. Other explicit targets, such as URLs or
documents, resolve through `open` without becoming application catalog entries.

## Automatic discovery

Windows Shell Applications-folder discovery is supplemented by Start Menu
shortcuts and shallow portable-program discovery under `D:\Program Files`,
`D:\Program Files (x86)`, `D:\Apps`, `D:\Games`, `C:\Program Files`, and
`C:\Program Files (x86)`. This preserves the retired scanner's directory coverage
and additionally checks direct files in each root. It does not import or maintain
an automatic alias cache. Scans run at load and explicit catalog refresh, not in
detached background threads. Native discovery failures retain last-good backend
snapshots. Linux/macOS native discovery remains unchanged.

## Isolated validation

```powershell
cmake -S plugins/application_plugin -B .codex-build/application-owned-aliases -G "Visual Studio 17 2022" -A x64 -DGRIM_DEPLOY_PLUGINS=OFF -DGRIM_APPLICATION_ALIAS_TESTS=ON
cmake --build .codex-build/application-owned-aliases --config Release --target application_plugin application_alias_tests --parallel 2
ctest --test-dir .codex-build/application-owned-aliases -C Release --output-on-failure
powershell -File tests/run_plugin_shutdown.ps1 -BuildApplicationFixture
```

Alias/storage tests use a private directory and synthetic FlatBuffers. DLL tests
use fake discovery and native launch implementations plus isolated configuration.
They do not launch applications or modify the real alias database. No deployment
occurs. API v12 requires matching host and plugin rebuilds before deployment.

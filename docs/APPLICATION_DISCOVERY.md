# Application discovery and provider resources

The Application plugin opts into a host-owned catalog during initialization.
After successful provider commit, the host scans applications and publishes
`application` resources owned by that exact plugin generation. Failed plugin
initialization performs no scan or publication. Unload/replacement removes the
generation's resources through the existing registry lifecycle.

## Organization

- `core/applications/application_catalog.*`: common records, backend composition,
  identity deduplication, generation tracking, atomic resource publication and
  guarded operations. No OS discovery details or plugin callbacks.
- `application_backends.cpp`: backend assembly and the existing cached alias adapter.
- `application_backend_windows.cpp`: Windows Shell Applications folder enumeration
  with `SHGetKnownFolderItem(FOLDERID_AppsFolder)` and `IEnumShellItems`. Includes
  desktop and packaged apps registered there. Launches the recorded shell item
  through `ShellExecuteExW` and a parsed PIDL. UTF-8/UTF-16 boundaries are explicit;
  COM initialization, interfaces and allocated metadata use scoped cleanup.
- `application_backend_linux.cpp`: XDG desktop entries, with per-user precedence,
  Hidden/NoDisplay, desktop visibility and TryExec checks. Flatpak and Snap export
  locations supplement XDG paths. Launch delegates to an existing `gio launch`
  through `posix_spawnp` argv, never a shell or a homegrown Exec parser.
- `application_backend_macos.mm`: application bundles in standard per-user,
  system and network application directories, with NSBundle identity and
  NSWorkspace/Launch Services activation.
- `plugins/application_plugin/application_tools.*`: tool contracts and host API
  forwarding. Existing raw-text `open` and `search` behavior remains compatible.

No third-party discovery package is required. A future appcatalog or other
scanner adapter can implement `Backend` without changing resource consumers.
Backends and launch dispatch are host-owned; do not retain DLL callbacks in them.

## Resource identity and refresh

Identity is a native namespace plus native ID, not a display name. The resource
ID is a deterministic hash of that pair; collisions abort publication. Records
with the same identity merge provenance in backend priority order. Equal names
with different identities remain separate. Windows Shell entries and legacy
file aliases use different namespaces and can describe the same application;
they are deliberately not merged by name or a guessed executable relationship.

A runtime ID remains unchanged across identical scans. Changed metadata or
removal/reappearance produces a new runtime ID. Launch requires the resource ID,
runtime ID and provider instance from a current observation. Running state is
`unknown`: an installed application is not a running process or window.

Successful scans replace that backend's snapshot. Failed scans preserve its last
good snapshot and return a backend error with `retained_previous`; failures must
not look like uninstallations. The merged resource batch replaces only the
current provider's `application` kind in one registry transaction. Other resource
kinds are preserved. Stale generations cannot republish.

Discovery runs on provider commit and explicit `applications.refresh`; there is
no filesystem watcher or periodic scanning yet. Calls are synchronous and serialized
per catalog. The alias adapter reads existing cached aliases, filters to existing
absolute Windows `.exe`/`.lnk` files and does not trigger the old directory scanner.
Refresh aliases through their existing service before refreshing this catalog.

## Tools

| Tool | JSON arguments |
| --- | --- |
| `applications.list` | `{}` or `{"offset":25,"limit":25}` |
| `applications.refresh` | `{}` |
| `applications.inspect` | `{"resource_id":"application:<id>"}` |
| `applications.launch` | `{"resource_id":"application:<id>","provider_instance":"<observed instance>","runtime_id":1}` |

The last two identity values must be copied from the actual resource, not the
example. Discovery/inspection require SYSTEM + FILESYSTEM. Launch also requires
PROCESS and is marked as requiring confirmation under normal tool policy.
Tools return a `grim-application-result` version 1 envelope with `success` and
either `value` or `error`. Catalog values include `applications`, `complete`, and
`errors`. Listings/refresh return the first 25 entries by default; list accepts
offset and limit (1-50), with a byte budget below the plugin result limit.
Follow `next_offset` until null; `total` is the full catalog size. All resources
are published regardless of pagination. A successful refresh can be incomplete;
inspect `complete` and `errors`.

The live world automatically projects these resources into the Application room.
Its manual viewer remains discovery/inspection only; this does not add production
tool execution to the viewer or change authored TextWorld episodes.

## Coverage and verification

This is a desktop application catalog, not every executable or installed package.
Unregistered portable apps need an alias or future custom-directory backend.
macOS custom locations and Linux unregistered AppImages are not scanned. Linux
currently uses the base desktop-entry name rather than locale-specific names;
`gio` must already exist to launch applications. Windows/macOS icon rendering is
not implemented; Linux returns the desktop entry's icon reference.

`tests/run_application_catalog.ps1` tests catalog identity, failed scans, removals,
generation guards and atomic publication, then runs read-only native Windows
discovery without launching apps. `tests/run_plugin_shutdown.ps1
-BuildApplicationFixture` tests the real plugin/API/registry lifecycle with a
fake OS backend and launcher. Linux/macOS backends need native-platform testing.

Platform references:
- https://learn.microsoft.com/windows/win32/api/shlobj_core/nf-shlobj_core-shgetknownfolderitem
- https://specifications.freedesktop.org/desktop-entry/latest-single/
- https://developer.apple.com/documentation/appkit/nsworkspace

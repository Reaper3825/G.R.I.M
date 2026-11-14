# OSINT provider — API v11

This DLL exclusively owns seven existing Ignore tools: `profile_person`,
`sherlock_sweep`, `osint_report`, `osint_status`, `osint_clear_cache`,
`osint_scan_secrets`, and `osint_show_secrets`. Command handlers, collectors,
report cache, retained async jobs and Python bridge sources live here.
No OSINT collector is compiled into the host. The UI permission is no longer
requested; FILESYSTEM, NETWORK and PROCESS are checked before tool execution.

Load captures the absolute working directory as the provider root without
starting Python, creating cache directories, or contacting websites. Bridges
are resolved under `<root>/plugins/osint_plugin/python`; existing installed
Python/Sherlock/scanner dependencies are used. No automatic installation occurs.
The bridge's local Sherlock discovery still uses `<root>/osint/sherlock`.
The Python scripts must accompany the provider source/runtime layout. Moving
only the DLL to an unrelated directory does not package its scripts.

The existing `<root>/cache/osint` durable cache survives unload. Its 24-hour
report expiry remains. Tool calls and storage access are serialized to avoid
collisions with active jobs. Unique temporary scan files are cleaned on every exit and use the cache directory,
not the Python source directory. Targets are bounded to 128 ASCII letters,
digits, `_`, `-`, `.`, `@`, `+`, rejecting traversal, reserved device filenames and option-like names.
Report export paths remain explicitly user-selected.

Child execution is shell-free. Windows uses a kill-on-close Job Object;
POSIX uses a child process group. Cancellation/timeout terminates child work.
Sherlock's outer timeout is max(300, per-site timeout * 15) + 30 seconds;
sensitive scans have a 600-second limit. Shutdown signals cancellation before
joining all retained async jobs. In-flight synchronous tools drain before the
host calls shutdown, so unloading can wait for their bounded execution.
Windows process behavior is tested; POSIX requires platform validation.

The retained question handler invokes `osint_report` and receives copied text.
The host findings panel invokes `osint_show_secrets`; its structured output is
`{format:"grim-osint-findings",version:1,username,entries:[...]}`. Entries retain
JSONL source metadata and findings filtered by the requested severity. Missing
reports return the legacy message without structured output. The UI consumes
only this copied snapshot. `osint_show_ui` remains unregistered.

## Validation (no deployment)

```powershell
cmake -S plugins/osint_plugin -B .codex-build/osint-owned -G "Visual Studio 17 2022" -A x64 -DGRIM_DEPLOY_PLUGINS=OFF -DGRIM_OSINT_PROVIDER_TESTS=ON
cmake --build .codex-build/osint-owned --config Release --target osint_plugin osint_provider_tests osint_process_tests --parallel 2
ctest --test-dir .codex-build/osint-owned -C Release --output-on-failure
```

Provider tests use a private temporary root and a fake process boundary.
Native process tests launch only the test executable itself, verifying argument
quoting, timeout, cancellation and restart. No websites, Python dependencies,
real cache, UI, GRIM or model runtimes are exercised.

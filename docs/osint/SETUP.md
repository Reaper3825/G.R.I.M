# OSINT Setup and Troubleshooting

Source-reviewed November 8, 2026. This guide does not install dependencies or
authorize network scanning. See [tools](TOOLS.md#commands) for self-audit scope.

## Prerequisites

- A host and loaded OSINT plugin with the required permissions; see
  [plugin deployment](../../plugins/README.md) and [API](../PLUGIN_API.md).
  Registration metadata is not a sandbox or a substitute for authorization.
- Launch from the intended repository/process working directory.
  [isSherlockAvailable](../../external_collector/osit.cpp#L46) looks there for
  `resources\python\osit_bridge.py`; cache and intermediate files also use it.
- A runnable Python interpreter and a Sherlock release with its own supported
  interpreter/dependencies. No universal "Python 3.7+" minimum is established.
  [getPythonExecutable](../../external_collector/osit.cpp#L24) takes the first
  `where python` result on Windows, otherwise attempts `python`; non-Windows
  uses `python3`. Internal `OSINTConfig.pythonPath` can override it, but the
  registered tools expose no interpreter-selection flag.
- Write access to `resources\python`, `cache\osint`, and any chosen export
  directory, plus authorized connectivity to the public sites being audited.

[find_sherlock](../../resources/python/osit_bridge.py#L13) tries, in order:
importable `sherlock_project`, importable `sherlock`, then local scripts at
`osint\sherlock\sherlock_project\sherlock.py`,
`osint\sherlock\sherlock\sherlock.py`, `osint\sherlock\sherlock.py`, and
`resources\python\sherlock\sherlock.py`.
The module route always invokes `sys.executable -m sherlock_project`, even when
only the alternative `sherlock` import succeeded. The local route changes its
working directory to `osint\sherlock`. Presence/import detection alone therefore
does not establish that the selected invocation works.

[Sensitive scanner imports](../../resources/python/sherlock_sensitive_scanner.py#L29)
require `requests`, `beautifulsoup4` (the `bs4` module), and `tldextract`.
spaCy plus `en_core_web_sm` enables optional NER; Playwright plus usable Chromium
browser assets enables optional rendering. An import message is not proof of a
working browser. Install/restore packages only through an explicitly authorized
dependency-maintenance workflow, not by blindly running old setup instructions.

## Read-only local checks

From the repository root, these PowerShell checks do not scan public sites:

```powershell
Get-Command python
where.exe python
Test-Path resources\python\osit_bridge.py
Test-Path resources\python\sherlock_sensitive_scanner.py
Test-Path osint\sherlock\sherlock_project\sherlock.py
Test-Path osint\sherlock\sherlock\sherlock.py
Test-Path osint\sherlock\sherlock.py
Test-Path resources\python\sherlock\sherlock.py
```

They establish paths/PATH candidates only, not importability, package versions,
or successful execution. A filesystem plugin's private `cd` is not proof that
the host process working directory changed.

## Troubleshooting

| Symptom | Check |
|---|---|
| Bridge missing / basic URL list | Correct working directory and bridge path; generated suggestions are unverified |
| Sherlock/module error | Interpreter actually selected, module-route mismatch, local layout, and that release's requirements |
| Empty or partial discovery | Bridge stderr and Sherlock exit code; found-line parsing can mask a failing subprocess |
| Discovery timeout | Network availability and per-site timeout; the bridge process deadline is `max(300, timeout * 15)`, not a duration promise |
| Scanner failed | Required packages, script path, output permissions, and subprocess result; C++ returns a generic `scan_failed` when no log path is returned |
| Browser/NER absent | Optional package/model/browser assets and stderr; ordinary requests/no NER can still be used |
| Unexpected scanner timeout or robots behavior | [Scanner CLI limitations](SCANNER.md#browser-and-request-behavior); some parsed options are not wired |
| Empty findings / "Clean" | Inspect each JSONL `status`; blocked/error pages and skipped malformed lines are not evidence of no exposure |
| Export/cache clear acknowledged but ineffective | Inspect files and permissions; several I/O failures are masked |
| UI cannot load | Confirm an existing `cache\osint\sensitive_<username>.jsonl` for the exact supplied username |

Do not elevate the host simply to suppress filesystem errors. Intermediate files
may survive failed discovery or failed cleanup; review them as sensitive data.
Logs append across scans and cache clearing deletes them; plan retention before
requesting either operation. Use [scanner storage/privacy](SCANNER.md#storage-and-privacy)
and [OSINT panel](../ui/OSINT_PANEL.md) for the remaining contracts.

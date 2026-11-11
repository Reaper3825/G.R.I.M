# OSINT Tools

Source-reviewed November 8, 2026. Use only for authorized self-audits. Requests
leave the machine; results are leads for manual review, not proof of identity.

## Commands

[osint_plugin registration](../../plugins/osint_plugin.cpp#L78) exposes eight
tools, with raw-text arguments rather than JSON parameter objects:

| Tool | Arguments | Behavior |
|---|---|---|
| `profile_person` | `<username> [--no-cache] [--verbose]` | Starts a background discovery job when the bridge is present |
| `sherlock_sweep` | `<username> [--async]` | Waits for discovery unless `--async` is supplied |
| `osint_report` | `<username> [--export <file>]` | Obtains an audit/cache report, formats it, optionally exports raw JSON |
| `osint_status` | `[username]` | Lists jobs or reads and consumes a named completed result |
| `osint_clear_cache` | none | Removes the OSINT cache directory, including sensitive logs |
| `osint_scan_secrets` | `<username>` | Obtains discovery results and scans found URLs for sensitive patterns |
| `osint_show_secrets` | `<username> [--severity critical\|high\|medium\|low\|all]` | Reads existing JSONL with a minimum-severity filter |

Aliases: **`sherlock` → `sherlock_sweep`**, **`osint` → `osint_report`**.
There is no `profile_self` registration. All descriptors request FILESYSTEM,
NETWORK, UI, and PROCESS permissions. Seven carry confirmation metadata;
only `osint_show_secrets` is informational. Status can consume state.
The source marks these tools `GRIM_TOOL_USAGE_IGNORE`; registration does not
promise normal model-catalog selection or literal console dispatch.

[handleCommand](../../commands/commands_core.cpp#L39) sends typed text to the
model. Natural-language requests such as "audit my username" or "scan my profiles
for exposed secrets" are requests, not guaranteed direct handler invocations.
Legacy NLP rules do not create registrations.

Flags are handler-specific. Discovery/report parsers split on whitespace;
`--export` reads one path token and does not implement shell-style quoting.
`osint_status` and `osint_scan_secrets` use the complete supplied
username string, whereas report/show-secrets parse its first token. Supply a
consistent trusted single-token username, without path or shell syntax:
the collector embeds it in filenames and a shell command without a general
username-validation boundary. Unknown flags are not comprehensively rejected.
`--verbose` sets a config field but the current collector does not consume it.

## Authorized workflow

The following are tool invocations to request, not commands run during this
documentation change:

```text
profile_person myusername
osint_status myusername
osint_report myusername --export my_audit.json
osint_scan_secrets myusername
osint_show_secrets myusername --severity high
```

Wait for the background result before requesting downstream work. Alternatively,
`sherlock myusername --async` starts an asynchronous sweep. A report or secrets
request on a cache miss may initiate new network discovery instead of waiting
for an existing job. `--no-cache` disables **both reads and writes** for that
profile job; its later report/secrets request can therefore scan again.

## Jobs and lifecycle

[cmdProfilePerson / cmdSherlockSweep](../../commands/commands_osint.cpp#L53)
run discovery with `std::async`. Profile always starts in the background;
synchronous sweep uses a worker but waits before returning.

[cmdOsintStatus](../../commands/commands_osint.cpp#L337) reports running elapsed
seconds or complete state, not incremental per-site progress. Empty arguments
list retained jobs without consuming them. A named successful or failed
completed job is removed after its result is returned. A missing job is a
successful informational response, not proof of an audit.

Async sweep avoids starting another running job for the same username; profile
does not have that duplicate guard and can replace the visible status entry.
All jobs remain retained for
[shutdown joining](../../commands/commands_osint.cpp#L641), including overwritten
entries. There is no registered cancellation tool. Cache clearing does not stop
jobs, and a running job can write results after a clear. Joining/cleanup can
swallow exceptions; do not treat every acknowledgment as fail-hard validation.

## Reports and cache

[runSelfAudit](../../external_collector/osit.cpp#L224) calls
[the Sherlock bridge](../../resources/python/osit_bridge.py#L73) on a cache miss.
The bridge parses only `[+] Platform: URL` found lines; it assigns status
`Claimed`, synthetic HTTP `"200"`, and zero response time. The C++ `totalChecked`
counts those parsed entries, not all platforms attempted. Displayed "success
rate" is consequently not detection accuracy or measured coverage. A nonzero
Sherlock exit code is logged but not always propagated by the bridge, so empty
or partial output can appear successful.

[cmdProfilePerson](../../commands/commands_osint.cpp#L93) returns successful,
**unverified URL suggestions** when the bridge is absent. This is not a Sherlock
scan. Availability detection checks bridge/layout presence, not a runnable
installation; see [setup](SETUP.md#prerequisites).

[cmdOsintReport](../../commands/commands_osint.cpp#L297) uses a footprint count
heuristic: HIGH for **more than 10** found entries, MODERATE for **more than 5**,
otherwise LOW. These labels are not security/privacy ratings. Its JSON export
writes `report.rawJson`, not the wrapper stored in cache; ordinary stream
failures are not reliably checked, and the overall result can remain successful
even if export fails. Verify the intended file independently.

[Cache functions](../../external_collector/osit.cpp#L132) use
`cache\osint\<username>.json` relative to the process working directory.
Expiry is checked only on lookup: the hour-truncated age must be **greater than
24 hours**. There is no exact scheduled TTL. Loaded reports receive a new host
timestamp, so "Generated" need not be the original scan time. Cache read/write/
delete failures can be swallowed; verify effects on disk. The wrapper stores
`rawJson`, counts, and findings. Clearing the directory also removes
`sensitive_<username>.jsonl`; it is not merely a discovery-cache reset.

## Review and presentation

Reports list privacy practices: review unused accounts/settings, use unique
passwords and 2FA, separate public/private emails, and monitor breaches.
HaveIBeenPwned, Holehe, and Social-Analyzer are referenced recommendations, not
implemented integrations in this collector.

[Sensitive scanning](SCANNER.md#sensitive-scanning) owns detector, severity,
storage, and redaction details. `osint_show_secrets --severity high` means
**score ≥7**, including critical findings; medium is ≥5, critical ≥9, low/all
≥0. Values are lowercase; an unrecognized value retains threshold zero.
Missing logs and skipped malformed JSONL can produce successful empty displays.

Use [OSINT panel](../ui/OSINT_PANEL.md) for presentation and
[generic panel operations](../ui/PANEL_OPERATIONS.md) for UI ownership.
No fixed scan duration, site count, response accuracy, export success, or
UI performance guarantee is established.

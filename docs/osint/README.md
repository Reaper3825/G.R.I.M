> API v11 ownership update: see [the current provider contract](../../plugins/osint_plugin/README.md). Older host/cache/process descriptions below are historical; the host now consumes copied tool results.

# OSINT Documentation

Defensive self-audits of accounts you own or are explicitly authorized to audit.
Discovery and sensitive scanning contact public websites; neither establishes
account ownership or authorizes access to somebody else's data.

| Guide | Scope |
|---|---|
| [Tools](TOOLS.md#commands) | Registered arguments, discovery, jobs, cache, reports, and limitations |
| [Setup](SETUP.md#prerequisites) | Interpreter/dependency prerequisites and troubleshooting |
| [Sensitive scanner](SCANNER.md#sensitive-scanning) | Detection, browser fallback, severity, privacy, and storage |
| [OSINT panel](../ui/OSINT_PANEL.md) | Host-owned findings presentation and controls |

## Source boundary

Source-reviewed November 8, 2026; no installation, network scan, build, or runtime
verification was performed for this consolidation.

```text
osint_plugin registration → OSINT command handlers → runSelfAudit/cache
  → Python Sherlock bridge → public sites → found-line JSON → report
found URLs → sensitive scanner → local JSONL → console / host results panel
```

- [Registration](../../plugins/osint_plugin/osint_plugin.cpp#L73) owns seven tools and two
  aliases. [Handlers](../../plugins/osint_plugin/commands_osint.cpp#L53) own scan jobs;
  [collector](../../plugins/osint_plugin/osit.cpp#L224) owns discovery/cache.
- Collector and bridge sources now live under `plugins/osint_plugin`; their
  existing `osit` filenames are retained.
- The specialized results panel remains host-owned through
  [cmdOsintShowUI](../../commands/commands_osint_ui.cpp#L9). Generic panel
  operations belong to [the UI guides](../ui/PANEL_OPERATIONS.md), not discovery.
- [Plugin API](../PLUGIN_API.md) owns ABI, permission, result, and reload rules;
  [plugin deployment](../../plugins/README.md) owns library deployment.

## Current versus historical claims

These guides consolidate the overlapping command, feature, improvement, setup,
summary, and scanner notes. `profile_person` is current; `profile_self` is not
registered. Old platform-count, timing, accuracy, production-readiness, and
universal Python-version claims are not verified contracts. Optional browser/NER
support is conditional, not always enabled. Missing-bridge URL suggestions,
masked cache/export errors, and partial redaction remain implementation limits,
not documentation fixes.

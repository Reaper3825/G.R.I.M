# OSINT Sensitive Scanner

Source-reviewed November 8, 2026. Pattern-based defensive review of authorized
public profiles, not credential validation, identity verification, breach
correlation, or permission to bypass authentication.

## Sensitive scanning

`osint_scan_secrets <username>` uses a cached audit or obtains one through
Sherlock. [cmdOsintScanSecrets](../../commands/commands_osint.cpp#L425) returns
an informational success without scanning when no accounts were found.
[runSensitiveDataScan](../../external_collector/osit_secrets.cpp#L18) converts
only `found` entries into site → `{url, status:"Claimed", http_status}` input,
runs the Python scanner synchronously, and returns its JSONL path.
The tool does not expose browser/timeout flags.

Authorized workflow:

```text
sherlock myusername
osint_scan_secrets myusername
osint_show_secrets myusername --severity critical
```

Wait for discovery if using an async tool. A cache miss can trigger another scan.
Review [tools/cache](TOOLS.md#reports-and-cache) and
[prerequisites](SETUP.md#prerequisites) first.

### Standalone input and options

[load_sherlock_json](../../resources/python/sherlock_sensitive_scanner.py#L317)
accepts site dictionaries with `status:"Claimed"` and `url`, `exists:true` and
`url`, or `url_main`; it also accepts site → URL strings and lists of URL
objects. The cache wrapper `cache\osint\<username>.json` is **not** that format.
Use a reviewed compatible file or the raw JSON exported by `osint_report`;
there is no guaranteed persistent `cache\osint\sherlock_<username>.json`.

Examples below are for an explicitly authorized future scan, not local checks
or commands executed by this documentation update:

```powershell
python resources\python\sherlock_sensitive_scanner.py my_audit.json --out cache\osint\sensitive_myusername_enhanced.jsonl --use-playwright
python resources\python\sherlock_sensitive_scanner.py my_audit.json --out cache\osint\sensitive_myusername_selected.jsonl --js-sites linkedin.com reddit.com
```

An arbitrary `--out` file is not automatically loaded by the username-based
console/panel readers. Without `--out`, output is `osint_sensitive_log.jsonl`
in the current directory. Ensure the output parent exists for standalone use.

| Parsed option | Actual effect |
|---|---|
| `--out <path>` | Chooses append-mode JSONL output |
| `--use-playwright` | Requests browser rendering, subject to availability/domain selection |
| `--js-sites <domains...>` | Selects exact registered domains for browser attempts, independently of the previous flag |
| `--no-delay` | Removes the delay between input URLs; do not use it to evade site constraints |
| `--max <n>` | Parsed but not used to enforce a pages-per-domain limit |
| `--timeout <seconds>` | Assigned to a local variable, not the global timeout used by requests/Playwright |
| `--no-robots` | Assigned to a local variable, not the global robots policy; it does not disable the current check |

These wiring limitations are visible in
[main](../../resources/python/sherlock_sensitive_scanner.py#L398).
There is no recursive crawl or link-following feature here: the scanner processes
supplied URLs sequentially, including repeated URLs. Defaults in the script are
15 seconds for page requests/browser navigation, 0.5 seconds between URLs,
TLS verification enabled, robots checks enabled, and 80 context characters on
each side. Editing script constants/patterns changes source behavior; it is not
a supported runtime configuration file. A custom detector belongs in
`RE_PATTERNS`, with severity/redaction reviewed separately.

## Detection and severity

[HTML extraction](../../resources/python/sherlock_sensitive_scanner.py#L145)
removes script/style/noscript content, includes pre/code text plus the overall
text, and scans that text rather than all HTML attributes or linked pages.
Code text can consequently be duplicated. `password_url` means a
`password/passwd/pwd=...` pattern in extracted text, not comprehensive inspection
of URLs or hidden form values.

[severity_for_tag](../../resources/python/sherlock_sensitive_scanner.py#L196)
and [summary buckets](../../external_collector/osit_secrets.cpp#L116) define:

| Detectors | Score | Display bucket |
|---|---|---|
| PEM private-key header, AWS/Google keys, GitHub PATs, JWTs, Slack/Stripe keys | 10 | CRITICAL |
| SSN-like, credit-card-like, password query patterns | 9 | CRITICAL |
| Firebase endpoint pattern | 8 | HIGH |
| Phone pattern | 7 | HIGH |
| Email; optional person-name finding | 6 | MEDIUM |
| Date pattern | 4 | LOW |
| Otherwise, including IPv4/base64-like patterns | Entropy >4.5 →9; >4.0 →7; otherwise 5 | CRITICAL/HIGH/MEDIUM |

Buckets are **CRITICAL ≥9, HIGH ≥7, MEDIUM ≥5, LOW <5**. Older comments
calling a score of 9 "HIGH" do not change these buckets. Prefix matching is
heuristic; a Stripe public key or Firebase endpoint is not necessarily a secret.
Cards receive no validity check, dates need not be DOBs, and IPv4 patterns are
not validated or limited to public addresses. There are no dedicated passport/
driver-license detectors or general national-ID validation. The PEM regex
recognizes its supported header forms, not a complete private-key block.

The additional entropy pass flags base64-like candidates longer than 40
characters with Shannon entropy >4.0 bits/character, using a heuristic duplicate
check. The generic base64 regex also runs in the ordinary pass. Entropy is
neither proof of a secret nor a guarantee of deduplication.

When installed, spaCy processes at most the first 100,000 text characters,
extracting PERSON/ORG/GPE/LOC/DATE entities. A first nearby entity start within
200 characters can add `nearby_entity`; at most five PERSON entities become
standalone `person_name` findings. Other entities supply context, not automatic
standalone findings. They may describe somebody else mentioned on the page.
NER errors log a warning and return no entities.

## Browser and request behavior

[scan_url](../../resources/python/sherlock_sensitive_scanner.py#L215) checks
robots, optionally attempts headless Chromium with `networkidle`, then uses
`requests.Session.get` if browser output is unavailable or fails.
Browser import success does not prove installed browser assets or page rendering.

With `--use-playwright` and an available Playwright import, the default selection
includes linkedin.com, twitter.com, facebook.com, instagram.com, reddit.com,
medium.com, dev.to, and stackoverflow.com. Explicit `--js-sites` selections are
matched against `tldextract` registered domains; combined use adds the default
set. The flag is not an unconditional "render every site" guarantee.
`rendered_with_js` records selected/available browser use, **not** confirmed
rendering success after a requests fallback. Likewise `used_ner` records
availability, not successful extraction.

[allowed_by_robots](../../resources/python/sherlock_sensitive_scanner.py#L131)
attempts robots.txt compliance but returns allowed on an exception. Robots checks
are therefore not fail-closed authorization, and neither the request nor browser
path guarantees site Terms-of-Service compliance. URLs without a scheme receive
`http://`. Public rendering does not log in, solve access controls, or guarantee
content behind a login/paywall. Third-party sites and browser subresources receive
network requests; "local processing" does not mean offline.

## Storage and privacy

The C++ integration writes `resources\python\temp_sherlock_<username>.json`;
the discovery collector uses `resources\python\sherlock_results_<username>.json`.
Cleanup is attempted but can fail silently; discovery failures can leave its
output behind. Neither is a secure temporary-storage guarantee.

Output is `cache\osint\sensitive_<username>.jsonl` for the tool, relative to the
process working directory. [main's writer](../../resources/python/sherlock_sensitive_scanner.py#L448)
**appends**, so repeated scans accumulate old/new findings and can inflate totals.
Console/panel summaries read the log, not only the latest invocation.
One line per attempted URL carries this shape (illustrative, not real findings):

```json
{"timestamp":"2026-11-08T00:00:00Z","source_url":"https://example.com/myusername","domain":"example.com","status":"ok","rendered_with_js":false,"used_ner":false,"findings":[{"tag":"github_pat","match":"[REDACTED]","context":"token = [REDACTED]","entropy":4.8,"severity":10,"location":"https://example.com/myusername","match_length":40}]}
```

Status may be `ok`, `robots_blocked`, `error:...`, or `exception:...`.
The console prints "Clean" whenever a URL has no findings, even for blocked/error
statuses. A zero scanner exit code, green result, or zero findings does not
certify successful retrieval or absence of exposure. Empty input returns zero
without creating a new log; the C++ existence check can then see an older log.
[C++ parsing](../../external_collector/osit_secrets.cpp#L93) and show-secrets skip
malformed JSONL; I/O failures are not all fail-hard.

[Redaction](../../resources/python/sherlock_sensitive_scanner.py#L249) masks
selected key/token/private-header/password matches and the extra entropy-pass
match. Emails, phones, SSNs, card-like strings, dates, IPs, names, Firebase, and
ordinary base64-like matches can remain **raw**. Context replaces only the
current match; surrounding values, nearby entities, and full source/location
URLs can still expose PII or credentials. UI truncation is not secure redaction.
There is no encryption, automatic retention schedule, or complete privacy
guarantee here. Restrict file access, review before sharing, and plan deletion;
`osint_clear_cache` removes these logs too but may mask deletion failure.

For a credible exposure, promptly rotate/revoke affected credentials, review
access logs, remove public copies, and review account privacy/2FA. Keep only an
access-controlled audit trail and use an appropriate disclosure/data-removal
process. Periodic or life-change audits are operational suggestions, not an
implemented scheduler. Breach-monitoring services are recommendations, not
scanner integrations. See [OSINT panel](../ui/OSINT_PANEL.md) for reviewing logs.

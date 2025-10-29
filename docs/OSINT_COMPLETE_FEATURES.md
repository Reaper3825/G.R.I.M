# ?? OSINT Complete Feature Summary

## ? What You Now Have

A complete OSINT (Open Source Intelligence) system with **10 commands** for defensive security auditing, including a **visual UI panel** for viewing results.

---

## ?? All OSINT Commands

| # | Command | Purpose | Output |
|---|---------|---------|--------|
| 1 | `profile_person <username>` | Profile digital footprint across 400+ platforms | Text summary |
| 2 | `sherlock_sweep <username> [--async]` | Comprehensive username scan | Text with found accounts |
| 3 | `osint_report <username> [--export file]` | Full OSINT report with privacy recommendations | Detailed report + optional JSON export |
| 4 | `osint_status [username]` | Check async scan status | Status update |
| 5 | `osint_clear_cache` | Clear cached scan results | Confirmation |
| 6 | `osint_scan_secrets <username>` | Scan discovered URLs for sensitive data (PII, API keys, etc.) | Summary of findings |
| 7 | `osint_show_secrets <username> [--severity level]` | View detailed findings in console | Formatted text output |
| 8 | **`osint_show_ui <username>`** | **Open visual UI panel with table** | **Interactive GUI** ? |
| 9 | `osint` (alias) | Shortcut for `osint_report` | Same as osint_report |
| 10 | `sherlock` (alias) | Shortcut for `sherlock_sweep` | Same as sherlock_sweep |

---

## ?? Complete Workflow

### **Basic Usage**
```sh
# 1. Profile a username
profile_person reaper3825

# 2. Check status
osint_status reaper3825

# 3. View full report
osint_report reaper3825
```

### **Advanced: Sensitive Data Scanning**
```sh
# 1. Profile username
profile_person reaper3825

# 2. Scan for sensitive data
osint_scan_secrets reaper3825

# 3. View in beautiful UI
osint_show_ui reaper3825

# 4. Or view in console
osint_show_secrets reaper3825 --severity critical
```

### **Export & Documentation**
```sh
# Export full report to JSON
osint_report reaper3825 --export reaper3825_audit.json

# Export filtered secrets
osint_show_secrets reaper3825 --severity high > high_priority.txt
```

---

## ?? Visual UI Panel Features

When you run `osint_show_ui <username>`, you get:

### **Summary Header**
```
Username: reaper3825
Total: 42 | CRITICAL: 3 | HIGH: 8 | MEDIUM: 15 | LOW: 16
Affected Domains: 12
```

### **Filter Buttons**
- [ALL] - Show everything
- [CRITICAL] - Only critical (API keys, private keys)
- [HIGH] - High and above (PII, emails, phones)
- [MEDIUM] - Medium and above
- [LOW] - Everything

### **Interactive Table**

| Severity | Type | Match | Domain | Context |
|----------|------|-------|--------|---------|
| ?? CRITICAL | aws_key | AKIAIO... | github.com | export AWS_... |
| ?? HIGH | email | user@... | twitter.com | Contact... |
| ?? MEDIUM | phone | +1-555... | linkedin.com | Call: +1... |
| ?? LOW | username | reaper3825 | steam.com | User: ... |

### **UI Controls**
- **Click & Drag** title bar to move
- **Click & Drag** bottom-right corner to resize
- **Click** filter buttons to change view
- **Scroll** with mouse wheel or arrow keys
- **Click** rows to select (future: copy/export)

---

## ?? What Gets Scanned

### **Platform Discovery (400+ sites)**
- Social media: Twitter, Instagram, Facebook, TikTok, Reddit
- Professional: LinkedIn, GitHub, GitLab, Stack Overflow
- Gaming: Steam, Xbox, PlayStation, Twitch, Discord
- Developer: Dev.to, Medium, HackerOne, CodePen
- And 380+ more...

### **Sensitive Data Detection**

| Severity | Findings |
|----------|----------|
| **CRITICAL (9-10)** | AWS keys, Google API keys, GitHub tokens, Private keys (RSA/PEM), JWT tokens, SSH keys |
| **HIGH (7-8)** | Emails, Phone numbers, SSN patterns, Credit card patterns, National IDs |
| **MEDIUM (5-6)** | Dates of birth, Addresses, IP addresses, Person names (NER) |
| **LOW (0-4)** | Usernames, Generic patterns, Common words |

---

## ?? File Locations

```
D:/G.R.I.M/
??? cache/osint/
?   ??? sherlock_reaper3825.json        # Sherlock scan results
?   ??? sensitive_reaper3825.jsonl      # Sensitive findings log
?   ??? reaper3825.json                 # Cached report (24h TTL)
??? resources/python/
?   ??? osit_bridge.py                  # Sherlock bridge
?   ??? sherlock_sensitive_scanner.py   # Sensitive data scanner
??? osint/sherlock/                     # Sherlock installation
??? commands/
?   ??? commands_osint.cpp              # OSINT command implementations
?   ??? commands_osint.hpp              # Command declarations
??? ui/
?   ??? ui_osint_results.cpp            # UI panel implementation
?   ??? ui_osint_results.hpp            # UI panel header
??? external_collector/
?   ??? osit.cpp                        # OSINT core logic
?   ??? osit_secrets.cpp                # Sensitive scanner logic
?   ??? *.hpp                           # Headers
??? plugins/
    ??? osint_plugin.cpp                # Plugin registration
```

---

## ??? Technical Details

### **Architecture**
- **Language**: C++17/20 with Python bridge
- **Backend**: Sherlock OSINT project (local installation)
- **Scanning**: Async background execution
- **Caching**: 24-hour cache with automatic expiry
- **UI**: Custom overlay renderer (GDI+)
- **Data Format**: JSONL for sensitive findings

### **Performance**
- **Scan time**: 1-5 minutes (400+ platforms)
- **Timeout**: 30s per platform (configurable)
- **Async support**: ? Yes (background execution)
- **Caching**: ? Yes (speeds up repeated queries)
- **UI rendering**: Real-time updates

### **Privacy & Security**
- **Defensive only**: For self-auditing your own accounts
- **Truncated display**: UI truncates sensitive data for safety
- **Local storage**: All data stays on your machine
- **No external services**: Runs entirely offline (except Sherlock HTTP requests)

---

## ?? Documentation Files

1. **[OSINT_COMMANDS.md](OSINT_COMMANDS.md)** - Full command reference
2. **[OSINT_UI_PANEL.md](OSINT_UI_PANEL.md)** - Visual UI guide
3. **[OSINT_SETUP.md](OSINT_SETUP.md)** - Installation & troubleshooting
4. **[OSINT_SUMMARY.md](OSINT_SUMMARY.md)** - Technical overview
5. **This file** - Complete feature summary

---

## ?? Use Cases

### **1. Personal Security Audit**
```sh
profile_person myusername
osint_scan_secrets myusername
osint_show_ui myusername
```

### **2. Regular Monitoring**
```sh
# Weekly check (uses cache)
sherlock_sweep myusername

# Monthly full refresh
profile_person myusername --no-cache
```

### **3. Incident Response**
```sh
# Quick scan
osint_scan_secrets compromised_user

# Open UI for triage
osint_show_ui compromised_user

# Filter critical issues
osint_show_secrets compromised_user --severity critical
```

### **4. Client Reports**
```sh
# Generate full report
osint_report client_username --export client_audit.json

# Screenshot UI panel for presentation
osint_show_ui client_username
```

---

## ?? Quick Start (60 Seconds)

```sh
# 1. Scan yourself (replace with your username)
profile_person myusername

# 2. Wait ~2 minutes for scan

# 3. Check for sensitive data
osint_scan_secrets myusername

# 4. Open beautiful UI
osint_show_ui myusername

# 5. Click CRITICAL button to see urgent issues

# Done! ??
```

---

## ?? Pro Tips

1. **Use UI for visual analysis** - Easier to spot patterns
2. **Use console for automation** - Script with `osint_show_secrets`
3. **Filter early** - Click CRITICAL first to focus
4. **Export reports** - Use `--export` for documentation
5. **Regular scans** - Monthly audits catch new leaks
6. **Clear cache** - Force fresh scans with `--no-cache`
7. **Async mode** - Use `sherlock_sweep --async` for background
8. **Multiple usernames** - Open multiple UI panels

---

## ?? Comparison

### Console Commands
? **Good for:**
- Quick checks
- Scripting / automation
- Copy-paste specific findings
- Terminal-only environments

### UI Panel
? **Good for:**
- Visual analysis
- Filtering & sorting
- Non-technical users
- Presentations / demos
- Comparing findings

**Best practice:** Use **both**! UI for analysis, console for automation.

---

## ?? Privacy Recommendations

Based on findings, take action:

### HIGH Footprint (10+ accounts)
- ?? Immediate audit
- ?? Deactivate unused accounts
- ?? Enable 2FA everywhere
- ?? Use password manager

### MODERATE Footprint (5-10 accounts)
- Review privacy settings
- Enable 2FA on critical accounts
- Monitor breaches (HaveIBeenPwned)

### LOW Footprint (<5 accounts)
- ? Maintain current practices
- ? Continue monitoring
- ? Keep credentials secure

---

## ?? UI Screenshots (Conceptual)

```
?????????????????????????????????????????????????????????
? OSINT Sensitive Findings                          [X] ?
?????????????????????????????????????????????????????????
? Username: reaper3825                                  ?
? Total: 42 | ?? 3 | ?? 8 | ?? 15 | ?? 16              ?
? Affected Domains: 12                                  ?
?????????????????????????????????????????????????????????
? [ALL] [CRITICAL] [HIGH] [MEDIUM] [LOW]              ?
????????????????????????????????????????????????????????
? Severity? Type     ? Match     ? Domain  ? Context   ?
????????????????????????????????????????????????????????
? ?? CRIT ? aws_key  ? AKIAIO... ? github  ? export... ?
? ?? CRIT ? gh_token ? ghp_16... ? github  ? TOKEN=... ?
? ?? HIGH ? email    ? user@ex...? twitter ? Contact...?
? ?? HIGH ? phone    ? +1-555... ? linkedin? Call: ... ?
? ?? MED  ? dob      ? 1990-01-01? reddit  ? Born on...?
? ?? LOW  ? username ? reaper... ? steam   ? User: ... ?
????????????????????????????????????????????????????????
```

---

## ?? Summary

You now have:

? **10 OSINT commands** for comprehensive security auditing
? **Visual UI panel** with interactive table
? **Sensitive data scanner** detecting 30+ pattern types
? **400+ platform coverage** via Sherlock integration
? **Severity-based filtering** for quick triage
? **Export capabilities** for documentation
? **Async scanning** for background operation
? **Smart caching** for faster repeat queries
? **Complete documentation** in 5 files

**All integrated into G.R.I.M's command system!** ??

---

**Stay safe, stay private!** ????

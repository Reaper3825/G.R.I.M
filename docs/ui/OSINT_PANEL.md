# OSINT UI Panel Guide

## ?? Viewing Sensitive Findings in a Chart/Table

Instead of scrolling through console text, you can now view sensitive data findings in a **beautiful, sortable, filterable UI panel**.

## ?? Quick Start

### Step 1: Run a Scan
```sh
osint_scan_secrets reaper3825
```

### Step 2: Open the UI Panel
```sh
osint_show_ui reaper3825
```

That's it! A visual panel will appear with all findings in a table.

## ?? UI Features

### **Summary Header**
- Username being scanned
- Total findings count
- Breakdown by severity (CRITICAL, HIGH, MEDIUM, LOW)
- Affected domains count

### **Filter Buttons**
Click to filter by severity:
- **ALL** - Show everything
- **CRITICAL** - Only critical findings (API keys, private keys)
- **HIGH** - High and above (PII, phone numbers, emails)
- **MEDIUM** - Medium and above
- **LOW** - Everything

### **Sortable Table**

| Severity | Type | Match | Domain | Context |
|----------|------|-------|--------|---------|
| CRITICAL | aws_access_key | AKIAIOSFODNN7... | github.com | export AWS_ACCESS... |
| HIGH | email | user@example.com | twitter.com | Contact me at... |
| MEDIUM | phone | +1-555-123-4567 | linkedin.com | Call: +1-555... |

**Color Coding:**
- ?? **CRITICAL** - Red
- ?? **HIGH** - Orange
- ?? **MEDIUM** - Yellow
- ?? **LOW** - Green

### **Table Columns**

1. **Severity** - Risk level (color-coded)
2. **Type** - What was found (api_key, email, phone, etc.)
3. **Match** - The actual data found (truncated for privacy)
4. **Domain** - Website where it was found
5. **Context** - Surrounding text (truncated)

### **Scrollbar**
- Appears when there are more findings than fit on screen
- Use mouse wheel or arrow keys to scroll
- Smooth scrolling for easy navigation

### **Row Selection**
- Hover over rows to highlight
- **Click to view full details** - Opens detail panel with:
  - **Full Match Value** (untruncated)
  - **Full Context** (complete surrounding text)
  - **Full URL**
  - Severity score, entropy, and warnings

## ?? Controls

### Keyboard
- **? / ?** - Scroll table
- **ESC** - Close detail panel

### Mouse
- **Click** filter buttons - Change severity filter
- **Click & Drag** title bar - Move panel
- **Click & Drag** bottom-right corner - Resize panel
- **Click** row - **Open detail panel with full values** ?
- **Scroll wheel** - Scroll table

## ?? Example Workflow

```sh
# 1. Profile username
profile_person reaper3825

# 2. Wait for scan to complete
osint_status reaper3825

# 3. Scan for sensitive data
osint_scan_secrets reaper3825

# 4. Open UI panel
osint_show_ui reaper3825

# 5. Use filter buttons to focus on critical issues
#    Click: CRITICAL button

# 6. Review findings in table

# 7. Click on a row to see FULL VALUES
#    Detail panel shows:
#    - Complete match (no truncation)
#    - Full context
#    - Complete URL
#    - Severity warnings

# 8. Press ESC to close detail panel

# 9. For text export, use:
osint_show_secrets reaper3825 --severity critical
```

## ?? UI Layout

```
?????????????????????????????????????????????????????????
? OSINT Sensitive Findings                          [X] ?
?????????????????????????????????????????????????????????
?                                                       ?
? Username: reaper3825                                  ?
? Total: 42 | CRITICAL: 3 | HIGH: 8 | MEDIUM: 15 | LOW: 16
? Affected Domains: 12                                  ?
?                                                       ?
?????????????????????????????????????????????????????????
? [ALL] [CRITICAL] [HIGH] [MEDIUM] [LOW]              ?
?????????????????????????????????????????????????????????
? Severity   ? Type          ? Match         ? Domain  ? Context
???????????????????????????????????????????????????????????????
? CRITICAL   ? aws_key       ? AKIAIO...     ? github  ? export AWS_...
? CRITICAL   ? github_token  ? ghp_16...     ? github  ? TOKEN=ghp_...
? HIGH       ? email         ? user@ex...    ? twitter ? Contact...
? HIGH       ? phone         ? +1-555...     ? linkedin? Call: +1...
? MEDIUM     ? date          ? 1990-01-01    ? reddit  ? Born on...
? LOW        ? username      ? reaper3825    ? steam   ? Username:...
?                                                       ?
?                                             [???????]? ? Scrollbar
?????????????????????????????????????????????????????????
```

## ?? Detail Panel (Click Any Row)

When you click a row, a detail panel appears showing **FULL VALUES**:

```
????????????????????????????????????????????
? Finding Details              [ESC to close] ?
????????????????????????????????????????????
? Severity: CRITICAL (10/10)               ?
? Type: aws_access_key                     ?
? Domain: github.com                       ?
? Entropy: 4.52                            ?
?                                          ?
? URL:                                     ?
? https://github.com/reaper3825/myproject/ ?
? blob/main/config.py                      ?
?                                          ?
? MATCH (Full Value):                      ?
? ?????????????????????????????????????? ?
? ? AKIAIOSFODNN7EXAMPLEKEY            ? ?
? ?????????????????????????????????????? ?
?                                          ?
? CONTEXT (Full Text):                     ?
? ?????????????????????????????????????? ?
? ? export AWS_ACCESS_KEY_ID=          ? ?
? ? AKIAIOSFODNN7EXAMPLEKEY            ? ?
? ? export AWS_SECRET_ACCESS_KEY=...   ? ?
? ?????????????????????????????????????? ?
?                                          ?
? ? CRITICAL: This is sensitive data!    ?
? Action required:                         ?
? • Rotate/revoke this credential now     ?
????????????????????????????????????????????
```

**The detail panel shows:**
- ? **Full untruncated match value** (not `AKIAIO...` but the complete key)
- ? **Full context text** (complete surrounding code/text)
- ? **Complete URL** (not truncated)
- ? **Exact severity score** (0-10 scale)
- ? **Entropy measurement** (randomness indicator)
- ? **Action warnings** for critical findings

**This is the actual data!** Use it to verify and take action.

## ?? Console vs UI

### Console (`osint_show_secrets`)
? Good for:
- Quick terminal checks
- Scripting / automation
- Copy-paste specific findings
- Saving to log files

### UI Panel (`osint_show_ui`)
? Good for:
- Visual analysis
- Comparing multiple findings
- Quick filtering
- Non-technical users
- Presentations/demos

## ?? Tips

1. **Filter Early** - Click CRITICAL first to focus on urgent issues
2. **Resize as Needed** - Drag corner to make table bigger
3. **Combine with Text** - Use both UI and console commands
4. **Multiple Scans** - Open UI for different usernames
5. **Take Screenshots** - Great for reporting/documentation

## ?? Commands Summary

| Command | Purpose |
|---------|---------|
| `osint_scan_secrets <username>` | Scan for sensitive data |
| `osint_show_ui <username>` | Open visual UI panel |
| `osint_show_secrets <username>` | Text-based view (console) |
| `osint_show_secrets <username> --severity critical` | Filter text output |

## ?? What You Can See

### Critical Findings (Red)
- AWS Access Keys
- Google API Keys
- GitHub Tokens
- Private Keys (RSA, PEM)
- JWT Tokens
- SSH Keys

### High Findings (Orange)
- Email Addresses
- Phone Numbers
- Social Security Numbers
- Credit Card Patterns
- National IDs

### Medium Findings (Yellow)
- Dates of Birth
- Addresses
- IP Addresses
- Person Names (via NER)

### Low Findings (Green)
- Usernames
- Generic Patterns
- Common Words

## ?? Privacy Note

The UI **truncates** sensitive data for display:
- Matches shown as "AKIAIO..." instead of full key
- Context shown as "export AWS_..." instead of full line
- Prevents accidental exposure during demos

For full details, check the log file:
```
cache/osint/sensitive_reaper3825.jsonl
```

## ?? Use Cases

1. **Security Audit** - Quick visual check for leaks
2. **Client Reports** - Screenshot for documentation
3. **Incident Response** - Rapid triage of exposed data
4. **Training** - Teach users about data exposure
5. **Monitoring** - Regular scans with visual tracking

Enjoy the visual OSINT experience! ???

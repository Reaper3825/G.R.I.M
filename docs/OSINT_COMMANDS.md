# OSINT Commands - G.R.I.M Self-Audit Tools

## Overview

G.R.I.M includes powerful OSINT (Open Source Intelligence) commands for defensive security auditing. These tools help you understand your digital footprint and identify accounts associated with your usernames across 300+ platforms.

?? **IMPORTANT**: These tools are for **self-auditing only**. Only scan usernames you own.

## Installation

### Prerequisites

1. **Python 3.7+** installed and in PATH
2. **Sherlock OSINT tool** - Already included in `osint/sherlock/` folder

### Verify Sherlock Installation

Sherlock should be located at: `D:\G.R.I.M\osint\sherlock\`

**Check the installation:**
```bash
# Navigate to the Sherlock directory
cd osint\sherlock

# Verify Sherlock is present
dir sherlock.py
# or
ls sherlock.py
```

**If Sherlock is missing**, clone it:
```bash
# From the G.R.I.M root directory
cd osint
git clone https://github.com/sherlock-project/sherlock.git

# Install dependencies
cd sherlock
pip install -r requirements.txt
```

**Alternative: System-wide installation**
```bash
pip install sherlock-project
```

### Directory Structure

```
G.R.I.M/
??? osint/sherlock/              # Sherlock tool (local installation)
?   ??? sherlock/
?   ?   ??? sherlock.py
?   ??? requirements.txt
?   ??? ...
??? resources/python/
?   ??? osit_bridge.py          # Python bridge to Sherlock
??? cache/osint/                 # Cached scan results (auto-created)
??? plugins/
    ??? osint_plugin.dll         # OSINT plugin
```

## Commands

### 1. `profile_self` - Basic Digital Footprint Scan

Searches for your username across 300+ platforms.

**Usage:**
```
profile_self <username> [options]
```

**Options:**
- `--no-cache` - Skip cache and force fresh scan
- `--verbose` - Show detailed output including raw JSON

**Examples:**
```
profile_self john_doe
profile_self alice_crypto --no-cache
profile_self bob_smith --verbose
```

**Output:**
- Total platforms checked
- Accounts found with URLs
- Response times per platform
- Summary statistics

---

### 2. `sherlock_sweep` - Comprehensive Username Enumeration

Performs a thorough scan with detailed results. Supports async operation.

**Usage:**
```
sherlock_sweep <username> [--async]
```

**Options:**
- `--async` - Run scan in background

**Examples:**
```
sherlock_sweep john_doe
sherlock_sweep alice_crypto --async
```

**Output:**
- Complete list of found accounts (up to 20 shown)
- Platform names and URLs
- Success rate statistics
- Suggestion to generate full report

**Async Mode:**
When using `--async`, the scan runs in the background. Check progress with:
```
osint_status john_doe
```

---

### 3. `osint_report` - Full OSINT Report with Recommendations

Generates a comprehensive security audit report.

**Usage:**
```
osint_report <username> [--export <file>]
```

**Options:**
- `--export <file>` - Export full JSON data to file

**Examples:**
```
osint_report john_doe
osint_report alice_crypto --export alice_report.json
```

**Output:**
- Detailed account findings
- Privacy risk assessment (LOW/MODERATE/HIGH)
- Security recommendations
- Export capability for further analysis

---

### 4. `osint_status` - Check Async Scan Status

Check the status of background scans.

**Usage:**
```
osint_status [username]
```

**Examples:**
```
osint_status                    # List all active scans
osint_status alice_crypto       # Get results for specific scan
```

**Output:**
- Active scan list
- Completion status
- Full results when ready

---

### 5. `osint_clear_cache` - Clear Cached Results

Remove all cached scan data to force fresh scans.

**Usage:**
```
osint_clear_cache
```

**Output:**
- Confirmation of cache clearance
- Next scans will be fresh queries

---

## Aliases

For convenience, the following aliases are available:

- `osint` ? `osint_report`
- `sherlock` ? `sherlock_sweep`

## Caching System

### How It Works

- Scan results are cached for **24 hours**
- Cache location: `cache/osint/<username>.json`
- Speeds up repeated queries
- Can be bypassed with `--no-cache` flag

### Cache Management

**View cache location:**
```
ls cache/osint
```

**Clear specific cache:**
Delete the file manually or use:
```
osint_clear_cache
```

## Privacy & Security Recommendations

Based on your scan results, G.R.I.M provides tailored recommendations:

### HIGH Footprint (10+ accounts)
- Immediate audit of all accounts
- Deactivate unused accounts
- Enable 2FA everywhere
- Use password manager with unique passwords

### MODERATE Footprint (5-10 accounts)
- Review account privacy settings
- Enable 2FA on critical accounts
- Monitor for breaches regularly

### LOW Footprint (<5 accounts)
- Maintain current practices
- Continue monitoring
- Keep credentials secure

## Technical Details

### Implementation

- **Language**: C++17/20 with Python bridge
- **Backend**: Sherlock OSINT project (local installation in `osint/sherlock/`)
- **Platforms**: 300+ sites including:
  - Social media (Twitter, Instagram, Facebook)
  - Professional networks (LinkedIn, GitHub)
  - Gaming platforms (Steam, Xbox, PlayStation)
  - Developer sites (Dev.to, Medium, HackerOne)
  - And many more...

### Performance

- **Scan time**: 30-60 seconds (depending on platform count)
- **Timeout**: 30 seconds per platform (configurable)
- **Async support**: Yes (background execution)
- **Caching**: 24-hour cache with automatic expiry

### Error Handling

The system gracefully handles:
- Missing Sherlock installation
- Network timeouts
- Invalid usernames
- Python environment issues
- File system errors

## Troubleshooting

### "Sherlock not found" or "Sherlock bridge not found"

**Problem**: The OSINT bridge can't locate Sherlock.

**Solution**:
1. Check if Sherlock exists in `osint/sherlock/`:
   ```bash
   dir osint\sherlock\sherlock\sherlock.py
   ```

2. If missing, install it:
   ```bash
   cd osint
   git clone https://github.com/sherlock-project/sherlock.git
   cd sherlock
   pip install -r requirements.txt
   ```

3. Or install system-wide:
   ```bash
   pip install sherlock-project
   ```

### "Python bridge not found"

**Problem**: The bridge script is missing.

**Solution**:
- Verify `resources/python/osit_bridge.py` exists
- Re-clone the repository if necessary

### "Scan timeout"

**Problem**: Scan takes longer than expected.

**Solution**:
- Use `--async` mode for long scans:
  ```
  sherlock_sweep username --async
  ```
- Check internet connection
- Some platforms may be slow to respond (this is normal)

### "No output JSON generated"

**Problem**: Sherlock ran but didn't create output.

**Solution**:
1. Check Python is in PATH:
   ```bash
   python --version
   ```

2. Verify Sherlock works standalone:
   ```bash
   cd osint\sherlock
   python sherlock\sherlock.py testuser --json
   ```

3. Check file permissions in `resources/python/`

### Scan returns no results

**Possible causes:**
- Username doesn't exist on platforms (expected)
- Network connectivity issues
- Sherlock database needs updating

**Solution**:
```bash
cd osint\sherlock
git pull origin master
pip install -r requirements.txt --upgrade
```

### "Invalid JSON from Sherlock"

**Problem**: Sherlock output is not valid JSON.

**Solution**:
- Update Sherlock to latest version
- Check Sherlock's dependencies are installed
- Verify internet connectivity

## Examples & Use Cases

### Personal Security Audit
```
# Quick check
profile_self myusername

# Full audit with export
osint_report myusername --export my_audit.json

# Review the JSON for detailed analysis
```

### Regular Monitoring
```
# Weekly scan (use cache)
sherlock_sweep myusername

# Monthly full refresh
profile_self myusername --no-cache
```

### Background Operations
```
# Start long scan
sherlock_sweep myusername --async

# Continue working...
# Check later:
osint_status myusername
```

## Integration with Other Tools

### Export to CSV

Use `--export` and convert JSON to CSV with Python:
```python
import json
import csv

with open('report.json') as f:
    data = json.load(f)

with open('report.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Platform', 'URL', 'Found'])
    
    for site, info in data.items():
        writer.writerow([site, info.get('url_main', ''), 
                        info.get('status') == 'FOUND'])
```

### HaveIBeenPwned Integration (Future)

Planned integration for email breach checking.

## Best Practices

1. **Run regular audits** - Monthly or quarterly
2. **Act on findings** - Don't just scan, take action
3. **Use unique usernames** - Avoid reusing usernames across platforms
4. **Monitor breaches** - Use HaveIBeenPwned for email monitoring
5. **Document changes** - Keep track of accounts you've closed
6. **Enable 2FA** - On all platforms that support it
7. **Use password manager** - Never reuse passwords

## Contributing

Found a bug or want to add a feature?
- File issues on GitHub: https://github.com/Reaper3825/G.R.I.M
- Submit pull requests
- Suggest new OSINT tools to integrate

## Legal & Ethical Use

?? **IMPORTANT NOTICE**

These tools are provided for **defensive security purposes only**:
- ? Audit your own usernames
- ? Check your own digital footprint
- ? Improve your privacy posture
- ? DO NOT scan others without permission
- ? DO NOT use for harassment or stalking
- ? DO NOT use for unauthorized access

**You are responsible for how you use these tools.**

## Credits

- **Sherlock Project**: https://github.com/sherlock-project/sherlock
- **G.R.I.M Team**: OSINT plugin development

## Version History

### v2.0.0 (Current)
- Real Sherlock integration
- Local Sherlock installation support (`osint/sherlock/`)
- Async scan support
- Improved caching system
- Better error handling
- Export functionality
- Status checking
- Performance improvements

### v1.0.0 (Legacy)
- Basic mock implementation
- Simulated platform checks
- Simple URL generation

---

**Stay safe, stay private! ???**
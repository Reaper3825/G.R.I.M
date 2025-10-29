# OSINT Quick Reference

## Installation
Sherlock is located at: `osint/sherlock/`

If missing:
```bash
cd osint
git clone https://github.com/sherlock-project/sherlock.git
cd sherlock
pip install -r requirements.txt
```

Or install system-wide:
```bash
pip install sherlock-project
```

## Commands

| Command | Description | Usage |
|---------|-------------|-------|
| `profile_self` | Basic footprint scan | `profile_self <username>` |
| `sherlock_sweep` | Comprehensive scan | `sherlock_sweep <username> [--async]` |
| `osint_report` | Full audit report | `osint_report <username> [--export <file>]` |
| `osint_status` | Check async scans | `osint_status [username]` |
| `osint_clear_cache` | Clear cache | `osint_clear_cache` |

## Aliases
- `osint` = `osint_report`
- `sherlock` = `sherlock_sweep`

## Common Workflows

### Quick Audit
```
profile_self myusername
```

### Full Report
```
osint_report myusername --export report.json
```

### Background Scan
```
sherlock_sweep myusername --async
# Later...
osint_status myusername
```

### Fresh Scan (No Cache)
```
profile_self myusername --no-cache
```

## Flags

- `--no-cache` - Force fresh scan
- `--verbose` - Detailed output
- `--async` - Background execution
- `--export <file>` - Export to JSON

## Cache

- Location: `cache/osint/`
- Expiry: 24 hours
- Clear: `osint_clear_cache`

## Output Colors

- ?? Cyan - Info/Profile
- ?? Magenta - Sherlock scans
- ?? Yellow - Reports/Warnings
- ?? Green - Success
- ?? Red - Errors

## Troubleshooting

### Sherlock Not Found
```bash
# Check installation
dir osint\sherlock\sherlock\sherlock.py

# If missing, install
cd osint
git clone https://github.com/sherlock-project/sherlock.git
cd sherlock
pip install -r requirements.txt
```

### Update Sherlock
```bash
cd osint\sherlock
git pull origin master
pip install -r requirements.txt --upgrade
```

## File Locations

- **Sherlock**: `osint/sherlock/`
- **Bridge**: `resources/python/osit_bridge.py`
- **Cache**: `cache/osint/`
- **Plugin**: `plugins/osint_plugin.dll`

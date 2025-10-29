# OSINT Setup Guide

## Quick Setup for Local Sherlock Installation

This guide helps you set up the OSINT commands with the locally installed Sherlock in `osint/sherlock/`.

## Current Installation Location

Your Sherlock installation should be at:
```
D:\G.R.I.M\osint\sherlock\
```

## Step-by-Step Setup

### 1. Verify Sherlock Installation

Open a terminal in the G.R.I.M directory and check if Sherlock is present:

```bash
# Check if Sherlock directory exists
dir osint\sherlock

# Check for the main script
dir osint\sherlock\sherlock\sherlock.py
```

### 2. If Sherlock is Missing

If the directory is empty or missing, install Sherlock:

```bash
# Navigate to osint directory
cd osint

# Clone Sherlock repository
git clone https://github.com/sherlock-project/sherlock.git

# Navigate to Sherlock directory
cd sherlock

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Verify Python Installation

Make sure Python 3.7+ is installed:

```bash
python --version
# Should show: Python 3.7.0 or higher
```

If Python is not in PATH, add it:
- Windows: Add Python installation directory to System Environment Variables
- Or use full path in commands

### 4. Test Sherlock Standalone

Before using through G.R.I.M, test Sherlock directly:

```bash
# Navigate to Sherlock directory
cd D:\G.R.I.M\osint\sherlock

# Run a test scan
python sherlock\sherlock.py testuser --json

# Or if sherlock.py is in the root:
python sherlock.py testuser --json
```

Expected output: JSON data with platform results

### 5. Test the Bridge

Test the Python bridge that connects G.R.I.M to Sherlock:

```bash
# Navigate to resources/python
cd D:\G.R.I.M\resources\python

# Run the bridge
python osit_bridge.py testuser test_output.json

# Check if output file was created
dir test_output.json

# View the results
type test_output.json
```

### 6. Test from G.R.I.M

Launch G.R.I.M and test the OSINT commands:

```
# In G.R.I.M console:
profile_self testuser
```

If successful, you should see scan results!

## Directory Structure

Your final structure should look like:

```
D:\G.R.I.M\
??? osint/
?   ??? sherlock/
?       ??? sherlock/
?       ?   ??? sherlock.py          # Main Sherlock script
?       ?   ??? sites.py
?       ?   ??? ...
?       ??? requirements.txt
?       ??? LICENSE
?       ??? README.md
??? resources/
?   ??? python/
?       ??? osit_bridge.py          # G.R.I.M bridge to Sherlock
??? cache/
?   ??? osint/                       # Auto-created cache directory
??? plugins/
?   ??? osint_plugin.dll
??? ...
```

## Common Issues & Solutions

### Issue 1: "Sherlock not found"

**Symptoms**: Error message saying Sherlock bridge or Sherlock itself not found

**Solutions**:

1. **Check directory structure**:
   ```bash
   dir osint\sherlock\sherlock\sherlock.py
   ```
   
2. **Verify bridge location**:
   ```bash
   dir resources\python\osit_bridge.py
   ```

3. **Install Sherlock if missing**:
   ```bash
   cd osint
   git clone https://github.com/sherlock-project/sherlock.git
   ```

### Issue 2: "Python not found"

**Symptoms**: System cannot find python command

**Solutions**:

1. **Install Python 3.7+** from python.org

2. **Add Python to PATH**:
   - Windows: System Properties ? Environment Variables ? Path
   - Add: `C:\Python39\` (or your Python installation path)
   - Add: `C:\Python39\Scripts\`

3. **Restart terminal** after adding to PATH

### Issue 3: "Module not found" errors

**Symptoms**: Python complains about missing modules (requests, etc.)

**Solutions**:

```bash
cd osint\sherlock
pip install -r requirements.txt

# Or install individually:
pip install requests
pip install beautifulsoup4
pip install PySocks
```

### Issue 4: "Permission denied" errors

**Symptoms**: Cannot write to cache or temp files

**Solutions**:

1. **Run G.R.I.M as Administrator** (Windows)

2. **Check directory permissions**:
   - Right-click on G.R.I.M folder
   - Properties ? Security
   - Ensure your user has Write permissions

### Issue 5: Slow scans or timeouts

**Symptoms**: Scans take very long or time out

**Solutions**:

1. **Use async mode**:
   ```
   sherlock_sweep username --async
   ```

2. **Check internet connection**

3. **Some platforms are slow** - this is normal
   - Sherlock checks 300+ sites
   - Some may be down or slow to respond

### Issue 6: "Invalid JSON" errors

**Symptoms**: Error about invalid JSON from Sherlock

**Solutions**:

1. **Update Sherlock**:
   ```bash
   cd osint\sherlock
   git pull origin master
   pip install -r requirements.txt --upgrade
   ```

2. **Test Sherlock directly**:
   ```bash
   python sherlock\sherlock.py testuser --json
   ```

3. **Check Python version** (needs 3.7+)

## Advanced Configuration

### Custom Python Path

If Python is not in PATH, you can specify it in the code:

Edit `external_collector/osit.cpp` and modify the `OSINTConfig`:

```cpp
OSINTConfig config;
config.pythonPath = "C:\\Python39\\python.exe";  // Your Python path
```

### Adjust Timeout

Default timeout is 30 seconds per platform. To change:

```cpp
OSINTConfig config;
config.timeoutSeconds = 60;  // Increase to 60 seconds
```

### Disable Cache

To always force fresh scans:

```cpp
OSINTConfig config;
config.useCache = false;
```

Or use the `--no-cache` flag:
```
profile_self username --no-cache
```

## Updating Sherlock

To get the latest platform definitions and bug fixes:

```bash
cd osint\sherlock
git pull origin master
pip install -r requirements.txt --upgrade
```

## Performance Tips

1. **Use caching**: Default 24-hour cache speeds up repeated scans
2. **Use async mode**: Run long scans in background
3. **Clear old cache**: Use `osint_clear_cache` periodically
4. **Update Sherlock**: New versions may be faster

## Testing Checklist

Before reporting issues, verify:

- [ ] Python 3.7+ is installed
- [ ] Sherlock exists at `osint/sherlock/`
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Sherlock works standalone: `python sherlock\sherlock.py testuser`
- [ ] Bridge works: `python osit_bridge.py testuser test.json`
- [ ] Internet connection is active
- [ ] No firewall blocking Python

## Getting Help

If you're still having issues:

1. **Check logs**: Look for error messages in G.R.I.M console
2. **Test each component** individually (Python ? Sherlock ? Bridge ? G.R.I.M)
3. **Report issues** on GitHub: https://github.com/Reaper3825/G.R.I.M/issues

Include:
- Error messages
- Python version: `python --version`
- Sherlock location: `dir osint\sherlock`
- What you tried from this guide

## Success Indicators

You'll know everything is working when:

? `profile_self testuser` returns scan results  
? No "Sherlock not found" errors  
? JSON output files are created in `cache/osint/`  
? Scans complete in 30-60 seconds  
? Platform URLs are displayed for found accounts  

---

**Ready to scan? Try it out:**
```
profile_self myusername
```

**Happy hunting! ?????**

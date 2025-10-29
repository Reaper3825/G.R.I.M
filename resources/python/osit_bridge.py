#!/usr/bin/env python3
"""
OSINT Bridge for G.R.I.M - Interfaces with Sherlock OSINT tool
Provides username enumeration across 300+ platforms
"""
import sys
import subprocess
import json
import os
import argparse
from pathlib import Path

def find_sherlock():
    """Locate the Sherlock executable or module"""
    # Try as installed package first (preferred method)
    try:
        import sherlock_project
        print(f"[OSINT Bridge] Found Sherlock as installed module", file=sys.stderr)
        return True, "module"
    except ImportError:
        pass
    
    # Try alternative module name
    try:
        import sherlock
        print(f"[OSINT Bridge] Found Sherlock as installed module (sherlock)", file=sys.stderr)
        return True, "module"
    except ImportError:
        pass
    
    # Try local installation in osint\sherlock folder (workspace root)
    script_dir = Path(__file__).parent  # resources/python
    workspace_root = script_dir.parent.parent  # Go up to workspace root
    
    # Check for sherlock_project structure (newer Sherlock versions)
    sherlock_dir = workspace_root / "osint" / "sherlock" / "sherlock_project"
    sherlock_script = sherlock_dir / "sherlock.py"
    
    if sherlock_script.exists():
        print(f"[OSINT Bridge] Found Sherlock at: {sherlock_script}", file=sys.stderr)
        return True, str(sherlock_script)
    
    # Check for older sherlock/sherlock.py structure
    sherlock_dir_old = workspace_root / "osint" / "sherlock" / "sherlock"
    sherlock_script_old = sherlock_dir_old / "sherlock.py"
    
    if sherlock_script_old.exists():
        print(f"[OSINT Bridge] Found Sherlock at: {sherlock_script_old}", file=sys.stderr)
        return True, str(sherlock_script_old)
    
    # Try even older structure
    sherlock_script_alt = workspace_root / "osint" / "sherlock" / "sherlock.py"
    if sherlock_script_alt.exists():
        print(f"[OSINT Bridge] Found Sherlock at: {sherlock_script_alt}", file=sys.stderr)
        return True, str(sherlock_script_alt)
    
    # Try relative to resources/python
    local_sherlock = script_dir / "sherlock" / "sherlock.py"
    if local_sherlock.exists():
        print(f"[OSINT Bridge] Found Sherlock at: {local_sherlock}", file=sys.stderr)
        return True, str(local_sherlock)
    
    print(f"[OSINT Bridge] Sherlock not found. Checked:", file=sys.stderr)
    print(f"  - Python module 'sherlock_project'", file=sys.stderr)
    print(f"  - Python module 'sherlock'", file=sys.stderr)
    print(f"  - {sherlock_script}", file=sys.stderr)
    print(f"  - {sherlock_script_old}", file=sys.stderr)
    print(f"  - {sherlock_script_alt}", file=sys.stderr)
    print(f"  - {local_sherlock}", file=sys.stderr)
    
    return False, None

def run_sherlock(username, output_file, timeout=60):
    """Execute Sherlock scan"""
    found, sherlock_path = find_sherlock()
    
    if not found:
        error_result = {
            "error": "Sherlock not found",
            "message": "Sherlock should be in osint/sherlock/ folder or installed via: pip install sherlock-project",
            "status": "ERROR"
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(error_result, f, indent=2)
        return 1
    
    try:
        # Store original directory
        original_dir = os.getcwd()
        
        if sherlock_path == "module":
            # Use installed module - capture output instead of using --json flag
            # The --json flag seems to crash in some versions
            cmd = [
                sys.executable, "-m", "sherlock_project", 
                username, 
                "--timeout", str(timeout),
                "--print-found",  # Only print found results
                "--no-color"  # Disable color codes
            ]
            
            print(f"[OSINT Bridge] Executing: {' '.join(cmd)}", file=sys.stderr)
            
            # Give plenty of time - Sherlock needs to check 400+ sites
            # Timeout should be much longer than the per-site timeout
            process_timeout = max(300, timeout * 15)  # At least 5 minutes or 15x the per-site timeout
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=process_timeout,
                check=False
            )
        else:
            # Use local installation
            script_dir = Path(__file__).parent
            workspace_root = script_dir.parent.parent
            sherlock_dir = workspace_root / "osint" / "sherlock"
            sherlock_script = Path(sherlock_path)
            
            os.chdir(str(sherlock_dir))
            
            cmd = [
                sys.executable, str(sherlock_script), 
                username, 
                "--timeout", str(timeout),
                "--print-found",
                "--no-color"
            ]
            
            print(f"[OSINT Bridge] Executing: {' '.join(cmd)}", file=sys.stderr)
            
            process_timeout = max(300, timeout * 15)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=process_timeout,
                check=False,
                cwd=str(sherlock_dir)
            )
        
        # Restore original directory
        os.chdir(original_dir)
        
        # Parse Sherlock text output and convert to JSON
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        
        print(f"[OSINT Bridge] Sherlock stdout length: {len(stdout)}", file=sys.stderr)
        
        # Parse the output
        data = {}
        lines = stdout.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for lines like: [+] GitHub: https://github.com/username
            if line.startswith('[+]'):
                # Remove the [+] prefix
                line = line[3:].strip()
                # Split by first colon
                parts = line.split(':', 1)
                if len(parts) == 2:
                    platform = parts[0].strip()
                    url = parts[1].strip()
                    # Remove "https://" or "http://" for cleaner platform names
                    if url.startswith('http'):
                        url = url.strip()
                    
                    data[platform] = {
                        "url_main": url,
                        "url": url,
                        "status": "Claimed",
                        "http_status": "200",
                        "response_time_s": 0
                    }
        
        # Write parsed results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        print(f"[OSINT Bridge] Parsed {len(data)} accounts from output", file=sys.stderr)
        print(f"[OSINT Bridge] Results written to: {output_file}", file=sys.stderr)
        print(f"[OSINT Bridge] Sherlock return code: {result.returncode}", file=sys.stderr)
        
        return 0
        
    except subprocess.TimeoutExpired:
        os.chdir(original_dir)
        error_result = {
            "error": "Scan timeout",
            "message": f"Scan exceeded {process_timeout} seconds (this is unusual - check network connectivity)",
            "status": "TIMEOUT"
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(error_result, f, indent=2)
        return 2
        
    except Exception as e:
        os.chdir(original_dir)
        error_result = {
            "error": str(e),
            "type": type(e).__name__,
            "status": "EXCEPTION",
            "traceback": str(e)
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(error_result, f, indent=2)
        return 1

def main():
    parser = argparse.ArgumentParser(description="OSINT Bridge for G.R.I.M")
    parser.add_argument("username", nargs='?', help="Username to search for")
    parser.add_argument("output_file", nargs='?', help="Output JSON file path")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds (default: 60)")
    
    # Handle both old and new argument styles
    if len(sys.argv) >= 3 and not sys.argv[1].startswith('-'):
        # Legacy mode: osit_bridge.py username output_file
        username = sys.argv[1]
        output_file = sys.argv[2]
        timeout = 60
        
        # Check for timeout flag
        for i, arg in enumerate(sys.argv[3:], start=3):
            if arg == "--timeout" and i + 1 < len(sys.argv):
                timeout = int(sys.argv[i + 1])
                break
    else:
        # New mode with argparse
        args = parser.parse_args()
        if not args.username or not args.output_file:
            print("Usage: osit_bridge.py <username> <output_file> [--timeout <seconds>]")
            print("Example: osit_bridge.py john_doe results.json --timeout 60")
            sys.exit(1)
        
        username = args.username
        output_file = args.output_file
        timeout = args.timeout
    
    print(f"[OSINT Bridge] Starting scan for: {username}", file=sys.stderr)
    print(f"[OSINT Bridge] Output file: {output_file}", file=sys.stderr)
    print(f"[OSINT Bridge] Timeout: {timeout}s", file=sys.stderr)
    
    exit_code = run_sherlock(username, output_file, timeout)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()

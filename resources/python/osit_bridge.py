import sys, subprocess, json, os

def main():
    if len(sys.argv) < 3:
        print("Usage: osit_bridge.py <username> <output_file>")
        sys.exit(1)

    username, output_file = sys.argv[1], sys.argv[2]
    sherlock_dir = os.path.join(os.path.dirname(__file__), "sherlock")
    sherlock_script = os.path.join(sherlock_dir, "sherlock.py")

    try:
        result = subprocess.run(
            [sys.executable, sherlock_script, username, "--json"],
            capture_output=True, text=True, check=False
        )
        stdout = result.stdout.strip() or "{}"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(stdout)
    except Exception as e:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"error": str(e)}, f)
        sys.exit(1)

if __name__ == "__main__":
    main()

# Training Log File Convention

## Problem Statement
When debugging training issues across multiple runs, it's easy to confuse observations from different training sessions. This leads to:
- Claiming current run has issues that only existed in old runs
- Wasting time investigating resolved issues
- Incorrect documentation mixing historical and current data

## Solution: Always Specify Which Log File

### Rule 1: Track Active Log at Top of Investigation Docs
Every investigation document MUST have a "CURRENT RUN TRACKING" section at the top:

```markdown
## 🎯 CURRENT RUN TRACKING (ALWAYS CHECK THIS FIRST!)

**Active Log File:** `training_17665346328529499.log`  
**Started:** December 23, 2025 at 19:03  
**Current Status:** Training in progress / recently completed  

**Current Run Observations:**
- [List observations specific to THIS log file]
```

### Rule 2: Every Observation Must Cite Source
When documenting ANY issue or observation, include:
1. **Log filename** (e.g., `training_17665346328529499.log`)
2. **Timestamp** from the log entry (e.g., `[2025-12-23 19:03:55]`)
3. **Mark historical vs current** (🔵 HISTORICAL or 🟢 CURRENT)

**Bad Example:**
```markdown
### Issue: Loss=0.0000 Events
Found in training logs...
```

**Good Example:**
```markdown
### Issue: Loss=0.0000 Events (🔵 HISTORICAL - Dec 22 run)
**Log File:** Previous run (Dec 22, 2025 14:26)
**NOT present in current run** (`training_17665346328529499.log`)

Found in historical log:
[2025-12-22 14:26:54] [GradTrace] POST-FORWARD loss=0.0000
```

### Rule 3: Use Current Log Symlink
Create a symlink or copy pointing to the active log:

```powershell
# Windows PowerShell
cd resources/models/GRIM-text/training/logs
Copy-Item training_17665346328529499.log current_run.log
```

Then reference `current_run.log` in documentation. Update the symlink when starting new training runs.

### Rule 4: Separate Historical from Current Sections
Structure investigation docs with clear boundaries:

```markdown
## 🟢 CURRENT RUN ANALYSIS (training_17665346328529499.log)
[Only observations from the active log file]

---

## 🔵 HISTORICAL ISSUES (Previous Training Runs)
[Archived observations from old logs, clearly dated]
```

### Rule 5: Verify Before Claiming
Before stating "the current run shows X", always:

```powershell
# Verify observation exists in CURRENT log
Get-Content "resources/models/GRIM-text/training/logs/training_17665346328529499.log" | Select-String "pattern"

# Double-check you're not looking at old logs
Get-ChildItem "resources/models/GRIM-text/training/logs/*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
```

## Quick Reference Commands

```powershell
# Find the most recent log file
$latest = Get-ChildItem "d:\G.R.I.M\resources\models\GRIM-text\training\logs\*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Write-Host "Latest log: $($latest.Name) (modified $($latest.LastWriteTime))"

# Search for pattern in specific log
Get-Content "d:\G.R.I.M\resources\models\GRIM-text\training\logs\training_17665346328529499.log" | Select-String "loss=0.0000"

# Compare observations across multiple logs
Get-ChildItem "d:\G.R.I.M\resources\models\GRIM-text\training\logs\*.log" | ForEach-Object {
    $count = (Get-Content $_.FullName | Select-String "loss=0.0000").Count
    if ($count -gt 0) {
        Write-Host "$($_.Name): $count occurrences"
    }
}
```

## For AI Assistants

When analyzing training issues:
1. **First action:** Read the "CURRENT RUN TRACKING" section of investigation doc
2. **Before any claim:** Verify the observation exists in the specified log file
3. **When confused:** Ask user "Which log file are we analyzing?" before proceeding
4. **Always include:** Log filename + timestamp when citing evidence
5. **Mark clearly:** Historical observations with 🔵 HISTORICAL tag

## Enforcement

When reviewing investigation documents or debugging discussions, check for:
- [ ] Current run clearly identified at top of document
- [ ] Every observation cites specific log file
- [ ] Historical vs current issues clearly separated
- [ ] No claims about "current run" without log file verification
- [ ] Timestamps included for all log excerpts

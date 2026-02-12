"""Quick script to catalog all bracket-prefixed tags in training_run.log."""
import sys

LOG = r"D:\G.R.I.M\resources\models\GRIM-text\training\logs\training_run.log"
OUT = r"D:\G.R.I.M\tag_catalog.txt"

tags = {}
with open(LOG, "r", encoding="utf-8", errors="replace") as fp:
    for i, line in enumerate(fp, 1):
        idx = line.find("[")
        if idx < 0 or idx >= 5:
            continue
        end = line.find("]", idx + 1)
        if end < 0 or (end - idx) >= 60:
            continue
        tag = line[idx : end + 1]
        if tag not in tags:
            tags[tag] = {"count": 0, "first": i, "sample": line.rstrip()}
        tags[tag]["count"] += 1

sorted_tags = sorted(tags.items(), key=lambda x: -x[1]["count"])

with open(OUT, "w", encoding="utf-8") as out:
    for tag, info in sorted_tags:
        out.write("%8d  first=%-8d  %s\n" % (info["count"], info["first"], tag))
        out.write("    SAMPLE: %s\n" % info["sample"][:200])

print("Wrote %d unique tags to %s" % (len(sorted_tags), OUT))
for tag, info in sorted_tags[:40]:
    print("%8d  first=%-8d  %s" % (info["count"], info["first"], tag))

#!/usr/bin/env python3
"""
Post-fix telemetry analysis.

Goal: determine whether the LM-head row-centering fix changed the underlying
collapse mechanism, and if rho is now an honest signal vs an artifact of
denominator collapse.

Reads the level-0 (per-step) telemetry rows for the streams that matter:
  5  rho_final
  6  rho_growth
  8  h_rms_growth                (mean rms ratio vs first layer)
 31  rho_raw_avg_abs_dot         (numerator)
 32  rho_raw_avg_norm_prod       (denominator)
 33  rho_raw_h_rms_min
 34  rho_raw_h_rms_max
 35  rms_gamma_pre_attn_rms
 36  rms_gamma_pre_ffn_rms
 37  rms_gamma_final_rms
"""

import csv
import sys
import math
from collections import defaultdict
from pathlib import Path

CSV = Path("resources/models/GRIM-text/training/logs/telemetry_1776548934293534998.csv")

WANT = {5, 6, 8, 31, 32, 33, 34, 35, 36, 37}

# step -> {stream_idx -> raw_observation}
data = defaultdict(dict)

with CSV.open() as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        # global_step,stream_idx,stream_name,level,stride,raw_observation,...
        level = int(row[3])
        if level != 0:
            continue
        s = int(row[1])
        if s not in WANT:
            continue
        step = int(row[0])
        try:
            data[step][s] = float(row[5])
        except ValueError:
            pass

steps_sorted = sorted(data.keys())
print(f"steps in CSV (level 0): {len(steps_sorted)}  range=[{steps_sorted[0]}, {steps_sorted[-1]}]")

def get_series(idx):
    xs, ys = [], []
    for s in steps_sorted:
        if idx in data[s]:
            xs.append(s)
            ys.append(data[s][idx])
    return xs, ys

def stats_at(idx, lo, hi):
    """min, max, mean of stream over [lo, hi)"""
    vals = [data[s][idx] for s in steps_sorted
            if lo <= s < hi and idx in data[s]]
    if not vals:
        return None, None, None
    return min(vals), max(vals), sum(vals) / len(vals)

def at_step(idx, target):
    """value of stream at step nearest to target"""
    best = min(steps_sorted, key=lambda s: abs(s - target))
    return data[best].get(idx), best

# ─── 1. rho-spread: did the fix work? ───────────────────────────────────
print("\n" + "=" * 70)
print("1. RMS SPREAD: did denominator collapse get fixed?")
print("=" * 70)
print(f"{'step':>6} {'rms_min':>9} {'rms_max':>9} {'spread':>8} {'denom':>9} {'|dot|':>9} {'rho':>7}")
for tgt in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
    s_actual = min(steps_sorted, key=lambda s: abs(s - tgt))
    d = data[s_actual]
    rmin = d.get(33, float("nan"))
    rmax = d.get(34, float("nan"))
    spread = rmax / rmin if rmin and rmin > 1e-9 else float("nan")
    denom = d.get(32, float("nan"))
    dot = d.get(31, float("nan"))
    rho = d.get(5, float("nan"))
    print(f"{s_actual:>6} {rmin:>9.4f} {rmax:>9.4f} {spread:>8.3f} "
          f"{denom:>9.2f} {dot:>9.2f} {rho:>7.4f}")

# Pre/post comparison
old_pk_min, _, _ = stats_at(33, 100, 300)
old_pk_max, _, _ = stats_at(34, 100, 300)
new_min, _, _ = stats_at(33, 15000, 20000)
new_max, _, _ = stats_at(34, 15000, 20000)
print(f"\nspread peak (steps 100-300): rms_min={old_pk_min:.3f} rms_max={old_pk_max:.3f}")
print(f"  → spread peak ≈ {old_pk_max/old_pk_min:.2f}x  (was 4.0x BEFORE the fix)")
print(f"spread end   (steps 15k-20k): rms_min={new_min:.3f} rms_max={new_max:.3f}")
print(f"  → spread end ≈ {new_max/new_min:.2f}x")

# ─── 2. is rho honest now? (numerator ratio test) ───────────────────────
print("\n" + "=" * 70)
print("2. IS RHO HONEST? (rho should equal |dot|/denom if both terms move)")
print("=" * 70)
print(f"{'step':>6} {'|dot|':>9} {'denom':>9} {'ratio':>9} {'rho':>9} {'rho-ratio':>11}")
for tgt in [50, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000]:
    s_actual = min(steps_sorted, key=lambda s: abs(s - tgt))
    d = data[s_actual]
    dot = d.get(31, float("nan"))
    denom = d.get(32, float("nan"))
    rho = d.get(5, float("nan"))
    ratio = dot / denom if denom > 0 else float("nan")
    print(f"{s_actual:>6} {dot:>9.2f} {denom:>9.2f} {ratio:>9.4f} "
          f"{rho:>9.4f} {rho - ratio:>+11.4f}")

# ─── 3. gamma drift (the next suspect) ──────────────────────────────────
print("\n" + "=" * 70)
print("3. GAMMA DRIFT (γ_final unbounded growth = logit temperature inflation)")
print("=" * 70)
print(f"{'step':>6} {'γ₁ pre-attn':>12} {'γ₂ pre-ffn':>12} {'γ_final':>10} "
      f"{'%drift_final':>13}")
for tgt in [100, 1000, 5000, 10000, 15000, 20000]:
    s_actual = min(steps_sorted, key=lambda s: abs(s - tgt))
    d = data[s_actual]
    g1 = d.get(35, float("nan"))
    g2 = d.get(36, float("nan"))
    gf = d.get(37, float("nan"))
    drift = (gf - 1.0) * 100.0 if not math.isnan(gf) else float("nan")
    print(f"{s_actual:>6} {g1:>12.4f} {g2:>12.4f} {gf:>10.4f} {drift:>+12.2f}%")

# ─── 4. correlation: does γ_final track h_rms_growth or rho? ────────────
print("\n" + "=" * 70)
print("4. CORRELATION: γ_final vs h_rms_growth, rho_final, denom")
print("=" * 70)

def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)

def aligned(idx_a, idx_b, lo=0, hi=None):
    if hi is None:
        hi = steps_sorted[-1] + 1
    xa, xb = [], []
    for s in steps_sorted:
        if not (lo <= s < hi):
            continue
        da, db = data[s].get(idx_a), data[s].get(idx_b)
        if da is None or db is None:
            continue
        if math.isnan(da) or math.isnan(db):
            continue
        xa.append(da); xb.append(db)
    return xa, xb

# Skip first 500 steps (warmup transient)
WINDOW_LO, WINDOW_HI = 500, 20000

pairs = [
    ("γ_final", 37, "h_rms_growth", 8),
    ("γ_final", 37, "rho_final",    5),
    ("γ_final", 37, "denom",        32),
    ("γ_final", 37, "|dot|",        31),
    ("γ_final", 37, "rms_max",      34),
    ("γ_final", 37, "rms_min",      33),
    ("h_rms_growth", 8, "rho_final", 5),
    ("h_rms_growth", 8, "denom",     32),
    ("h_rms_growth", 8, "|dot|",     31),
    ("rms_max",   34, "rms_min",    33),
]
for name_a, ia, name_b, ib in pairs:
    xa, xb = aligned(ia, ib, WINDOW_LO, WINDOW_HI)
    r = pearson(xa, xb)
    print(f"  ρ({name_a:>13s}, {name_b:>13s}) = {r:+.4f}   (n={len(xa)})")

# ─── 5. logit temperature surrogate ─────────────────────────────────────
# Effective LM-head pre-projection scale = γ_final × h_rms_after_centering
# We don't have rms_after_centering directly, but we have rms_avg via slot 8 / 33-34
print("\n" + "=" * 70)
print("5. LOGIT TEMPERATURE SURROGATE: γ_final * (rms_min+rms_max)/2")
print("=" * 70)
print("(this is what scales logits before softmax — driver of logit_max growth)")
print(f"{'step':>6} {'γ_final':>9} {'rms_avg':>9} {'temp_proxy':>11}")
for tgt in [100, 500, 1000, 5000, 10000, 15000, 20000]:
    s_actual = min(steps_sorted, key=lambda s: abs(s - tgt))
    d = data[s_actual]
    gf = d.get(37, float("nan"))
    rmin = d.get(33, float("nan"))
    rmax = d.get(34, float("nan"))
    avg = (rmin + rmax) / 2.0
    temp = gf * avg
    print(f"{s_actual:>6} {gf:>9.4f} {avg:>9.4f} {temp:>11.4f}")

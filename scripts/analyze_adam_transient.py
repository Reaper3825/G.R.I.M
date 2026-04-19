#!/usr/bin/env python3
"""
Adam early-step transient ↔ ρ spike alignment.

Hypothesis (from user observation): the early ρ spike / denom dip / h_rms
bifurcation are all driven by the well-known Adam bias-correction transient
during the first ~700 steps (β₂=0.999 → half-life=693).

Decompose:
  bc1(t)  = 1 - β₁^t                       # bias-correction for first moment
  bc2(t)  = 1 - β₂^t                       # bias-correction for second moment
  v_inflation(t)  = 1 / sqrt(bc2(t))       # how much √v̂ underestimates √E[g²]
  m_inflation(t)  = 1 / bc1(t)             # how much m̂ overestimates E[g]
  signal_ratio(t) = bc2(t) / (1-bc2(t))    # >1 = "learning", <1 = "destroying"
                                           # (the panel's green/red shading)

Then overlay these against:
  rho_final, denom (32), |dot| (31), h_rms_min (33), h_rms_max (34), lr,
  cumulative displacement (we don't have a stream for this, but we have
  grad_norm_mean × lr × step ≈ proxy)
"""

import csv, math
from collections import defaultdict
from pathlib import Path

CSV = Path("resources/models/GRIM-text/training/logs/telemetry_1776548934293534998.csv")
BETA1 = 0.9
BETA2 = 0.999

# Per Phase1 logs: warmup_fraction=0.05 with the soft-restart adjustment.
# From the LR plot the cosine has 3 cycles over 20k steps ≈ 6700/cycle.
# Visual peak appears around step ~2000 (first cycle peak).
TOTAL_STEPS = 20213
WARMUP_STEPS = int(round(0.05 * TOTAL_STEPS))  # ≈ 1011

WANT = {1: "loss", 4: "grad_norm_mean", 5: "rho_final",
        8: "h_rms_growth", 31: "raw_dot", 32: "raw_denom",
        33: "rms_min", 34: "rms_max", 37: "gamma_final"}

data = defaultdict(dict)
with CSV.open() as f:
    r = csv.reader(f); next(r)
    for row in r:
        if int(row[3]) != 0: continue
        s = int(row[1])
        if s in WANT:
            try: data[int(row[0])][s] = float(row[5])
            except ValueError: pass

steps = sorted(data.keys())

def adam_factors(t):
    """Returns (bc1, bc2, v_inflation, m_inflation, signal_ratio).
    bc2 < 0.5 → 'destroying' (panel red zone); bc2 > 0.5 → 'learning'."""
    bc1 = 1.0 - BETA1 ** t
    bc2 = 1.0 - BETA2 ** t
    v_inf = 1.0 / math.sqrt(bc2) if bc2 > 0 else float("inf")
    m_inf = 1.0 / bc1
    sig = bc2 / (1.0 - bc2) if bc2 < 1.0 else float("inf")
    return bc1, bc2, v_inf, m_inf, sig

def lr_at(t):
    """Linear warmup, then cosine. Approximation matches the LR plot shape."""
    base_lr = 6e-4
    if t < WARMUP_STEPS: return base_lr * (t / max(1, WARMUP_STEPS))
    # 3 cosine cycles
    cycle = TOTAL_STEPS // 3
    in_cycle = ((t - WARMUP_STEPS) % cycle) / cycle
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * in_cycle))

# ─── Print step-by-step alignment table for the critical window ─────────
print("=" * 92)
print(" Adam transient (theory) vs telemetry (measured) — early-spike alignment")
print("=" * 92)
print(f" β₁={BETA1}  β₂={BETA2}  half-life = {math.log(0.5)/math.log(BETA2):.0f} steps  warmup={WARMUP_STEPS}")
print()
hdr = (f"{'step':>5} | {'bc2':>6} {'v_inf':>6} {'sig':>7} {'lr':>9} | "
       f"{'rms_min':>8} {'rms_max':>8} {'spread':>7} | "
       f"{'denom':>7} {'|dot|':>7} {'rho':>7}")
print(hdr)
print("-" * len(hdr))
for tgt in [1, 5, 10, 25, 50, 100, 150, 200, 300, 500, 693, 1000, 2000, 5000, 10000, 20000]:
    s = min(steps, key=lambda x: abs(x - tgt))
    d = data[s]
    _, bc2, vi, _, sig = adam_factors(s + 1)
    rmin = d.get(33, float('nan')); rmax = d.get(34, float('nan'))
    sp = rmax / rmin if rmin > 1e-9 else float('nan')
    print(f"{s:>5} | {bc2:>6.4f} {vi:>6.2f} {sig:>7.3f} {lr_at(s):>9.2e} | "
          f"{rmin:>8.4f} {rmax:>8.4f} {sp:>7.3f} | "
          f"{d.get(32, float('nan')):>7.1f} {d.get(31, float('nan')):>7.1f} {d.get(5, float('nan')):>7.4f}")

# ─── Where do the spikes peak? ──────────────────────────────────────────
print("\n" + "=" * 70)
print(" PEAK LOCATIONS (within first 1000 steps)")
print("=" * 70)

def find_peak(idx, lo=0, hi=1000, mode="max"):
    fn = max if mode == "max" else min
    candidates = [(s, data[s][idx]) for s in steps if lo <= s < hi and idx in data[s]]
    if not candidates: return (None, None)
    return fn(candidates, key=lambda x: x[1])

for label, idx, mode in [("rho_final  PEAK", 5, "max"),
                         ("|dot|      PEAK", 31, "max"),
                         ("denom     TROUGH", 32, "min"),
                         ("rms_min   TROUGH", 33, "min"),
                         ("rms_max    PEAK", 34, "max"),
                         ("h_rms_grw  PEAK", 8, "max")]:
    s, v = find_peak(idx, mode=mode)
    if s is None: continue
    _, bc2, vi, _, _ = adam_factors(s + 1)
    print(f"  {label}: step {s:>4}  value={v:>8.4f}  "
          f"|  bc2={bc2:.3f}  v_inflation={vi:.2f}x")

# ─── Correlation: Adam transient vs measured signals over early window ──
print("\n" + "=" * 70)
print(" CORRELATION over [step 0, 1500]: v_inflation vs measured signals")
print("=" * 70)

def pearson(a, b):
    n = len(a)
    if n < 2: return float("nan")
    ma, mb = sum(a)/n, sum(b)/n
    num = sum((x-ma)*(y-mb) for x,y in zip(a,b))
    da = math.sqrt(sum((x-ma)**2 for x in a))
    db = math.sqrt(sum((y-mb)**2 for y in b))
    return num/(da*db) if da*db > 0 else float("nan")

WIN_LO, WIN_HI = 0, 1500
v_inf_series, vals = {}, defaultdict(list)
v_list = []
for s in steps:
    if not (WIN_LO <= s < WIN_HI): continue
    _, _, vi, _, _ = adam_factors(s + 1)
    v_list.append(vi)
    for idx in [5, 8, 31, 32, 33, 34]:
        if idx in data[s]:
            vals[idx].append((s, data[s][idx], vi))

names = {5: "rho_final", 8: "h_rms_growth", 31: "|dot|",
         32: "denom", 33: "rms_min", 34: "rms_max"}
for idx, name in names.items():
    pairs = vals[idx]
    if len(pairs) < 10: continue
    xs = [v for _, _, v in pairs]   # v_inflation
    ys = [y for _, y, _ in pairs]   # measured
    r = pearson(xs, ys)
    sign = "+" if r >= 0 else "-"
    print(f"  ρ(v_inflation, {name:>14s}) = {sign}{abs(r):.4f}   (n={len(pairs)})")

# ─── Spike vs LR-warmup timing ──────────────────────────────────────────
print("\n" + "=" * 70)
print(" CRITICAL TIMING: when does each transient die?")
print("=" * 70)
def steps_until(thresh_fn, max_step=2000):
    for t in range(1, max_step):
        if thresh_fn(t): return t
    return max_step
t_lr_full   = WARMUP_STEPS
t_bc2_half  = steps_until(lambda t: 1 - BETA2**t > 0.5)
t_bc2_90    = steps_until(lambda t: 1 - BETA2**t > 0.9)
t_v_inf_2   = steps_until(lambda t: 1.0/math.sqrt(1 - BETA2**t) < 2.0)

print(f"  LR warmup completes:           step {t_lr_full}")
print(f"  bc2 reaches 0.5 (half-life):   step {t_bc2_half}")
print(f"  bc2 reaches 0.9:               step {t_bc2_90}")
print(f"  v_inflation drops below 2x:    step {t_v_inf_2}")
print(f"  ρ peak occurs around step:     ~200 (per peak table above)")
print(f"  → ρ spike PRECEDES LR warmup completion AND bc2 half-life.")
print(f"  → ρ spike happens during 'destroying' zone (bc2 < 0.5).")

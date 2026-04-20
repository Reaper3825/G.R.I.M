#!/usr/bin/env python3
"""
Causation-vs-correlation interrogation of the early-window spike.

The user is correctly skeptical: matching DEATH-TIME does not prove cause.
Need to match SHAPE (a spike that rises and falls) and TEMPORAL ORDER
(cause must lead effect).

Hypotheses to test:
 H1. Adam v̂ underestimation → unbounded effective step → spike
     Predicts: monotonically decreasing magnitude. Should NOT show a peak.
 H2. LR-warmup × first-useful-gradient → first big update → spike then decay
     Predicts: peak when LR × grad_norm is maximized in early window.
 H3. Phase transition: model stumbles into local minimum, escapes
     Predicts: discrete event in loss / grad_norm that LEADS the rho spike.
 H4. Embedding-init artefact decaying as embeddings move away from Xavier
     Predicts: hidden-state norm DRIFTING from init, monotonic, no peak.

Streams (verified from TelemetryLattice_GPU.hpp):
  0  loss
  1  grad_norm_mean
  2  grad_norm_max
  3  learning_rate
  4  tokens_per_batch
  5  rho_final
  8  h_rms_growth
 10  adam_signal_dominance     (bc2 / (1-bc2))
 11  adam_cumulative_disp      (Σlr)
 12  adam_disruption_emb       (cumulative_disp / xavier_emb_scale)
 31  raw_dot
 32  raw_denom
 33  rms_min
 34  rms_max
"""
import csv, math
from collections import defaultdict
from pathlib import Path

CSV = Path("resources/models/GRIM-text/training/logs/telemetry_1776548934293534998.csv")

WANT = {0,1,2,3,5,8,10,11,12,31,32,33,34}
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

def get_series(idx, lo=0, hi=None):
    if hi is None: hi = steps[-1] + 1
    return [(s, data[s][idx]) for s in steps
            if lo <= s < hi and idx in data[s] and not math.isnan(data[s][idx])]

def find_extremum(idx, lo=0, hi=2000, mode="max"):
    pts = get_series(idx, lo, hi)
    if not pts: return None
    return max(pts, key=lambda p: p[1]) if mode == "max" else min(pts, key=lambda p: p[1])

# ─── 1. Cross-correlation lag analysis (causation needs cause to LEAD effect) ─────
print("=" * 78)
print("1. WHO LEADS WHOM? (cross-correlation lag, window steps 0..1500)")
print("=" * 78)
print("If grad_norm_max LEADS rho_final by k steps, that's evidence")
print("grad_norm spike CAUSES rho spike (not vice versa).\n")

WIN_LO, WIN_HI = 0, 1500

def aligned_pairs(idx_a, idx_b, lo, hi):
    a = {s: v for s, v in get_series(idx_a, lo, hi)}
    b = {s: v for s, v in get_series(idx_b, lo, hi)}
    common = sorted(set(a) & set(b))
    return common, [a[s] for s in common], [b[s] for s in common]

def pearson(a, b):
    n = len(a)
    if n < 2: return float("nan")
    ma, mb = sum(a)/n, sum(b)/n
    num = sum((x-ma)*(y-mb) for x,y in zip(a,b))
    da = math.sqrt(sum((x-ma)**2 for x in a))
    db = math.sqrt(sum((y-mb)**2 for y in b))
    return num/(da*db) if da*db > 0 else float("nan")

def lagged_corr(idx_cause, idx_effect, max_lag=50, lo=WIN_LO, hi=WIN_HI):
    """For lag>0: cause LEADS effect by `lag` steps (cause[t] → effect[t+lag])."""
    common, ca, ef = aligned_pairs(idx_cause, idx_effect, lo, hi)
    if len(common) < 100: return []
    # Build dict for fast lookup
    cause_dict = dict(zip(common, ca))
    effect_dict = dict(zip(common, ef))
    results = []
    for lag in range(-max_lag, max_lag + 1, 2):
        xs, ys = [], []
        for s in common:
            if s + lag in effect_dict:
                xs.append(cause_dict[s])
                ys.append(effect_dict[s + lag])
        if len(xs) >= 50:
            results.append((lag, pearson(xs, ys), len(xs)))
    return results

NAMES = {0:"loss", 1:"grad_norm_mean", 2:"grad_norm_max", 3:"learning_rate",
         5:"rho_final", 8:"h_rms_growth", 31:"|dot|", 32:"denom",
         33:"rms_min", 34:"rms_max", 12:"adam_disruption_emb"}

def report_lag(cause_idx, effect_idx):
    pairs = lagged_corr(cause_idx, effect_idx, max_lag=40)
    if not pairs:
        print(f"  {NAMES[cause_idx]:>20s} → {NAMES[effect_idx]:<15s}: insufficient data")
        return
    best = max(pairs, key=lambda p: abs(p[1]))
    lag, r, n = best
    direction = "LEADS" if lag > 0 else ("LAGS" if lag < 0 else "SIMUL")
    print(f"  {NAMES[cause_idx]:>20s} → {NAMES[effect_idx]:<15s}: best lag={lag:>+3d}  r={r:+.3f}  n={n}  ({direction} effect by {abs(lag)} steps)")

# Test: does grad_norm spike lead the rho spike?
report_lag(1, 5)   # grad_norm_mean → rho_final
report_lag(2, 5)   # grad_norm_max  → rho_final
report_lag(3, 5)   # LR             → rho_final
report_lag(2, 31)  # grad_norm_max  → |dot|
report_lag(2, 34)  # grad_norm_max  → rms_max
report_lag(2, 33)  # grad_norm_max  → rms_min   (negative? trough)
report_lag(0, 5)   # loss           → rho_final
print()
report_lag(5, 1)   # rho → grad_norm (reverse)
report_lag(5, 0)   # rho → loss (reverse)

# ─── 2. SHAPE check: which signals actually have a peak in [0, 1000]? ─────────
print("\n" + "=" * 78)
print("2. SHAPE CHECK: peak-and-decay signature (rise > fall > settle)")
print("=" * 78)
print("A signal with this shape has a discrete cause; a monotonic signal does not.\n")

def peak_metric(idx, lo=0, hi=1500):
    pts = get_series(idx, lo, hi)
    if len(pts) < 50: return None
    init = sum(v for _, v in pts[:10]) / 10
    peak_step, peak_val = max(pts, key=lambda p: abs(p[1] - init))
    end = sum(v for _, v in pts[-50:]) / 50
    rise = abs(peak_val - init)
    fall = abs(peak_val - end)
    overshoot = rise / max(abs(end - init), 1e-6)
    return (peak_step, init, peak_val, end, rise, fall, overshoot)

print(f"{'signal':>20s}  {'pk_step':>7s}  {'init':>9s}  {'peak':>9s}  {'end':>9s}  "
      f"{'rise':>8s}  {'fall':>8s}  {'overshoot':>10s}")
print("-" * 100)
for idx in [1, 2, 5, 8, 12, 31, 32, 33, 34]:
    m = peak_metric(idx)
    if m is None: continue
    ps, ini, pk, en, ri, fa, ov = m
    flag = " ← spike-and-decay" if ov > 1.5 else ""
    print(f"{NAMES.get(idx, str(idx)):>20s}  {ps:>7d}  {ini:>9.3f}  {pk:>9.3f}  {en:>9.3f}  "
          f"{ri:>8.3f}  {fa:>8.3f}  {ov:>10.2f}{flag}")

# ─── 3. The smoking gun: gradient norm trace in the spike window ──────────────
print("\n" + "=" * 78)
print("3. GRAD_NORM TRACE through the spike window")
print("=" * 78)
print("If grad_norm has a discrete spike that PRECEDES rho, that is the cause.\n")
print(f"{'step':>5}  {'lr':>9s}  {'gn_mean':>9s}  {'gn_max':>9s}  {'lr×gn_max':>10s}  "
      f"{'rho':>7s}  {'rms_max':>8s}  {'denom':>7s}")
for tgt in [1, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500, 700, 1000]:
    s = min(steps, key=lambda x: abs(x - tgt))
    d = data[s]
    lr = d.get(3, float('nan')); gnm = d.get(1, float('nan')); gnx = d.get(2, float('nan'))
    print(f"{s:>5}  {lr:>9.2e}  {gnm:>9.3e}  {gnx:>9.3e}  {lr*gnx if not math.isnan(gnx) else 0:>10.2e}  "
          f"{d.get(5,0):>7.4f}  {d.get(34,0):>8.4f}  {d.get(32,0):>7.1f}")

# ─── 4. Adam disruption stream (the GPU's own measurement) ─────────────────────
print("\n" + "=" * 78)
print("4. ADAM_DISRUPTION_EMB stream (the actual measured displacement)")
print("=" * 78)
print("This is `cumulative_disp / xavier_emb_scale` — the actual y-axis of the")
print("'Weight Disruption vs Xavier Init Scale' plot, measured on GPU.\n")
m = peak_metric(12, lo=0, hi=20000)
if m:
    ps, ini, pk, en, ri, fa, ov = m
    print(f"  initial:  {ini:.3f}  (step ~5)")
    print(f"  peak:     {pk:.3f}  at step {ps}")
    print(f"  end:      {en:.3f}  (last 50 steps avg)")
    print(f"  overshoot ratio: {ov:.2f}x  ({'spike-and-decay' if ov > 1.5 else 'monotonic'})")

# Compare ADAM_SIGNAL_DOMINANCE measured vs what theory predicts
print("\n  ADAM_SIGNAL_DOMINANCE (GPU stream 10) vs theoretical bc2/(1-bc2):")
print(f"  {'step':>5}  {'measured':>10s}  {'theory':>10s}  {'ratio':>7s}")
for tgt in [1, 50, 200, 500, 1000, 5000, 10000]:
    s = min(steps, key=lambda x: abs(x - tgt))
    if 10 not in data[s]: continue
    measured = data[s][10]
    bc2 = 1 - 0.999**(s+1)
    theory = bc2 / (1 - bc2) if bc2 < 1 else float('inf')
    ratio = measured / theory if theory > 0 else float('nan')
    print(f"  {s:>5}  {measured:>10.4f}  {theory:>10.4f}  {ratio:>7.3f}")

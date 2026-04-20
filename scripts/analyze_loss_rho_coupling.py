#!/usr/bin/env python3
"""
Investigate the loss ↔ rho_final correlation (r=-0.47, simultaneous, strongest in early window).

Questions:
 Q1. Is r=-0.47 a window artifact, or persistent across training?
 Q2. Is the relationship linear, or does it have a regime change?
 Q3. Does loss actually drop FASTER when rho is rising, or do they just both move?
 Q4. What component of loss drives it? cross-entropy on frequent tokens vs rare?
 Q5. Conditioning on lr/warmup: is r=-0.47 just because both depend on time?
 Q6. The "uniform attractor" hypothesis predicts: when rho peaks, model output is
     close to the unigram distribution → loss should be near unigram-CE entropy.
     What is the loss value at the rho peak (step 430)?  Is it ≈ unigram entropy?
"""
import csv, math
from collections import defaultdict
from pathlib import Path

CSV = Path("resources/models/GRIM-text/training/logs/telemetry_1776548934293534998.csv")
WANT = {0, 1, 2, 3, 5, 8, 31, 32, 33, 34, 38}
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
def series(idx, lo=0, hi=None):
    hi = hi if hi is not None else steps[-1]+1
    return [(s, data[s][idx]) for s in steps
            if lo <= s < hi and idx in data[s] and not math.isnan(data[s][idx])]
def pearson(xs, ys):
    n = len(xs)
    if n < 2: return float('nan')
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx = math.sqrt(sum((x-mx)**2 for x in xs))
    dy = math.sqrt(sum((y-my)**2 for y in ys))
    return num/(dx*dy) if dx*dy>0 else float('nan')
def aligned(a_idx, b_idx, lo, hi):
    a = dict(series(a_idx, lo, hi)); b = dict(series(b_idx, lo, hi))
    common = sorted(set(a)&set(b))
    return [a[s] for s in common], [b[s] for s in common], common

# ─── Q1. Window-by-window correlation: is r=-0.47 universal or local? ─────────
print("="*78); print("Q1. Loss↔ρ correlation by training window"); print("="*78)
print(f"{'window':>15}  {'r(loss,rho)':>12}  {'n':>6}  {'loss range':>20}  {'rho range':>15}")
windows = [(0,200),(200,500),(500,1000),(1000,2000),(2000,5000),
           (5000,10000),(10000,15000),(15000,20000),(0,20000)]
for lo, hi in windows:
    a, b, _ = aligned(0, 5, lo, hi)
    if len(a) < 30: continue
    r = pearson(a, b)
    lr_str = f"{min(a):.2f}→{max(a):.2f}"
    rr_str = f"{min(b):.3f}→{max(b):.3f}"
    print(f"  [{lo:>5},{hi:>5})  {r:>12.3f}  {len(a):>6}  {lr_str:>20}  {rr_str:>15}")

# ─── Q2. Phase plot: where in (loss, rho) space does training live? ───────────
print("\n"+"="*78); print("Q2. Joint trajectory through (loss, rho) phase space"); print("="*78)
a, b, sb = aligned(0, 5, 0, 20000)
phases = [(0,100,"warmup"),(100,500,"early-spike"),(500,2000,"recovery"),
          (2000,10000,"plateau-1"),(10000,20000,"plateau-2")]
print(f"{'phase':>15}  {'steps':>10}  {'mean loss':>10}  {'mean rho':>10}  "
      f"{'Δloss':>10}  {'Δrho':>10}")
prev_loss, prev_rho = None, None
for lo, hi, name in phases:
    pts = [(s, l, r) for s, l, r in zip(sb, a, b) if lo <= s < hi]
    if not pts: continue
    mean_loss = sum(p[1] for p in pts)/len(pts)
    mean_rho = sum(p[2] for p in pts)/len(pts)
    dl = mean_loss - prev_loss if prev_loss is not None else float('nan')
    dr = mean_rho - prev_rho if prev_rho is not None else float('nan')
    print(f"  {name:>15}  [{lo:>4},{hi:>4})  {mean_loss:>10.3f}  {mean_rho:>10.3f}  "
          f"{dl:>+10.3f}  {dr:>+10.3f}")
    prev_loss, prev_rho = mean_loss, mean_rho

# ─── Q3. Velocity correlation: does Δloss correlate with Δrho? ────────────────
print("\n"+"="*78); print("Q3. VELOCITY correlation (rate of change)"); print("="*78)
print("If 'loss drops faster when rho rising' is a real coupling,")
print("then Δloss and Δrho should correlate, not just levels.\n")
def velocity(idx, lo, hi, smooth=10):
    pts = series(idx, lo, hi)
    out = []
    for i in range(smooth, len(pts)-smooth):
        s_mid = pts[i][0]
        avg_before = sum(pts[i-j][1] for j in range(1,smooth+1))/smooth
        avg_after  = sum(pts[i+j][1] for j in range(1,smooth+1))/smooth
        out.append((s_mid, avg_after - avg_before))
    return out
for lo, hi in [(0,500),(500,2000),(2000,10000),(10000,20000)]:
    vl = dict(velocity(0, lo, hi))
    vr = dict(velocity(5, lo, hi))
    common = sorted(set(vl)&set(vr))
    if len(common) < 30: continue
    a = [vl[s] for s in common]; b = [vr[s] for s in common]
    r = pearson(a, b)
    print(f"  [{lo:>5},{hi:>5})  r(Δloss, Δrho)={r:>+.3f}  n={len(common)}")

# ─── Q4. Loss at rho peak — does it match unigram entropy? ────────────────────
print("\n"+"="*78); print("Q4. Uniform-attractor test: loss at ρ peak"); print("="*78)
# Vocab is 10262 from your config
import math as _m
H_uniform = _m.log(10262)
# Approximate unigram entropy: typical English text ≈ 6.5–7.5 nats over a 10k vocab
# Without the actual unigram dist we can bound it
H_unigram_typical = 5.5  # nats; common-token-heavy distributions
print(f"  Vocab size: 10262")
print(f"  Uniform-distribution loss (max possible): ln(10262) = {H_uniform:.3f}")
print(f"  Typical unigram entropy on natural text:  ~5.5–6.5 nats (heavy tail of frequents)")
print(f"  Expected loss at 'predict unigram everywhere' attractor: ~5.5–6.5\n")
peak_step = 430
window = [s for s in steps if abs(s - peak_step) < 10]
losses = [data[s][0] for s in window if 0 in data[s]]
rhos   = [data[s][5] for s in window if 5 in data[s]]
print(f"  Loss at ρ peak (step {peak_step} ±10):  mean={sum(losses)/len(losses):.3f}")
print(f"  ρ at ρ peak (step {peak_step} ±10):     mean={sum(rhos)/len(rhos):.3f}")
print(f"  → If loss ≈ 5.5–6.5 here, model HAS converged on unigram-attractor")

# Trace loss at rho-peak vs other key moments
print(f"\n  {'step':>6}  {'loss':>7}  {'rho':>7}  {'rms_max':>8}  {'denom':>8}  {'spread':>7}")
for tgt in [10, 100, 200, 226, 304, 430, 500, 700, 1000, 2000, 5000, 10000, 19000]:
    s = min(steps, key=lambda x: abs(x-tgt))
    d = data[s]
    print(f"  {s:>6}  {d.get(0,0):>7.3f}  {d.get(5,0):>7.3f}  {d.get(34,0):>8.3f}  "
          f"{d.get(32,0):>8.1f}  {d.get(38,0):>7.3f}")

# ─── Q5. Partial-correlation: condition out time/lr ────────────────────────────
print("\n"+"="*78); print("Q5. Partial correlation (condition out LR)"); print("="*78)
print("If r(loss,rho) survives after removing LR's effect, the coupling is real,")
print("not just 'both depend on training step'.\n")
def linreg(xs, ys):
    n = len(xs); mx, my = sum(xs)/n, sum(ys)/n
    sxy = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    sxx = sum((x-mx)**2 for x in xs)
    if sxx == 0: return 0, my, ys
    slope = sxy/sxx; intercept = my - slope*mx
    resid = [y - (slope*x + intercept) for x,y in zip(xs,ys)]
    return slope, intercept, resid

for lo, hi in [(0,500),(500,2000),(2000,10000)]:
    # Get loss, rho, lr aligned
    aL = dict(series(0, lo, hi)); aR = dict(series(5, lo, hi)); aLR = dict(series(3, lo, hi))
    common = sorted(set(aL)&set(aR)&set(aLR))
    if len(common) < 30: continue
    L = [aL[s] for s in common]; R = [aR[s] for s in common]; LR = [aLR[s] for s in common]
    raw_r = pearson(L, R)
    # Residualize loss and rho on lr
    _, _, L_res = linreg(LR, L)
    _, _, R_res = linreg(LR, R)
    partial_r = pearson(L_res, R_res)
    print(f"  [{lo:>5},{hi:>5})  raw r(loss,rho)={raw_r:>+.3f}  "
          f"partial r(loss,rho|lr)={partial_r:>+.3f}  diff={partial_r-raw_r:>+.3f}")

# ─── Q6. Direct test of the attractor: is loss-rate-of-decrease GREATER during ρ peak? ─
print("\n"+"="*78); print("Q6. Does loss drop FASTER while ρ is elevated?"); print("="*78)
print("If ρ-peak corresponds to model collapsing to unigram (uniform attractor),")
print("then loss should drop ESPECIALLY FAST at that moment (free CE wins).\n")
def avg_drop(lo, hi):
    pts = series(0, lo, hi)
    if len(pts) < 20: return float('nan')
    return (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0])
print(f"  Steps [   0, 100):  loss-drop rate = {avg_drop(0,100):>+.5f}/step")
print(f"  Steps [ 100, 200):  loss-drop rate = {avg_drop(100,200):>+.5f}/step  (ρ rising)")
print(f"  Steps [ 200, 300):  loss-drop rate = {avg_drop(200,300):>+.5f}/step  (ρ at peak)")
print(f"  Steps [ 300, 500):  loss-drop rate = {avg_drop(300,500):>+.5f}/step  (ρ at peak)")
print(f"  Steps [ 500,1000):  loss-drop rate = {avg_drop(500,1000):>+.5f}/step  (ρ relaxing)")
print(f"  Steps [1000,2000):  loss-drop rate = {avg_drop(1000,2000):>+.5f}/step")
print(f"  Steps [2000,5000):  loss-drop rate = {avg_drop(2000,5000):>+.5f}/step")

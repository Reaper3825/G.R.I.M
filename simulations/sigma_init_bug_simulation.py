"""
Sigma Initialization Bug Simulation — TelemetryLattice
========================================================
The real problem is NOT the epsilon floor (that's a minor secondary issue).
The real problem is initialization:

BUG 1 (denominator): sigma_prev = 0 at init. First real update computes:
  delta_hat = (x_1 - x_0) / (0 + 1e-7) = O(1e5) for loss-scale streams

BUG 2 (seed bias): mu and m2 seeded from single sample x_0.
  EMA at beta=0.95 takes ~20 steps to forget seed.
  During ramp: m2 - mu² reflects seed→true-mean drift, not actual variance.
  sigma is inflated for entire warmup window.

PASS CONDITION: no artificial gradnorm jump without hacks.
  step 2 delta_hat must be O(1), not O(1e5).
  sigma during first 20 steps must not be orders-of-magnitude above steady state.

Reproduces EXACT kernel logic from TelemetryLattice_GPU.cu.
"""

import math
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# Hyperparameters (exact match to TelemetryHyperParams)
# ═══════════════════════════════════════════════════════════════════════════════
HP = {
    'beta_mu':    0.95,
    'beta_a':     0.995,
    'beta_delta': 0.90,
    'beta_r':     0.85,
    'beta_run':   0.80,
    'beta_v':     0.90,
    'k_out0':     2.5,
    'alpha_v':    1.5,
    'epsilon':    1e-7,
}

N_STEPS = 200
np.random.seed(42)

def make_streams():
    streams = {}
    # 1. Loss: ~3.7 ± 0.03 (0.8% CV) — the 300K delta_hat stream
    streams['loss'] = 3.7 + 0.03 * np.random.randn(N_STEPS)
    # 2. grad_norm: ~5.5e-5 ± 1e-5 (18% CV) — small magnitude
    streams['grad_norm'] = np.abs(5.5e-5 + 1e-5 * np.random.randn(N_STEPS))
    # 3. rho: ~0.30 ± 0.06 (20% CV) — medium variance
    streams['rho'] = np.clip(0.30 + 0.06 * np.random.randn(N_STEPS), 0.01, 0.99)
    # 4. lr constant: 6e-5 (zero variance)
    streams['lr_constant'] = np.full(N_STEPS, 6e-5)
    return streams

# ═══════════════════════════════════════════════════════════════════════════════
# State + helpers
# ═══════════════════════════════════════════════════════════════════════════════
def fresh_state():
    return {
        'mu': 0.0, 'm2': 0.0, 'sigma': 0.0, 'sigma_tilde': 0.0,
        'mu_a': 0.0, 'sigma_a': 0.0, 'delta_mu': 0.0, 'delta_sigma': 0.0,
        'v_sigma': 0.0, 'sigma_prev': 0.0,
        'delta_bar': 0.0, 'p': 0.0, 'mu_prev': 0.0,
        'r_out': 0.0, 'ell_out': 0.0, 'mu_ex': 0.0,
        'k_out': HP['k_out0'], 'c_out': 0.0,
        'step_count': 0, 'initialized': False,
        # Welford accumulator (only used by FIX 2)
        '_w_n': 0, '_w_mean': 0.0, '_w_m2': 0.0,
    }

def safe_sqrt(x, eps):
    return math.sqrt(max(x, eps))

def safe_sign(x):
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)

# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE: exact reproduction of current kernel init + update
# ═══════════════════════════════════════════════════════════════════════════════
def update_BASELINE(s, x_t, hp):
    eps = hp['epsilon']

    if not s['initialized']:
        s['mu'] = x_t
        s['m2'] = x_t * x_t
        s['sigma'] = 0.0               # ← BUG 1: sigma_prev will be 0
        s['sigma_tilde'] = 0.0
        s['mu_a'] = x_t
        s['sigma_a'] = 0.0
        s['delta_mu'] = 0.0
        s['delta_sigma'] = 0.0
        s['v_sigma'] = 0.0
        s['sigma_prev'] = 0.0          # ← BUG 1: zero denominator next step
        s['delta_bar'] = 0.0
        s['p'] = 0.0
        s['mu_prev'] = x_t             # ← BUG 2: single-sample seed
        s['r_out'] = 0.0
        s['ell_out'] = 0.0
        s['mu_ex'] = 0.0
        s['k_out'] = hp['k_out0']
        s['c_out'] = x_t
        s['step_count'] = 1
        s['initialized'] = True
        return 0.0  # delta_hat for recording

    # STEP 1
    beta_mu = hp['beta_mu']
    s['mu'] = beta_mu * s['mu'] + (1.0 - beta_mu) * x_t
    s['m2'] = beta_mu * s['m2'] + (1.0 - beta_mu) * (x_t * x_t)
    variance = max(s['m2'] - s['mu'] * s['mu'], eps)
    s['sigma'] = safe_sqrt(variance, eps)
    s['sigma_tilde'] = s['sigma'] / (abs(s['mu']) + eps)

    # STEP 2
    sigma_delta = s['sigma'] - s['sigma_prev']
    s['v_sigma'] = hp['beta_v'] * s['v_sigma'] + (1.0 - hp['beta_v']) * sigma_delta * sigma_delta
    s['sigma_prev'] = s['sigma']

    # STEP 3
    v_sigma_sqrt = safe_sqrt(s['v_sigma'], eps)
    s['k_out'] = hp['k_out0'] * (1.0 + hp['alpha_v'] * v_sigma_sqrt)
    s['c_out'] = s['mu'] + s['k_out'] * s['sigma']

    # STEP 4
    delta_hat = (x_t - s['mu_prev']) / (s['sigma_prev'] + eps)
    s['delta_bar'] = hp['beta_delta'] * s['delta_bar'] + (1.0 - hp['beta_delta']) * delta_hat
    s['p'] = hp['beta_delta'] * s['p'] + (1.0 - hp['beta_delta']) * safe_sign(delta_hat)
    s['mu_prev'] = s['mu']

    # STEP 5
    outlier_arg = max(min((x_t - s['c_out']) / (s['sigma'] + eps), 80.0), -80.0)
    w_out = 1.0 / (1.0 + math.exp(-outlier_arg))
    s['r_out'] = hp['beta_r'] * s['r_out'] + (1.0 - hp['beta_r']) * w_out

    # STEP 6
    eta = max(0.0, x_t - s['c_out'])
    s['mu_ex'] = beta_mu * s['mu_ex'] + (1.0 - beta_mu) * eta

    # STEP 7
    s['ell_out'] = hp['beta_run'] * s['ell_out'] + (1.0 - hp['beta_run']) * w_out

    # STEP 8
    s['mu_a'] = hp['beta_a'] * s['mu_a'] + (1.0 - hp['beta_a']) * s['mu']
    s['sigma_a'] = hp['beta_a'] * s['sigma_a'] + (1.0 - hp['beta_a']) * s['sigma']
    s['delta_mu'] = s['mu'] - s['mu_a']
    s['delta_sigma'] = s['sigma'] - s['sigma_a']

    s['step_count'] += 1
    return delta_hat

# ═══════════════════════════════════════════════════════════════════════════════
# FIX 1: Patch sigma_prev on first real update
#
# Change: After computing sigma on the FIRST post-init update, set
#   sigma_prev = sigma BEFORE computing delta_hat.
# This eliminates the 0-denominator at step 2.
# Does NOT fix the single-sample seed bias (bug 2 remains).
#
# Math at step 2:
#   sigma_prev was 0 → patched to sigma (just computed)
#   delta_hat = (x_1 - mu_prev) / (sigma + eps)
#   For loss: ≈ 0.03 / 0.03 ≈ 1.0  (not 300,000)
# ═══════════════════════════════════════════════════════════════════════════════
def update_FIX1(s, x_t, hp):
    eps = hp['epsilon']

    if not s['initialized']:
        s['mu'] = x_t
        s['m2'] = x_t * x_t
        s['sigma'] = 0.0
        s['sigma_tilde'] = 0.0
        s['mu_a'] = x_t
        s['sigma_a'] = 0.0
        s['delta_mu'] = 0.0
        s['delta_sigma'] = 0.0
        s['v_sigma'] = 0.0
        s['sigma_prev'] = 0.0
        s['delta_bar'] = 0.0
        s['p'] = 0.0
        s['mu_prev'] = x_t
        s['r_out'] = 0.0
        s['ell_out'] = 0.0
        s['mu_ex'] = 0.0
        s['k_out'] = hp['k_out0']
        s['c_out'] = x_t
        s['step_count'] = 1
        s['initialized'] = True
        return 0.0

    beta_mu = hp['beta_mu']
    s['mu'] = beta_mu * s['mu'] + (1.0 - beta_mu) * x_t
    s['m2'] = beta_mu * s['m2'] + (1.0 - beta_mu) * (x_t * x_t)
    variance = max(s['m2'] - s['mu'] * s['mu'], eps)
    s['sigma'] = safe_sqrt(variance, eps)
    s['sigma_tilde'] = s['sigma'] / (abs(s['mu']) + eps)

    # ── FIX 1: if sigma_prev is still 0, seed it from first computed sigma ──
    if s['sigma_prev'] == 0.0:
        s['sigma_prev'] = s['sigma']

    sigma_delta = s['sigma'] - s['sigma_prev']
    s['v_sigma'] = hp['beta_v'] * s['v_sigma'] + (1.0 - hp['beta_v']) * sigma_delta * sigma_delta
    s['sigma_prev'] = s['sigma']

    v_sigma_sqrt = safe_sqrt(s['v_sigma'], eps)
    s['k_out'] = hp['k_out0'] * (1.0 + hp['alpha_v'] * v_sigma_sqrt)
    s['c_out'] = s['mu'] + s['k_out'] * s['sigma']

    delta_hat = (x_t - s['mu_prev']) / (s['sigma_prev'] + eps)
    s['delta_bar'] = hp['beta_delta'] * s['delta_bar'] + (1.0 - hp['beta_delta']) * delta_hat
    s['p'] = hp['beta_delta'] * s['p'] + (1.0 - hp['beta_delta']) * safe_sign(delta_hat)
    s['mu_prev'] = s['mu']

    outlier_arg = max(min((x_t - s['c_out']) / (s['sigma'] + eps), 80.0), -80.0)
    w_out = 1.0 / (1.0 + math.exp(-outlier_arg))
    s['r_out'] = hp['beta_r'] * s['r_out'] + (1.0 - hp['beta_r']) * w_out

    eta = max(0.0, x_t - s['c_out'])
    s['mu_ex'] = beta_mu * s['mu_ex'] + (1.0 - beta_mu) * eta

    s['ell_out'] = hp['beta_run'] * s['ell_out'] + (1.0 - hp['beta_run']) * w_out

    s['mu_a'] = hp['beta_a'] * s['mu_a'] + (1.0 - hp['beta_a']) * s['mu']
    s['sigma_a'] = hp['beta_a'] * s['sigma_a'] + (1.0 - hp['beta_a']) * s['sigma']
    s['delta_mu'] = s['mu'] - s['mu_a']
    s['delta_sigma'] = s['sigma'] - s['sigma_a']

    s['step_count'] += 1
    return delta_hat

# ═══════════════════════════════════════════════════════════════════════════════
# FIX 2: Welford warmup (N samples) before starting EMA
#
# Change: For first N observations, accumulate mean/variance via Welford's
# online algorithm. At step N, seed EMA state with Welford estimates.
# Before N, all sigma-derived outputs stay at 0.
#
# This fixes BOTH bugs:
#   - sigma_prev is seeded from real variance (not 0)
#   - mu, m2 are seeded from N-sample statistics (not single sample)
#
# Math: Welford gives exact (unbiased) mean and variance from N samples.
#   mu_seed = mean(x_0..x_{N-1})
#   var_seed = var(x_0..x_{N-1})
#   sigma_seed = sqrt(var_seed)
# ═══════════════════════════════════════════════════════════════════════════════
WELFORD_N = 20

def update_FIX2(s, x_t, hp):
    eps = hp['epsilon']

    # ── Welford accumulation phase ──
    if s['_w_n'] < WELFORD_N:
        s['_w_n'] += 1
        delta = x_t - s['_w_mean']
        s['_w_mean'] += delta / s['_w_n']
        delta2 = x_t - s['_w_mean']
        s['_w_m2'] += delta * delta2

        if s['_w_n'] == WELFORD_N:
            # Seed EMA state from Welford statistics
            w_var = s['_w_m2'] / s['_w_n']  # population variance (matches EMA bias)
            w_sigma = math.sqrt(max(w_var, 0.0))

            s['mu'] = s['_w_mean']
            s['m2'] = s['_w_mean'] * s['_w_mean'] + w_var  # E[x²] = mu² + var
            s['sigma'] = w_sigma
            s['sigma_tilde'] = w_sigma / (abs(s['_w_mean']) + eps) if w_sigma > 0 else 0.0
            s['mu_a'] = s['_w_mean']
            s['sigma_a'] = w_sigma
            s['sigma_prev'] = w_sigma       # ← fixes bug 1
            s['mu_prev'] = s['_w_mean']     # ← fixes bug 2
            s['c_out'] = s['_w_mean'] + hp['k_out0'] * w_sigma
            s['k_out'] = hp['k_out0']
            s['step_count'] = WELFORD_N
            s['initialized'] = True

        return 0.0  # no delta_hat during warmup

    # ── Normal EMA update (identical to baseline after init) ──
    beta_mu = hp['beta_mu']
    s['mu'] = beta_mu * s['mu'] + (1.0 - beta_mu) * x_t
    s['m2'] = beta_mu * s['m2'] + (1.0 - beta_mu) * (x_t * x_t)
    variance = max(s['m2'] - s['mu'] * s['mu'], eps)
    s['sigma'] = safe_sqrt(variance, eps)
    s['sigma_tilde'] = s['sigma'] / (abs(s['mu']) + eps)

    sigma_delta = s['sigma'] - s['sigma_prev']
    s['v_sigma'] = hp['beta_v'] * s['v_sigma'] + (1.0 - hp['beta_v']) * sigma_delta * sigma_delta
    s['sigma_prev'] = s['sigma']

    v_sigma_sqrt = safe_sqrt(s['v_sigma'], eps)
    s['k_out'] = hp['k_out0'] * (1.0 + hp['alpha_v'] * v_sigma_sqrt)
    s['c_out'] = s['mu'] + s['k_out'] * s['sigma']

    delta_hat = (x_t - s['mu_prev']) / (s['sigma_prev'] + eps)
    s['delta_bar'] = hp['beta_delta'] * s['delta_bar'] + (1.0 - hp['beta_delta']) * delta_hat
    s['p'] = hp['beta_delta'] * s['p'] + (1.0 - hp['beta_delta']) * safe_sign(delta_hat)
    s['mu_prev'] = s['mu']

    outlier_arg = max(min((x_t - s['c_out']) / (s['sigma'] + eps), 80.0), -80.0)
    w_out = 1.0 / (1.0 + math.exp(-outlier_arg))
    s['r_out'] = hp['beta_r'] * s['r_out'] + (1.0 - hp['beta_r']) * w_out

    eta = max(0.0, x_t - s['c_out'])
    s['mu_ex'] = beta_mu * s['mu_ex'] + (1.0 - beta_mu) * eta

    s['ell_out'] = hp['beta_run'] * s['ell_out'] + (1.0 - hp['beta_run']) * w_out

    s['mu_a'] = hp['beta_a'] * s['mu_a'] + (1.0 - hp['beta_a']) * s['mu']
    s['sigma_a'] = hp['beta_a'] * s['sigma_a'] + (1.0 - hp['beta_a']) * s['sigma']
    s['delta_mu'] = s['mu'] - s['mu_a']
    s['delta_sigma'] = s['sigma'] - s['sigma_a']

    s['step_count'] += 1
    return delta_hat

# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════
FIELDS = ['sigma', 'sigma_tilde', 'delta_bar', 'v_sigma', 'k_out',
          'r_out', 'mu', 'mu_a', 'delta_mu', 'delta_hat']

def run_variant(streams, update_fn, label):
    results = {}
    for name, data in streams.items():
        s = fresh_state()
        history = {f: [] for f in FIELDS}
        for x_t in data:
            dh = update_fn(s, x_t, HP)
            for f in FIELDS:
                if f == 'delta_hat':
                    history[f].append(dh)
                else:
                    history[f].append(s[f])
        results[name] = history
    return results

# ═══════════════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def analyze(results, label, streams):
    L = []
    L.append(f"\n{'='*80}")
    L.append(f"  {label}")
    L.append(f"{'='*80}")

    pass_all = True

    for name, hist in results.items():
        data = streams[name]
        sigma = np.array(hist['sigma'])
        mu = np.array(hist['mu'])
        mu_a = np.array(hist['mu_a'])
        dh = np.array(hist['delta_hat'])
        v_sigma = np.array(hist['v_sigma'])
        delta_bar = np.array(hist['delta_bar'])
        delta_mu = np.array(hist['delta_mu'])

        # True stream statistics for reference
        true_mean = np.mean(data)
        true_std = np.std(data)

        L.append(f"\n  ── {name} (true μ={true_mean:.6e}, true σ={true_std:.6e}) ──")

        # --- Startup contamination metrics ---

        # 1. delta_hat at step 2 (the critical one)
        dh_step2 = dh[1] if len(dh) > 1 else 0.0
        L.append(f"    delta_hat[step 2]:     {dh_step2:.6e}")

        # 2. Max |delta_hat| in first 5 steps
        dh_max_early = np.max(np.abs(dh[:5])) if len(dh) >= 5 else np.max(np.abs(dh))
        L.append(f"    max|delta_hat| [0:5]:  {dh_max_early:.6e}")

        # 3. sigma at steps 2, 5, 10, 20, 50 vs steady state (step 150+)
        checkpoints = [2, 5, 10, 20, 50]
        sigma_ss = np.mean(sigma[150:]) if len(sigma) > 150 else sigma[-1]
        L.append(f"    sigma steady-state:    {sigma_ss:.6e}")
        for cp in checkpoints:
            if cp < len(sigma):
                ratio = sigma[cp] / sigma_ss if sigma_ss > 0 else float('inf')
                L.append(f"    sigma[{cp:>3d}]:            {sigma[cp]:.6e}  ({ratio:.2f}× steady)")

        # 4. mu - mu_a drift at step 5 vs steady state
        dmu_5 = abs(delta_mu[5]) if len(delta_mu) > 5 else 0
        dmu_ss = np.mean(np.abs(delta_mu[150:])) if len(delta_mu) > 150 else abs(delta_mu[-1])
        L.append(f"    |delta_mu[5]|:         {dmu_5:.6e}  (steady={dmu_ss:.6e})")

        # 5. delta_hat RMS: steps 2-20 vs steps 50-200
        dh_early = dh[2:20]
        dh_late = dh[50:]
        rms_early = np.sqrt(np.mean(dh_early**2)) if len(dh_early) > 0 else 0
        rms_late = np.sqrt(np.mean(dh_late**2)) if len(dh_late) > 0 else 0
        L.append(f"    delta_hat RMS [2:20]:  {rms_early:.4f}")
        L.append(f"    delta_hat RMS [50+]:   {rms_late:.4f}")

        # 6. Finite check
        for fname in ['sigma', 'delta_bar', 'v_sigma', 'delta_hat']:
            arr = np.array(hist[fname])
            if not np.all(np.isfinite(arr)):
                L.append(f"    ✗ FAIL: {fname} contains NaN/Inf")
                pass_all = False

        # --- PASS/FAIL ---
        stream_pass = True

        # TEST A: delta_hat at step 2 must be < 100 (O(1), not O(1e5))
        if abs(dh_step2) > 100:
            L.append(f"    ✗ FAIL: delta_hat[2] = {dh_step2:.1f} (want < 100)")
            stream_pass = False
        else:
            L.append(f"    ✓ PASS: delta_hat[2] = {dh_step2:.4f} (no explosion)")

        # TEST B: sigma at step 5 must be < 10× steady state
        if len(sigma) > 5 and sigma_ss > 0:
            ratio_5 = sigma[5] / sigma_ss
            if ratio_5 > 10.0:
                L.append(f"    ✗ FAIL: sigma[5]/sigma_ss = {ratio_5:.1f}× (seed bias)")
                stream_pass = False
            else:
                L.append(f"    ✓ PASS: sigma[5]/sigma_ss = {ratio_5:.2f}× (reasonable)")

        # TEST C: delta_hat RMS ratio (early/late) should be < 10×
        if rms_late > 0:
            rms_ratio = rms_early / rms_late
            if rms_ratio > 10.0:
                L.append(f"    ✗ FAIL: early/late delta_hat RMS = {rms_ratio:.1f}× (init contamination)")
                stream_pass = False
            elif rms_ratio > 3.0:
                L.append(f"    ~ WARN: early/late delta_hat RMS = {rms_ratio:.1f}× (mild contamination)")
            else:
                L.append(f"    ✓ PASS: early/late delta_hat RMS = {rms_ratio:.2f}× (clean)")
        elif name != 'lr_constant':
            L.append(f"    ~ SKIP: rms_late=0 (constant stream)")

        if not stream_pass:
            pass_all = False
        L.append(f"    {'✓ STREAM PASS' if stream_pass else '✗ STREAM FAIL'}")

    L.append(f"\n  {'═'*60}")
    L.append(f"  OVERALL: {'✓ ALL PASS' if pass_all else '✗ FAIL'}")
    L.append(f"  {'═'*60}")
    return '\n'.join(L), pass_all

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    streams = make_streams()

    out = []
    out.append("=" * 80)
    out.append("  SIGMA INITIALIZATION BUG SIMULATION")
    out.append("  TelemetryLattice — exact kernel reproduction")
    out.append("  Pass condition: no artificial gradnorm jump without hacks")
    out.append("=" * 80)
    out.append("")
    out.append("  Streams:")
    out.append("    loss:        3.7 ± 0.03 (0.8% CV)")
    out.append("    grad_norm:   5.5e-5 ± 1e-5 (18% CV)")
    out.append("    rho:         0.30 ± 0.06 (20% CV)")
    out.append("    lr_constant: 6e-5 constant")
    out.append("")
    out.append("  Bugs under test:")
    out.append("    BUG 1: sigma_prev=0 at init → delta_hat=(x1-x0)/(0+1e-7) = O(1e5)")
    out.append("    BUG 2: single-sample seed → m2,mu wrong for ~20 steps → sigma inflated")
    out.append("")
    out.append("  FIX 1: Patch sigma_prev → sigma on first post-init update (fixes bug 1 only)")
    out.append("  FIX 2: Welford warmup N=20 → seed EMA from real statistics (fixes both)")
    out.append("")
    out.append(f"  HP: {', '.join(f'{k}={v}' for k,v in HP.items())}")
    out.append(f"  Welford N = {WELFORD_N}")

    # Baseline
    r0 = run_variant(streams, update_BASELINE, "BASELINE")
    rpt0, p0 = analyze(r0, "BASELINE: current init (sigma_prev=0, single-sample seed)", streams)
    out.append(rpt0)

    # Fix 1
    r1 = run_variant(streams, update_FIX1, "FIX 1")
    rpt1, p1 = analyze(r1, "FIX 1: sigma_prev patch (first update seeds sigma_prev=sigma)", streams)
    out.append(rpt1)

    # Fix 2
    r2 = run_variant(streams, update_FIX2, "FIX 2")
    rpt2, p2 = analyze(r2, "FIX 2: Welford warmup N=20 (accumulate then seed EMA)", streams)
    out.append(rpt2)

    # Comparative
    out.append(f"\n{'='*80}")
    out.append("  COMPARATIVE SUMMARY")
    out.append(f"{'='*80}")
    out.append("")

    out.append("  delta_hat[step 2] (want O(1), not O(1e5)):")
    for lbl, res in [("BASELINE", r0), ("FIX 1", r1), ("FIX 2", r2)]:
        vals = {n: res[n]['delta_hat'][1] for n in streams}
        parts = [f"{n}={v:.2e}" for n, v in vals.items()]
        out.append(f"    {lbl:12s}  {', '.join(parts)}")

    out.append("")
    out.append("  sigma[5] / sigma_steady (want ≈ 1.0):")
    for lbl, res in [("BASELINE", r0), ("FIX 1", r1), ("FIX 2", r2)]:
        parts = []
        for n in streams:
            s5 = res[n]['sigma'][5]
            ss = np.mean(np.array(res[n]['sigma'][150:]))
            ratio = s5 / ss if ss > 0 else 0
            parts.append(f"{n}={ratio:.2f}×")
        out.append(f"    {lbl:12s}  {', '.join(parts)}")

    out.append("")
    out.append("  delta_hat RMS [2:20] / RMS [50+] (want ≈ 1.0):")
    for lbl, res in [("BASELINE", r0), ("FIX 1", r1), ("FIX 2", r2)]:
        parts = []
        for n in streams:
            dh = np.array(res[n]['delta_hat'])
            re = np.sqrt(np.mean(dh[2:20]**2)) if len(dh) > 20 else 0
            rl = np.sqrt(np.mean(dh[50:]**2)) if len(dh) > 50 else 1
            ratio = re / rl if rl > 0 else 0
            parts.append(f"{n}={ratio:.2f}×")
        out.append(f"    {lbl:12s}  {', '.join(parts)}")

    out.append("")
    out.append(f"  VERDICTS:")
    out.append(f"    BASELINE:  {'PASS' if p0 else 'FAIL'}")
    out.append(f"    FIX 1:     {'PASS' if p1 else 'FAIL'}")
    out.append(f"    FIX 2:     {'PASS' if p2 else 'FAIL'}")

    full = '\n'.join(out)
    print(full)

    with open('simulations/sigma_init_bug_simulation.txt', 'w') as f:
        f.write(full + '\n')
    print(f"\n[Written to simulations/sigma_init_bug_simulation.txt]")

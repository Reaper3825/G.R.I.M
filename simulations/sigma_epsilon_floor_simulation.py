"""
Sigma Epsilon Floor Simulation — TelemetryLattice
===================================================
Tests two proposed fixes for the epsilon floor bug in sigma computation.

BUG: sigma = sqrt(max(m2 - mu², epsilon)) where epsilon=1e-7
     → For constant/small streams, sigma = sqrt(1e-7) = 3.16e-4
     → sigma/mu = 5.27x for learning_rate (6e-5), 5.7x for grad_norm (5.5e-5)
     → ALL downstream metrics corrupted (delta_hat, delta_bar, v_sigma, k_out, ...)

FIX 1: Scale-aware epsilon — fmaxf(variance, epsilon * mu²)
FIX 2: Step-count warmup gate — sigma = 0 until step_count > warmup_N

PASS CONDITION: No artificial gradnorm jump without hacks.
  Specifically: for a stream with true small noise, delta_hat must reflect
  ACTUAL variation, not be crushed/inflated by a fabricated sigma floor.

Reproduces the EXACT kernel logic from TelemetryLattice_GPU.cu lines 49-155.
"""

import math
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# Hyperparameters (exact match to TelemetryHyperParams defaults + ai_config)
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

# ═══════════════════════════════════════════════════════════════════════════════
# Test streams (pulled from actual training run telemetry_1776118051332766179)
# ═══════════════════════════════════════════════════════════════════════════════
N_STEPS = 500
np.random.seed(42)

def make_streams():
    """4 streams spanning the range of real training signals."""
    streams = {}
    
    # 1. Constant stream: learning_rate = 6e-5 (zero variance)
    streams['lr_constant'] = np.full(N_STEPS, 6e-5)
    
    # 2. Small noisy: grad_norm_mean ~ 5.5e-5 ± 1e-5 (18% CV)
    streams['grad_norm'] = np.abs(5.5e-5 + 1e-5 * np.random.randn(N_STEPS))
    
    # 3. Medium: loss ~ 3.7 with 0.03 noise (0.8% CV)
    streams['loss'] = 3.7 + 0.03 * np.random.randn(N_STEPS)
    
    # 4. Large noisy: rho_final ~ 0.30 ± 0.06 (20% CV)  
    streams['rho'] = np.clip(0.30 + 0.06 * np.random.randn(N_STEPS), 0.01, 0.99)
    
    # 5. Step function: grad_norm jumps from 5e-5 to 2e-4 at step 250
    #    (tests whether fix handles real regime change)
    gn_jump = np.full(N_STEPS, 5e-5)
    gn_jump[250:] = 2e-4
    gn_jump += 5e-6 * np.random.randn(N_STEPS)  # small noise
    streams['grad_norm_jump'] = np.abs(gn_jump)
    
    return streams

# ═══════════════════════════════════════════════════════════════════════════════
# Telemetry state (mirrors TelemetryState struct exactly)
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
    }

def safe_sqrt(x, eps):
    return math.sqrt(max(x, eps))

def safe_sign(x):
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)

# ═══════════════════════════════════════════════════════════════════════════════
# THREE SIGMA VARIANTS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_sigma_CURRENT(s, hp):
    """CURRENT: variance = max(m2 - mu², epsilon)"""
    variance = max(s['m2'] - s['mu'] * s['mu'], hp['epsilon'])
    sigma = safe_sqrt(variance, hp['epsilon'])
    sigma_tilde = sigma / (abs(s['mu']) + hp['epsilon'])
    return sigma, sigma_tilde

def compute_sigma_FIX1(s, hp):
    """FIX 1: Scale-aware epsilon — variance = max(m2 - mu², epsilon * mu²)
    
    For constant stream mu=6e-5:
      scale_eps = 1e-7 * (6e-5)^2 = 3.6e-16
      sigma = sqrt(3.6e-16) = 1.897e-8
      sigma/|mu| = sqrt(epsilon) = 3.16e-4
    
    This is a RELATIVE floor: sigma_tilde is always >= sqrt(epsilon) ≈ 3.16e-4.
    No hidden absolute floors.
    """
    raw_variance = s['m2'] - s['mu'] * s['mu']
    scale_eps = hp['epsilon'] * s['mu'] * s['mu']
    variance = max(raw_variance, scale_eps)
    if variance <= 0.0:
        return 0.0, 0.0
    sigma = math.sqrt(variance)
    sigma_tilde = sigma / (abs(s['mu']) + hp['epsilon'])
    return sigma, sigma_tilde

def compute_sigma_FIX2(s, hp, warmup_N=20):
    """FIX 2: Warmup gate — use raw variance (no epsilon floor on variance itself)
    until step_count > warmup_N. After warmup, use current epsilon."""
    raw_variance = s['m2'] - s['mu'] * s['mu']
    if s['step_count'] < warmup_N:
        # During warmup: no artificial floor on variance. Floor at 0 only.
        variance = max(raw_variance, 0.0)
        sigma = math.sqrt(variance) if variance > 0 else 0.0
    else:
        # After warmup: current behavior (epsilon floor for numerical safety)
        variance = max(raw_variance, hp['epsilon'])
        sigma = safe_sqrt(variance, hp['epsilon'])
    sigma_tilde = sigma / (abs(s['mu']) + hp['epsilon']) if sigma > 0 else 0.0
    return sigma, sigma_tilde

# ═══════════════════════════════════════════════════════════════════════════════
# FULL UPDATE STEP (mirrors kernel Steps 1-8)
# ═══════════════════════════════════════════════════════════════════════════════

def update_step(s, x_t, hp, sigma_fn, **sigma_kwargs):
    """Exact reproduction of updateTelemetryStateKernel."""
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
        return
    
    # STEP 1: Fast magnitude statistics
    beta_mu = hp['beta_mu']
    s['mu'] = beta_mu * s['mu'] + (1.0 - beta_mu) * x_t
    s['m2'] = beta_mu * s['m2'] + (1.0 - beta_mu) * (x_t * x_t)
    
    s['sigma'], s['sigma_tilde'] = sigma_fn(s, hp, **sigma_kwargs)
    
    # STEP 2: Volatility-of-volatility
    sigma_delta = s['sigma'] - s['sigma_prev']
    beta_v = hp['beta_v']
    s['v_sigma'] = beta_v * s['v_sigma'] + (1.0 - beta_v) * (sigma_delta * sigma_delta)
    s['sigma_prev'] = s['sigma']
    
    # STEP 3: Adaptive outlier threshold
    v_sigma_sqrt = safe_sqrt(s['v_sigma'], eps)
    s['k_out'] = hp['k_out0'] * (1.0 + hp['alpha_v'] * v_sigma_sqrt)
    s['c_out'] = s['mu'] + s['k_out'] * s['sigma']
    
    # STEP 4: Normalized slope + direction
    delta_hat = (x_t - s['mu_prev']) / (s['sigma_prev'] + eps)
    beta_d = hp['beta_delta']
    s['delta_bar'] = beta_d * s['delta_bar'] + (1.0 - beta_d) * delta_hat
    s['p'] = beta_d * s['p'] + (1.0 - beta_d) * safe_sign(delta_hat)
    s['mu_prev'] = s['mu']
    
    # STEP 5: Soft outlier gating
    outlier_arg = (x_t - s['c_out']) / (s['sigma'] + eps)
    # Clamp to avoid overflow in exp
    outlier_arg = max(min(outlier_arg, 80.0), -80.0)
    w_out = 1.0 / (1.0 + math.exp(-outlier_arg))
    beta_r = hp['beta_r']
    s['r_out'] = beta_r * s['r_out'] + (1.0 - beta_r) * w_out
    
    # STEP 6: Excess magnitude
    eta = max(0.0, x_t - s['c_out'])
    s['mu_ex'] = beta_mu * s['mu_ex'] + (1.0 - beta_mu) * eta
    
    # STEP 7: Persistence
    beta_run = hp['beta_run']
    s['ell_out'] = beta_run * s['ell_out'] + (1.0 - beta_run) * w_out
    
    # STEP 8: Slow anchor drift
    beta_a = hp['beta_a']
    s['mu_a'] = beta_a * s['mu_a'] + (1.0 - beta_a) * s['mu']
    s['sigma_a'] = beta_a * s['sigma_a'] + (1.0 - beta_a) * s['sigma']
    s['delta_mu'] = s['mu'] - s['mu_a']
    s['delta_sigma'] = s['sigma'] - s['sigma_a']
    
    # STEP 9: Metadata
    s['step_count'] += 1

# ═══════════════════════════════════════════════════════════════════════════════
# RUN ONE VARIANT ON ALL STREAMS
# ═══════════════════════════════════════════════════════════════════════════════

def run_variant(streams, sigma_fn, label, **sigma_kwargs):
    """Returns {stream_name: {field: [values_per_step]}}"""
    results = {}
    fields_to_track = ['sigma', 'sigma_tilde', 'delta_bar', 'v_sigma', 'k_out', 'r_out', 'mu', 'delta_hat_raw']
    
    for name, data in streams.items():
        s = fresh_state()
        history = {f: [] for f in fields_to_track}
        
        for t, x_t in enumerate(data):
            sigma_prev_before = s['sigma_prev']
            mu_prev_before = s['mu_prev']
            
            update_step(s, x_t, HP, sigma_fn, **sigma_kwargs)
            
            # Record delta_hat manually (it's local in kernel, not stored in state)
            if t > 0:
                dh = (x_t - mu_prev_before) / (sigma_prev_before + HP['epsilon'])
            else:
                dh = 0.0
            
            for f in fields_to_track:
                if f == 'delta_hat_raw':
                    history[f].append(dh)
                else:
                    history[f].append(s[f])
        
        results[name] = history
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS: Detect jumps and compute diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_results(results, label):
    """Compute key diagnostics per stream."""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"  {label}")
    lines.append(f"{'='*80}")
    
    pass_all = True
    
    for stream_name, history in results.items():
        sigma = np.array(history['sigma'])
        sigma_tilde = np.array(history['sigma_tilde'])
        delta_bar = np.array(history['delta_bar'])
        delta_hat = np.array(history['delta_hat_raw'])
        v_sigma = np.array(history['v_sigma'])
        mu = np.array(history['mu'])
        
        # Compute actual input statistics for comparison
        true_sigma_ratio = sigma[-1] / (abs(mu[-1]) + HP['epsilon']) if abs(mu[-1]) > 0 else float('inf')
        
        lines.append(f"\n  ── {stream_name} ──")
        lines.append(f"    mu (final):            {mu[-1]:.6e}")
        lines.append(f"    sigma (final):         {sigma[-1]:.6e}")
        lines.append(f"    sigma_tilde (final):   {sigma_tilde[-1]:.6e}")
        lines.append(f"    v_sigma (final):       {v_sigma[-1]:.6e}")
        
        # KEY TEST 1: sigma/mu ratio — should reflect actual variance, not epsilon
        lines.append(f"    sigma/|mu| ratio:      {true_sigma_ratio:.4f}")
        
        # KEY TEST 2: delta_hat magnitude — should be O(1) if sigma tracks real variance
        #   If sigma is inflated, delta_hat is crushed toward 0
        #   If sigma is too small, delta_hat explodes
        dh_post_init = delta_hat[2:]  # skip init steps
        if len(dh_post_init) > 0:
            dh_rms = np.sqrt(np.mean(dh_post_init**2))
            dh_max = np.max(np.abs(dh_post_init))
            lines.append(f"    delta_hat RMS:         {dh_rms:.4f}")
            lines.append(f"    delta_hat |max|:       {dh_max:.4f}")
        
        # KEY TEST 3: Max step-to-step sigma jump (should be smooth, no discontinuities)
        sigma_diffs = np.abs(np.diff(sigma))
        max_sigma_jump = np.max(sigma_diffs) if len(sigma_diffs) > 0 else 0
        max_jump_step = np.argmax(sigma_diffs) + 1 if len(sigma_diffs) > 0 else -1
        lines.append(f"    max sigma jump:        {max_sigma_jump:.6e} at step {max_jump_step}")
        
        # KEY TEST 4: delta_hat max jump (the actual "gradnorm jump" test)
        dh_diffs = np.abs(np.diff(delta_hat[1:]))  # skip step 0
        max_dh_jump = np.max(dh_diffs) if len(dh_diffs) > 0 else 0
        max_dh_step = np.argmax(dh_diffs) + 2 if len(dh_diffs) > 0 else -1
        lines.append(f"    max delta_hat jump:    {max_dh_jump:.4f} at step {max_dh_step}")
        
        # ── PASS/FAIL checks ──
        stream_pass = True
        
        # For constant stream: sigma/mu should be << 1 (not 5.27x)
        if stream_name == 'lr_constant':
            if true_sigma_ratio > 0.1:
                lines.append(f"    ✗ FAIL: sigma/|mu| = {true_sigma_ratio:.2f}x for CONSTANT stream (want < 0.1)")
                stream_pass = False
            else:
                lines.append(f"    ✓ PASS: sigma/|mu| = {true_sigma_ratio:.4f} (properly small for constant stream)")
        
        # For noisy streams: delta_hat RMS should be O(1), not O(1000) or O(0.001)
        if stream_name in ('grad_norm', 'loss', 'rho'):
            if len(dh_post_init) > 0:
                if dh_rms < 0.01:
                    lines.append(f"    ✗ FAIL: delta_hat RMS = {dh_rms:.6f} (crushed — sigma too large)")
                    stream_pass = False
                elif dh_rms > 100:
                    lines.append(f"    ✗ FAIL: delta_hat RMS = {dh_rms:.1f} (exploding — sigma too small)")
                    stream_pass = False
                else:
                    lines.append(f"    ✓ PASS: delta_hat RMS = {dh_rms:.4f} (O(1) — sigma properly scaled)")
        
        # For jump stream: sigma should respond to the jump but not have artificial jumps BEFORE step 250
        if stream_name == 'grad_norm_jump':
            pre_jump_sigma_diffs = np.abs(np.diff(sigma[:245]))
            max_pre = np.max(pre_jump_sigma_diffs) if len(pre_jump_sigma_diffs) > 0 else 0
            post_jump_response = sigma[300] - sigma[245]
            lines.append(f"    pre-jump max sigma Δ:  {max_pre:.6e}")
            lines.append(f"    post-jump sigma rise:  {post_jump_response:.6e}")
            if max_pre > 1e-4 and stream_name == 'grad_norm_jump':
                lines.append(f"    ✗ FAIL: artificial sigma jumps BEFORE real regime change")
                stream_pass = False
            elif post_jump_response < 1e-6:
                lines.append(f"    ✗ FAIL: sigma did not respond to real regime change")
                stream_pass = False
            else:
                lines.append(f"    ✓ PASS: sigma smooth pre-jump, responds to real change")
        
        # For ALL streams: no NaN/Inf
        for field_name in ['sigma', 'delta_bar', 'v_sigma']:
            arr = np.array(history[field_name])
            if not np.all(np.isfinite(arr)):
                lines.append(f"    ✗ FAIL: {field_name} contains NaN/Inf")
                stream_pass = False
        
        if not stream_pass:
            pass_all = False
        
        lines.append(f"    {'✓ STREAM PASS' if stream_pass else '✗ STREAM FAIL'}")
    
    lines.append(f"\n  {'═'*60}")
    lines.append(f"  OVERALL: {'✓ ALL PASS' if pass_all else '✗ FAIL — see above'}")
    lines.append(f"  {'═'*60}")
    
    return '\n'.join(lines), pass_all


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN: Run all three variants
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    streams = make_streams()
    
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("  SIGMA EPSILON FLOOR SIMULATION")
    output_lines.append("  TelemetryLattice — exact kernel reproduction")
    output_lines.append("  Pass condition: no gradnorm jump without hacks")
    output_lines.append("=" * 80)
    output_lines.append("")
    output_lines.append("  Test streams:")
    output_lines.append("    1. lr_constant:     6e-5 constant (zero variance)")
    output_lines.append("    2. grad_norm:       5.5e-5 ± 1e-5 (18% CV)")
    output_lines.append("    3. loss:            3.7 ± 0.03 (0.8% CV)")
    output_lines.append("    4. rho:             0.30 ± 0.06 (20% CV)")  
    output_lines.append("    5. grad_norm_jump:  5e-5 → 2e-4 step change at t=250")
    output_lines.append("")
    output_lines.append("  Hyperparameters: " + ", ".join(f"{k}={v}" for k, v in HP.items()))
    output_lines.append("")
    
    # ── BASELINE (current broken behavior) ──
    r_current = run_variant(streams, compute_sigma_CURRENT, "CURRENT")
    report_current, pass_current = analyze_results(r_current, "BASELINE: current epsilon floor (epsilon=1e-7)")
    output_lines.append(report_current)
    
    # ── FIX 1: Scale-aware epsilon ──
    r_fix1 = run_variant(streams, compute_sigma_FIX1, "FIX 1")
    report_fix1, pass_fix1 = analyze_results(r_fix1, "FIX 1: Scale-aware epsilon — max(variance, epsilon * mu²)")
    output_lines.append(report_fix1)
    
    # ── FIX 2: Warmup gate ──
    r_fix2 = run_variant(streams, compute_sigma_FIX2, "FIX 2", warmup_N=20)
    report_fix2, pass_fix2 = analyze_results(r_fix2, "FIX 2: Warmup gate (N=20) — no epsilon floor on variance during warmup")
    output_lines.append(report_fix2)
    
    # ── COMPARATIVE SUMMARY ──
    output_lines.append(f"\n{'='*80}")
    output_lines.append(f"  COMPARATIVE SUMMARY")
    output_lines.append(f"{'='*80}")
    output_lines.append("")
    
    # Side-by-side sigma/mu for the constant stream
    output_lines.append("  sigma/|mu| ratio for lr_constant (want ≈ 0):")
    for lbl, res in [("CURRENT", r_current), ("FIX 1", r_fix1), ("FIX 2", r_fix2)]:
        sigma_final = res['lr_constant']['sigma'][-1]
        mu_final = res['lr_constant']['mu'][-1]
        ratio = sigma_final / (abs(mu_final) + HP['epsilon'])
        output_lines.append(f"    {lbl:12s}  sigma={sigma_final:.6e}  ratio={ratio:.4f}")
    
    output_lines.append("")
    output_lines.append("  delta_hat RMS for grad_norm (want O(1)):")
    for lbl, res in [("CURRENT", r_current), ("FIX 1", r_fix1), ("FIX 2", r_fix2)]:
        dh = np.array(res['grad_norm']['delta_hat_raw'][2:])
        rms = np.sqrt(np.mean(dh**2))
        output_lines.append(f"    {lbl:12s}  delta_hat_rms={rms:.4f}")
    
    output_lines.append("")
    output_lines.append("  delta_hat RMS for loss (want O(1)):")
    for lbl, res in [("CURRENT", r_current), ("FIX 1", r_fix1), ("FIX 2", r_fix2)]:
        dh = np.array(res['loss']['delta_hat_raw'][2:])
        rms = np.sqrt(np.mean(dh**2))
        output_lines.append(f"    {lbl:12s}  delta_hat_rms={rms:.4f}")
    
    output_lines.append("")
    output_lines.append("  grad_norm_jump: sigma at t=245 (pre) vs t=300 (post):")
    for lbl, res in [("CURRENT", r_current), ("FIX 1", r_fix1), ("FIX 2", r_fix2)]:
        s_pre = res['grad_norm_jump']['sigma'][245]
        s_post = res['grad_norm_jump']['sigma'][300]
        output_lines.append(f"    {lbl:12s}  pre={s_pre:.6e}  post={s_post:.6e}  rise={s_post-s_pre:.6e}")
    
    output_lines.append("")
    output_lines.append(f"  VERDICTS:")
    output_lines.append(f"    CURRENT:  {'PASS' if pass_current else 'FAIL'}")
    output_lines.append(f"    FIX 1:    {'PASS' if pass_fix1 else 'FAIL'}")
    output_lines.append(f"    FIX 2:    {'PASS' if pass_fix2 else 'FAIL'}")
    
    full_output = '\n'.join(output_lines)
    print(full_output)
    
    # Write report
    with open('simulations/sigma_epsilon_floor_simulation.txt', 'w') as f:
        f.write(full_output + '\n')
    print(f"\n[Written to simulations/sigma_epsilon_floor_simulation.txt]")

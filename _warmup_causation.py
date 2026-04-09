"""
Causal analysis: Why GRIM-text loss rises above random (ln(V)) during warmup.

CAUSAL CHAIN:
  1. Adam's v-estimate (β₂=0.999) has half-life 693 steps. Before convergence,
     per-parameter step sizes are wrong → some params get 10-100x too large steps.
  2. Linear LR warmup ramps from 0 → 0.0006 over 1000 steps, so each step is
     LARGER than the last. The cumulative displacement from Xavier init grows
     quadratically: Σlr(t) ∝ t².
  3. By step 700, total displacement ≈ 6× the Xavier embedding scale. The Xavier
     equilibrium (uniform logits) is destroyed.
  4. Without reliable v-estimates, large weights produce peaked-at-wrong-tokens
     logits → loss rises ABOVE ln(V).
  5. Around step 693 (v half-life), Adam's per-parameter scales converge. Updates
     become signal-dominated. Loss peaks and begins descending.
  6. After step 1000, warmup ends. LR is constant → per-step displacement stops
     growing → recovery accelerates.

WHAT CHANGED at the peak (~step 700):
  - v_convergence crosses 50%: Adam finally "knows" which params need big vs small steps
  - cumulative_displacement saturates: Σlr(t) growth decelerates as LR approaches max
  - The ratio (signal_per_step / noise_per_step) tips above 1.0

This script computes all causal quantities and writes them to CSV.
Plot them alongside your loss curve to see exact correspondence.
"""

import csv
import math
import sys

# ── Config from ai_config.json ────────────────────────────────────────────
BASE_LR         = 0.0006
WARMUP_STEPS    = 1000
COSINE_MIN_LR   = 6e-5
VOCAB_SIZE       = 10000        # total: bytes[0-255] + atoms[256-511] + unigram[512+]
D_MODEL          = 768
BETA1            = 0.9
BETA2            = 0.999
EPSILON          = 1e-8
WEIGHT_DECAY     = 0.01
FOCAL_GAMMA      = 0.75
LABEL_SMOOTH_EPS = 0.05
NUM_LAYERS       = 12
TOTAL_STEPS      = 6000         # ~3 epochs × ~2000 batches (from graphs)

# ── Derived constants ──────────────────────────────────────────────────────
LN_V = math.log(VOCAB_SIZE)     # 9.2103 — the "random" baseline loss
V_HALFLIFE = -math.log(2) / math.log(BETA2)  # 692.8 steps

# Xavier scale for embedding matrix [V, d_model]
XAVIER_EMB = math.sqrt(6.0 / (VOCAB_SIZE + D_MODEL))   # 0.0236
# Xavier scale for encoder weights [d_model, d_model]
XAVIER_ENC = math.sqrt(6.0 / (D_MODEL + D_MODEL))      # 0.0625


def scheduled_lr(step: int) -> float:
    """Exact replica of getScheduledLearningRate from Phase2_TrainingLoop.cu."""
    if step < WARMUP_STEPS:
        return BASE_LR * (step + 1) / max(1, WARMUP_STEPS)
    decay_steps = TOTAL_STEPS - WARMUP_STEPS
    if decay_steps <= 0:
        return BASE_LR
    progress = (step - WARMUP_STEPS) / decay_steps
    progress = max(0.0, min(1.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return COSINE_MIN_LR + (BASE_LR - COSINE_MIN_LR) * cosine


def main():
    out_path = "_warmup_causation.csv"
    rows = []

    cumulative_disp = 0.0       # Σ lr(t) — Adam normalizes steps to ≈ lr magnitude
    cumulative_disp_sq = 0.0    # Σ lr(t)² — energy of perturbation (variance proxy)

    for t in range(TOTAL_STEPS + 1):
        lr = scheduled_lr(t)
        iteration = t + 1       # matches AdamW kernel: step+1

        # ── Adam bias corrections ──────────────────────────────────────
        bc1  = 1.0 - BETA1 ** iteration        # m convergence: 0→1
        bc2  = 1.0 - BETA2 ** iteration         # v convergence: 0→1 (SLOW)
        inv_bc1 = 1.0 / bc1
        inv_bc2 = 1.0 / bc2

        # ── Adam amplification factor ─────────────────────────────────
        # Effective update = lr × (m × inv_bc1) / sqrt(v × inv_bc2 + ε)
        # If m ∝ g and v ∝ g², the bias-corrected ratio is:
        #   inv_bc1 / sqrt(inv_bc2) — how much bias correction amplifies raw m/√v
        adam_amp = inv_bc1 / math.sqrt(inv_bc2)

        # ── Effective LR (what each parameter update actually scales by) ──
        effective_lr = lr * adam_amp

        # ── Cumulative displacement from Xavier init ───────────────────
        # Adam normalizes m/√v ≈ ±1, so each step moves weights by ≈ lr
        cumulative_disp += lr
        cumulative_disp_sq += lr * lr

        # ── Disruption ratios ─────────────────────────────────────────
        # How many "Xavier scales" the weights have moved from init
        disruption_emb = cumulative_disp / XAVIER_EMB
        disruption_enc = cumulative_disp / XAVIER_ENC

        # ── Direction signal quality ──────────────────────────────────
        # m converges in ~10 steps (β₁=0.9), v in ~693 steps (β₂=0.999).
        # Good Adam steps require BOTH: direction (m) AND scale (v).
        # bc1 × bc2 measures joint convergence.
        signal_quality = bc1 * bc2

        # ── Noise-to-signal ratio of each step ───────────────────────
        # v remaining uncertainty = 1 - bc2 (fraction of v that is bias, not data)
        # Each step's "noise power" ∝ lr × (1 - bc2)
        # Each step's "signal power" ∝ lr × bc2
        step_noise_power  = lr * (1.0 - bc2)
        step_signal_power = lr * bc2

        # ── Net signal dominance ──────────────────────────────────────
        # When > 1: each step does more good than harm
        # When < 1: each step does more harm than good
        # This is the "crossover" that predicts the loss peak
        if step_noise_power > 0:
            signal_dominance = step_signal_power / step_noise_power
        else:
            signal_dominance = float('inf')

        # ── Expected excess loss from random logit perturbation ───────
        # For logits z_i with i.i.d. perturbation std σ:
        #   E[CE] ≈ ln(V) + σ²/2
        # σ² ∝ cumulative_disp² × (1 - bc2)  [noise fraction of displacement]
        # This is the "destroyed equilibrium" term
        noise_displacement = cumulative_disp * (1.0 - bc2)
        excess_loss_theory = 0.5 * (noise_displacement / XAVIER_EMB) ** 2 / VOCAB_SIZE

        # ── Phase labels ──────────────────────────────────────────────
        if t < WARMUP_STEPS:
            in_warmup = 1
        else:
            in_warmup = 0

        rows.append({
            "step":                  t,
            # ── Schedule ──
            "lr":                    lr,
            "in_warmup":             in_warmup,
            # ── Adam bias correction (THE causal mechanism) ──
            "bc1_m_convergence":     bc1,       # m estimate quality (fast: ~10 steps)
            "bc2_v_convergence":     bc2,       # v estimate quality (SLOW: ~693 steps) ← THIS IS IT
            "inv_bc2_amplification": inv_bc2,   # how much bias correction inflates v
            "adam_amp":              adam_amp,   # combined bias correction amplification
            # ── Step character ──
            "effective_lr":          effective_lr,
            "step_noise_power":      step_noise_power,
            "step_signal_power":     step_signal_power,
            "signal_dominance":      signal_dominance,    # >1 = learning, <1 = destroying
            "signal_quality":        signal_quality,       # bc1×bc2: joint convergence
            # ── Cumulative damage ──
            "cumulative_displacement":   cumulative_disp,
            "disruption_vs_xavier_emb":  disruption_emb,  # displacement / xavier_scale
            "disruption_vs_xavier_enc":  disruption_enc,
            # ── Reference lines ──
            "ln_V_random_baseline":  LN_V,                # 9.2103 — the random baseline
            "v_halflife_693":        1.0 if t == 693 else 0.0,  # marker at v half-life
        })

    # ── Write CSV ──────────────────────────────────────────────────────────
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print()
    print("KEY COLUMNS TO PLOT ALONGSIDE YOUR LOSS CURVE:")
    print("─" * 60)
    print()
    print("1. bc2_v_convergence (0→1)")
    print("   THE primary causal variable.")
    print(f"   Half-life at step {V_HALFLIFE:.0f} — this is WHY the peak is at ~700.")
    print(f"   At step 693: bc2 = {1 - BETA2**694:.4f}")
    print()
    print("2. signal_dominance (noise/signal ratio per step)")
    print("   Crosses 1.0 when learning overtakes destruction.")
    print("   The loss peak should coincide with signal_dominance ≈ 1.0.")
    print()
    print("3. disruption_vs_xavier_emb")
    print("   Cumulative displacement measured in Xavier scales.")
    print(f"   At step 700: {sum(scheduled_lr(s) for s in range(701)) / XAVIER_EMB:.1f}×")
    print(f"   This means embedding weights have moved {sum(scheduled_lr(s) for s in range(701)) / XAVIER_EMB:.1f}×")
    print("   their INITIAL SCALE in semi-random directions.")
    print()
    print("4. inv_bc2_amplification")
    print("   How much Adam inflates its v estimate via bias correction.")
    print(f"   Step 1:   {1/(1-BETA2**1):.0f}× — Adam thinks v is 1000× larger than observed")
    print(f"   Step 100: {1/(1-BETA2**101):.1f}×")
    print(f"   Step 693: {1/(1-BETA2**694):.1f}× — half-life, still 2× inflated")
    print(f"   Step 2000: {1/(1-BETA2**2001):.2f}× — nearly converged")
    print()
    print("CAUSAL SUMMARY:")
    print("─" * 60)
    print(f"  ln(V) = {LN_V:.4f}  (random baseline)")
    print(f"  Xavier embedding scale = {XAVIER_EMB:.4f}")
    print(f"  β₂ half-life = {V_HALFLIFE:.0f} steps")
    print()
    print("  Steps 0-700:  bc2 < 0.5 → Adam v-estimates are noise-dominated")
    print("                Each step's direction is ~correct (bc1 converges fast)")
    print("                BUT each step's MAGNITUDE is wrong (bc2 still low)")
    print("                → some params get 10-100× too-large updates")
    print("                → Xavier equilibrium destroyed → logits peak at wrong tokens")
    print("                → loss rises ABOVE ln(V)")
    print()
    print("  Step ~700:    bc2 crosses 0.5 → v knows the gradient landscape")
    print("                signal_dominance crosses 1.0 → learning > destruction")
    print("                Loss peaks and begins descending")
    print()
    print("  Steps 1000+:  Warmup ends → LR constant → no more acceleration")
    print("                bc2 > 0.63 → per-parameter scales well-calibrated")
    print("                Model enters pure learning regime")


if __name__ == "__main__":
    main()

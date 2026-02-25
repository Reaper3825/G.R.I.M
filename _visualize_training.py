"""
GRIM-text Training Run Visualization
Session: 17718958646177576
Generates 6 diagnostic charts from extracted CSV data.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

BASE = r"d:\G.R.I.M"

def load_csv(name, dtype=float):
    path = os.path.join(BASE, name)
    data = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append([dtype(x) for x in line.split(',')])
    return np.array(data)

# ═══════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════
loss_data = load_csv('_viz_loss.csv')           # batch, loss, valid_tokens
grad_data = load_csv('_viz_grad.csv')           # batch, emb, attn, ffn, rmsnorm, sb
collapse_data = load_csv('_viz_collapse.csv')   # batch, token, count, total, pct
logit_data = load_csv('_viz_logitsignal.csv')   # batch, top2_margin, top1_frac, unique_argmax
rho_data = load_csv('_viz_rho.csv')             # batch, emb_rho, emb_hrms, L0_rho, L0_hrms, L11_rho, L11_hrms

# Running averages for smoothing
def smooth(y, window=50):
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='valid')

def smooth_x(x, window=50):
    if len(x) < window:
        return x
    return x[window//2 : window//2 + len(x) - window + 1]

# ═══════════════════════════════════════════════════════════════
# CHART 1: Training Loss
# The fundamental metric - is the model learning?
# ═══════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(14, 5))
batches = loss_data[:, 0]
losses = loss_data[:, 1]

ax1.plot(batches, losses, alpha=0.15, color='steelblue', linewidth=0.5, label='Raw loss')
ax1.plot(smooth_x(batches), smooth(losses), color='steelblue', linewidth=2, label='Smoothed (50-batch)')
ax1.axhline(y=np.log(2533), color='red', linestyle='--', alpha=0.7, label=f'Random baseline ln(2533)={np.log(2533):.2f}')
ax1.axhline(y=3.85, color='orange', linestyle='--', alpha=0.7, label='Plateau region ~3.85')

ax1.set_xlabel('Batch', fontsize=12)
ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
ax1.set_title('CHART 1: Training Loss — Where Does Learning Stall?', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(3.0, 8.5)

# Annotate phases
ax1.annotate('Rapid\nlearning', xy=(200, 5.5), fontsize=10, ha='center', color='green',
            fontweight='bold')
ax1.annotate('Plateau\n(no progress)', xy=(1500, 4.1), fontsize=10, ha='center', color='red',
            fontweight='bold')

fig1.tight_layout()
fig1.savefig(os.path.join(BASE, 'chart1_loss.png'), dpi=150)
print("Saved chart1_loss.png")

# ═══════════════════════════════════════════════════════════════
# CHART 2: Gradient Components Over Time
# Which parts of the model are still learning?
# ═══════════════════════════════════════════════════════════════
fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

gb = grad_data[:, 0]
emb_g = grad_data[:, 1]
attn_g = grad_data[:, 2]
ffn_g = grad_data[:, 3]
rmsn_g = grad_data[:, 4]
sb_g = grad_data[:, 5]

# Top: All components (log scale)
ax2a.semilogy(smooth_x(gb), smooth(emb_g), linewidth=2, label='emb/lm_head (tied)', color='#e74c3c')
ax2a.semilogy(smooth_x(gb), smooth(ffn_g), linewidth=2, label='FFN', color='#3498db')
ax2a.semilogy(smooth_x(gb), smooth(rmsn_g), linewidth=2, label='RMSNorm', color='#2ecc71')
ax2a.semilogy(smooth_x(gb), smooth(attn_g), linewidth=2, label='Attention', color='#9b59b6')
ax2a.semilogy(smooth_x(gb), smooth(sb_g), linewidth=2, label='ScratchBlock', color='#f39c12')

ax2a.set_ylabel('Gradient RMS (log scale)', fontsize=12)
ax2a.set_title('CHART 2a: Gradient Components — Who Is Learning?', fontsize=14, fontweight='bold')
ax2a.legend(fontsize=9, loc='upper right')
ax2a.grid(True, alpha=0.3)

# Bottom: Ratio of emb to encoder (the imbalance)
enc_total = np.sqrt(attn_g**2 + ffn_g**2 + rmsn_g**2 + sb_g**2)
ratio = emb_g / (enc_total + 1e-12)
ax2b.plot(smooth_x(gb), smooth(ratio), linewidth=2, color='#e74c3c')
ax2b.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Balanced (ratio=1)')
ax2b.fill_between(smooth_x(gb), 0, smooth(ratio), alpha=0.1, color='#e74c3c')

ax2b.set_xlabel('Batch', fontsize=12)
ax2b.set_ylabel('emb_grad / encoder_grad', fontsize=12)
ax2b.set_title('CHART 2b: Gradient Imbalance — Embedding vs Encoder', fontsize=14, fontweight='bold')
ax2b.legend(fontsize=10)
ax2b.grid(True, alpha=0.3)
ax2b.set_ylim(0, max(10, np.percentile(ratio, 99)))

fig2.tight_layout()
fig2.savefig(os.path.join(BASE, 'chart2_gradients.png'), dpi=150)
print("Saved chart2_gradients.png")

# ═══════════════════════════════════════════════════════════════
# CHART 3: Token 36 Collapse
# Is the model collapsing to predict one token?
# ═══════════════════════════════════════════════════════════════
fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

cb = collapse_data[:, 0]
cpct = collapse_data[:, 4]
ctok = collapse_data[:, 1]

# Top: Dominant token percentage over time
ax3a.scatter(cb, cpct, s=8, alpha=0.4, color='#e74c3c', zorder=2)
if len(cb) > 20:
    ax3a.plot(smooth_x(cb, 20), smooth(cpct, 20), linewidth=2.5, color='darkred', label='Smoothed (20-sample)', zorder=3)
ax3a.axhline(y=100/2533*100, color='green', linestyle='--', alpha=0.7, label=f'Uniform baseline ({100/2533:.1f}%)')
ax3a.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='Concern threshold (10%)')

ax3a.set_ylabel('Dominant Token %', fontsize=12)
ax3a.set_title('CHART 3a: Token Collapse — One Token Dominates Predictions', fontsize=14, fontweight='bold')
ax3a.legend(fontsize=10)
ax3a.grid(True, alpha=0.3)
ax3a.set_ylim(0, 65)

# Bottom: Which token is dominant
ax3b.scatter(cb, ctok, s=8, alpha=0.5, color='#3498db')
ax3b.set_xlabel('Batch', fontsize=12)
ax3b.set_ylabel('Dominant Token ID', fontsize=12)
ax3b.set_title('CHART 3b: Which Token Is Dominant?', fontsize=14, fontweight='bold')
ax3b.grid(True, alpha=0.3)

# Annotate tok36
tok36_mask = ctok == 36
if np.any(tok36_mask):
    ax3b.scatter(cb[tok36_mask], ctok[tok36_mask], s=15, alpha=0.7, color='red', label='Token 36', zorder=3)
    ax3b.legend(fontsize=10)

fig3.tight_layout()
fig3.savefig(os.path.join(BASE, 'chart3_collapse.png'), dpi=150)
print("Saved chart3_collapse.png")

# ═══════════════════════════════════════════════════════════════
# CHART 4: Logit Quality — Specialization & Diversity
# Is the model developing distinct predictions?
# ═══════════════════════════════════════════════════════════════
fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

lb = logit_data[:, 0]
margin = logit_data[:, 1]
unique = logit_data[:, 3]

ax4a.plot(lb, margin, alpha=0.15, color='#3498db', linewidth=0.5)
ax4a.plot(smooth_x(lb), smooth(margin), linewidth=2, color='#3498db', label='top2 margin (smoothed)')
ax4a.set_ylabel('Top-2 Logit Margin', fontsize=12)
ax4a.set_title('CHART 4a: Prediction Confidence — How Sure Is the Model?', fontsize=14, fontweight='bold')
ax4a.legend(fontsize=10)
ax4a.grid(True, alpha=0.3)
ax4a.annotate('Higher = more confident\n(not always better)', xy=(2000, 1.1), fontsize=9, 
             ha='center', color='gray')

ax4b.plot(lb, unique, alpha=0.15, color='#2ecc71', linewidth=0.5)
ax4b.plot(smooth_x(lb), smooth(unique), linewidth=2, color='#2ecc71', label='unique argmax tokens (smoothed)')
ax4b.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Batch 1 baseline (random)')
ax4b.set_xlabel('Batch', fontsize=12)
ax4b.set_ylabel('Unique Argmax Tokens', fontsize=12)
ax4b.set_title('CHART 4b: Vocabulary Diversity — How Many Tokens Does Model Use?', fontsize=14, fontweight='bold')
ax4b.legend(fontsize=10)
ax4b.grid(True, alpha=0.3)
ax4b.set_ylim(0, 55)
ax4b.annotate('Dropped from 50 → ~25\nModel uses half the vocab', xy=(1500, 35), fontsize=9,
             ha='center', color='red', fontweight='bold')

fig4.tight_layout()
fig4.savefig(os.path.join(BASE, 'chart4_logit_quality.png'), dpi=150)
print("Saved chart4_logit_quality.png")

# ═══════════════════════════════════════════════════════════════
# CHART 5: Directional Similarity (ρ) — The Core Problem
# Are hidden states collapsing to the same direction?
# ═══════════════════════════════════════════════════════════════
fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

rb = rho_data[:, 0]
emb_rho = rho_data[:, 1]
l0_rho = rho_data[:, 3]
l11_rho = rho_data[:, 5]
emb_hrms = rho_data[:, 2]
l0_hrms = rho_data[:, 4]
l11_hrms = rho_data[:, 6]

# Top: Directional similarity per layer
ax5a.plot(smooth_x(rb, 30), smooth(emb_rho, 30), linewidth=2, label='ρ(emb) — embedding', color='#3498db')
ax5a.plot(smooth_x(rb, 30), smooth(l0_rho, 30), linewidth=2, label='ρ(L0) — first encoder', color='#e74c3c')
ax5a.plot(smooth_x(rb, 30), smooth(l11_rho, 30), linewidth=2, label='ρ(L11) — last encoder', color='#e67e22')

ax5a.set_ylabel('ρ = avg|cos(h_i, h_j)|', fontsize=12)
ax5a.set_title('CHART 5a: Directional Similarity ρ — Are Hidden States Aligning?', fontsize=14, fontweight='bold')
ax5a.legend(fontsize=10)
ax5a.grid(True, alpha=0.3)
ax5a.set_ylim(0, 0.6)
ax5a.annotate('ρ → 1.0 = collapse\nρ → 0.0 = diverse', xy=(200, 0.50), fontsize=9,
             ha='center', color='gray', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# Bottom: h_rms per layer
ax5b.plot(smooth_x(rb, 30), smooth(emb_hrms, 30), linewidth=2, label='h_rms(emb)', color='#3498db')
ax5b.plot(smooth_x(rb, 30), smooth(l0_hrms, 30), linewidth=2, label='h_rms(L0)', color='#e74c3c')
ax5b.plot(smooth_x(rb, 30), smooth(l11_hrms, 30), linewidth=2, label='h_rms(L11)', color='#e67e22')

ax5b.set_xlabel('Batch', fontsize=12)
ax5b.set_ylabel('h_rms (activation scale)', fontsize=12)
ax5b.set_title('CHART 5b: Activation Magnitude Per Layer', fontsize=14, fontweight='bold')
ax5b.legend(fontsize=10)
ax5b.grid(True, alpha=0.3)
ax5b.annotate('Normal pre-norm growth\n(final RMSNorm rescales)', xy=(2000, 5), fontsize=9,
             ha='center', color='gray')

fig5.tight_layout()
fig5.savefig(os.path.join(BASE, 'chart5_rho_direction.png'), dpi=150)
print("Saved chart5_rho_direction.png")

# ═══════════════════════════════════════════════════════════════
# CHART 6: Combined Dashboard — The Big Picture
# Everything on one page to see correlations
# ═══════════════════════════════════════════════════════════════
fig6 = plt.figure(figsize=(18, 14))
gs = GridSpec(3, 2, figure=fig6, hspace=0.35, wspace=0.25)

# Loss
ax6a = fig6.add_subplot(gs[0, 0])
ax6a.plot(smooth_x(batches), smooth(losses), linewidth=2, color='steelblue')
ax6a.axhline(y=np.log(2533), color='red', linestyle='--', alpha=0.5, linewidth=1)
ax6a.set_title('Loss', fontsize=12, fontweight='bold')
ax6a.set_ylabel('CE Loss')
ax6a.grid(True, alpha=0.3)
ax6a.set_ylim(3.0, 8.5)

# Token collapse
ax6b = fig6.add_subplot(gs[0, 1])
if len(cb) > 20:
    ax6b.plot(smooth_x(cb, 20), smooth(cpct, 20), linewidth=2, color='#e74c3c')
ax6b.axhline(y=10, color='orange', linestyle='--', alpha=0.5)
ax6b.set_title('Tok36 Dominance %', fontsize=12, fontweight='bold')
ax6b.set_ylabel('% argmax')
ax6b.grid(True, alpha=0.3)

# Gradient components
ax6c = fig6.add_subplot(gs[1, 0])
ax6c.semilogy(smooth_x(gb), smooth(emb_g), linewidth=2, label='emb', color='#e74c3c')
ax6c.semilogy(smooth_x(gb), smooth(ffn_g), linewidth=2, label='FFN', color='#3498db')
ax6c.semilogy(smooth_x(gb), smooth(sb_g), linewidth=2, label='SB', color='#f39c12')
ax6c.set_title('Gradient RMS by Component', fontsize=12, fontweight='bold')
ax6c.set_ylabel('Grad RMS (log)')
ax6c.legend(fontsize=8)
ax6c.grid(True, alpha=0.3)

# Unique argmax
ax6d = fig6.add_subplot(gs[1, 1])
ax6d.plot(smooth_x(lb), smooth(unique), linewidth=2, color='#2ecc71')
ax6d.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
ax6d.set_title('Unique Argmax Tokens', fontsize=12, fontweight='bold')
ax6d.set_ylabel('Count')
ax6d.grid(True, alpha=0.3)
ax6d.set_ylim(0, 55)

# Rho per layer
ax6e = fig6.add_subplot(gs[2, 0])
ax6e.plot(smooth_x(rb, 30), smooth(l0_rho, 30), linewidth=2, label='ρ(L0)', color='#e74c3c')
ax6e.plot(smooth_x(rb, 30), smooth(l11_rho, 30), linewidth=2, label='ρ(L11)', color='#e67e22')
ax6e.plot(smooth_x(rb, 30), smooth(emb_rho, 30), linewidth=2, label='ρ(emb)', color='#3498db')
ax6e.set_title('Directional Similarity ρ', fontsize=12, fontweight='bold')
ax6e.set_ylabel('ρ')
ax6e.set_xlabel('Batch')
ax6e.legend(fontsize=8)
ax6e.grid(True, alpha=0.3)

# h_rms per layer
ax6f = fig6.add_subplot(gs[2, 1])
ax6f.plot(smooth_x(rb, 30), smooth(l11_hrms, 30), linewidth=2, label='h_rms(L11)', color='#e67e22')
ax6f.plot(smooth_x(rb, 30), smooth(l0_hrms, 30), linewidth=2, label='h_rms(L0)', color='#e74c3c')
ax6f.set_title('Activation Scale (h_rms)', fontsize=12, fontweight='bold')
ax6f.set_ylabel('h_rms')
ax6f.set_xlabel('Batch')
ax6f.legend(fontsize=8)
ax6f.grid(True, alpha=0.3)

fig6.suptitle('GRIM-text Training Dashboard — Session 17718958646177576\n'
              '2706 batches, Epoch 1/3, vocab=2533, d_model=768, 12 layers',
              fontsize=14, fontweight='bold', y=0.99)

fig6.savefig(os.path.join(BASE, 'chart6_dashboard.png'), dpi=150, bbox_inches='tight')
print("Saved chart6_dashboard.png")

print("\n✓ All 6 charts saved to", BASE)
print("  chart1_loss.png         — Loss curve")
print("  chart2_gradients.png    — Gradient components + imbalance")
print("  chart3_collapse.png     — Token 36 collapse")
print("  chart4_logit_quality.png — Confidence & diversity")
print("  chart5_rho_direction.png — Directional similarity (ρ) + h_rms")
print("  chart6_dashboard.png    — Combined overview")

"""
Readout Architecture Simulation
================================
HONEST test of the claim: "A non-dot-product readout (MLP head, mixture-of-
softmaxes, factorized head) breaks the Yang et al. softmax-rank bottleneck
that mathematically forces h into the column span of W's most-frequent-token
directions."

Specifically, we are testing whether changing the readout architecture
DIRECTIONALLY changes the gradient on h (and therefore prevents the
hidden-state-aligns-with-unigram-direction collapse), or whether it only
changes magnitude / fitting capacity.

Setup (toy, but faithful to the underlying mechanic):
  - Vocab V = 200 tokens with Zipfian unigram p(y) ~ 1/rank^1.0  (NL-like)
  - d_model = 32
  - Inputs: x ~ Uniform[V]
  - True conditional:
        p(y | x) = alpha * delta(y == NEXT(x))  +  (1 - alpha) * p_unigram(y)
    with alpha = 0.7 and NEXT(x) a fixed random permutation.
    => There is REAL conditional signal a working encoder can learn,
       PLUS a unigram component that creates the collapse pressure.
  - "Encoder": h(x) = tanh(W_h @ E[x]); W_h learned, E learned.
  - 4 readouts compared:
        A) TIED   : logits = h @ E^T
        B) UNTIED : logits = h @ W^T          (W independent of E)
        C) MLP    : logits = W2 @ ReLU(W1 h + b1) + b2   (NON-dot-product)
        D) MoS    : logits = logsumexp_k( log pi_k(h) + W_k h )
                    K = 4 mixture components, all learned.
  - Loss: cross-entropy on (x, y).

Key metrics, measured every EVAL_EVERY steps:
  M1) loss                 -- compared to two baselines:
                              loss_unigram = H(p_unigram | true)
                              loss_oracle  = H(p_true | p_true)
  M2) cos_unigram_dir      = cos( mean_x h(x) , e_unigram_dir )
                              where e_unigram_dir = Σ_y p_unigram(y) * E[y]
                              normalized.
                              This is the EXACT directional collapse metric.
                              Random baseline ≈ 1/sqrt(d_model) = 0.177.
  M3) eff_rank             = exp(entropy of normalized singular values of H)
                              where H stacks h(x) for x in a held-out set.
                              Random baseline ≈ d_model = 32.
                              Collapsed encoder => eff_rank → 1.
  M4) cond_KL              = mean_x KL( p_model(.|x) || p_true(.|x) )
                              cond_KL ~ 0  => model learned the conditional.
                              cond_KL ~ KL(p_unigram || p_true) => collapsed.

PASS CRITERIA (honest, derived from correctness — NOT tuned to validate
the hypothesis):
  P1 (DIRECTIONAL):   cos_unigram_dir <= 0.30 at end of training
                      (random baseline 0.18; 0.30 = noticeable but not collapsed)
  P2 (RANK):          eff_rank        >= 8.0  (= 25% of d_model used)
  P3 (CONDITIONAL):   cond_KL         <= 0.5 * KL(p_unigram || p_true)
                      (model is at least halfway from "predict marginal" to
                       "predict conditional")
  P4 (LOSS):          loss            <= 0.7 * loss_unigram + 0.3 * loss_oracle
                      (model is at least 30% of the way from unigram baseline
                       to oracle — same threshold for all readouts)

A readout "BREAKS the unigram pull" iff P1 AND P2 AND P3 AND P4 all pass.
A readout "FITS BETTER but still collapses directionally" iff P3+P4 pass
but P1 fails — this would prove the user's hypothesis: better readouts
delay/mask but do not change direction.

NO knobs are tuned per-readout. Same lr, same steps, same batch size,
same init scale, same data, same seed. Differences are architectural.
"""

import numpy as np

# --------------------------------------------------------------------------
# Config (single source of truth, identical for all 4 readouts)
# --------------------------------------------------------------------------
V          = 200          # vocab
D          = 32           # d_model
K_MOS      = 4            # MoS mixture components
H_MLP      = 128          # MLP head hidden width
ALPHA      = 0.7          # signal strength of conditional vs unigram
ZIPF_S     = 1.0          # Zipfian exponent
LR         = 0.05
N_STEPS    = 4000
BATCH      = 256
EVAL_EVERY = 500
EVAL_BATCH = 1024
INIT_SCALE = 0.1
SEED       = 1234

rng = np.random.default_rng(SEED)

# --------------------------------------------------------------------------
# Synthetic ground-truth distribution
# --------------------------------------------------------------------------
ranks      = np.arange(1, V + 1)
p_unigram  = (1.0 / ranks ** ZIPF_S)
p_unigram /= p_unigram.sum()

# Random permutation: NEXT[x] is the "true next token" for input x
NEXT = rng.permutation(V)

def true_cond(x_batch):
    """p(y | x) for a batch of x. Shape (B, V)."""
    B = x_batch.shape[0]
    out = np.broadcast_to(p_unigram, (B, V)).copy() * (1.0 - ALPHA)
    out[np.arange(B), NEXT[x_batch]] += ALPHA
    return out

def sample_targets(x_batch):
    """y ~ p(y | x)."""
    P = true_cond(x_batch)
    # Inverse-CDF sample per row
    cum = P.cumsum(axis=1)
    u   = rng.random(x_batch.shape[0])[:, None]
    return (u < cum).argmax(axis=1)

# Baseline losses
loss_oracle  = -(true_cond(np.arange(V)) *
                 np.log(true_cond(np.arange(V)) + 1e-30)).sum(axis=1).mean()
loss_unigram = -(true_cond(np.arange(V)) *
                 np.log(p_unigram + 1e-30)).sum(axis=1).mean()
KL_unigram_true = (true_cond(np.arange(V)) *
                   (np.log(true_cond(np.arange(V)) + 1e-30) -
                    np.log(p_unigram + 1e-30))).sum(axis=1).mean()

# --------------------------------------------------------------------------
# Numerical helpers
# --------------------------------------------------------------------------
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=-1, keepdims=True)

def logsoftmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

def logsumexp(z, axis):
    m = z.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=axis, keepdims=True))).squeeze(axis)

def init(*shape):
    return rng.standard_normal(shape) * INIT_SCALE

# --------------------------------------------------------------------------
# Encoder (shared across readouts: same architecture, same init seed)
# --------------------------------------------------------------------------
def make_encoder():
    return {
        "E":   init(V, D),       # token embeddings
        "Wh":  init(D, D),       # encoder transform
    }

def encode(enc, x):
    """h(x) = tanh(E[x] @ Wh).  Shape (B, D)."""
    pre = enc["E"][x] @ enc["Wh"]
    return np.tanh(pre), pre  # return pre for backward

def encoder_backward(enc, x, dh, pre, lr):
    """Backprop dh through tanh and Wh and gather grad on E rows."""
    dpre = dh * (1.0 - np.tanh(pre) ** 2)        # (B, D)
    Ex   = enc["E"][x]                            # (B, D)
    dWh  = Ex.T @ dpre                            # (D, D)
    dEx  = dpre @ enc["Wh"].T                     # (B, D)
    enc["Wh"] -= lr * dWh / x.shape[0]
    np.add.at(enc["E"], x, -lr * dEx / x.shape[0])

# --------------------------------------------------------------------------
# Readouts: each implements forward (loss, p_model) and backward (dh, update).
# All update only their OWN params from dlogits; dh is returned for encoder.
# --------------------------------------------------------------------------

# ---- A) TIED dot product: logits = h @ E^T  (E is the encoder embedding)
class TiedHead:
    name = "TIED"
    def __init__(self, enc): self.enc = enc
    def forward(self, h):
        logits = h @ self.enc["E"].T              # (B, V)
        return logits
    def loss_and_dlogits(self, h, y):
        logits = self.forward(h)
        p = softmax(logits)
        nll = -logsoftmax(logits)[np.arange(y.size), y].mean()
        dlogits = p.copy()
        dlogits[np.arange(y.size), y] -= 1.0
        return nll, dlogits, p
    def backward(self, h, dlogits, lr):
        # logits = h @ E^T  =>  dh = dlogits @ E ;  dE += dlogits^T @ h
        dh = dlogits @ self.enc["E"]              # (B, D)
        dE = dlogits.T @ h                        # (V, D)
        self.enc["E"] -= lr * dE / h.shape[0]     # E updated by tied path
        return dh

# ---- B) UNTIED dot product: logits = h @ W^T, W independent
class UntiedHead:
    name = "UNTIED"
    def __init__(self, enc):
        self.enc = enc
        self.W = init(V, D)
    def forward(self, h):
        return h @ self.W.T
    def loss_and_dlogits(self, h, y):
        logits = self.forward(h)
        p = softmax(logits)
        nll = -logsoftmax(logits)[np.arange(y.size), y].mean()
        dlogits = p.copy()
        dlogits[np.arange(y.size), y] -= 1.0
        return nll, dlogits, p
    def backward(self, h, dlogits, lr):
        dh = dlogits @ self.W
        dW = dlogits.T @ h
        self.W -= lr * dW / h.shape[0]
        return dh

# ---- C) MLP head (NON-dot-product): logits = W2 ReLU(W1 h + b1) + b2
class MLPHead:
    name = "MLP"
    def __init__(self, enc):
        self.enc = enc
        self.W1 = init(D, H_MLP)
        self.b1 = np.zeros(H_MLP)
        self.W2 = init(H_MLP, V)
        self.b2 = np.zeros(V)
    def forward(self, h, return_intermediates=False):
        z1 = h @ self.W1 + self.b1                  # (B, H)
        a1 = np.maximum(z1, 0.0)                    # ReLU
        logits = a1 @ self.W2 + self.b2             # (B, V)
        if return_intermediates:
            return logits, z1, a1
        return logits
    def loss_and_dlogits(self, h, y):
        logits, z1, a1 = self.forward(h, return_intermediates=True)
        self._z1, self._a1 = z1, a1
        p = softmax(logits)
        nll = -logsoftmax(logits)[np.arange(y.size), y].mean()
        dlogits = p.copy()
        dlogits[np.arange(y.size), y] -= 1.0
        return nll, dlogits, p
    def backward(self, h, dlogits, lr):
        # logits = a1 @ W2 + b2
        dW2 = self._a1.T @ dlogits / h.shape[0]
        db2 = dlogits.mean(axis=0)
        da1 = dlogits @ self.W2.T                    # (B, H)
        dz1 = da1 * (self._z1 > 0.0)                 # ReLU'
        dW1 = h.T @ dz1 / h.shape[0]
        db1 = dz1.mean(axis=0)
        dh  = dz1 @ self.W1.T                        # (B, D)
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        return dh

# ---- D) Mixture of Softmaxes: p(y|h) = Σ_k π_k(h) softmax(W_k h)
#         Implemented in log-space for stability.
class MoSHead:
    name = "MoS"
    def __init__(self, enc):
        self.enc = enc
        self.Ws  = [init(V, D)  for _ in range(K_MOS)]
        self.Wpi = init(D, K_MOS)            # gating logits
    def _components(self, h):
        comp_logits = np.stack([h @ W.T for W in self.Ws], axis=1)  # (B, K, V)
        comp_logp   = comp_logits - logsumexp(comp_logits, axis=2)[..., None]
        gate_logits = h @ self.Wpi                                    # (B, K)
        gate_logp   = gate_logits - logsumexp(gate_logits, axis=1)[..., None]
        # log p(y|h) = logsumexp_k( gate_logp[k] + comp_logp[k, y] )
        joint = comp_logp + gate_logp[:, :, None]                    # (B, K, V)
        logp  = logsumexp(joint, axis=1)                              # (B, V)
        return logp, joint, comp_logp, gate_logp, comp_logits
    def loss_and_dlogits(self, h, y):
        logp, joint, comp_logp, gate_logp, comp_logits = self._components(h)
        nll = -logp[np.arange(y.size), y].mean()
        # We backprop through the mixture by autograd-by-hand:
        # post[k] = exp(joint[k, y] - logp[y])    posterior over components
        # d/dcomp_logits[k, v] L = post[k] * (softmax(comp_logits[k])[v] - 1{v==y})
        # d/dgate_logits[k]    L = post[k] - softmax(gate_logits)[k]
        post = np.exp(joint[np.arange(y.size), :, y] -
                      logp[np.arange(y.size), y][:, None])            # (B, K)
        comp_p = np.exp(comp_logp)                                    # (B, K, V)
        # dcomp_logits per k
        dcomp = post[:, :, None] * comp_p
        dcomp[np.arange(y.size), :, y] -= post
        # dgate_logits
        gate_p = np.exp(gate_logp)
        dgate  = post - gate_p                                        # (B, K)
        # NEGATIVE log-likelihood => we minimize, so gradients above are for
        # +log p(y); flip sign for NLL.
        dcomp = -dcomp
        dgate = -dgate
        # Stash for backward()
        self._h     = h
        self._dcomp = dcomp        # (B, K, V)
        self._dgate = dgate        # (B, K)
        # Return a dummy dlogits = 0 (we handle dh ourselves in backward)
        p_model = np.exp(logp)
        return nll, None, p_model
    def backward(self, h, _unused, lr):
        B = h.shape[0]
        dh = np.zeros_like(h)
        # comp_logits[k] = h @ W_k^T  =>  dh += dcomp[k] @ W_k ;  dW_k += dcomp[k]^T @ h
        for k in range(K_MOS):
            dh += self._dcomp[:, k, :] @ self.Ws[k]
            dWk = self._dcomp[:, k, :].T @ h / B
            self.Ws[k] -= lr * dWk
        # gate_logits = h @ Wpi   =>  dh += dgate @ Wpi^T ;  dWpi += h^T @ dgate
        dh += self._dgate @ self.Wpi.T
        dWpi = h.T @ self._dgate / B
        self.Wpi -= lr * dWpi
        return dh

# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def unigram_dir(E):
    """e_unigram_dir = normalize( Σ_y p_unigram(y) * E[y] )."""
    v = (p_unigram[:, None] * E).sum(axis=0)
    n = np.linalg.norm(v) + 1e-12
    return v / n

def cos_to(v, u):
    """cos angle between two vectors."""
    return float(v @ u / (np.linalg.norm(v) * np.linalg.norm(u) + 1e-12))

def effective_rank(H):
    """exp(entropy(normalized singular values))."""
    s = np.linalg.svd(H, compute_uv=False)
    s = s + 1e-12
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))

def evaluate(enc, head):
    x_eval = rng.integers(0, V, size=EVAL_BATCH)
    h, _   = encode(enc, x_eval)
    # loss + p_model
    y_eval = sample_targets(x_eval)
    nll, _, p_model = head.loss_and_dlogits(h, y_eval)
    # cond KL: mean_x KL(p_model || p_true)
    p_true = true_cond(x_eval)
    cond_kl = (p_model * (np.log(p_model + 1e-30) -
                          np.log(p_true + 1e-30))).sum(axis=1).mean()
    # directional alignment
    e_uf  = unigram_dir(enc["E"])
    h_bar = h.mean(axis=0)
    cos_u = abs(cos_to(h_bar, e_uf))
    # effective rank
    eff_r = effective_rank(h)
    return float(nll), cos_u, eff_r, float(cond_kl)

# --------------------------------------------------------------------------
# Training loop (identical for all readouts)
# --------------------------------------------------------------------------
def train(head_class):
    enc  = make_encoder()
    head = head_class(enc)
    history = []
    for step in range(1, N_STEPS + 1):
        x = rng.integers(0, V, size=BATCH)
        y = sample_targets(x)
        h, pre = encode(enc, x)
        nll, dlogits, _ = head.loss_and_dlogits(h, y)
        dh = head.backward(h, dlogits, LR)
        encoder_backward(enc, x, dh, pre, LR)
        if step % EVAL_EVERY == 0 or step == 1:
            metrics = evaluate(enc, head)
            history.append((step, *metrics))
    return head.name, history

# --------------------------------------------------------------------------
# Run all four
# --------------------------------------------------------------------------
def fmt_row(step, nll, cos_u, eff_r, cond_kl):
    return (f"  step={step:5d}  loss={nll:6.3f}  "
            f"cos_unigram_dir={cos_u:5.3f}  "
            f"eff_rank={eff_r:6.2f}  cond_KL={cond_kl:6.3f}")

print("=" * 78)
print(" READOUT ARCHITECTURE SIMULATION")
print("=" * 78)
print(f" V={V} D={D} alpha={ALPHA} zipf_s={ZIPF_S}  steps={N_STEPS}  lr={LR}")
print(f" BASELINES:")
print(f"   loss_oracle           = {loss_oracle:.4f}")
print(f"   loss_unigram          = {loss_unigram:.4f}")
print(f"   KL(unigram || true)   = {KL_unigram_true:.4f}")
print(f"   random baseline cos   = 1/sqrt(D) = {1.0 / np.sqrt(D):.4f}")
print()

# Re-seed before each run so encoder init is identical across readouts.
results = {}
for head_class in [TiedHead, UntiedHead, MLPHead, MoSHead]:
    rng = np.random.default_rng(SEED)
    NEXT = rng.permutation(V)  # rebuild with same seed
    name, hist = train(head_class)
    results[name] = hist
    print(f"--- {name} " + "-" * (74 - len(name)))
    for row in hist:
        print(fmt_row(*row))
    print()

# --------------------------------------------------------------------------
# Pass / fail summary against honest criteria
# --------------------------------------------------------------------------
print("=" * 78)
print(" PASS / FAIL  (criteria stated in file header, NOT tuned per readout)")
print("=" * 78)
target_cos     = 0.30
target_rank    = 8.0
target_kl      = 0.5 * KL_unigram_true
target_loss    = 0.7 * loss_unigram + 0.3 * loss_oracle
print(f" Thresholds:")
print(f"   P1 cos_unigram_dir <= {target_cos:.3f}")
print(f"   P2 eff_rank        >= {target_rank:.2f}")
print(f"   P3 cond_KL         <= {target_kl:.3f}   (= 0.5 * KL_unigram_true)")
print(f"   P4 loss            <= {target_loss:.3f}   (= 0.7*unigram + 0.3*oracle)")
print()
print(f" {'readout':<8} {'loss':>7}  {'cos_u':>6}  {'eff_r':>6}  {'cond_KL':>8}  "
      f"{'P1':>3} {'P2':>3} {'P3':>3} {'P4':>3}  verdict")
for name, hist in results.items():
    _, nll, cos_u, eff_r, cond_kl = hist[-1]
    p1 = cos_u   <= target_cos
    p2 = eff_r   >= target_rank
    p3 = cond_kl <= target_kl
    p4 = nll     <= target_loss
    all_pass = p1 and p2 and p3 and p4
    verdict  = "BREAKS unigram pull" if all_pass else (
               "fits but DIRECTIONALLY collapsed" if (p3 and p4 and not p1)
               else "FAILS")
    print(f" {name:<8} {nll:7.3f}  {cos_u:6.3f}  {eff_r:6.2f}  {cond_kl:8.3f}  "
          f"{'Y' if p1 else 'N':>3} {'Y' if p2 else 'N':>3} "
          f"{'Y' if p3 else 'N':>3} {'Y' if p4 else 'N':>3}  {verdict}")

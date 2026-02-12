"""
GRIM-text Training Session Analyzer (v2)
==========================================
Auto-finds the latest training session log and its paired training_run.log.
Parses ALL data from both files. Saves report to file.
Uses string search only (no regex). Reports facts, not interpretations.

Model Equation:
  logit[v] = (h . W[v]) where h = Encoder(Embed(tokens) + SinPos)
  loss = CE(softmax(logits), targets) + numeric_loss + reg
  grad_W[v] = sum_t( hidden[t] * grad_logits[t,v] )
  W_new = W - lr * m / sqrt(v_hat + eps)   (AdamW)
"""

import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# 1.  Locate latest logs (auto-detect, not hardcoded)
# ---------------------------------------------------------------------------
LOG_DIR = Path(r"D:\G.R.I.M\resources\models\GRIM-text\training\logs")


def find_latest_session_log():
    """Find latest session log (training_NNNN.log), NOT training_run.log."""
    candidates = []
    for f in LOG_DIR.iterdir():
        name = f.name
        if (name.startswith("training_")
                and name.endswith(".log")
                and name != "training_run.log"):
            candidates.append(f)
    if not candidates:
        print("ERROR: No training session log found in", LOG_DIR)
        sys.exit(1)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def find_training_run_log():
    """Find training_run.log (console output)."""
    p = LOG_DIR / "training_run.log"
    if p.exists():
        return p
    return None


# ---------------------------------------------------------------------------
# 2.  String-based field extractors (NO regex)
# ---------------------------------------------------------------------------

def extract_field(line, key):
    """Extract value after 'key=' until next space/comma/tab/newline."""
    idx = line.find(key + "=")
    if idx < 0:
        return None
    start = idx + len(key) + 1
    end = start
    while end < len(line) and line[end] not in (" ", "\t", "\n", "\r", ","):
        end += 1
    return line[start:end]


def extract_after(line, prefix):
    """Extract everything after prefix until end of line, stripped."""
    idx = line.find(prefix)
    if idx < 0:
        return None
    return line[idx + len(prefix):].strip()


def extract_between(line, left, right):
    """Extract text between left and right markers."""
    a = line.find(left)
    if a < 0:
        return None
    a += len(left)
    b = line.find(right, a)
    if b < 0:
        return line[a:].strip()
    return line[a:b].strip()


def safe_float(s):
    if s is None:
        return None
    s = s.strip().rstrip(",").rstrip("%")
    try:
        return float(s)
    except ValueError:
        return None


def safe_int(s):
    if s is None:
        return None
    s = s.strip().rstrip(",")
    try:
        return int(s)
    except ValueError:
        f = safe_float(s)
        return int(f) if f is not None else None


# ---------------------------------------------------------------------------
# 3.  Data structures
# ---------------------------------------------------------------------------

class BatchRecord:
    """Holds all metrics for a single batch from the session log."""
    __slots__ = [
        "batch_num",
        # --- Loss ---
        "loss_mean", "loss_sum", "valid_tokens", "masked_tokens", "total_tokens",
        "text_ce", "text_ce_weight", "numeric_loss", "numeric_weight",
        "reg_term", "total_loss",
        # --- Logits ---
        "logit_mean", "logit_max", "logit_min", "avg_max_logit", "top2_margin",
        "unique_argmax", "top_argmax_str",
        # --- Hidden cosine ---
        "avg_cos", "cos_pairs_str",
        # --- LM head norms ---
        "lm_head_norms_str",
        # --- Gradients ---
        "grad_total", "grad_emb_lm_tied", "grad_attn", "grad_ffn",
        "grad_rms", "grad_num",
        # --- Token 277 POST-OPT ---
        "w277_pre_norm", "w277_post_norm", "w277_delta", "w277_increased",
        # --- Token 277 Diag ---
        "space_logit_mean", "space_logit_min", "space_logit_max", "space_is_argmax",
        "global_max_logit", "global_argmax_token",
        # --- Weight gradient equation ---
        "w277_grad_norm", "w277_grad_sum", "w277_grad_mean",
        "w277_weight_norm", "w277_weight_mean",
        "w277_target_count", "w277_target_ratio",
        "w277_prediction",
        # --- Hidden state equation ---
        "h_mean", "h_norm_mean", "h_std",
        "h_at_277_norm", "h_at_other_norm",
        "grad_logits_277_at_target", "grad_logits_277_at_other",
        "grad_w277_from_277_norm", "grad_w277_from_other_norm",
        "grad_w277_ratio", "grad_w277_cos",
        # --- Feedback loop ---
        "fb_h_norm", "fb_w277_norm", "fb_cos_mean",
        "fb_hidden_corr",
        "fb_growth_h", "fb_growth_w", "fb_growth_cos", "fb_growth_logit",
        "fb_logit_277_at_target", "fb_logit_277_at_other", "fb_logit_delta",
        # --- PtPvDump ---
        "ptpv_n_positions", "ptpv_avg_pt", "ptpv_avg_p277_other",
        "ptpv_uniform_baseline",
        # --- Optimizer ---
        "opt_m_rms", "opt_v_rms",
        "update_rms", "param_rms",
        # --- Grad clipping ---
        "post_accum_norm", "clipped",
        # --- Learning rate ---
        "lr",
        # --- GradTrace extra ---
        "grad_scale", "preclip_grad_norm",
        # --- BOUNDARY_DIAGNOSTIC ---
        "boundary_summary",
        # --- PREDICTION ---
        "prediction_lines",
    ]

    def __init__(self, batch_num):
        self.batch_num = batch_num
        for s in self.__slots__[1:]:
            setattr(self, s, None)
        self.prediction_lines = []


# ---------------------------------------------------------------------------
# 4.  Parse session log (training_NNNNN.log)
# ---------------------------------------------------------------------------

def strip_timestamp(line):
    """Remove [2026-02-10 HH:MM:SS] prefix if present."""
    if len(line) > 22 and line.startswith("[20") and "] " in line[:25]:
        end = line.find("] ")
        if end > 0:
            return line[end + 2:]
    return line


def parse_session_log(log_path):
    """Parse session log file into dict of batch_num -> BatchRecord."""
    records = {}
    current_batch = None

    # Multi-line block state
    block_type = None  # "weight_grad", "hidden_state", "feedback_loop", "ptpv", "boundary"
    block_batch = None

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            raw_line = raw_line.rstrip("\n\r")
            line = strip_timestamp(raw_line)

            # ---- Batch boundary from POST-BACKWARD ----
            if "[GradTrace] POST-BACKWARD" in line:
                bn = safe_int(extract_field(line, "batch"))
                if bn is not None:
                    current_batch = bn
                    if bn not in records:
                        records[bn] = BatchRecord(bn)
                continue

            # ---- [LossStats] ----
            if "[LossStats]" in line:
                bn = safe_int(extract_field(line, "batch"))
                if bn is not None:
                    if bn not in records:
                        records[bn] = BatchRecord(bn)
                    r = records[bn]
                    current_batch = bn
                    r.loss_mean = safe_float(extract_field(line, "loss_mean"))
                    r.loss_sum = safe_float(extract_field(line, "loss_sum"))
                    r.valid_tokens = safe_int(extract_field(line, "valid_tokens"))
                    r.masked_tokens = safe_int(extract_field(line, "masked_tokens"))
                    r.total_tokens = safe_int(extract_field(line, "total_tokens"))
                continue

            # ---- [LossComponents] ----
            if "[LossComponents]" in line:
                bn = current_batch
                if bn is not None:
                    if bn not in records:
                        records[bn] = BatchRecord(bn)
                    r = records[bn]
                    r.text_ce = safe_float(extract_field(line, "text_ce"))
                    r.numeric_loss = safe_float(extract_field(line, "numeric"))
                    r.total_loss = safe_float(extract_field(line, "total"))
                    # weights are in (w=X.XXX) format
                    idx = line.find("(w=")
                    if idx > 0:
                        tw = extract_between(line, "(w=", ")")
                        r.text_ce_weight = safe_float(tw)
                    r.reg_term = safe_float(extract_field(line, "reg"))
                continue

            # ---- [LogitSignal] ----
            if "[LogitSignal]" in line:
                bn = safe_int(extract_field(line, "batch"))
                if bn is not None:
                    if bn not in records:
                        records[bn] = BatchRecord(bn)
                    r = records[bn]
                    r.logit_mean = safe_float(extract_field(line, "logit_mean"))
                    r.logit_max = safe_float(extract_field(line, "logit_max"))
                    r.logit_min = safe_float(extract_field(line, "logit_min"))
                    r.avg_max_logit = safe_float(extract_field(line, "avg_max_logit"))
                    r.top2_margin = safe_float(extract_field(line, "top2_margin"))
                    r.unique_argmax = safe_int(extract_field(line, "unique_argmax"))
                    ta = extract_between(line, "top_argmax=[", "]")
                    if ta:
                        r.top_argmax_str = ta
                continue

            # ---- [HiddenCosine] ----
            if "[HiddenCosine]" in line:
                bn = safe_int(extract_field(line, "batch"))
                if bn is not None:
                    if bn not in records:
                        records[bn] = BatchRecord(bn)
                    r = records[bn]
                    r.avg_cos = safe_float(extract_field(line, "avg_cos"))
                    cp = extract_between(line, "cos(h_i,h_j)=[", "]")
                    if cp:
                        r.cos_pairs_str = cp
                continue

            # ---- [LMHeadNorm] ----
            if "[LMHeadNorm]" in line:
                bn = safe_int(extract_field(line, "batch"))
                if bn is not None:
                    if bn not in records:
                        records[bn] = BatchRecord(bn)
                    norms = extract_between(line, "||W[v]||=[", "]")
                    if norms:
                        records[bn].lm_head_norms_str = norms
                continue

            # ---- [GradTrace] COMPUTED COMPONENTS ----
            if "COMPUTED COMPONENTS:" in line:
                bn = safe_int(extract_field(line, "batch"))
                if bn is None:
                    bn = current_batch
                if bn is not None:
                    if bn not in records:
                        records[bn] = BatchRecord(bn)
                    r = records[bn]
                    r.grad_total = safe_float(extract_field(line, "total"))
                    r.grad_emb_lm_tied = safe_float(extract_field(line, "emb_lm_tied"))
                    r.grad_attn = safe_float(extract_field(line, "attn"))
                    r.grad_ffn = safe_float(extract_field(line, "ffn"))
                    r.grad_rms = safe_float(extract_field(line, "rms"))
                    r.grad_num = safe_float(extract_field(line, "num"))
                continue

            # ---- [Token277] POST-OPT ----
            if "[Token277] POST-OPT" in line:
                bn = safe_int(extract_field(line, "batch"))
                if bn is not None:
                    if bn not in records:
                        records[bn] = BatchRecord(bn)
                    r = records[bn]
                    r.w277_pre_norm = safe_float(extract_field(line, "pre_norm"))
                    r.w277_post_norm = safe_float(extract_field(line, "post_norm"))
                    r.w277_delta = safe_float(extract_field(line, "delta_norm"))
                    r.w277_increased = "NORM_INCREASED" in line
                continue

            # ---- [Token277Diag] ----
            if "[Token277Diag]" in line:
                bn = safe_int(extract_field(line, "batch"))
                if bn is not None:
                    if bn not in records:
                        records[bn] = BatchRecord(bn)
                    r = records[bn]
                    r.space_logit_mean = safe_float(extract_field(line, "mean"))
                    r.space_logit_min = safe_float(extract_field(line, "min"))
                    r.space_logit_max = safe_float(extract_field(line, "max"))
                    r.space_is_argmax = extract_field(line, "is_argmax")
                    r.global_max_logit = safe_float(extract_field(line, "global_max_logit"))
                    r.global_argmax_token = safe_int(extract_field(line, "global_argmax_token"))
                continue

            # ---- [PostAccumClip] ----
            if "[PostAccumClip]" in line:
                bn = safe_int(extract_field(line, "batch"))
                if bn is not None:
                    if bn not in records:
                        records[bn] = BatchRecord(bn)
                    r = records[bn]
                    r.post_accum_norm = safe_float(extract_field(line, "post_accum_norm"))
                    r.clipped = "clipped=YES" in line
                continue

            # ---- [OptState] ----
            if "[OptState]" in line:
                bn = safe_int(extract_field(line, "batch"))
                if bn is not None:
                    if bn not in records:
                        records[bn] = BatchRecord(bn)
                    r = records[bn]
                    r.opt_m_rms = safe_float(extract_field(line, "m_rms"))
                    r.opt_v_rms = safe_float(extract_field(line, "v_rms"))
                continue

            # ---- [UpdateMag] ----
            if "[UpdateMag]" in line:
                bn = safe_int(extract_field(line, "batch"))
                if bn is not None:
                    if bn not in records:
                        records[bn] = BatchRecord(bn)
                    r = records[bn]
                    r.update_rms = safe_float(extract_field(line, "update_rms"))
                    r.param_rms = safe_float(extract_field(line, "param_rms"))
                continue

            # ---- [GradScaleDiag] ----
            if "[GradScaleDiag]" in line:
                bn = safe_int(extract_field(line, "batch"))
                if bn is not None:
                    if bn not in records:
                        records[bn] = BatchRecord(bn)
                    records[bn].grad_scale = safe_float(extract_field(line, "computed_grad_scale"))
                continue

            # ---- [Step N] loss=X lr=Y ----
            if "[Step " in line and "loss=" in line:
                bn_str = extract_between(line, "[Step ", "]")
                if bn_str is not None:
                    bn = safe_int(bn_str)
                    if bn is not None:
                        if bn not in records:
                            records[bn] = BatchRecord(bn)
                        r = records[bn]
                        r.lr = safe_float(extract_field(line, "lr"))
                        if r.loss_mean is None:
                            r.loss_mean = safe_float(extract_field(line, "loss"))
                continue

            # ---- [GradTrace] PRE-OPTIMIZER ----
            if "[GradTrace] PRE-OPTIMIZER" in line:
                bn = safe_int(extract_field(line, "batch"))
                if bn is not None:
                    if bn not in records:
                        records[bn] = BatchRecord(bn)
                    r = records[bn]
                    r.lr = safe_float(extract_field(line, "lr"))
                    r.preclip_grad_norm = safe_float(extract_field(line, "grad_norm"))
                continue

            # ============================================================
            #  Multi-line blocks
            # ============================================================

            # ---- [WEIGHT_GRADIENT_EQUATION] ----
            if "[WEIGHT_GRADIENT_EQUATION]" in line:
                block_type = "weight_grad"
                block_batch = current_batch
                continue

            # ---- [HIDDEN_STATE_EQUATION] ----
            if "[HIDDEN_STATE_EQUATION]" in line:
                block_type = "hidden_state"
                block_batch = current_batch
                continue

            # ---- [FEEDBACK_LOOP_EQUATION] ----
            if "[FEEDBACK_LOOP_EQUATION]" in line:
                block_type = "feedback_loop"
                block_batch = current_batch
                continue

            # ---- [PtPvDump] ----
            if "[PtPvDump]" in line:
                bn = safe_int(extract_field(line, "batch"))
                if bn is not None:
                    block_type = "ptpv"
                    block_batch = bn
                    if bn not in records:
                        records[bn] = BatchRecord(bn)
                    records[bn].ptpv_n_positions = safe_int(extract_field(line, "positions"))
                continue

            # ---- [BOUNDARY_DIAGNOSTIC] ----
            if "[BOUNDARY_DIAGNOSTIC]" in line:
                bn = safe_int(extract_field(line, "batch"))
                if bn is None:
                    bn = current_batch
                if bn is not None:
                    block_type = "boundary"
                    block_batch = bn
                    if bn not in records:
                        records[bn] = BatchRecord(bn)
                    records[bn].boundary_summary = line.strip()
                continue

            # ---- [PREDICTION] standalone ----
            if "[PREDICTION]" in line and block_type is None:
                bn = current_batch
                if bn is not None and bn in records:
                    records[bn].prediction_lines.append(line.strip())
                continue

            # ---- Process multi-line blocks ----
            if block_type and block_batch is not None and block_batch in records:
                r = records[block_batch]

                if block_type == "weight_grad":
                    if "GRAD_W[277]:" in line and "||grad||" in line:
                        r.w277_grad_norm = safe_float(extract_field(line, "||grad||"))
                        r.w277_grad_sum = safe_float(extract_field(line, "sum"))
                        r.w277_grad_mean = safe_float(extract_field(line, "mean"))
                    elif "WEIGHT[277]:" in line:
                        r.w277_weight_norm = safe_float(extract_field(line, "||W[277]||"))
                        r.w277_weight_mean = safe_float(extract_field(line, "mean"))
                    elif "token_277_count=" in line:
                        tc = extract_field(line, "token_277_count")
                        if tc and "/" in tc:
                            r.w277_target_count = safe_int(tc.split("/")[0])
                        r.w277_target_ratio = safe_float(extract_field(line, "ratio"))
                    elif "[PREDICTION]" in line:
                        r.w277_prediction = line.strip()
                        r.prediction_lines.append(line.strip())
                    elif line.strip() == "":
                        block_type = None

                elif block_type == "hidden_state":
                    if "HIDDEN STATES (encoder output):" in line:
                        r.h_mean = safe_float(extract_field(line, "mean"))
                        r.h_norm_mean = safe_float(extract_field(line, "||h||_mean"))
                        r.h_std = safe_float(extract_field(line, "std"))
                    elif "AT_277_TARGETS" in line:
                        r.h_at_277_norm = safe_float(extract_field(line, "||h||"))
                    elif "AT_OTHER_TARGETS" in line:
                        r.h_at_other_norm = safe_float(extract_field(line, "||h||"))
                    elif "GRAD_LOGITS[277]:" in line:
                        r.grad_logits_277_at_target = safe_float(extract_field(line, "at_277_targets"))
                        r.grad_logits_277_at_other = safe_float(extract_field(line, "at_other_targets"))
                    elif "from_277_targets:" in line:
                        r.grad_w277_from_277_norm = safe_float(extract_field(line, "||g||"))
                    elif "from_other_targets:" in line:
                        r.grad_w277_from_other_norm = safe_float(extract_field(line, "||g||"))
                    elif "||g_277||/||g_other||=" in line:
                        ratio_str = extract_field(line, "||g_277||/||g_other||")
                        if ratio_str and ratio_str.endswith("x"):
                            r.grad_w277_ratio = safe_float(ratio_str[:-1])
                        cos_part = extract_between(line, "cos(g_277, g_other)=", " ")
                        if cos_part:
                            r.grad_w277_cos = safe_float(cos_part)
                        block_type = None

                elif block_type == "feedback_loop":
                    if "INPUT h (encoder output):" in line:
                        r.fb_h_norm = safe_float(extract_field(line, "||h||_mean"))
                    elif "INPUT W[277] (LM head row):" in line:
                        r.fb_w277_norm = safe_float(extract_field(line, "||W[277]||"))
                    elif "ALIGNMENT:" in line:
                        r.fb_cos_mean = safe_float(extract_field(line, "cos(h, W[277])_mean"))
                    elif "When target=277:" in line:
                        r.fb_logit_277_at_target = safe_float(extract_field(line, "logit[277]_mean"))
                    elif "When target" in line and "277:" in line and "target=277" not in line:
                        r.fb_logit_277_at_other = safe_float(extract_field(line, "logit[277]_mean"))
                    elif "DELTA =" in line:
                        r.fb_logit_delta = safe_float(extract_between(line, "DELTA = ", " "))
                    elif "GROWTH_RATES:" in line:
                        gh = extract_field(line, "||h||")
                        if gh and "%" in gh:
                            r.fb_growth_h = safe_float(gh.replace("%", "").replace("+", ""))
                        gw = extract_field(line, "||W||")
                        if gw and "%" in gw:
                            r.fb_growth_w = safe_float(gw.replace("%", "").replace("+", ""))
                        cos_idx = line.find("cos=")
                        if cos_idx >= 0:
                            cos_val = line[cos_idx + 4:]
                            pct_idx = cos_val.find("%")
                            if pct_idx >= 0:
                                r.fb_growth_cos = safe_float(cos_val[:pct_idx])
                        logit_idx = line.find("logit=")
                        if logit_idx >= 0:
                            logit_val = line[logit_idx + 6:]
                            pct_idx = logit_val.find("%")
                            if pct_idx >= 0:
                                r.fb_growth_logit = safe_float(logit_val[:pct_idx])
                    elif "HIDDEN_CORRELATION:" in line:
                        hc = extract_field(line, "avg|cos(h_i,h_j)|")
                        if hc:
                            r.fb_hidden_corr = safe_float(hc)
                    elif "CONTRIBUTION_FACTORS:" in line or line.strip() == "":
                        if "CONTRIBUTION_FACTORS:" in line or (
                                r.fb_hidden_corr is not None and line.strip() == ""):
                            block_type = None

                elif block_type == "ptpv":
                    if "SUMMARY:" in line:
                        r.ptpv_avg_pt = safe_float(extract_field(line, "avg_p_t"))
                        ap = extract_field(line, "avg_p_277_at_other")
                        if ap:
                            r.ptpv_avg_p277_other = safe_float(ap)
                        r.ptpv_uniform_baseline = safe_float(extract_field(line, "uniform_baseline"))
                        block_type = None

                elif block_type == "boundary":
                    if line.strip() == "" or line.startswith("["):
                        block_type = None
                    else:
                        r.boundary_summary = (r.boundary_summary or "") + " | " + line.strip()

    return records


# ---------------------------------------------------------------------------
# 5.  Parse training_run.log (kernel-level diagnostics)
# ---------------------------------------------------------------------------

def parse_run_log(log_path):
    """Parse training_run.log for per-layer forward/backward kernel diagnostics.

    Returns a dict with summary data organized by tag type.
    """
    data = {
        "qkv_equations": [],       # list of dicts per [QKV_EQUATION] block
        "attn_scores": [],         # list of dicts per [ATTN_SCORE_EQUATION] block
        "fwd_lse_perhead": [],     # list of per-head LSE arrays
        "fwd_lse_summary": [],     # list of summary LSE dicts
        "fwd_alibi": [],           # list of ALiBi info dicts
        "layer0_cosine": [],       # list of layer 0 cosine dicts
        "bwd_sdpa": [],            # list of backward SDPA dicts
        "bwd_lse_perhead": [],     # list of backward per-head LSE
        "grad_a_equations": [],    # list of GRAD_A dicts
        "matmul_bwd": [],          # list of MATMUL-BWD dicts
        "loss_components": [],     # list of LossComponents dicts
        "issue93_fwd": [],         # list of per-layer forward stats
        "logsoftmax_bwd": [],      # list of LOGSOFTMAX_BWD dicts
        "bias_add_fwd": [],        # list of BIAS-ADD-FWD entries
        "issue77_cached": [],      # list of Issue77-CachedActivation entries
        "split_qkv": [],           # SplitQKV entries
        "fa_fwd_in": [],           # FA-FWD-IN entries
        "fa_fwd_out": [],          # FA-FWD-OUT entries
    }

    block = None
    current_block_data = {}
    current_head_list = []

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n\r")

            # ---- [QKV_EQUATION] ----
            if "[QKV_EQUATION]" in line:
                block = "qkv"
                current_block_data = {"header": line.strip()}
                continue

            if block == "qkv":
                if "ln1_out (RMSNorm output):" in line:
                    current_block_data["ln1_rms"] = safe_float(extract_field(line, "rms"))
                    current_block_data["ln1_row_norm"] = safe_float(extract_field(line, "avg_row_norm"))
                elif "W_qkv (QKV weights):" in line:
                    current_block_data["wqkv_rms"] = safe_float(extract_field(line, "rms"))
                    current_block_data["wqkv_row_norm"] = safe_float(extract_field(line, "avg_row_norm"))
                elif "EXPECTED" in line:
                    current_block_data["expected_row_norm"] = safe_float(extract_field(line, "expected_row_norm"))
                elif "ACTUAL" in line and "row_norm" in line:
                    current_block_data["actual_row_norm"] = safe_float(extract_field(line, "qkv_row_norm"))
                    current_block_data["inflation"] = safe_float(extract_field(line, "inflation"))
                    data["qkv_equations"].append(current_block_data)
                    block = None
                continue

            # ---- [ATTN_SCORE_EQUATION] ----
            if "[ATTN_SCORE_EQUATION]" in line:
                block = "attn_score"
                current_block_data = {"header": line.strip()}
                continue

            if block == "attn_score":
                if "Q (" in line:
                    current_block_data["q_rms"] = safe_float(extract_field(line, "rms"))
                    current_block_data["q_row_norm"] = safe_float(extract_field(line, "avg_row_norm"))
                elif "K (" in line:
                    current_block_data["k_rms"] = safe_float(extract_field(line, "rms"))
                    current_block_data["k_row_norm"] = safe_float(extract_field(line, "avg_row_norm"))
                elif "PARAMETERS:" in line:
                    current_block_data["scale"] = safe_float(extract_field(line, "scale"))
                    current_block_data["alibi_max"] = safe_float(extract_field(line, "alibi_slope_max"))
                elif "EXPECTED" in line and "score_magnitude" in line:
                    # Try to extract after the last "= "
                    idx = line.rfind("= ")
                    if idx >= 0:
                        current_block_data["expected_score"] = safe_float(line[idx + 2:].split()[0])
                elif "ACTUAL" in line and "score" in line:
                    current_block_data["actual_score_rms"] = safe_float(extract_field(line, "rms"))
                    current_block_data["actual_score_max"] = safe_float(extract_field(line, "max"))
                    data["attn_scores"].append(current_block_data)
                    block = None
                continue

            # ---- [FA-FWD-LSE-PERHEAD] ----
            if "[FA-FWD-LSE-PERHEAD]" in line:
                block = "fwd_lse_perhead"
                current_head_list = []
                continue

            if block == "fwd_lse_perhead":
                if "head[" in line:
                    mean_val = safe_float(extract_field(line, "mean"))
                    if mean_val is not None:
                        current_head_list.append(mean_val)
                elif line.strip() == "" or (line.strip() and "[" in line[:5] and "head[" not in line):
                    if current_head_list:
                        data["fwd_lse_perhead"].append(current_head_list)
                    block = None
                continue

            # ---- [FA-FWD-LSE-SUMMARY] ----
            if "[FA-FWD-LSE-SUMMARY]" in line:
                d = {
                    "nan": safe_int(extract_field(line, "nan")),
                    "inf": safe_int(extract_field(line, "inf")),
                    "mean": safe_float(extract_field(line, "mean")),
                }
                rng = extract_between(line, "range=[", "]")
                if rng and "," in rng:
                    parts = rng.split(",")
                    d["range_min"] = safe_float(parts[0].strip())
                    d["range_max"] = safe_float(parts[1].strip())
                data["fwd_lse_summary"].append(d)
                continue

            # ---- [FA-FWD-ALIBI] ----
            if "[FA-FWD-ALIBI]" in line:
                d = {}
                sr = extract_between(line, "slope_range=[", "]")
                if sr and "," in sr:
                    parts = sr.split(",")
                    d["slope_min"] = safe_float(parts[0].strip())
                    d["slope_max"] = safe_float(parts[1].strip())
                data["fwd_alibi"].append(d)
                continue

            # ---- [FA-FWD-IN] ----
            if "[FA-FWD-IN]" in line:
                d = {
                    "call": safe_int(extract_field(line, "call")),
                    "b": safe_int(extract_field(line, "b")),
                    "s": safe_int(extract_field(line, "s")),
                    "h": safe_int(extract_field(line, "h")),
                    "d": safe_int(extract_field(line, "d")),
                }
                data["fa_fwd_in"].append(d)
                continue

            # ---- [FA-FWD-OUT] ----
            if "[FA-FWD-OUT]" in line:
                d = {
                    "call": safe_int(extract_field(line, "call")),
                    "out_max": safe_float(extract_field(line, "out_max")),
                    "out_rms": safe_float(extract_field(line, "out_rms")),
                }
                data["fa_fwd_out"].append(d)
                continue

            # ---- [LAYER_0_COSINE_EQUATION] ----
            if "[LAYER_0_COSINE_EQUATION]" in line:
                block = "layer0_cos"
                current_block_data = {"header": line.strip()}
                continue

            if block == "layer0_cos":
                if "LAYERSCALE:" in line:
                    current_block_data["ls1"] = safe_float(extract_field(line, "LS1"))
                    current_block_data["ls2"] = safe_float(extract_field(line, "LS2"))
                elif "ACTUAL avg_cos=" in line:
                    current_block_data["avg_cos"] = safe_float(extract_field(line, "avg_cos"))
                    pairs_str = extract_between(line, "(pairs=", ")")
                    current_block_data["pairs"] = safe_int(pairs_str)
                    data["layer0_cosine"].append(current_block_data)
                    block = None
                continue

            # ---- [FA-BWD-SAVED-LSE-PERHEAD] ----
            if "[FA-BWD-SAVED-LSE-PERHEAD]" in line:
                block = "bwd_lse_perhead"
                current_head_list = []
                continue

            if block == "bwd_lse_perhead":
                if "head[" in line:
                    mean_val = safe_float(extract_field(line, "mean"))
                    if mean_val is not None:
                        current_head_list.append(mean_val)
                elif line.strip() == "" or (line.strip() and "[" in line[:5] and "head[" not in line):
                    if current_head_list:
                        data["bwd_lse_perhead"].append(current_head_list)
                    block = None
                continue

            # ---- [SDPA-BWD-ISSUE76] ----
            if "[SDPA-BWD-ISSUE76]" in line:
                d = {
                    "call": safe_int(extract_field(line, "call")),
                    "seqlen": safe_int(extract_field(line, "seqlen")),
                }
                block = "sdpa_bwd"
                current_block_data = d
                continue

            if block == "sdpa_bwd":
                if "dQ_max=" in line:
                    current_block_data["dQ_max"] = safe_float(extract_field(line, "dQ_max"))
                    current_block_data["dK_max"] = safe_float(extract_field(line, "dK_max"))
                    current_block_data["dV_max"] = safe_float(extract_field(line, "dV_max"))
                elif "dQ_rms=" in line:
                    current_block_data["dQ_rms"] = safe_float(extract_field(line, "dQ_rms"))
                    current_block_data["dK_rms"] = safe_float(extract_field(line, "dK_rms"))
                    current_block_data["dV_rms"] = safe_float(extract_field(line, "dV_rms"))
                elif "softmax_lse:" in line:
                    rng = extract_between(line, "range=[", "]")
                    if rng and "," in rng:
                        parts = rng.split(",")
                        current_block_data["lse_min"] = safe_float(parts[0].strip())
                        current_block_data["lse_max"] = safe_float(parts[1].strip())
                    current_block_data["lse_mean"] = safe_float(extract_field(line, "mean"))
                    data["bwd_sdpa"].append(current_block_data)
                    block = None
                continue

            # ---- [GRAD_A_EQUATION] ----
            if "[GRAD_A_EQUATION]" in line:
                block = "grad_a"
                current_block_data = {"header": line.strip()}
                continue

            if block == "grad_a":
                if "grad_C:" in line and "shape=" in line:
                    current_block_data["grad_c_rms"] = safe_float(extract_field(line, "rms"))
                    current_block_data["grad_c_max"] = safe_float(extract_field(line, "max"))
                elif "B(weights):" in line:
                    current_block_data["b_rms"] = safe_float(extract_field(line, "rms"))
                    current_block_data["b_std"] = safe_float(extract_field(line, "std"))
                elif "EXPECTED" in line:
                    idx = line.rfind("= ")
                    if idx >= 0:
                        current_block_data["expected"] = safe_float(line[idx + 2:].split()[0])
                elif "ACTUAL" in line and "grad_A:" in line:
                    current_block_data["actual_rms"] = safe_float(extract_field(line, "rms"))
                    current_block_data["actual_max"] = safe_float(extract_field(line, "max"))
                    data["grad_a_equations"].append(current_block_data)
                    block = None
                continue

            # ---- [LOGSOFTMAX_BWD_EQUATION] ----
            if "[LOGSOFTMAX_BWD_EQUATION]" in line:
                block = "logsoftmax_bwd"
                current_block_data = {"header": line.strip()}
                continue

            if block == "logsoftmax_bwd":
                if "grad_output (loss backward):" in line:
                    current_block_data["grad_out_rms"] = safe_float(extract_field(line, "rms"))
                    current_block_data["grad_out_max"] = safe_float(extract_field(line, "max"))
                elif "ACTUAL grad_logits:" in line:
                    current_block_data["grad_logits_rms"] = safe_float(extract_field(line, "rms"))
                    current_block_data["grad_logits_max"] = safe_float(extract_field(line, "max"))
                    data["logsoftmax_bwd"].append(current_block_data)
                    block = None
                continue

            # ---- [LossComponents] in run log ----
            if "[LossComponents]" in line:
                d = {
                    "text_ce": safe_float(extract_field(line, "text_ce")),
                    "numeric": safe_float(extract_field(line, "numeric")),
                    "reg": safe_float(extract_field(line, "reg")),
                    "total": safe_float(extract_field(line, "total")),
                }
                data["loss_components"].append(d)
                continue

            # ---- [MATMUL-BWD-IN] ----
            if "[MATMUL-BWD-IN]" in line:
                d = {
                    "call": safe_int(extract_field(line, "call")),
                    "grad_c_max": safe_float(extract_field(line, "max")),
                    "grad_c_rms": safe_float(extract_field(line, "rms")),
                }
                # Extract label between call=N and M=
                call_str = extract_field(line, "call")
                if call_str:
                    ci = line.find("call=" + call_str)
                    if ci >= 0:
                        after_call = line[ci + len("call=") + len(call_str):].strip()
                        space_idx = after_call.find(" M=")
                        if space_idx > 0:
                            d["label"] = after_call[:space_idx].strip()
                        else:
                            d["label"] = after_call.split()[0] if after_call else ""
                data["matmul_bwd"].append(d)
                continue

            # ---- [BIAS-ADD-FWD] ----
            if "[BIAS-ADD-FWD]" in line:
                d = {
                    "target": extract_field(line, "target"),
                    "layer": safe_int(extract_field(line, "layer")),
                    "rms": safe_float(extract_field(line, "rms")),
                    "max": safe_float(extract_field(line, "max")),
                }
                data["bias_add_fwd"].append(d)
                continue

            # ---- [Issue77-CachedActivation] ----
            if "[Issue77-CachedActivation]" in line:
                d = {
                    "call": safe_int(extract_field(line, "call")),
                    "what": extract_field(line, "what"),
                    "rms": safe_float(extract_field(line, "rms")),
                    "max": safe_float(extract_field(line, "max")),
                }
                data["issue77_cached"].append(d)
                continue

            # ---- [Issue93-FWD-*] ----
            if "[Issue93-FWD-" in line:
                d = {
                    "tag": extract_between(line, "[Issue93-FWD-", "]"),
                    "layer": safe_int(extract_field(line, "layer")),
                    "min": safe_float(extract_field(line, "min")),
                    "max": safe_float(extract_field(line, "max")),
                    "abs_max": safe_float(extract_field(line, "abs_max")),
                    "mean": safe_float(extract_field(line, "mean")),
                    "rms": safe_float(extract_field(line, "rms")),
                }
                data["issue93_fwd"].append(d)
                continue

            # ---- SplitQKV ----
            if "[SplitQKV-" in line:
                tag = extract_between(line, "[SplitQKV-", "]")
                d = {
                    "component": tag,
                    "rms": safe_float(extract_field(line, "rms")),
                    "max": safe_float(extract_field(line, "max")),
                }
                data["split_qkv"].append(d)
                continue

    return data


# ---------------------------------------------------------------------------
# 6.  Report generation -> writes to file
# ---------------------------------------------------------------------------

def fmt(val, decimals=6, width=12):
    if val is None:
        return "N/A".rjust(width)
    if isinstance(val, float):
        return f"{val:.{decimals}f}".rjust(width)
    return str(val).rjust(width)


def fmt_pct(val, width=10):
    if val is None:
        return "N/A".rjust(width)
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.2f}%".rjust(width)


class ReportWriter:
    """Buffer report output and write to file at end."""

    def __init__(self):
        self.lines = []

    def w(self, text=""):
        self.lines.append(text)

    def section(self, title):
        self.w()
        self.w("=" * 130)
        self.w("  " + title)
        self.w("=" * 130)

    def subsection(self, title):
        self.w()
        self.w("--- " + title + " ---")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))
        print(f"Report saved to: {path}")
        print(f"  Lines: {len(self.lines)}")


def generate_report(records, run_data, session_log_path, run_log_path, out):
    """Generate full report into ReportWriter out."""
    if not records:
        out.w("No batch records found.")
        return

    sorted_batches = sorted(records.keys())
    total_batches = len(sorted_batches)
    first_batch = sorted_batches[0]
    last_batch = sorted_batches[-1]

    out.w()
    out.w("###############################################################################")
    out.w("#                                                                             #")
    out.w("#              GRIM-text Training Session Analysis Report                     #")
    out.w("#                                                                             #")
    out.w("###############################################################################")
    out.w()
    out.w(f"  Session log:  {session_log_path.name}")
    out.w(f"  Run log:      {run_log_path.name if run_log_path else 'NOT FOUND'}")
    out.w(f"  Batches:      {first_batch} to {last_batch} ({total_batches} total)")

    r_first = records.get(first_batch)
    r_last = records.get(last_batch)
    if r_first and r_last and r_first.loss_mean is not None and r_last.loss_mean is not None:
        out.w(f"  Loss:         {r_first.loss_mean:.4f} (batch {first_batch}) -> {r_last.loss_mean:.4f} (batch {last_batch})")
        out.w(f"  Loss delta:   {r_last.loss_mean - r_first.loss_mean:+.4f}")

    # =========================================================================
    #  SECTION 1: Core Loss
    # =========================================================================
    out.section("1. LOSS EQUATION: L = text_ce * w_text + numeric * w_num + reg")

    out.subsection("loss per batch (ALL batches)")
    header = ("batch".rjust(8) + "loss_mean".rjust(12) + "text_ce".rjust(12)
              + "numeric".rjust(12) + "reg".rjust(12) + "total".rjust(12)
              + "valid_tok".rjust(10) + "lr".rjust(12))
    out.w(header)
    out.w("-" * len(header))
    for b in sorted_batches:
        r = records[b]
        out.w(f"{b:>8}{fmt(r.loss_mean, 4)}{fmt(r.text_ce, 4)}"
              f"{fmt(r.numeric_loss, 4)}{fmt(r.reg_term, 4)}"
              f"{fmt(r.total_loss, 4)}{fmt(r.valid_tokens, 0, 10)}"
              f"{fmt(r.lr, 8)}")

    # =========================================================================
    #  SECTION 2: Logit Distribution
    # =========================================================================
    out.section("2. LOGIT EQUATION: logit[v] = h @ W[v].T")

    out.subsection("logit statistics per batch")
    header = ("batch".rjust(8) + "logit_mean".rjust(12) + "logit_max".rjust(12)
              + "logit_min".rjust(12) + "avg_max".rjust(12) + "top2_marg".rjust(12)
              + "uniq_argm".rjust(12))
    out.w(header)
    out.w("-" * len(header))
    for b in sorted_batches:
        r = records[b]
        out.w(f"{b:>8}{fmt(r.logit_mean, 4)}{fmt(r.logit_max, 4)}"
              f"{fmt(r.logit_min, 4)}{fmt(r.avg_max_logit, 4)}"
              f"{fmt(r.top2_margin, 4)}{fmt(r.unique_argmax, 0)}")

    out.subsection("argmax token distribution")
    for b in sorted_batches:
        r = records[b]
        if r.top_argmax_str:
            ua = int(r.unique_argmax) if r.unique_argmax else "?"
            out.w(f"  batch={b:>5}  unique={ua:>3}  top_tokens: {r.top_argmax_str}")

    # =========================================================================
    #  SECTION 3: Hidden State
    # =========================================================================
    out.section("3. HIDDEN STATE EQUATION: h = Encoder(Embed(x) + SinPos)")

    out.subsection("hidden state norms and alignment")
    header = ("batch".rjust(8) + "||h||_mean".rjust(12) + "h_std".rjust(10)
              + "||h||@277".rjust(12) + "||h||@oth".rjust(12)
              + "avg_cos".rjust(12) + "fb_hid_cor".rjust(12))
    out.w(header)
    out.w("-" * len(header))
    for b in sorted_batches:
        r = records[b]
        out.w(f"{b:>8}{fmt(r.h_norm_mean, 4)}{fmt(r.h_std, 4)}"
              f"{fmt(r.h_at_277_norm, 4)}{fmt(r.h_at_other_norm, 4)}"
              f"{fmt(r.avg_cos, 6)}{fmt(r.fb_hidden_corr, 6)}")

    out.subsection("pairwise cosine pairs")
    for b in sorted_batches:
        r = records[b]
        if r.cos_pairs_str:
            out.w(f"  batch={b:>5}  {r.cos_pairs_str}")

    # =========================================================================
    #  SECTION 4: LM Head Weight Norms
    # =========================================================================
    out.section("4. LM HEAD WEIGHTS: ||W[v]|| for top-argmax tokens")

    for b in sorted_batches:
        r = records[b]
        if r.lm_head_norms_str:
            out.w(f"  batch={b:>5}  {r.lm_head_norms_str}")

    out.subsection("dominant token tracking")
    all_top_tokens = {}
    for b in sorted_batches:
        r = records[b]
        if r.top_argmax_str:
            for piece in r.top_argmax_str.split(","):
                piece = piece.strip()
                colon = piece.find(":")
                if colon > 0:
                    tok_name = piece[:colon]
                    count = safe_int(piece[colon + 1:])
                    if tok_name not in all_top_tokens:
                        all_top_tokens[tok_name] = 0
                    if count:
                        all_top_tokens[tok_name] += count
    top_tokens_sorted = sorted(all_top_tokens.items(), key=lambda x: -x[1])[:15]
    out.w(f"  Most frequent argmax tokens across {total_batches} batches:")
    for tok, count in top_tokens_sorted:
        out.w(f"    {tok}: appeared as argmax {count} times total")

    # =========================================================================
    #  SECTION 5: Token 277 Feedback Loop
    # =========================================================================
    out.section("5. TOKEN 277 FEEDBACK LOOP: logit[277] = ||h|| * ||W[277]|| * cos(h, W[277])")

    out.subsection("component tracking")
    header = ("batch".rjust(8) + "||h||".rjust(12) + "||W[277]||".rjust(12)
              + "cos(h,W)".rjust(12) + "hid_corr".rjust(12)
              + "dh%".rjust(10) + "dW%".rjust(10) + "dcos%".rjust(12)
              + "dlogit%".rjust(12))
    out.w(header)
    out.w("-" * len(header))
    for b in sorted_batches:
        r = records[b]
        if r.fb_h_norm is not None or r.fb_w277_norm is not None:
            out.w(f"{b:>8}{fmt(r.fb_h_norm, 4)}{fmt(r.fb_w277_norm, 6)}"
                  f"{fmt(r.fb_cos_mean, 6)}{fmt(r.fb_hidden_corr, 6)}"
                  f"{fmt_pct(r.fb_growth_h)}{fmt_pct(r.fb_growth_w)}"
                  f"{fmt_pct(r.fb_growth_cos)}{fmt_pct(r.fb_growth_logit)}")

    out.subsection("W[277] norm trajectory (post-optimizer)")
    header = ("batch".rjust(8) + "pre_norm".rjust(12) + "post_norm".rjust(12)
              + "delta".rjust(12) + "increased".rjust(10))
    out.w(header)
    out.w("-" * len(header))
    for b in sorted_batches:
        r = records[b]
        if r.w277_pre_norm is not None:
            inc = "YES" if r.w277_increased else "no"
            out.w(f"{b:>8}{fmt(r.w277_pre_norm, 6)}{fmt(r.w277_post_norm, 6)}"
                  f"{fmt(r.w277_delta, 6)}{inc:>10}")

    out.subsection("Token277Diag space logit statistics")
    header = ("batch".rjust(8) + "sp_mean".rjust(12) + "sp_min".rjust(12)
              + "sp_max".rjust(12) + "is_argmax".rjust(12)
              + "glob_max".rjust(12) + "glob_argm".rjust(12))
    out.w(header)
    out.w("-" * len(header))
    for b in sorted_batches:
        r = records[b]
        if r.space_logit_mean is not None:
            out.w(f"{b:>8}{fmt(r.space_logit_mean, 4)}{fmt(r.space_logit_min, 4)}"
                  f"{fmt(r.space_logit_max, 4)}{str(r.space_is_argmax):>12}"
                  f"{fmt(r.global_max_logit, 4)}{fmt(r.global_argmax_token, 0)}")

    out.subsection("feedback loop discrimination")
    header = ("batch".rjust(8) + "logit@277tgt".rjust(14)
              + "logit@other".rjust(14) + "DELTA".rjust(12))
    out.w(header)
    out.w("-" * len(header))
    for b in sorted_batches:
        r = records[b]
        if r.fb_logit_277_at_target is not None:
            out.w(f"{b:>8}{fmt(r.fb_logit_277_at_target, 4, 14)}"
                  f"{fmt(r.fb_logit_277_at_other, 4, 14)}"
                  f"{fmt(r.fb_logit_delta, 4)}")

    # =========================================================================
    #  SECTION 6: Gradient Equation
    # =========================================================================
    out.section("6. GRADIENT EQUATION: grad_W = h.T @ grad_logits")

    out.subsection("gradient component norms")
    header = ("batch".rjust(8) + "total".rjust(10) + "emb+lm".rjust(10)
              + "attn".rjust(10) + "ffn".rjust(10) + "rms".rjust(10)
              + "num_head".rjust(10) + "clipped".rjust(10))
    out.w(header)
    out.w("-" * len(header))
    for b in sorted_batches:
        r = records[b]
        if r.grad_total is not None:
            clip = "YES" if r.clipped else "no"
            out.w(f"{b:>8}{fmt(r.grad_total, 4, 10)}{fmt(r.grad_emb_lm_tied, 4, 10)}"
                  f"{fmt(r.grad_attn, 4, 10)}{fmt(r.grad_ffn, 4, 10)}"
                  f"{fmt(r.grad_rms, 4, 10)}{fmt(r.grad_num, 4, 10)}"
                  f"{clip:>10}")

    out.subsection("gradient group ratios")
    header = ("batch".rjust(8) + "emb+lm%".rjust(10) + "attn%".rjust(10)
              + "ffn%".rjust(10) + "rms%".rjust(10) + "num%".rjust(10))
    out.w(header)
    out.w("-" * len(header))
    for b in sorted_batches:
        r = records[b]
        if r.grad_total and r.grad_total > 0:
            e_pct = 100 * (r.grad_emb_lm_tied or 0) / r.grad_total
            a_pct = 100 * (r.grad_attn or 0) / r.grad_total
            f_pct = 100 * (r.grad_ffn or 0) / r.grad_total
            rm_pct = 100 * (r.grad_rms or 0) / r.grad_total
            n_pct = 100 * (r.grad_num or 0) / r.grad_total
            out.w(f"{b:>8}{e_pct:>9.1f}%{a_pct:>9.1f}%{f_pct:>9.1f}%"
                  f"{rm_pct:>9.1f}%{n_pct:>9.1f}%")

    # =========================================================================
    #  SECTION 7: Token 277 Gradient Decomposition
    # =========================================================================
    out.section("7. TOKEN 277 GRADIENT: grad_W[277] = sum_t(h[t] * (p[277] - y[277]))")

    out.subsection("gradient decomposition")
    header = ("batch".rjust(8) + "||gW277||".rjust(12) + "gW_sum".rjust(12)
              + "g_from277".rjust(12) + "g_from_oth".rjust(12)
              + "ratio".rjust(10) + "cos(g,g')".rjust(12)
              + "t277_cnt".rjust(10) + "t277_%".rjust(10))
    out.w(header)
    out.w("-" * len(header))
    for b in sorted_batches:
        r = records[b]
        if r.w277_grad_norm is not None:
            ratio_str = f"{r.grad_w277_ratio:.1f}x" if r.grad_w277_ratio else "N/A"
            out.w(f"{b:>8}{fmt(r.w277_grad_norm, 6)}{fmt(r.w277_grad_sum, 6)}"
                  f"{fmt(r.grad_w277_from_277_norm, 6)}{fmt(r.grad_w277_from_other_norm, 6)}"
                  f"{ratio_str:>10}{fmt(r.grad_w277_cos, 6)}"
                  f"{fmt(r.w277_target_count, 0)}{fmt(r.w277_target_ratio, 2)}")

    out.subsection("grad_logits[277] at target vs non-target positions")
    header = ("batch".rjust(8) + "at_277tgt".rjust(14) + "at_other".rjust(14))
    out.w(header)
    out.w("-" * len(header))
    for b in sorted_batches:
        r = records[b]
        if r.grad_logits_277_at_target is not None:
            out.w(f"{b:>8}{fmt(r.grad_logits_277_at_target, 6, 14)}"
                  f"{fmt(r.grad_logits_277_at_other, 6, 14)}")

    out.subsection("predictions")
    for b in sorted_batches:
        r = records[b]
        if r.prediction_lines:
            for pl in r.prediction_lines:
                out.w(f"  batch={b:>5}  {pl}")

    # =========================================================================
    #  SECTION 8: PtPvDump
    # =========================================================================
    out.section("8. PtPvDump: p(target) vs p(277) at sampled positions")

    out.subsection("summary per batch")
    header = ("batch".rjust(8) + "n_pos".rjust(10) + "avg_p_t".rjust(14)
              + "avg_p277_oth".rjust(14) + "uniform".rjust(14))
    out.w(header)
    out.w("-" * len(header))
    for b in sorted_batches:
        r = records[b]
        if r.ptpv_avg_pt is not None:
            out.w(f"{b:>8}{fmt(r.ptpv_n_positions, 0, 10)}"
                  f"{fmt(r.ptpv_avg_pt, 8, 14)}"
                  f"{fmt(r.ptpv_avg_p277_other, 8, 14)}"
                  f"{fmt(r.ptpv_uniform_baseline, 8, 14)}")

    # =========================================================================
    #  SECTION 9: Optimizer State
    # =========================================================================
    out.section("9. OPTIMIZER: W_new = W - lr * m / sqrt(v_hat + eps)  (AdamW)")

    out.subsection("optimizer state")
    header = ("batch".rjust(8) + "m_rms".rjust(14) + "v_rms".rjust(14)
              + "update_rms".rjust(14) + "param_rms".rjust(14)
              + "upd/param".rjust(14))
    out.w(header)
    out.w("-" * len(header))
    for b in sorted_batches:
        r = records[b]
        if r.opt_m_rms is not None:
            ratio = ""
            if r.update_rms and r.param_rms and r.param_rms > 0:
                ratio = f"{r.update_rms / r.param_rms:.8f}"
            out.w(f"{b:>8}{fmt(r.opt_m_rms, 10, 14)}{fmt(r.opt_v_rms, 10, 14)}"
                  f"{fmt(r.update_rms, 10, 14)}{fmt(r.param_rms, 10, 14)}"
                  f"{ratio:>14}")

    # =========================================================================
    #  SECTION 10: Boundary Diagnostic
    # =========================================================================
    boundary_batches = [b for b in sorted_batches if records[b].boundary_summary]
    if boundary_batches:
        out.section("10. BOUNDARY DIAGNOSTIC")
        for b in boundary_batches:
            out.w(f"  batch={b:>5}  {records[b].boundary_summary}")

    # =========================================================================
    #  SECTION 11: training_run.log kernel-level diagnostics
    # =========================================================================
    if run_data:
        out.section("11. KERNEL DIAGNOSTICS (from training_run.log)")

        # QKV equations
        if run_data["qkv_equations"]:
            out.subsection("QKV Projection: qkv_out = ln1_out @ W_qkv.T + b_qkv")
            header = ("  idx".rjust(6) + "ln1_rms".rjust(12) + "wqkv_rms".rjust(12)
                      + "expected".rjust(12) + "actual".rjust(12) + "inflation".rjust(12))
            out.w(header)
            out.w("  " + "-" * (len(header) - 2))
            for i, q in enumerate(run_data["qkv_equations"]):
                out.w(f"  {i:>4}{fmt(q.get('ln1_rms'), 6)}{fmt(q.get('wqkv_rms'), 6)}"
                      f"{fmt(q.get('expected_row_norm'), 4)}{fmt(q.get('actual_row_norm'), 4)}"
                      f"{fmt(q.get('inflation'), 2)}")

        # Attention scores
        if run_data["attn_scores"]:
            out.subsection("Attention Scores: score = (Q @ K.T) / sqrt(d) + ALiBi")
            header = ("  idx".rjust(6) + "q_rms".rjust(12) + "k_rms".rjust(12)
                      + "scale".rjust(12) + "score_rms".rjust(12) + "score_max".rjust(12))
            out.w(header)
            out.w("  " + "-" * (len(header) - 2))
            for i, a in enumerate(run_data["attn_scores"]):
                out.w(f"  {i:>4}{fmt(a.get('q_rms'), 6)}{fmt(a.get('k_rms'), 6)}"
                      f"{fmt(a.get('scale'), 6)}{fmt(a.get('actual_score_rms'), 6)}"
                      f"{fmt(a.get('actual_score_max'), 6)}")

        # Forward LSE per-head
        if run_data["fwd_lse_perhead"]:
            out.subsection("Forward LSE per-head means (12 heads)")
            for i, heads in enumerate(run_data["fwd_lse_perhead"]):
                heads_str = " ".join(f"{h:.2f}" for h in heads)
                out.w(f"  call={i:>3}  [{heads_str}]")

        # Forward LSE summary
        if run_data["fwd_lse_summary"]:
            out.subsection("Forward LSE summary")
            header = ("  idx".rjust(6) + "mean".rjust(10) + "range_min".rjust(12)
                      + "range_max".rjust(12) + "nan".rjust(6) + "inf".rjust(6))
            out.w(header)
            for i, s in enumerate(run_data["fwd_lse_summary"]):
                out.w(f"  {i:>4}{fmt(s.get('mean'), 4, 10)}"
                      f"{fmt(s.get('range_min'), 4)}{fmt(s.get('range_max'), 4)}"
                      f"{fmt(s.get('nan'), 0, 6)}{fmt(s.get('inf'), 0, 6)}")

        # ALiBi
        if run_data["fwd_alibi"]:
            out.subsection("ALiBi slopes")
            for i, a in enumerate(run_data["fwd_alibi"]):
                out.w(f"  call={i:>3}  slope_range=[{a.get('slope_min', 'N/A')}, {a.get('slope_max', 'N/A')}]")

        # Layer 0 cosine
        if run_data["layer0_cosine"]:
            out.subsection("Layer 0 cosine (output diversity)")
            header = ("  idx".rjust(6) + "avg_cos".rjust(12) + "LS1".rjust(10)
                      + "LS2".rjust(10) + "pairs".rjust(8))
            out.w(header)
            for i, c in enumerate(run_data["layer0_cosine"]):
                out.w(f"  {i:>4}{fmt(c.get('avg_cos'), 6)}"
                      f"{fmt(c.get('ls1'), 4, 10)}{fmt(c.get('ls2'), 4, 10)}"
                      f"{fmt(c.get('pairs'), 0, 8)}")

        # Backward SDPA
        if run_data["bwd_sdpa"]:
            out.subsection("Backward FlashAttention dQ/dK/dV (SDPA-BWD-ISSUE76)")
            header = ("  call".rjust(7) + "dQ_max".rjust(14) + "dK_max".rjust(14)
                      + "dV_max".rjust(14) + "dK/dV".rjust(10)
                      + "lse_min".rjust(12) + "lse_max".rjust(12))
            out.w(header)
            out.w("  " + "-" * (len(header) - 2))
            for s in run_data["bwd_sdpa"]:
                dk_dv = ""
                if s.get("dK_max") and s.get("dV_max") and s["dV_max"] > 0:
                    dk_dv = f"{s['dK_max'] / s['dV_max']:.2f}"
                out.w(f"  {fmt(s.get('call'), 0, 5)}"
                      f"{fmt(s.get('dQ_max'), 10, 14)}"
                      f"{fmt(s.get('dK_max'), 10, 14)}"
                      f"{fmt(s.get('dV_max'), 10, 14)}"
                      f"{dk_dv:>10}"
                      f"{fmt(s.get('lse_min'), 4)}"
                      f"{fmt(s.get('lse_max'), 4)}")

        # GRAD_A_EQUATION
        if run_data["grad_a_equations"]:
            out.subsection("GRAD_A (LM head backward: grad_A = grad_C @ B)")
            header = ("  idx".rjust(6) + "grad_c_rms".rjust(14)
                      + "b_rms".rjust(14) + "expected".rjust(14)
                      + "actual_rms".rjust(14) + "actual_max".rjust(14))
            out.w(header)
            for i, g in enumerate(run_data["grad_a_equations"]):
                out.w(f"  {i:>4}{fmt(g.get('grad_c_rms'), 10, 14)}"
                      f"{fmt(g.get('b_rms'), 10, 14)}"
                      f"{fmt(g.get('expected'), 10, 14)}"
                      f"{fmt(g.get('actual_rms'), 10, 14)}"
                      f"{fmt(g.get('actual_max'), 10, 14)}")

        # LOGSOFTMAX_BWD
        if run_data["logsoftmax_bwd"]:
            out.subsection("LogSoftmax backward")
            header = ("  idx".rjust(6) + "grad_out_rms".rjust(14)
                      + "grad_out_max".rjust(14) + "grad_logits_rms".rjust(16)
                      + "grad_logits_max".rjust(16))
            out.w(header)
            for i, ls in enumerate(run_data["logsoftmax_bwd"]):
                out.w(f"  {i:>4}{fmt(ls.get('grad_out_rms'), 10, 14)}"
                      f"{fmt(ls.get('grad_out_max'), 10, 14)}"
                      f"{fmt(ls.get('grad_logits_rms'), 10, 16)}"
                      f"{fmt(ls.get('grad_logits_max'), 10, 16)}")

        # MATMUL-BWD by label
        if run_data["matmul_bwd"]:
            out.subsection("MATMUL-BWD grad_C magnitudes by operation")
            label_groups = {}
            for m in run_data["matmul_bwd"]:
                label = m.get("label", "unknown")
                if label not in label_groups:
                    label_groups[label] = []
                label_groups[label].append(m)
            for label, items in sorted(label_groups.items()):
                maxes = [i["grad_c_max"] for i in items if i.get("grad_c_max") is not None]
                rmses = [i["grad_c_rms"] for i in items if i.get("grad_c_rms") is not None]
                if maxes:
                    out.w(f"  {label:>20}  n={len(items):>3}"
                          f"  grad_c_max: [{min(maxes):.2e} .. {max(maxes):.2e}]"
                          f"  grad_c_rms: [{min(rmses):.2e} .. {max(rmses):.2e}]")

        # BIAS-ADD-FWD summary
        if run_data["bias_add_fwd"]:
            out.subsection("BIAS-ADD-FWD summary by target")
            target_groups = {}
            for item in run_data["bias_add_fwd"]:
                t = item.get("target", "?")
                if t not in target_groups:
                    target_groups[t] = []
                target_groups[t].append(item)
            for target, items in sorted(target_groups.items()):
                rmses = [i["rms"] for i in items if i.get("rms") is not None]
                maxes = [i["max"] for i in items if i.get("max") is not None]
                if rmses:
                    out.w(f"  {target:>15}  n={len(items):>3}"
                          f"  rms: [{min(rmses):.2e} .. {max(rmses):.2e}]"
                          f"  max: [{min(maxes):.2e} .. {max(maxes):.2e}]")

        # Issue77 cached activation summary
        if run_data["issue77_cached"]:
            out.subsection("Issue77 cached activation magnitudes")
            what_groups = {}
            for item in run_data["issue77_cached"]:
                w = item.get("what", "?")
                if w not in what_groups:
                    what_groups[w] = []
                what_groups[w].append(item)
            for what, items in sorted(what_groups.items()):
                rmses = [i["rms"] for i in items if i.get("rms") is not None]
                if rmses:
                    out.w(f"  {what:>25}  n={len(items):>3}"
                          f"  rms: [{min(rmses):.4f} .. {max(rmses):.4f}]")

        # SplitQKV summary
        if run_data["split_qkv"]:
            out.subsection("SplitQKV component magnitudes")
            comp_groups = {}
            for item in run_data["split_qkv"]:
                c = item.get("component", "?")
                if c not in comp_groups:
                    comp_groups[c] = []
                comp_groups[c].append(item)
            for comp, items in sorted(comp_groups.items()):
                rmses = [i["rms"] for i in items if i.get("rms") is not None]
                maxes = [i["max"] for i in items if i.get("max") is not None]
                if rmses:
                    out.w(f"  {comp:>5}  n={len(items):>3}"
                          f"  rms: [{min(rmses):.4f} .. {max(rmses):.4f}]"
                          f"  max: [{min(maxes):.4f} .. {max(maxes):.4f}]")

        # Issue93 forward pass
        if run_data["issue93_fwd"]:
            out.subsection("Issue93 forward pass activation stats")
            tag_groups = {}
            for item in run_data["issue93_fwd"]:
                tag = item.get("tag", "?")
                if tag not in tag_groups:
                    tag_groups[tag] = []
                tag_groups[tag].append(item)
            for tag in sorted(tag_groups.keys()):
                items = tag_groups[tag]
                out.w(f"  {tag}:")
                for item in items:
                    out.w(f"    layer={item.get('layer', '?')}"
                          f"  min={fmt(item.get('min'), 4, 8)}"
                          f"  max={fmt(item.get('max'), 4, 8)}"
                          f"  rms={fmt(item.get('rms'), 4, 8)}"
                          f"  mean={fmt(item.get('mean'), 4, 8)}")

        # Loss components from run log
        if run_data["loss_components"]:
            out.subsection("LossComponents from training_run.log")
            for i, lc in enumerate(run_data["loss_components"]):
                out.w(f"  call={i:>3}"
                      f"  text_ce={fmt(lc.get('text_ce'), 4, 8)}"
                      f"  numeric={fmt(lc.get('numeric'), 4, 8)}"
                      f"  reg={fmt(lc.get('reg'), 4, 8)}"
                      f"  total={fmt(lc.get('total'), 4, 8)}")

        # FA-FWD-IN summary
        if run_data["fa_fwd_in"]:
            out.subsection("FlashAttention Forward inputs (shape)")
            for i, d in enumerate(run_data["fa_fwd_in"][:5]):
                out.w(f"  call={d.get('call','?')}"
                      f"  b={d.get('b','?')} s={d.get('s','?')}"
                      f"  h={d.get('h','?')} d={d.get('d','?')}")
            if len(run_data["fa_fwd_in"]) > 5:
                out.w(f"  ... ({len(run_data['fa_fwd_in'])} total calls)")

        # FA-FWD-OUT summary
        if run_data["fa_fwd_out"]:
            out.subsection("FlashAttention Forward output magnitudes")
            out_maxes = [d["out_max"] for d in run_data["fa_fwd_out"] if d.get("out_max") is not None]
            out_rmses = [d["out_rms"] for d in run_data["fa_fwd_out"] if d.get("out_rms") is not None]
            if out_maxes:
                out.w(f"  n={len(run_data['fa_fwd_out'])}"
                      f"  out_max: [{min(out_maxes):.4f} .. {max(out_maxes):.4f}]"
                      f"  out_rms: [{min(out_rmses):.4f} .. {max(out_rmses):.4f}]")

    # =========================================================================
    #  SECTION 12: Summary Statistics
    # =========================================================================
    out.section("12. SUMMARY STATISTICS (factual)")

    # Loss
    losses = [(b, records[b].loss_mean) for b in sorted_batches if records[b].loss_mean is not None]
    if losses:
        min_loss = min(losses, key=lambda x: x[1])
        max_loss = max(losses, key=lambda x: x[1])
        out.w(f"  Loss range:       [{min_loss[1]:.4f} (batch {min_loss[0]})] to [{max_loss[1]:.4f} (batch {max_loss[0]})]")
        out.w(f"  First batch loss: {losses[0][1]:.4f}")
        out.w(f"  Final batch loss: {losses[-1][1]:.4f}")
        n5 = min(5, len(losses))
        avg_first = sum(l[1] for l in losses[:n5]) / n5
        avg_last = sum(l[1] for l in losses[-n5:]) / n5
        out.w(f"  Avg first {n5}:     {avg_first:.4f}")
        out.w(f"  Avg last {n5}:      {avg_last:.4f}")

    # unique_argmax
    ua_vals = [(b, records[b].unique_argmax) for b in sorted_batches if records[b].unique_argmax is not None]
    if ua_vals:
        out.w()
        min_ua = min(ua_vals, key=lambda x: x[1])
        max_ua = max(ua_vals, key=lambda x: x[1])
        out.w(f"  unique_argmax:    [{int(min_ua[1])} (batch {min_ua[0]})] to [{int(max_ua[1])} (batch {max_ua[0]})]")

    # W[277] norm
    w277_norms = [(b, records[b].w277_post_norm) for b in sorted_batches if records[b].w277_post_norm is not None]
    if w277_norms:
        out.w()
        out.w(f"  ||W[277]|| start: {w277_norms[0][1]:.6f} (batch {w277_norms[0][0]})")
        out.w(f"  ||W[277]|| end:   {w277_norms[-1][1]:.6f} (batch {w277_norms[-1][0]})")
        peak = max(w277_norms, key=lambda x: x[1])
        out.w(f"  ||W[277]|| peak:  {peak[1]:.6f} (batch {peak[0]})")
        inc_count = sum(1 for b in sorted_batches if records[b].w277_increased)
        out.w(f"  W[277] increased: {inc_count}/{total_batches} batches")

    # Hidden state norms
    h_norms = [(b, records[b].h_norm_mean) for b in sorted_batches if records[b].h_norm_mean is not None]
    if h_norms:
        out.w()
        out.w(f"  ||h|| start:      {h_norms[0][1]:.4f} (batch {h_norms[0][0]})")
        out.w(f"  ||h|| end:        {h_norms[-1][1]:.4f} (batch {h_norms[-1][0]})")
        peak = max(h_norms, key=lambda x: x[1])
        trough = min(h_norms, key=lambda x: x[1])
        out.w(f"  ||h|| peak:       {peak[1]:.4f} (batch {peak[0]})")
        out.w(f"  ||h|| trough:     {trough[1]:.4f} (batch {trough[0]})")

    # avg_cos
    cos_vals = [(b, records[b].avg_cos) for b in sorted_batches if records[b].avg_cos is not None]
    if cos_vals:
        out.w()
        out.w(f"  avg_cos start:    {cos_vals[0][1]:.6f}")
        out.w(f"  avg_cos end:      {cos_vals[-1][1]:.6f}")
        peak = max(cos_vals, key=lambda x: x[1])
        trough = min(cos_vals, key=lambda x: x[1])
        out.w(f"  avg_cos peak:     {peak[1]:.6f} (batch {peak[0]})")
        out.w(f"  avg_cos trough:   {trough[1]:.6f} (batch {trough[0]})")

    # fb_hidden_corr
    hc_vals = [(b, records[b].fb_hidden_corr) for b in sorted_batches if records[b].fb_hidden_corr is not None]
    if hc_vals:
        out.w()
        out.w(f"  hidden_corr start: {hc_vals[0][1]:.6f}")
        out.w(f"  hidden_corr end:   {hc_vals[-1][1]:.6f}")
        out.w(f"  (expected ~0.036 for uncorrelated 768-dim vectors)")

    # Gradient clipping
    clipped_count = sum(1 for b in sorted_batches if records[b].clipped)
    out.w()
    out.w(f"  Gradient clipped: {clipped_count}/{total_batches} batches")

    # cos(h, W[277])
    cos_hw = [(b, records[b].fb_cos_mean) for b in sorted_batches if records[b].fb_cos_mean is not None]
    if cos_hw:
        out.w()
        out.w(f"  cos(h, W[277]) start: {cos_hw[0][1]:.6f}")
        out.w(f"  cos(h, W[277]) end:   {cos_hw[-1][1]:.6f}")
        peak = max(cos_hw, key=lambda x: x[1])
        out.w(f"  cos(h, W[277]) peak:  {peak[1]:.6f} (batch {peak[0]})")

    # Gradient dominance
    grad_records = [(b, records[b]) for b in sorted_batches if records[b].grad_total is not None]
    if grad_records:
        out.w()
        emb_dom = sum(1 for _, r in grad_records if r.grad_emb_lm_tied and r.grad_total and r.grad_emb_lm_tied > 0.5 * r.grad_total)
        out.w(f"  emb+lm > 50% of total grad: {emb_dom}/{len(grad_records)} batches")

    # grad_w277 cos
    gcos_vals = [(b, records[b].grad_w277_cos) for b in sorted_batches if records[b].grad_w277_cos is not None]
    if gcos_vals:
        out.w()
        avg_gcos = sum(v for _, v in gcos_vals) / len(gcos_vals)
        out.w(f"  cos(g_277, g_other) avg: {avg_gcos:.6f}  (negative = opposing)")
        out.w(f"  cos(g_277, g_other) end: {gcos_vals[-1][1]:.6f}")

    # PtPvDump summary
    ptpv_records = [(b, records[b]) for b in sorted_batches if records[b].ptpv_avg_pt is not None]
    if ptpv_records:
        out.w()
        avg_pt_all = sum(r.ptpv_avg_pt for _, r in ptpv_records) / len(ptpv_records)
        p277_vals = [r.ptpv_avg_p277_other for _, r in ptpv_records if r.ptpv_avg_p277_other is not None]
        avg_p277_all = sum(p277_vals) / max(1, len(p277_vals))
        out.w(f"  PtPv avg p(target): {avg_pt_all:.8f}")
        out.w(f"  PtPv avg p(277) at other: {avg_p277_all:.8f}")
        if ptpv_records[0][1].ptpv_uniform_baseline:
            out.w(f"  PtPv uniform baseline: {ptpv_records[0][1].ptpv_uniform_baseline:.8f}")

    out.w()
    out.w("=" * 130)
    out.w("  END OF REPORT")
    out.w("=" * 130)


# ---------------------------------------------------------------------------
# 7.  Main
# ---------------------------------------------------------------------------

def main():
    session_log = find_latest_session_log()
    run_log = find_training_run_log()

    print(f"Session log: {session_log.name} ({session_log.stat().st_size / 1024:.0f} KB)")
    if run_log:
        print(f"Run log:     {run_log.name} ({run_log.stat().st_size / (1024*1024):.1f} MB)")

    print("Parsing session log...")
    records = parse_session_log(session_log)
    print(f"  Parsed {len(records)} batch records")

    run_data = None
    if run_log:
        print("Parsing training_run.log...")
        run_data = parse_run_log(run_log)
        counts = {k: len(v) for k, v in run_data.items() if v}
        print(f"  Parsed: {counts}")

    # Generate report
    out = ReportWriter()
    generate_report(records, run_data, session_log, run_log, out)

    # Save to file next to the session log
    session_id = session_log.stem.replace("training_", "")
    report_path = LOG_DIR / f"analysis_{session_id}.txt"
    out.save(report_path)

    # Also print to console
    print()
    print("=" * 60)
    print("Report preview (first 50 lines):")
    print("=" * 60)
    for line in out.lines[:50]:
        print(line)
    print(f"\n... ({len(out.lines)} total lines, see {report_path.name})")


if __name__ == "__main__":
    main()

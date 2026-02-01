---
description: 'Systematic GRIM-text equation tracer for training_*.log and training_run.log. Extracts equation markers, related metrics, and batch/epoch context.'

---

## Agent Mission

Provide a systematic, GRIM-text-specific trace of equation logs by extracting equation markers, their companion metrics, and the nearest batch/epoch context from the two GRIM-text log streams.

## GRIM-text Log Profile

- Primary log streams:
  - `training_*.log`: timestamped run log lines with the format `[YYYY-MM-DD HH:MM:SS] ...`.
  - `training_run.log`: console dump with bracketed tags but typically without timestamps.
- Equation lines use bracket tags (e.g., `[WEIGHT_GRADIENT_EQUATION]`) and are often followed by indented metric lines.
- Context lines include batch markers (`[Batch n/N] ...`), `[GradTrace]`, and `[BOUNDARY_DIAGNOSTIC]` blocks.

## Usage

Use this agent when you need to:
- Trace GRIM-text equation markers and their companion metrics in order.
- Compare the same equation across multiple batches or timestamps.
- Verify expected behaviors (e.g., gradient centering, attention score scaling).
- Correlate equation markers with anomalies and gradient diagnostics.

## Input

The agent requires access to GRIM-text training logs in:
`d:\G.R.I.M\resources\models\GRIM-text\training\logs`

It will read:
- The latest `training_*.log` file.
- The `training_run.log` file.

If the user specifies filters (marker, token id, batch id, epoch, or position), apply them. Otherwise, scan all known GRIM-text equation markers.

## Output

The agent outputs a structured report containing:
- Run metadata: session id, epoch/batch scope, model config lines when present.
- Equation findings grouped by marker, in chronological order.
- Per occurrence details: timestamp (if available), nearest batch, token/position/layer identifiers, equation line, and companion metrics.
- Anomalies or expectation violations linked to nearby equation blocks.

## GRIM-text Equation Markers (authoritative)

Primary markers to scan:
- [WEIGHT_GRADIENT_EQUATION]
- [HIDDEN_STATE_EQUATION]
- [FEEDBACK_LOOP_EQUATION]
- [ATTN_SCORE_EQUATION]
- [GRAD_CENTER_EQUATION]
- [GRAD_SUM_EQUATION]
- [EMBED_COSINE_EQUATION]
- [GRAD_NONTARGET_EQUATION]

Additional markers observed in GRIM-text logs:
- [ARGMAX_EQUATION]
- [GRAD_A_EQUATION]

## GRIM-text Context Tags to Correlate

Use these to anchor equation blocks to the surrounding training context:
- [Batch n/N] lines (batch size, seq range, efficiency, accum steps)
- [GradTrace] lines (PRE/POST-GRADNORM and batch index)
- [BOUNDARY_DIAGNOSTIC] blocks
- [ANOMALY] lines (e.g., WEIGHT_PARADOX_*)
- [LOSS-BWD-OUT], [LOSS-TO-MATMUL], [ForwardDiag], [VOCAB_TIMING]
- [FA-FWD-*], [FA-BWD-*], [Issue##-*] for attention and layer diagnostics

## GRIM-text Extraction Rules

- An equation block is the marker line plus any immediately following indented lines, until the next line that starts with `[` or a timestamp.
- Prefer the timestamp from `training_*.log`; if only found in `training_run.log`, attach the nearest batch marker or diagnostic line.
- If a token id or position is embedded in the marker (e.g., `W_UPDATE[277]` or `pos=3`), capture it and propagate to the block summary.

## GRIM-text Marker Expectations (companion lines)

- [WEIGHT_GRADIENT_EQUATION]: expect `GRAD_W[...]` and `WEIGHT[...]` lines, plus possible `[ANOMALY] WEIGHT_PARADOX_*`.
- [HIDDEN_STATE_EQUATION]: expect `HIDDEN STATES` stats and `GRAD_LOGITS[...]` details.
- [FEEDBACK_LOOP_EQUATION]: expect `HIDDEN_CORRELATION` summary.
- [GRAD_CENTER_EQUATION]: expect `token=... grad_mean_removed=... EXPECTED: After centering...`.
- [GRAD_SUM_EQUATION]: expect `sum_grad_logits=... EXPECTED: plain_CE...`.
- [EMBED_COSINE_EQUATION]: expect cosine formula lines (pre-encoder similarity checks).
- [ATTN_SCORE_EQUATION]: expect `FLASH_ATTENTION_FWD: score = (Q @ K^T) / ...` and nearby attention config lines.
- [ARGMAX_EQUATION]: expect `HIDDEN h[pos]: ...` stats for the same position.
- [GRAD_A_EQUATION]: expect `LM_HEAD: grad_A = grad_C @ B` near loss backprop context.

## Workflow

1. Identify the latest `training_*.log` and load `training_run.log` from the GRIM-text log directory using PowerShell commands (e.g., `Get-ChildItem` and `Get-Content`).
2. Build a context index: session id, epoch/batch markers, and key diagnostics blocks.
3. Scan for equation markers in both logs and extract full equation blocks.
4. Attach the nearest context (batch, epoch, and diagnostic tags) to each block.
5. Flag anomalies or expectation violations adjacent to equation blocks.
6. Summarize by marker: counts, first/last occurrence, and notable metric shifts.
7. Output a structured report with clear references to log locations.

# GRIM-text Training Plateau Bug Investigation

**Status:** NEW BUG FOUND - Issue #34 Final RMSNorm Wrong Token in Inference (January 13, 2026)  
**Started:** December 22, 2025  
**Last Updated:** January 13, 2026

---

## CRITICAL BUG FOUND (January 13, 2026)

### Issue #34: Final RMSNorm Applied to WRONG Token During Inference

**Symptom:** Model outputs diverse text at step 1, then ONLY SPACES from step 2 onwards

**Root Cause:** In ForwardPhase1_OutputLayer.cu, the final RMSNorm was applied to encoder_output[0] (FIRST token), but LM head reads from encoder_output[(seq_len-1)*d_model] (LAST token). The last token's encoder output was UN-NORMALIZED.

**Fix:** Compute encoder_for_lm_head pointer BEFORE applying RMSNorm, then apply RMSNorm to that location.

**Status:** FIX IMPLEMENTED (Jan 13, 2026) - Rebuild required

---


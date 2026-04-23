# Configuration (`ai_config.json`)

All model and training configuration lives in `ai_config.json`.

## Fail-loud defaults (Rule 20)
Algorithmic config fields MUST default to `0` and throw if not loaded. Hardcoded defaults like `int max_seq_len = 512;` are forbidden — they hide silent fallbacks.

```cpp
int max_seq_len = 0;  // ✅
// ...load from JSON...
if (max_seq_len == 0)
    throw std::runtime_error("max_seq_len missing from ai_config.json");
```

## Validation token budget
Validation MUST use `ctx.model->getConfig().max_tokens_per_batch`, never a hardcoded constant — buffer overflow crash otherwise.

## `per_token_grad_scale=true`
Required. See [Encoder.md](Encoder.md).

## Loss config
`training.config.loss` — see [Loss.md](Loss.md).

## LM head centering
`training.config.lm_head_centering.freeze_final_rms_gamma=true` — see [LMHead.md](LMHead.md).

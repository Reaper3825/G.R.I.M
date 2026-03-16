# Training Data: Batches per Epoch & Vocab Size

## 1. Batches per epoch (~4850 vs expected ~6992)

**Your setup:** 55,935 entries, batch length 1024, batch size 8.

- **Expected batches per epoch:** `ceil(55935 / 8) = 6992` (each batch has at most 8 sequences, so you need at least that many batches to cover all sequences).
- **If you see ~4850:** That would mean only about `4850 × 8 ≈ 38,800` sequences are being used per epoch, so ~17k sequences would never be seen.

**What actually determines batch count:**

- Batches are built in **dynamic batching** (`Batching_GPU.cu::buildBatches`): sequences are packed into batches with a **token budget** `batch_size × max_seq_len` (e.g. 8 × 1024 = 8192) and `max_batch_size = 8`. Every sequence with `seq_length > 0` is included; none are dropped by the packing logic.
- So the number of batches is determined by **how many sequences are in the train catalog** when the epoch runs. That catalog is built from `train_views` / `train_seqs` **after**:
  - Sliding window (can increase sequence count)
  - **filterOverlong** (removes sequences with `token_ids.size() > max_seq_len`)
  - **filterShortSequences** (removes sequences with too few valid targets if `min_seq_valid_tokens > 0`)

**What to check:**

1. **Logs at startup:** Look for:
   - `"Train sequences: XXXXX"` — this is the count **after** all filters. If this is ~38,800 instead of 55,935, the difference is from overlong/short filtering.
   - `"Created Y dynamic batches"` at the start of each epoch — Y should equal `ceil(Train sequences / 8)` (roughly, depending on packing).

2. **If "Train sequences" is 55,935 but you still see ~4850 batches:** Then something else is wrong (e.g. a cap or mis-read of the number). The code path does not cap the number of batches; it builds one batch per “slot” of sequences up to the token budget and batch size.

3. **If "Train sequences" is ~38,800:** Then ~17k sequences are being removed by:
   - **Overlong:** Increase `max_seq_len` in config, or ensure your data isn’t longer than 1024 after sliding window.
   - **Short:** If `min_seq_valid_tokens > 0`, lower it or set to 0 so short sequences aren’t dropped.

So: confirm the actual **Train sequences** and **Created Y dynamic batches** in the logs; that will tell you whether the “missing” data is from filtering (then fix max_seq_len / min_seq_valid_tokens) or from a different bug.

---

## 2. Vocab only 2250 (expected larger, e.g. 10000)

**Where vocab size comes from:**

- **At training time:** Vocab size is **not** read from `ai_config.json` at runtime. It is read from the **.grmt file header** when the training data is loaded (`training_data_loader.hpp`: `vocab_size_` from the GRMT header). So whatever vocab size was used **when the .grmt was encoded** is what you get.
- When the **DataLoader** (or encode pipeline) builds/encodes data, it uses the **tokenizer’s `totalVocabSize()`** and writes that into the .grmt header (`DataLoader.cu` ~657–663). So:
  - **Vocab size in training = tokenizer’s totalVocabSize() at the time the .grmt was written.**

**Why you might see 2250:**

- **totalVocabSize()** = special tokens (4) + bytes (256) + atoms (2) + **unigram pieces** = **262 + unigram pieces**. So 2250 ⇒ **~1988 unigram pieces**.
- So the **vocab.bin** that was used when encoding your current .grmt only had ~1988 pieces. That can happen if:
  1. **Existing vocab.bin** was built with an older/smaller config (e.g. smaller `vocab_size` or less data), and you never regenerated it.
  2. **Tokenizer training** was run with a small corpus or strict **min_subword_freq** (e.g. 3), so the unigram algorithm never reached 10000 pieces and stopped at ~1988.

**What to do:**

1. **Regenerate vocab and .grmt together** so they match and use the desired size:
   - Delete the existing **vocab** and **.grmt** (e.g. `vocab.bin` and `training_data.grmt` in your training data directory).
   - Run the **DataLoader / encode pipeline** again. It will:
     - Train a new tokenizer with `target_vocab_size` from config (e.g. 10000 in `ai_config.json` → up to 10000 **unigram pieces** → totalVocabSize = 262 + 10000 = 10262), **or** load an existing vocab if you don’t force rebuild.
     - Encode the training data and write the new **totalVocabSize()** into the .grmt header.
   - Then start training again; the trainer will read the new (larger) vocab size from the .grmt.

2. **If after rebuild you still get fewer than 10000 pieces**, the tokenizer is hitting a natural limit (e.g. corpus size or frequency pruning):
   - Lower **min_subword_freq** (e.g. to 1 or 2) so more subwords are kept.
   - Add more/better training text so the unigram algorithm has more types to grow the vocab.

**Config reference:** In `ai_config.json` you have `"tokenizer": { "vocab_size": 10000, "max_vocab_size": 10000, ... }`. The DataLoader uses this for **training** the tokenizer; the **runtime** vocab size for training the LM is always whatever is stored in the .grmt header (i.e. the tokenizer that was used when the .grmt was created).

---

## Summary

| Issue | Cause | Fix |
|-------|--------|-----|
| ~4850 batches instead of ~6992 | Fewer sequences after overlong/short filters, or mis-reading the number | Check logs for "Train sequences" and "Created Y dynamic batches"; adjust max_seq_len and min_seq_valid_tokens if needed. |
| Vocab 2250 instead of 10000 | .grmt was encoded with a tokenizer that had only ~1988 unigram pieces (total 2250) | Delete vocab.bin and .grmt; re-run DataLoader/encode so a new tokenizer (e.g. 10000 pieces) is built and .grmt is re-encoded; optionally lower min_subword_freq or add data if vocab still small. |

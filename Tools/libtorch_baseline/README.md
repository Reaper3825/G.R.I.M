# Libtorch Baseline (C++)

Minimal GPT-style training loop in C++ using libtorch. This is meant as a known-good baseline to verify training and text generation on your JSONL data.

## Build (Windows, VS 2022)

From repo root:

```powershell
cmake -S tools/libtorch_baseline -B tools/libtorch_baseline/build -DTorch_DIR="D:/G.R.I.M/venv_rl/Lib/site-packages/torch/share/cmake/Torch"
cmake --build tools/libtorch_baseline/build --config Release
```

## Run

Make sure the torch DLLs are on PATH:

```powershell
$env:PATH = "D:/G.R.I.M/venv_rl/Lib/site-packages/torch/lib;$env:PATH"
.\tools\libtorch_baseline\build\Release\grim_libtorch_baseline.exe --data resources/models/GRIM-text/training/data/merged_verified_cache.jsonl --field content
```

## Common args

```text
--data <path>           JSONL dataset path
--field <name>          field name (default: content)
--seq_len 256
--batch_size 8
--steps 1000
--n_layer 6 --n_head 8 --n_embd 512
--lr 3e-4
--prompt "Hello world"
--device cuda|cpu
```

The tokenizer is byte-level (0-255) with an EOS token (256). This is only for baseline validation.

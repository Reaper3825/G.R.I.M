$python = '.\.venv\Scripts\python.exe'
$script = @"
from pathlib import Path
from huggingface_hub import HfApi

target = Path(r"D:/G.R.I.M/resources/models/GRIM-text/quality/deberta-v3-base-mnli")
target.mkdir(parents=True, exist_ok=True)

api = HfApi()
for fname in ("config.json", "pytorch_model.bin", "tokenizer.json", "vocab.txt"):
    api.hf_hub_download(
        repo_id="MoritzLaurer/deberta-v3-base-mnli",
        filename=fname,
        local_dir=target,
        local_dir_use_symlinks=False,
        force_download=True,
    )
    print("downloaded", fname)
"@
& $python - <<"PY"
$script
PY

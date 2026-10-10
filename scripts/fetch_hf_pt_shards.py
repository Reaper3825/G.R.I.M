"""Download one pinned Parquet shard per PT source using existing dependencies."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import os
import time
import urllib.parse
import urllib.request

from generate_hf_pt import DATA, SOURCES, request_json, atomic_bytes, json_bytes

ROOT = DATA / "hf_pt_provenance" / "expansion_8800"


def fetch_source(source):
    key, dataset, config, _, _ = source
    hub, _ = request_json("https://huggingface.co/api/datasets/" + dataset)
    revision = hub["sha"]
    folder = "data/wikihow" if key == "howto" else config
    listing, _ = request_json(f"https://huggingface.co/api/datasets/{dataset}/tree/{revision}/{folder}?recursive=false")
    shard = sorted((f for f in listing if f["path"].endswith(".parquet")), key=lambda f:f["path"])[0]
    expected = shard["lfs"]["oid"]
    path = ROOT / "parquet" / (key + "_" + expected + ".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        url = f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{shard['path']}?download=true&grim_pt={int(time.time())}"
        temp = path.with_suffix(".partial")
        print(f"Downloading {key}: {shard['size']/1024/1024:.1f} MiB", flush=True)
        h = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(url, timeout=60) as response, temp.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > shard["size"]:
                    raise ValueError("Shard exceeds declared size")
                h.update(chunk)
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        if size != shard["size"] or h.hexdigest() != expected:
            raise ValueError("Shard size or LFS SHA-256 mismatch")
        temp.replace(path)
    else:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                h.update(chunk)
        if path.stat().st_size != shard["size"] or h.hexdigest() != expected:
            raise ValueError("Cached shard integrity mismatch")
    print(f"Verified {key} shard SHA-256", flush=True)
    return key, dict(dataset=dataset, config=config, revision=revision, hub=hub,
        path=str(path.resolve()), repository_path=shard["path"], sha256=expected, size=shard["size"])


def main():
    with ThreadPoolExecutor(max_workers=3) as pool:
        manifest = dict(pool.map(fetch_source, SOURCES))
    atomic_bytes(ROOT / "parquet_manifest.json", json_bytes(manifest))
    print("Pinned shard manifest saved", flush=True)


if __name__ == "__main__":
    main()

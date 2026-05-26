#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
TRAINING_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[5]
DEFAULT_CACHE_DIR = REPO_ROOT / "external" / "vcpkg" / "downloads"
DEFAULT_DEPS_ROOT = TRAINING_ROOT / "third_party"

DEPS = (
    {
        "name": "nlohmann_json",
        "archive": "nlohmann-json-v3.12.0.tar.gz",
        "url": "https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.tar.gz",
        "target": Path("nlohmann_json/include/nlohmann/json.hpp"),
        "member_suffixes": (
            "/single_include/nlohmann/json.hpp",
            "/include/nlohmann/json.hpp",
        ),
    },
    {
        "name": "flatbuffers",
        "archive": "google-flatbuffers-v25.2.10.tar.gz",
        "url": "https://github.com/google/flatbuffers/archive/refs/tags/v25.2.10.tar.gz",
        "target": Path("flatbuffers/include/flatbuffers/flatbuffers.h"),
        "prefix_marker": "/include/flatbuffers/",
    },
    {
        "name": "cpp-httplib",
        "archive": "yhirose-cpp-httplib-v0.26.0.tar.gz",
        "url": "https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.26.0.tar.gz",
        "target": Path("cpp-httplib/include/httplib.h"),
        "member_suffixes": (
            "/httplib.h",
        ),
    },
)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[manual-deps] Downloading {url} -> {destination}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def ensure_archive(dep: dict, cache_dir: Path, allow_download: bool) -> Path:
    archive_path = cache_dir / dep["archive"]
    if archive_path.exists():
        print(f"[manual-deps] Using cached archive {archive_path}")
        return archive_path
    if not allow_download:
        raise RuntimeError(
            f"Required archive {archive_path} is missing and downloads are disabled"
        )
    download_file(dep["url"], archive_path)
    return archive_path


def extract_single_file(archive_path: Path, target_path: Path, member_suffixes: tuple[str, ...]) -> None:
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            member_name = f"/{member.name}"
            if not any(member_name.endswith(suffix) for suffix in member_suffixes):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Failed to read {member.name} from {archive_path}")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("wb") as output:
                shutil.copyfileobj(extracted, output)
            return
    raise RuntimeError(
        f"Could not find any of {member_suffixes} inside {archive_path.name}"
    )


def extract_prefixed_tree(archive_path: Path, deps_root: Path, prefix_marker: str) -> None:
    wrote_any = False
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            marker_index = member.name.find(prefix_marker.lstrip("/"))
            if marker_index < 0:
                continue
            prefix = member.name[marker_index:]
            if not prefix.startswith(prefix_marker.lstrip("/")):
                continue
            relative_path = prefix[len(prefix_marker.lstrip("/")):]
            destination = deps_root / "flatbuffers" / "include" / "flatbuffers" / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Failed to read {member.name} from {archive_path}")
            with destination.open("wb") as output:
                shutil.copyfileobj(extracted, output)
            wrote_any = True
    if not wrote_any:
        raise RuntimeError(
            f"Could not find files under {prefix_marker} inside {archive_path.name}"
        )


def stage_dependency(dep: dict, archive_path: Path, deps_root: Path) -> None:
    stage_dir = deps_root / dep["target"].parts[0]
    if stage_dir.exists():
        shutil.rmtree(stage_dir)

    if dep["name"] == "flatbuffers":
        extract_prefixed_tree(archive_path, deps_root, dep["prefix_marker"])
    else:
        extract_single_file(archive_path, deps_root / dep["target"], dep["member_suffixes"])

    expected_path = deps_root / dep["target"]
    if not expected_path.exists():
        raise RuntimeError(f"Staging {dep['name']} failed; missing {expected_path}")
    print(f"[manual-deps] Staged {dep['name']} -> {expected_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage manual GRIM-text training dependencies from cached source archives"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Directory containing dependency source archives (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--deps-root",
        type=Path,
        default=DEFAULT_DEPS_ROOT,
        help=f"Output root for staged headers (default: {DEFAULT_DEPS_ROOT})",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Fail instead of downloading an archive that is not already cached",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_dir = args.cache_dir.resolve()
    deps_root = args.deps_root.resolve()
    deps_root.mkdir(parents=True, exist_ok=True)

    print(f"[manual-deps] Cache dir: {cache_dir}")
    print(f"[manual-deps] Output dir: {deps_root}")

    for dep in DEPS:
        archive_path = ensure_archive(dep, cache_dir, allow_download=not args.no_download)
        stage_dependency(dep, archive_path, deps_root)

    print("[manual-deps] Manual dependency staging complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - fail loud with context for build prep
        print(f"[manual-deps] ERROR: {exc}", file=sys.stderr)
        raise

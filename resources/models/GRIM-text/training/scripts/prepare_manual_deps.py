#!/usr/bin/env python3

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


def download_file(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    print("[manual-deps] Downloading {0} -> {1}".format(url, destination))
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def ensure_archive(dep, cache_dir, allow_download):
    archive_path = cache_dir / dep["archive"]
    if archive_path.exists():
        print("[manual-deps] Using cached archive {0}".format(archive_path))
        return archive_path
    if not allow_download:
        raise RuntimeError(
            "Required archive {0} is missing and downloads are disabled".format(archive_path)
        )
    download_file(dep["url"], archive_path)
    return archive_path


def extract_single_file(archive_path, target_path, member_suffixes):
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            member_name = "/{0}".format(member.name)
            if not any(member_name.endswith(suffix) for suffix in member_suffixes):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError("Failed to read {0} from {1}".format(member.name, archive_path))
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("wb") as output:
                shutil.copyfileobj(extracted, output)
            return
    raise RuntimeError(
        "Could not find any of {0} inside {1}".format(member_suffixes, archive_path.name)
    )


def extract_prefixed_tree(archive_path, deps_root, prefix_marker):
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
                raise RuntimeError("Failed to read {0} from {1}".format(member.name, archive_path))
            with destination.open("wb") as output:
                shutil.copyfileobj(extracted, output)
            wrote_any = True
    if not wrote_any:
        raise RuntimeError(
            "Could not find files under {0} inside {1}".format(prefix_marker, archive_path.name)
        )


def stage_dependency(dep, archive_path, deps_root):
    stage_dir = deps_root / dep["target"].parts[0]
    if stage_dir.exists():
        shutil.rmtree(stage_dir)

    if dep["name"] == "flatbuffers":
        extract_prefixed_tree(archive_path, deps_root, dep["prefix_marker"])
    else:
        extract_single_file(archive_path, deps_root / dep["target"], dep["member_suffixes"])

    expected_path = deps_root / dep["target"]
    if not expected_path.exists():
        raise RuntimeError("Staging {0} failed; missing {1}".format(dep["name"], expected_path))
    print("[manual-deps] Staged {0} -> {1}".format(dep["name"], expected_path))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage manual GRIM-text training dependencies from cached source archives"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory containing dependency source archives (default: {0})".format(DEFAULT_CACHE_DIR),
    )
    parser.add_argument(
        "--deps-root",
        type=Path,
        default=DEFAULT_DEPS_ROOT,
        help="Output root for staged headers (default: {0})".format(DEFAULT_DEPS_ROOT),
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Fail instead of downloading an archive that is not already cached",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cache_dir = args.cache_dir.resolve()
    deps_root = args.deps_root.resolve()
    deps_root.mkdir(parents=True, exist_ok=True)

    print("[manual-deps] Cache dir: {0}".format(cache_dir))
    print("[manual-deps] Output dir: {0}".format(deps_root))

    for dep in DEPS:
        archive_path = ensure_archive(dep, cache_dir, allow_download=not args.no_download)
        stage_dependency(dep, archive_path, deps_root)

    print("[manual-deps] Manual dependency staging complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # fail loud with context for build prep
        print("[manual-deps] ERROR: {0}".format(exc), file=sys.stderr)
        raise

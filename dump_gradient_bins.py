#!/usr/bin/env python3
"""
Dump .bin files in a directory to human-readable .txt files.
- Tries multiple numeric dtypes (float32, float64, int32, int64, uint8)
- Writes a header with file info and a stats section for each attempted dtype
- Writes up to --max-elements values (default 100000) unless --full

Usage:
    python dump_gradient_bins.py --input-dir "D:/G.R.I.M/gradient_dumps" --out-dir "D:/G.R.I.M/gradient_dumps/text" 

"""
import argparse
from pathlib import Path
import sys
import numpy as np
import math

CANDIDATE_DTYPES = [
    (np.float32, 'float32'),
    (np.float64, 'float64'),
    (np.int32, 'int32'),
    (np.int64, 'int64'),
    (np.uint8, 'uint8'),
    (np.int8, 'int8'),
]


def summarize_array(arr):
    if arr.size == 0:
        return {'count': 0}
    count = int(arr.size)
    # attempt to compute stats safely
    try:
        arrf = arr.astype(np.float64)
        mean = float(np.mean(arrf))
        std = float(np.std(arrf))
        mini = float(np.min(arrf))
        maxi = float(np.max(arrf))
    except Exception:
        mean = std = mini = maxi = None
    return {'count': count, 'mean': mean, 'std': std, 'min': mini, 'max': maxi}


def dump_file(bin_path: Path, out_dir: Path, dtypes=None, max_elements=100000, full=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    base = bin_path.stem
    out_path = out_dir / (base + '.txt')

    tried = []
    with out_path.open('w', encoding='utf-8') as fout:
        fout.write(f"Source file: {bin_path}\n")
        fout.write(f"Size bytes: {bin_path.stat().st_size}\n")
        fout.write('\n')

        dtypes_to_try = dtypes if dtypes is not None else CANDIDATE_DTYPES
        for dtype, name in dtypes_to_try:
            fout.write(f"=== Interpretation as {name} ===\n")
            try:
                arr = np.fromfile(str(bin_path), dtype=dtype)
            except Exception as e:
                fout.write(f"Failed to read as {name}: {e}\n\n")
                continue

            stats = summarize_array(arr)
            fout.write(f"Element count: {stats.get('count', 0)}\n")
            if stats.get('count', 0) > 0:
                fout.write(f"Mean: {stats.get('mean')}  Std: {stats.get('std')}\n")
                fout.write(f"Min: {stats.get('min')}  Max: {stats.get('max')}\n")

            # decide how many elements to write
            if full:
                n = arr.size
            else:
                n = min(arr.size, max_elements)

            fout.write(f"Writing {n} elements{' (full)' if full else ''}\n")

            if n > 0:
                # format numbers compactly
                if np.issubdtype(arr.dtype, np.floating):
                    fmt = '%.6g'
                else:
                    fmt = '%d'

                # write values one per line to keep it readable and streamable
                for i in range(n):
                    try:
                        fout.write(fmt % arr[i] + '\n')
                    except Exception:
                        fout.write(repr(arr[i]) + '\n')

                if not full and arr.size > n:
                    fout.write(f"... (truncated, {arr.size - n} more elements)\n")

            fout.write('\n')
            tried.append(name)

        if not tried:
            fout.write('No dtype interpretation succeeded. File may be non-numeric or corrupted.\n')

    return out_path


def main():
    parser = argparse.ArgumentParser(description='Dump .bin files to human-readable text')
    parser.add_argument('--input-dir', '-i', required=True, help='Input directory containing .bin files')
    parser.add_argument('--out-dir', '-o', required=False, help='Output directory for .txt files (defaults to <input-dir>/text)')
    parser.add_argument('--max-elements', type=int, default=100000, help='Max values to write per dtype (default 100000)')
    parser.add_argument('--full', action='store_true', help='Write entire arrays (may produce very large files)')
    parser.add_argument('--dtype', choices=[name for _, name in CANDIDATE_DTYPES], help='Only try a single dtype (e.g. float32, int32)')
    parser.add_argument('--pattern', default='*.bin', help='Glob pattern to match files (default *.bin)')

    args = parser.parse_args()
    inp = Path(args.input_dir)
    if not inp.exists():
        print(f"Input directory does not exist: {inp}", file=sys.stderr)
        sys.exit(2)
    out = Path(args.out_dir) if args.out_dir else inp / 'text'

    if args.dtype:
        # filter candidate list to only that dtype
        dtypes = [(dt, name) for dt, name in CANDIDATE_DTYPES if name == args.dtype]
    else:
        dtypes = CANDIDATE_DTYPES

    files = sorted(inp.glob(args.pattern))
    if not files:
        print(f"No files matching {args.pattern} in {inp}")
        sys.exit(0)

    for f in files:
        try:
            print(f"Processing {f} -> {out}")
            out_path = dump_file(f, out, dtypes=dtypes, max_elements=args.max_elements, full=args.full)
            print(f"Wrote: {out_path}")
        except Exception as e:
            print(f"Failed to process {f}: {e}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Analyze gradient .bin files for trend features:
- Supports files written by the training exporter: [uint64 count][float32 * count]
- Also supports raw float32 arrays saved with no header.

For each file computes:
- basic stats (count, mean, std, min, max)
- largest single increase/decrease (value and index)
- longest monotonic increasing/decreasing run (length, total delta, start index)
- linear regression slope (trend) and R^2
- number of trend reversals and last reversal index/value

Outputs JSON summary and a CSV summary row to an output directory (default: <input>/analysis).

Usage:
    python analyze_gradient_trends.py -i "D:/G.R.I.M/gradient_dumps" 

Optional args:
    --pattern (glob pattern, default *.bin)
    --out-dir
    --min-run-length (ignore runs shorter than this when reporting longest run)
    --max-elements (truncate huge arrays for performance, default 5e6)

"""
from pathlib import Path
import argparse
import struct
import json
import csv
import numpy as np
from math import sqrt


def read_bin_file(path: Path):
    """Read .bin file written as: uint64 count followed by count float32 values
    If header doesn't match, fall back to reading entire file as float32 array.
    Returns numpy array of dtype float32.
    """
    data = path.read_bytes()
    nbytes = len(data)
    if nbytes < 4:
        return np.array([], dtype=np.float32)

    # try parse first 8 bytes as uint64 count (little-endian)
    if nbytes >= 8:
        count = struct.unpack_from('<Q', data, 0)[0]
        expected = 8 + count * 4
        if expected == nbytes:
            # valid header
            arr = np.frombuffer(data, dtype=np.float32, offset=8)
            return arr.copy()
        # sometimes size_t may be 4 bytes (unlikely on 64-bit build), try uint32
        count32 = struct.unpack_from('<I', data, 0)[0]
        expected32 = 4 + count32 * 4
        if expected32 == nbytes:
            arr = np.frombuffer(data, dtype=np.float32, offset=4)
            return arr.copy()

    # fallback: treat whole file as raw float32 array
    try:
        arr = np.frombuffer(data, dtype=np.float32)
        return arr.copy()
    except Exception:
        return np.array([], dtype=np.float32)


def longest_monotonic_run(diff, direction=1, min_len=1):
    """Given diff = arr[i+1]-arr[i], find longest consecutive run where sign(diff)==direction (direction=1 for increasing, -1 for decreasing)
    Returns (length, total_delta, start_index_of_run)
    start_index refers to index in original arr where run begins.
    """
    best_len = 0
    best_total = 0.0
    best_start = None
    cur_len = 0
    cur_total = 0.0
    cur_start = 0
    for i, d in enumerate(diff):
        ok = (d > 0) if direction == 1 else (d < 0)
        if ok:
            if cur_len == 0:
                cur_start = i
                cur_total = d
                cur_len = 1
            else:
                cur_len += 1
                cur_total += d
        else:
            if cur_len >= min_len and cur_len > best_len:
                best_len = cur_len
                best_total = cur_total
                best_start = cur_start
            cur_len = 0
            cur_total = 0.0
    # tail
    if cur_len >= min_len and cur_len > best_len:
        best_len = cur_len
        best_total = cur_total
        best_start = cur_start
    return best_len, float(best_total), (int(best_start) if best_start is not None else None)


def linear_trend(arr):
    """Compute linear slope (per-index) and R^2 via least squares fit to arr.
    Returns slope, intercept, r2
    """
    n = arr.size
    if n < 2:
        return 0.0, float(arr[0]) if n==1 else 0.0, 0.0
    x = np.arange(n, dtype=np.float64)
    y = arr.astype(np.float64)
    xm = x.mean()
    ym = y.mean()
    xv = np.sum((x - xm) ** 2)
    cov = np.sum((x - xm) * (y - ym))
    slope = cov / xv if xv != 0 else 0.0
    intercept = ym - slope * xm
    ss_tot = np.sum((y - ym) ** 2)
    ss_res = np.sum((y - (slope * x + intercept)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
    return float(slope), float(intercept), float(r2)


def analyze_array(arr: np.ndarray, min_run_len=2):
    out = {}
    n = int(arr.size)
    out['count'] = n
    if n == 0:
        return out
    # basic stats
    a64 = arr.astype(np.float64)
    out['mean'] = float(a64.mean())
    out['std'] = float(a64.std())
    out['min'] = float(a64.min())
    out['max'] = float(a64.max())

    # diffs
    if n >= 2:
        diff = np.diff(a64)
        # largest single increase/decrease
        max_inc = diff.max()
        max_inc_idx = int(np.argmax(diff))
        max_dec = diff.min()
        max_dec_idx = int(np.argmin(diff))
        out['max_single_increase'] = {'value': float(max_inc), 'index': max_inc_idx}
        out['max_single_decrease'] = {'value': float(max_dec), 'index': max_dec_idx}

        # longest monotonic runs
        inc_len, inc_total, inc_start = longest_monotonic_run(diff, direction=1, min_len=min_run_len)
        dec_len, dec_total, dec_start = longest_monotonic_run(diff, direction=-1, min_len=min_run_len)
        out['longest_increasing_run'] = {'length': inc_len, 'total_delta': inc_total, 'start_index': inc_start}
        out['longest_decreasing_run'] = {'length': dec_len, 'total_delta': dec_total, 'start_index': dec_start}

        # trend reversals: number of sign changes in diff (ignoring zeros)
        signs = np.sign(diff)
        nonzero = signs[signs != 0]
        if nonzero.size == 0:
            reversals = 0
        else:
            reversals = int(np.sum(nonzero[1:] * nonzero[:-1] < 0))
        out['trend_reversals'] = reversals

        # last reversal index and info
        last_rev_idx = None
        for i in range(len(signs)-1, 0, -1):
            if signs[i] * signs[i-1] < 0:
                last_rev_idx = i
                break
        if last_rev_idx is not None:
            out['last_reversal_index'] = int(last_rev_idx)
            out['last_reversal_value_before'] = float(a64[last_rev_idx])
            out['last_reversal_value_after'] = float(a64[last_rev_idx+1]) if last_rev_idx + 1 < n else None
        else:
            out['last_reversal_index'] = None

    else:
        out['max_single_increase'] = None
        out['max_single_decrease'] = None
        out['longest_increasing_run'] = None
        out['longest_decreasing_run'] = None
        out['trend_reversals'] = 0
        out['last_reversal_index'] = None

    # linear trend
    slope, intercept, r2 = linear_trend(a64)
    out['linear_trend'] = {'slope': slope, 'intercept': intercept, 'r2': r2}

    return out


def process_file(path: Path, out_dir: Path, min_run_len=2, max_elements=5000000):
    arr = read_bin_file(path)
    if arr.size == 0:
        print(f"Warning: {path} appears empty or unreadable")
    # optionally truncate very large arrays
    if arr.size > max_elements:
        print(f"Truncating {path.name} from {arr.size} to {int(max_elements)} elements for speed")
        arr = arr[:int(max_elements)].copy()
    summary = analyze_array(arr, min_run_len=min_run_len)
    out_dir.mkdir(parents=True, exist_ok=True)
    # write json
    jpath = out_dir / (path.stem + '.json')
    with jpath.open('w', encoding='utf-8') as jf:
        json.dump({'file': str(path), 'summary': summary}, jf, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description='Analyze gradient .bin files for trend features')
    parser.add_argument('--input-dir', '-i', required=True, help='Input directory containing .bin files')
    parser.add_argument('--pattern', default='*.bin', help='Glob pattern (default: *.bin)')
    parser.add_argument('--out-dir', '-o', help='Output directory for summaries (default: <input>/analysis)')
    parser.add_argument('--min-run-length', type=int, default=2, help='Minimum run length to consider')
    parser.add_argument('--max-elements', type=int, default=5000000, help='Max elements to analyze per file to avoid memory issues')
    args = parser.parse_args()

    inp = Path(args.input_dir)
    if not inp.exists():
        print(f"Input dir does not exist: {inp}")
        return
    out = Path(args.out_dir) if args.out_dir else inp / 'analysis'

    files = sorted(inp.glob(args.pattern))
    if not files:
        print(f"No files matching {args.pattern} in {inp}")
        return

    # prepare CSV summary
    csv_path = out / 'summary.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'file','count','mean','std','min','max',
        'max_single_increase_val','max_single_increase_idx',
        'max_single_decrease_val','max_single_decrease_idx',
        'longest_inc_len','longest_inc_total','longest_inc_start',
        'longest_dec_len','longest_dec_total','longest_dec_start',
        'trend_reversals','last_reversal_index','linear_slope','linear_r2'
    ]
    with csv_path.open('w', newline='', encoding='utf-8') as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for f in files:
            print(f"Processing {f}")
            summary = process_file(f, out, min_run_len=args.min_run_length, max_elements=args.max_elements)
            row = {
                'file': str(f.name),
                'count': summary.get('count'),
                'mean': summary.get('mean'),
                'std': summary.get('std'),
                'min': summary.get('min'),
                'max': summary.get('max'),
            }
            msi = summary.get('max_single_increase') or {}
            msd = summary.get('max_single_decrease') or {}
            lir = summary.get('longest_increasing_run') or {}
            ldr = summary.get('longest_decreasing_run') or {}
            lt = summary.get('linear_trend') or {}
            row.update({
                'max_single_increase_val': msi.get('value'),
                'max_single_increase_idx': msi.get('index'),
                'max_single_decrease_val': msd.get('value'),
                'max_single_decrease_idx': msd.get('index'),
                'longest_inc_len': lir.get('length'),
                'longest_inc_total': lir.get('total_delta'),
                'longest_inc_start': lir.get('start_index'),
                'longest_dec_len': ldr.get('length'),
                'longest_dec_total': ldr.get('total_delta'),
                'longest_dec_start': ldr.get('start_index'),
                'trend_reversals': summary.get('trend_reversals'),
                'last_reversal_index': summary.get('last_reversal_index'),
                'linear_slope': lt.get('slope'),
                'linear_r2': lt.get('r2')
            })
            writer.writerow(row)
    print(f"Wrote CSV summary to {csv_path}")

if __name__ == '__main__':
    main()

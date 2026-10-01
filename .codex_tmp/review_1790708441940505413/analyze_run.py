"""Offline log/CSV analysis only; no model or training execution."""
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import json
from pathlib import Path
import re
import statistics

root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
logs = root / 'resources/models/GRIM-text/training/logs'
text = (logs / 'training_1790708441940505413.log').read_text(encoding='utf-8')
expected = {
    'decrease_removed': ('-', 120, 84), 'decrease_start': ('+', 36, 84),
    'decrease_remaining': ('-', 120, 36), 'increase_start': ('-', 42, 19),
    'increase_added': ('-', 42, 23), 'increase_final': ('+', 23, 19),
    'comparison_smaller': ('-', 50, 18), 'comparison_difference': ('-', 50, 32),
    'comparison_larger': ('+', 32, 18), 'groups_group_count': ('/', 72, 9),
    'groups_per_group': ('/', 72, 8), 'groups_total': ('*', 8, 9),
}
samples = []
for match in re.finditer(r'^\[([^\n]+)\] \[Sample\] step=(\d+)([^\n]*)\n', text, re.M):
    end = text.find('\n[', match.end() + 1)
    body = text[match.end():end if end >= 0 else len(text)]
    case = re.search(r'case=(\w+)', match[3])[1]
    determine = re.search(r'<determine>\s*(.*?)\s*</determine>', body, re.S)
    bindings = dict((name, Decimal(value)) for name, value in
                    re.findall(r'\$\{(\w+)\}\s*->\s*(\d+(?:\.\d+)?);', body))
    tool = re.search(r'<TOOL>\s*\(?\s*\$\{(\w+)\}\s*([+*/-])\s*\$\{(\w+)\}\s*\)?\s*</TOOL>\s*->\s*\$\{(\w+)\}', body)
    correct = False
    op = None
    if tool:
        left, op, right, result = tool.groups()
        wanted, a, b = expected[case]
        operands = (bindings.get(left), bindings.get(right))
        correct = op == wanted and (operands == (a, b) or (op in '+*' and operands == (b, a)))
    samples.append(dict(step=int(match[2]), case=case, line=text.count('\n', 0, match.start()) + 1,
                        body=body, determine=determine[1] if determine else '',
                        parsed_operator=op, correct_expression=correct,
                        complete_sections=all(f'</{tag}>' in body for tag in ('determine','define','execute','answer'))))
(out / 'samples.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in samples), encoding='utf-8')
streams = defaultdict(list)
with (logs / 'telemetry_1790708441940505413.csv').open(newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        if row['level'] == '0':
            streams[row['stream_name']].append((int(row['global_step']), float(row['raw_observation'])))
stats = {}
for name, rows in streams.items():
    values = [v for _, v in rows]
    stats[name] = dict(count=len(values), first=values[0], last=values[-1],
                       min=min(values), max=max(values), median=statistics.median(values),
                       last500_mean=statistics.mean(values[-500:]))
loss = streams['loss']
failures = text.count('[Sample] generation failed:')
last = {s['case']: s for s in samples}
summary = dict(successful_generations=len(samples), failed_generations=failures,
               attempted_generations=len(samples)+failures,
               optimizer_range=[samples[0]['step'], samples[-1]['step']],
               telemetry_step_range=[loss[0][0], loss[-1][0]],
               last60_operator_counts=Counter(s['parsed_operator'] or 'unparseable' for s in samples[-60:]),
               last60_correct_expressions=sum(s['correct_expression'] for s in samples[-60:]),
               last60_complete_sections=sum(s['complete_sections'] for s in samples[-60:]),
               latest_case_expression_correct={k:v['correct_expression'] for k,v in last.items()},
               latest_case_lines={k:v['line'] for k,v in last.items()},
               determine_counts=Counter(s['determine'] for s in samples),
               first_loss_below_001=next((row for row in loss if row[1] < 0.001), None),
               streams=stats)
(out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
short = {k:v for k,v in summary.items() if k not in ('streams','determine_counts')}
short['selected_streams'] = {k:v for k,v in stats.items() if k in
    ('loss','learning_rate','grad_norm_mean','rho_final','rho_atom_only','h_rms_growth',
     'rms_gamma_pre_attn_rms','rms_gamma_pre_ffn_rms','rms_gamma_final_rms','rho_raw_rms_spread')}
print(json.dumps(short, indent=2))

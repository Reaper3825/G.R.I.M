"""Read-only, bounded-memory audit; run over SSH stdin, never starts training."""
import collections
import json
import mmap
from pathlib import Path
import struct

root = Path('/ocean/projects/cis250124p/uwadkins/G.R.I.M')
data = root / 'resources/models/GRIM-text/training/data'
fields = {'id': 0, 'name': 1, 'prompt': 2, 'answer': 4,
          'source_sequence_id': 10, 'determine': 16, 'execute': 17, 'define': 19}
report = {}
with (data / 'concept_blocks.fb').open('rb') as file:
    with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as buf:
        def scalar(fmt, pos):
            return struct.unpack_from(fmt, buf, pos)[0]
        def field(table, index):
            vt = table - scalar('<i', table)
            offset = 4 + 2 * index
            if offset >= scalar('<H', vt):
                return None
            relative = scalar('<H', vt + offset)
            return table + relative if relative else None
        def indirect(pos):
            return pos + scalar('<I', pos)
        def string(table, index):
            ptr = field(table, index)
            if ptr is None:
                return ''
            start = indirect(ptr)
            size = scalar('<I', start)
            return buf[start + 4:start + 4 + size].decode('utf-8')
        assert buf[4:8] == b'GRCB'
        table = scalar('<I', 0)
        vec = indirect(field(table, 1))
        size = scalar('<I', vec)
        counts = collections.Counter()
        determines = collections.Counter()
        examples = []
        # The original v1 append placed its 60k rows at the end. Inspect a
        # bounded window instead of paging the whole large FlatBuffer remotely.
        for i in range(max(0, size - 60000), min(size, max(0, size - 60000) + 24)):
            block = indirect(vec + 4 + i * 4)
            block_id = string(block, 0)
            if not block_id.startswith('ssatv1_'):
                continue
            provenance = string(block, 10)
            counts[provenance.split(':')[0]] += 1
            determines[string(block, 16)] += 1
            if len(examples) < 4:
                example = {name: string(block, index) for name, index in fields.items()}
                gp = field(block, 12)
                example['goal_target'] = string(indirect(gp), 0) if gp else ''
                examples.append(example)
        report['flatbuffer'] = dict(total_blocks=size, sampled_provenance_counts=counts,
                                   determine_counts=determines, examples=examples)

counts = collections.Counter()
examples = []
with (data / 'concept_blocks.jsonl').open('rb') as file:
    file.seek(max(0, (data / 'concept_blocks.jsonl').stat().st_size - 1000000))
    file.readline()  # Discard a possible partial row.
    for line in file:
        if b'ssatv1_' not in line:
            continue
        row = json.loads(line)
        if not row.get('id', '').startswith('ssatv1_'):
            continue
        counts[row['source_sequence_id'].split(':')[0]] += 1
        examples.append({name: row.get(name) for name in fields})
        if len(examples) == 4:
            break
report['jsonl'] = dict(sampled_provenance_counts=counts, examples=examples)

# Decode the first GRMT row directly from its frozen vocabulary. The v34
# corpus-level layout precedes sequence lengths. Only the token array is read.
with (data / 'vocab.bin').open('rb') as file:
    assert file.read(4) in (b'KTMG', b'GMTK')
    version = struct.unpack('<H', file.read(2))[0]
    file.read(4)
    count = struct.unpack('<I', file.read(4))[0]
    file.read(4 + 3 + 4)
    pieces = {}
    for i in range(count):
        size = struct.unpack('<I', file.read(4))[0]
        text = file.read(size).decode('utf-8')
        file.read(4)
        stored_id = struct.unpack('<I', file.read(4))[0] if version >= 3 else 277 + i
        if version >= 9:
            file.read(1)
        pieces[stored_id] = text
with (data / 'single_step_arithmetic_tool_v1.grmt').open('rb') as file:
    header = struct.unpack('<IIII', file.read(16))
    assert header[1] == 34
    size = struct.unpack('<I', file.read(4))[0]
    assert size < 1048576
    layout = file.read(size).decode('utf-8')
    id_length = struct.unpack('<I', file.read(4))[0]
    assert id_length < 10000
    block_id = file.read(id_length).decode('utf-8')
    length = struct.unpack('<I', file.read(4))[0]
    assert length < 10000
    tokens = struct.unpack('<' + 'I' * length, file.read(length * 4))
    assert all(t < header[3] for t in tokens)
    decoded = ''.join(pieces.get(t, chr(t - 4) if 4 <= t < 260 else '<ID%d>' % t)
                      for t in tokens).replace('\u2581', ' ')
    report['grmt'] = dict(header=header, first_concept_block_id=block_id,
                         first_sequence_length=length, first_sequence_decoded=decoded)
print(json.dumps(report, indent=2))

import json
import os
import struct
from pathlib import Path


def read_grmt_header(path: Path):
    with path.open('rb') as f:
        header = f.read(16)
    if len(header) < 16:
        raise ValueError(f"GRMT too small: {path}")
    magic, version, num_sequences, vocab_size = struct.unpack('<IIII', header)
    return {
        'magic_hex': hex(magic),
        'magic_ascii': header[:4][::-1].decode('latin1', errors='replace'),  # little-end view
        'version': version,
        'num_sequences': num_sequences,
        'vocab_size': vocab_size,
    }


def read_vocab_bin_header(path: Path):
    with path.open('rb') as f:
        magic = f.read(4)
        if len(magic) != 4:
            raise ValueError(f"vocab.bin too small: {path}")
        (version,) = struct.unpack('<H', f.read(2))
        out = {
            'magic_ascii': magic.decode('latin1', errors='replace'),
            'magic_hex': magic.hex(),
            'version': version,
        }
        # v2 is the intended format. Some older files were written with v2 fields
        # but stamped as version=1; treat v1 as compatible for inspection.
        if version >= 2 or version == 1:
            checksum = struct.unpack('<I', f.read(4))[0]
            config_vocab_size = struct.unpack('<i', f.read(4))[0]
            max_length = struct.unpack('<i', f.read(4))[0]
            # 3 bools (written as sizeof(bool) in C++; on MSVC it's 1 byte)
            flags = f.read(3)
            actual_vocab_size = struct.unpack('<I', f.read(4))[0]
            out.update({
                'checksum': checksum,
                'config_vocab_size': config_vocab_size,
                'max_length': max_length,
                'flags_bytes': flags.hex(),
                'actual_vocab_size': actual_vocab_size,
            })
        return out


def main():
    root = Path(__file__).resolve().parent
    config_path = root / 'ai_config.json'
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    grim_paths = cfg.get('paths', {}).get('grim_text', {})

    training_data = Path(grim_paths.get('training_data', ''))
    vocab_bin = Path(grim_paths.get('vocab', ''))
    model_bin = Path(grim_paths.get('model', ''))

    print(f"Config: {config_path}")
    print(f"training_data: {training_data}")
    print(f"vocab.bin: {vocab_bin}")
    print(f"model: {model_bin}")

    ok = True

    if training_data.exists():
        grmt = read_grmt_header(training_data)
        print("\nGRMT header:")
        for k, v in grmt.items():
            print(f"  {k}: {v}")
    else:
        print("\nGRMT header: MISSING")
        ok = False
        grmt = None

    if vocab_bin.exists():
        vb = read_vocab_bin_header(vocab_bin)
        print("\nVocab.bin header:")
        for k, v in vb.items():
            print(f"  {k}: {v}")
    else:
        print("\nVocab.bin header: MISSING")
        ok = False
        vb = None

    if grmt and vb and 'actual_vocab_size' in vb:
        if grmt['vocab_size'] != vb['actual_vocab_size']:
            ok = False
            print("\nMISMATCH: training_data.grmt vocab_size != vocab.bin actual_vocab_size")
        else:
            print("\nOK: training_data.grmt vocab_size matches vocab.bin actual_vocab_size")

    # We can't reliably parse grim_text.bin without knowing its format here.
    # But we can at least flag if it doesn't exist.
    if not model_bin.exists():
        print("\nNOTE: model file missing (can't validate model-vocab shape alignment)")

    print(f"\nRESULT: {'LIKELY SAFE TO TRAIN' if ok else 'NOT SAFE YET'}")


if __name__ == '__main__':
    main()

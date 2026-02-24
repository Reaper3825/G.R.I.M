"""Analyze vocab.txt for duplicates, near-duplicates, and other issues."""
import re
from collections import defaultdict

tokens = {}
with open('resources/models/GRIM-text/training/data/vocab.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.rstrip('\n')
        if '\t' in line:
            parts = line.split('\t')
            tok = parts[0]
            score = float(parts[1])
            if tok in tokens:
                print(f'EXACT DUPLICATE: "{tok}" scores: {tokens[tok]} and {score}')
            tokens[tok] = score

print(f'Total tokens: {len(tokens)}')
print()

# 1. Find tokens that only differ by trailing/leading punctuation
punct_stripped = defaultdict(list)
for tok in tokens:
    stripped = tok.strip('.,;:!?()[]{}/ -')
    if stripped and stripped != tok and len(stripped) >= 2:
        punct_stripped[stripped].append(tok)

print('=== TOKENS THAT ARE SAME WORD WITH PUNCTUATION VARIANTS ===')
count = 0
for base, variants in sorted(punct_stripped.items()):
    all_forms = []
    if base in tokens:
        all_forms.append(base)
    all_forms.extend(variants)
    if len(all_forms) >= 2:
        items = [(t, f'{tokens[t]:.4f}') for t in all_forms]
        print(f'  Base "{base}": {items}')
        count += 1
print(f'  Total groups with punctuation variants: {count}')
print()

# 2. Tokens that are identical words (full words appearing multiple times with different affixes)
print('=== MULTI-WORD TOKENS THAT CONTAIN SINGLE-WORD TOKENS ===')
multi_word = [(tok, s) for tok, s in tokens.items() if ' ' in tok]
single_word = {tok: s for tok, s in tokens.items() if ' ' not in tok}
for mw, ms in sorted(multi_word, key=lambda x: x[0]):
    words = mw.split()
    contained = [w for w in words if w in single_word and len(w) >= 3]
    if contained:
        print(f'  "{mw}" ({ms:.4f}) contains: {[(w, f"{single_word[w]:.4f}") for w in contained]}')
print()

# 3. Find near-duplicate substrings (token A is prefix/suffix of token B, very close)
print('=== TOKENS WHERE ONE IS OTHER + 1 CHAR (potential waste) ===')
token_set = set(tokens.keys())
overlaps = []
for tok in tokens:
    if len(tok) >= 3:
        prefix = tok[:-1]
        if prefix in token_set and len(prefix) >= 2:
            overlaps.append((prefix, tok, tokens[prefix], tokens[tok]))

overlaps.sort(key=lambda x: x[0])
for prefix, full, s1, s2 in overlaps:
    print(f'  "{prefix}" ({s1:.4f}) -> "{full}" ({s2:.4f})')
print(f'  Total overlapping pairs: {len(overlaps)}')
print()

# 4. Astronomy-specific domain tokens that may waste space for general use
print('=== DOMAIN-SPECIFIC (ASTRONOMY) TOKENS ===')
astro_terms = [
    'stellar', 'solar', 'orbital', 'dust', 'density', 'radius', 'flux',
    'velocity', 'saturn', 'neptune', 'jupiter', 'mars', 'moon', 'comet',
    'asteroid', 'planet', 'telescope', 'spectral', 'albedo', 'eclips',
    'accretion', 'photometr', 'icarus', 'mnras', 'apj', 'keplerian',
    'exoplanet', 'protoplanet', 'planetesimal', 'helio', 'semimajor',
    'eccentricit', 'inclination', 'torque', 'gravitation', 'tidal',
    'migration', 'luminosit', 'dwarf', 'transiting', 'transit', 'wasp',
    'meteorit', 'asteroid', 'grains', 'disks', 'discs', 'debris',
    'corot', 'pulsar', 'quasar', 'nebula', 'galax',
    'the disk', 'the star', 'the mass', 'the planet', 'the transit',
    'the stellar', 'the solar', 'the orbital', 'the dust', 'the inner',
    'the outer', 'the radial', 'the disc', 'light curve', 'solar system',
    'surface density', 'angular momentum', 'semi-major axis', 'semi-major',
    'of the planet', 'the planetary', 'the evolution',
]
found_astro = []
for tok, score in sorted(tokens.items()):
    for term in astro_terms:
        if tok == term or tok.startswith(term) or term.startswith(tok):
            found_astro.append((tok, score))
            break
    else:
        # Check full words
        if any(term in tok for term in ['planet', 'orbit', 'stellar', 'solar', 'lunar',
                                         'comet', 'astro', 'meteor', 'spit', 'mnras',
                                         'icarus', 'apj', 'keplerian']):
            found_astro.append((tok, score))

print(f'  Found {len(found_astro)} astronomy-specific tokens:')
for tok, score in sorted(found_astro, key=lambda x: x[1]):
    print(f'    "{tok}" ({score:.4f})')
print()

# 5. Fragmented tokens with very similar log-probs that likely compete
print('=== VERY FRAGMENTED OVERLAPPING SEQUENCES (3+ tokens form a chain) ===')
chains = defaultdict(list)
for tok in tokens:
    if len(tok) >= 4:
        for i in range(2, len(tok)):
            sub = tok[:i]
            if sub in token_set:
                chains[tok].append(sub)

for tok, subs in sorted(chains.items(), key=lambda x: -len(x[1])):
    if len(subs) >= 3:
        sub_info = [(s, f'{tokens[s]:.2f}') for s in subs]
        print(f'  "{tok}" ({tokens[tok]:.2f}) has sub-tokens: {sub_info}')

print()

# 6. Suspicious patterns - tokens ending with fragments that suggest training data issues
print('=== TOKENS THAT LOOK LIKE TRUNCATED WORDS (potential training data artifacts) ===')
trunc_count = 0
for tok, score in sorted(tokens.items()):
    # Long tokens that end mid-word (not at morpheme boundary)
    if len(tok) >= 6 and not tok.endswith(('.', ',', ' ', '-')) and tok[-1].isalpha():
        # Check if any token starts with this and extends it
        extensions = [t for t in token_set if t.startswith(tok) and t != tok]
        if not extensions and tok not in ['figure', 'which', 'these', 'between', 'have',
                                           'from', 'where', 'more', 'also', 'using',
                                           'were', 'their', 'this', 'that', 'will',
                                           'data', 'than']:
            # Likely a truncated word
            if trunc_count < 80:
                print(f'  "{tok}" ({score:.4f})')
            trunc_count += 1

print(f'  Total truncated-looking tokens: {trunc_count}')

# 7. Check for the specific pathological duplication patterns
print()
print('=== PATHOLOGICAL FRAGMENTATION (same word appears as multiple overlapping pieces) ===')
# Look for cases like "metallicity" being split many ways
problem_words = defaultdict(list)
for tok in tokens:
    if len(tok) >= 5:
        for tok2 in tokens:
            if tok2 != tok and len(tok2) >= 5 and tok in tok2 and abs(len(tok) - len(tok2)) <= 3:
                problem_words[tok2].append(tok)

for word, fragments in sorted(problem_words.items(), key=lambda x: -len(x[1])):
    if len(fragments) >= 2:
        frag_info = [(f, f'{tokens[f]:.2f}') for f in fragments]
        print(f'  "{word}" ({tokens[word]:.2f}) overlaps with: {frag_info}')

# Hugging Face PT pilot

`scripts/generate_hf_pt.py` collects a bounded sample from three datasets:

| Dataset/config | Course | Pilot sequences |
| --- | --- | ---: |
| HuggingFaceTB/cosmopedia / wikihow | Practical how-to (Cosmopedia) | 400 |
| HuggingFaceTB/smollm-corpus / cosmopedia-v2 | General education (Cosmopedia v2) | 400 |
| HuggingFaceTB/finemath / finemath-4plus | Mathematics (FineMath 4+) | 400 |

Cosmopedia articles are synthetic. The wikiHow subset contains generated tutorials,
not original wikiHow articles. FineMath is filtered web text. The dataset license
declaration and upstream metadata are recorded; these are not a per-document rights audit.

Run from the repository root using existing Python; no new dependencies:

```powershell
python scripts/generate_hf_pt.py
python scripts/verify_hf_pt.py --append
python -m unittest discover -s scripts -p "test_generate_*pt.py"
```

The first command stages. The second independently verifies every chunk against its
cached source slice, document digest, and local vocabulary counts, then appends
offline to `Pre-Trainingv1`, maintaining course, registry, and manifest membership.
Omit `--append` from the verifier for a read-only corpus check. Alternatively,
`generate_hf_pt.py --append` collects and appends in one run.
It uses the existing append journal/lock and rollback mechanism. Each course is a
separate transaction; interrupted multi-course runs can be retried. Identical existing
text is reused and already-present PT members are not added again. Existing members
are not reassigned between courses.

## Sampling and curation

- Fixed seed; randomly selected 50-row pages across each source, rather than just
  the beginning of a dataset. This is page sampling, not independent document sampling.
- Viewer `x-revision` must match the Hub commit. Complete responses are cached under
  revision-specific keys. A stale viewer aborts rather than silently mixing versions.
- Only the source `text` enters PT. Generation prompts, original URLs, audience,
  license declaration, scores, and row identifiers remain in provenance.
- Practical how-to selection uses title keywords for organization, household tasks,
  planning, travel, cooking, money, studying, and communication. This heuristic does
  not establish balanced coverage of those topics.
- FineMath additionally uses a host allowlist of educational/math references and
  `.edu`/`.ac.uk` domains. Domain selection improves relevance but does not certify
  factual correctness or remove every extraction defect.
- Rejects truncated rows, short documents, selected essay-service spam, replacement
  characters, authored atom delimiters, heavy non-Latin text, excessive repeated
  lines, topic indexes, selected worksheet SEO patterns, and severely fragmented extraction. These checks do not establish factual
  correctness or fully eliminate spam and malformed formulas.
- Uses the local KTMG v9 vocabulary CPU counter. Each chunk has 128–1022 content
  tokens, reserving two boundary tokens within the 1024 maximum. Native runtime
  parity is not tested by this utility.
- Preserves source paragraph/heading structure where it fits. Splits oversized
  prose at line/whitespace boundaries; skips oversized/unclosed code fences.
  No cross-document packing or overlap. Up to four accepted chunks per document;
  long documents and short tails can be partially omitted. Later chunks may depend
  on earlier context, so these are PT passages, not standalone instruction examples.
- Exact text deduplication across the corpus; no semantic or near-duplicate pass.

The default output is
`resources/models/GRIM-text/training/data/hf_pt_provenance/pilot_v1/`.
`report.json` describes the latest successful run. Its `artifacts` path points to
versioned candidates, document metadata, provenance, and report. Source snapshots
and cached original responses remain under the output root for audit.

## Recommended next increment

After reviewing the pilot, grow these three sources to **10,000 total sequences**:

| Source | Share of this increment | Target sequences |
| --- | ---: | ---: |
| Practical Cosmopedia how-to | 40% | 4,000 |
| Cosmopedia v2 education | 40% | 4,000 |
| FineMath 4+ | 20% | 2,000 |

These targets include the pilot; they imply 8,800 additional unique sequences if
all 1,200 pilot rows are newly added. They are a proposed expansion mix, not a
replacement for the full FineWeb/developer/culture/communication curriculum.
At the 1024 cap this increment has at most 10.24M token positions; actual useful
tokens depend on passage lengths. Source proportions should ultimately be monitored
by content tokens as well as sequence count.

Keep FineWeb as the broad-language base. These sources add educational prose and
practical knowledge but do not teach calendar/email tools or conversational behavior
by themselves. Dedicated assistant behavior examples belong in subsequent training.

For evaluation, reserve 2% of document families (approximately 98/1/1 train/validation/test).
Group chunks from the same document and related synthetic prompts/seeds together;
near-duplicate families must not cross splits. This importer does not create or alter
trainer evaluation splits. All appended rows are currently PT curriculum candidates.
Before a larger run, add family-aware splitting and stronger topic/near-duplicate curation.
No training or model runtime is launched by the importer.

## Expanding the pilot to 10,000 entries

The requested second batch adds 3,600 how-to, 3,600 education, and 1,600 mathematics
entries. Existing corpus content is excluded before counting these quotas, so
the batch contains 8,800 novel entries, rather than repeating the pilot.

```powershell
python scripts/generate_hf_pt.py --howto-sequences 3600 --education-sequences 3600 --math-sequences 1600 --exclude-existing --page-size 100 --max-pages 1000 --output resources/models/GRIM-text/training/data/hf_pt_provenance/expansion_8800
python scripts/verify_hf_pt.py --output resources/models/GRIM-text/training/data/hf_pt_provenance/expansion_8800 --append
```

To retry an already staged batch, rerun the verifier/append command. Re-running
collection with `--exclude-existing` after a successful append requests another
batch of novel content; it does not mean "bring the total to 10,000."

Under section 8 of `docs/GRIM_REASONING_DATA_CONTRACT.md`, nonempty `raw` replaces
structured rendering. The verifier requires `format_type: "raw"`, nonempty raw
text, and empty or absent prompt, answer, goal, state, reasoning phase, and legacy
reasoning fields. The destination must have `training_stage: "pt"` and
`format_as_concept: false`. IDs are added to their source course and to the flattened
curriculum membership in both the registry and `Pre-Trainingv1.json`.

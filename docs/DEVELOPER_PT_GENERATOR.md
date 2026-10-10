# Developer PT generator

`scripts/generate_developer_pt.py` collects versioned developer documents and
produces raw ConceptBlocks for the existing `Pre-Trainingv1` curriculum. It uses
only the installed Python standard library and the repository's CPU Viterbi
helper. It does not execute downloaded examples or run model/training targets.

## Run

From the repository root, stage a bounded collection:

```powershell
python scripts/generate_developer_pt.py --max-sequences 1000
```

Append that collection (cached source requests are reused):

```powershell
python scripts/generate_developer_pt.py --max-sequences 1000 --append
```

Choose particular sources and increase collection limits:

```powershell
python scripts/generate_developer_pt.py --source python-docs --source pytest-docs --max-documents-per-source 100 --max-sequences-per-source 1000 --max-sequences 2000 --append
```

Defaults:

- Vocabulary: `resources/models/GRIM-text/training/data/vocab.bin`.
- Target: `Pre-Trainingv1`, under that same data directory.
- Sequence cap: 1,024 tokens, reserving two for BOS/EOS.
- Minimum: 64 content tokens.
- Per source: at most 20 documents and 200 accepted sequences.
- Overall: at most 1,000 accepted sequences.
- Sources: `DataCollection/developer_pt_sources.json`.
- Staging/cache: `temp/developer_pt`.

Counts are upper bounds, not promised yields. The overall limit is also divided
equally among selected source streams; unused quotas are not filled by repeating
documents. A repeat run with identical settings is idempotent. To expand further,
increase limits or add source paths. This version does not implement a resumable
cursor that automatically moves past previous collection limits.

## Sources and extraction

The starter manifest covers C++ Core Guidelines, Python tutorials/HOWTOs, CMake
guides, PowerShell conceptual/about documentation, and pytest guides. Python,
CMake, and pytest use explicit release tags. Moving branches are resolved to an
immutable commit and cached. To refresh one, use a new `--output` cache directory
or change its `ref` to a new explicit commit/tag.

Requests download repository trees, selected text files, and license files, not
whole repository checkouts. Set `GITHUB_TOKEN` in the environment if authenticated
GitHub API quota is needed; the script does not print it or send it to raw-content
URLs. GitHub tree truncation and missing sources fail explicitly.

Sphinx `include` and `literalinclude` content is fetched at the same commit,
including common line/marker selectors. Code directives become fenced blocks.
Raw HTML and table-of-contents directives are removed. Unsupported include
selectors stop the run instead of silently selecting incorrect code. Remaining
RST prose/roles are retained; this is intentionally not a complete Sphinx renderer.
Images are not interpreted. Available alt text is retained.

C++ collection begins at its first substantive philosophy rule. Markdown front
matter, empty anchors, and standalone internal navigation links are removed.
Indentation, Unicode, numeric literals, and scientific notation are retained.

Chunks are contiguous, non-overlapping slices of the cleaned document. Paragraphs
and fenced code blocks are preferred boundaries. Long prose falls back to line
and then character boundaries, re-counting every emitted slice. Oversized or
unclosed fenced code blocks are skipped and counted; no partial fenced example
is manufactured. Adjacent chunks retain a shared source identity in provenance.
Titles/headings are not automatically repeated on continuation chunks, so manual
review of context-dependent fragments remains necessary. No cross-document
packing or synthetic rewriting is performed.

## Token counting

`scripts/pt_vocab.py` reads the current KTMG v9 artifact, including exact-piece
flags and scores. It mirrors the unannotated raw-text path in `UniByte.cu`:

1. Convert horizontal spacing to the word-boundary marker; preserve LF tokens.
2. Encode decimal literals with the fixed numeric vocabulary (including grouped
   thousands, exponents, and repeated-digit tokens).
3. Match user-defined exact pieces with lexical end boundaries.
4. Apply float32 Viterbi and byte fallback to remaining text.

The shared `vocab_playground.py` Viterbi implementation is reused, but its older
vocabulary reader and numeric-atom pipeline are not used. Authored typed atom
delimiters cause the source document to be skipped for review. This script does
not infer or create atom annotations.

Each candidate is re-counted and must fit `content_tokens + 2 <= max_tokens`.
The vocabulary SHA-256 is recorded and checked again before append. A changed
vocabulary requires regeneration. Tests cover the CPU port and current format;
native CUDA tokenizer parity has not been executed as part of this utility work.

## Append integrity and provenance

The generator validates existing registry/course/manifest agreement and scans
the corpus without rewriting old rows. Exact normalized-content duplicates are
reused; new IDs use full SHA-256 hashes. It does not perform near-duplicate or
semantic deduplication, benchmark decontamination, or train/validation splitting.

New memberships go into a dedicated `Developer assistance` course. The target's
flat membership projection and standalone manifest are updated to match. Other
courses and metadata are preserved. Existing malformed JSON, duplicate IDs,
missing target members, and membership disagreement fail before writing.

An exclusive lock prevents concurrent instances of this generator. Stop other
DataHub/curriculum writers during append; their code does not share this lock.
The generator also checks for observed corpus/metadata changes before committing.

Appending uses a durable transaction journal, data fsync, and atomic metadata
replacement. Normal write failures roll back. After a process crash, ensure the
old process is stopped, remove only its stale `.developer_pt.lock`, then run:

```powershell
python scripts/generate_developer_pt.py --recover
```

Recovery refuses to truncate an unexpected corpus tail or overwrite metadata
changed by another writer. Preserve the journal for investigation in that case.

Each run has a content-addressed directory with:

- `candidates.jsonl`: generated raw ConceptBlocks.
- `documents.jsonl`: source URLs, immutable revisions, topics, hashes, include
  dependencies, and captured license text (once per source document).
- `provenance.jsonl`: candidate-to-document mapping, cleaned-text character
  offsets, and token counts.
- `report.json`: collection/rejection counts, vocabulary hash, and append result.

On append, these artifacts are also retained under
`training/data/developer_pt_provenance/<run-id>/`. The record is an intake audit,
not a blanket licensing assertion for every file in a repository.

## Offline input and validation

`--input-jsonl <path>` accepts already-rendered documents with non-empty `text`,
`title`, `source_url`, `revision`, and `licenses` fields. Optional `source_id`,
`topic`, and other metadata are preserved in the document manifest. External
includes in offline text are not fetched or interpreted.

Run focused tests:

```powershell
python -m unittest discover -s scripts -p test_generate_developer_pt.py -v
```

The first append is a small pipeline pilot, not the complete 300,000-sequence
developer allocation. Expand and review the source mix before a large release.

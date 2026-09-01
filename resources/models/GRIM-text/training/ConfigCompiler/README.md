# GRIM model configuration compiler

`compile_model_config` is a host-only compiler for immutable `.grimcfg`
artifacts. It reads the single human-authored root object in
`model_config.json`, computes model-derived values, validates cross-field
invariants, and emits the `GCFG` FlatBuffer
defined in `../schemas/grim_compiled_hyperparameters.fbs`.

It intentionally does not include or link `Shared/HyperParameters`, tokenizer
runtime classes, CUDA, or training startup code. Formula and validation changes
belong here and require a `semantic_version` increment.

## Standalone build

Configure with a package prefix or toolchain that supplies FlatBuffers/`flatc`
and nlohmann-json:

```text
cmake -S resources/models/GRIM-text/training/ConfigCompiler -B build/config-compiler \
  -DCMAKE_PREFIX_PATH=<dependency-prefix>
cmake --build build/config-compiler --target compile_model_config --config Release
```

The TrainingLoop build also exposes the same target through `add_subdirectory`.

## Usage

```text
compile_model_config \
  --input model_config.json \
  --output model.grimcfg
```

Vocabulary size is intentionally late-bound by training or inference startup
before model allocation. Exact vocabulary identity belongs to trained-weight or
checkpoint compatibility metadata, represented separately by
`../schemas/grim_model_manifest.fbs`, not to this reusable model preset. The
input must be the model object itself; `ai_config.json` and `training.config`
wrappers are not accepted.

`atom_insertion_enabled` selects the separate full-context byte-gap model. It
requires `causal_mask=false` and `max_seq_len > 1`; ordinary causal LM presets
must author it as `false`.

`local_atom_retrieval_enabled` selects the causal sequence-local typed-atom
retrieval path across batching, parameter ownership, training, validation, and
inference. It requires `causal_mask=true`,
`tokenizer_enable_atom_reasoning=true`, and
`atom_insertion_enabled=false`. Its training loss weight remains procedure
configuration and is not part of the compiled model artifact.

Artifact integrity is source-independent: SHA-256 and model-compatibility
xxHash64 are computed over the serialized artifact after normalizing all three
integrity values to zero. A loader can therefore verify a `.grimcfg` without
access to the source JSON or compiler formulas.

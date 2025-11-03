# Grammar-Based NLP Pipeline for G.R.I.M

This directory contains the grammar-based NLP system that uses treebank-derived rules for improved natural language understanding.

## Overview

The grammar-based NLP system uses:
- **Treebank annotations** (Universal Dependencies) for extracting linguistic patterns
- **Frequency-based weighting** to prioritize common constructions
- **Pipeline categorization** to route commands/queries/banter appropriately
- **FlatBuffer serialization** for efficient binary loading

## Quick Start

### 1. Install Dependencies

```bash
# Activate Python environment
.venv\Scripts\Activate.ps1

# Install required packages
pip install conllu flatbuffers
```

### 2. Use Pre-built Grammar (Already Done!)

The system now includes `resources/nlp_grammar.json` with starter rules. Your application should work immediately:

```bash
./out/build/release/grim.exe
```

You should see:
```
[DEBUG][GrammarParser] Loaded grammar: X components, Y verbs, Z objects...
[DEBUG][Main] Grammar parser initialized
```

## Advanced: Generate from Treebank Data

### Option A: Download Universal Dependencies Treebank

1. Download a UD treebank for English:
   - Visit: https://universaldependencies.org/
   - Download: `UD_English-EWT` (recommended)
   - Extract to: `data/treebanks/UD_English-EWT/`

2. Extract grammar rules:
```bash
python scripts/extract_grammar.py \
  --treebank data/treebanks/UD_English-EWT/en_ewt-ud-train.conllu \
  --output resources/nlp_grammar.json \
  --min-verb-freq 5 \
  --min-pattern-freq 10
```

### Option B: Use Your Own Annotated Data

If you have custom command/query data:

1. Annotate your data in CoNLL-U format using tools like:
   - Stanza: https://stanfordnlp.github.io/stanza/
   - spaCy with UD converter
   - Manual annotation

2. Run the extractor on your data:
```bash
python scripts/extract_grammar.py \
  --treebank data/my_commands.conllu \
  --output resources/nlp_grammar.json
```

## Pipeline Categories

The system automatically categorizes inputs into:

- **command**: Actions to execute (`open`, `close`, `start`, `delete`)
- **query**: Information requests (`search`, `find`, `show`, `tell`)
- **banter**: Social interactions (`hi`, `thanks`, `bye`)
- **conversation**: Dialogue (`tell me about`, `talk about`)

### Customizing Categories

Edit `scripts/extract_grammar.py` and modify `pipeline_rules`:

```python
self.pipeline_rules = {
    'command': ['open', 'close', 'start', 'stop', ...],
    'query': ['what', 'when', 'where', ...],
    'custom_category': ['your', 'keywords'],
}
```

## Grammar File Structure

`resources/nlp_grammar.json` contains:

```json
{
  "grammar_components": {
    "greeting": { "patterns": [...], "optional": true },
    ...
  },
  "command_verbs": {
    "open": {
      "intent": "command_open",
      "synonyms": ["launch", "start"],
      "pipeline_category": "command",
      "weight": 0.95
    }
  },
  "command_objects": { ... },
  "sentence_templates": { ... },
  "context_rules": { ... },
  "learning_config": { ... }
}
```

### Key Fields:

- **weight**: Priority/probability (0.0-1.0) derived from frequency
- **frequency**: Raw count from treebank
- **pipeline_category**: Routing category
- **synonyms**: Variations that map to canonical form

## FlatBuffer Compilation (Optional)

For production, compile to binary format:

### 1. Generate FlatBuffer Code

```bash
# Generate C++ headers
flatc --cpp -o nlp/ nlp/grammar_rules.fbs

# Generate Python bindings
flatc --python -o nlp/ nlp/grammar_rules.fbs
```

### 2. Compile Grammar to Binary

```bash
python scripts/compile_grammar_fb.py \
  --input resources/nlp_grammar.json \
  --output resources/grammar_rules.fb
```

### 3. Update Bootstrap to Load Binary

Modify `bootstrap/bootstrap_config.cpp`:

```cpp
// Add binary loading option
fs::path grammarBinary = fs::path(getResourcePath()) / "grammar_rules.fb";
if (fs::exists(grammarBinary)) {
    if (GRIM::g_grammarParser.loadBinary(grammarBinary.string())) {
        LOG_PHASE("Grammar binary loaded", true);
        return;
    }
}

// Fallback to JSON
fs::path grammarPath = fs::path(getResourcePath()) / "nlp_grammar.json";
...
```

## Weights & Scoring

### How Weights Work

1. **Frequency Extraction**: Count occurrences in treebank
2. **Normalization**: Convert to probability (count / total)
3. **Application**: Boost confidence scores in parsing

Example:
```json
"open": {
  "frequency": 150,  // appeared 150 times
  "weight": 0.95     // 150 / total_verbs (highly common)
}
```

### Tuning Weights

Manually adjust in `nlp_grammar.json`:

```json
"search": {
  "weight": 1.2  // Boost search commands (>1.0 = higher priority)
}
```

## Integration with Existing NLP

The grammar parser works alongside regex-based rules:

1. **Grammar parser** tries first (higher precision)
2. **Regex fallback** if grammar match fails (coverage)

Both results are combined with confidence scoring.

## Extending the Grammar

### Add New Verb

```json
"command_verbs": {
  "analyze": {
    "intent": "command_analyze",
    "synonyms": ["examine", "inspect"],
    "requires_object": true,
    "pipeline_category": "command",
    "frequency": 30,
    "weight": 0.5
  }
}
```

### Add New Template

```json
"sentence_templates": {
  "template_my_pattern": {
    "structure": "<verb> all [article] <object>",
    "examples": ["delete all files", "close all windows"],
    "pipeline_category": "command",
    "weight": 0.7
  }
}
```

### Add Context Rule

```json
"context_rules": {
  "morning_greeting": {
    "condition": "time.hour < 12 && input.contains('hi', 'hello')",
    "action": "add_greeting_response",
    "priority": 80,
    "weight_modifier": 1.3
  }
}
```

## Licensing & Sharing

If you generate grammar from public treebanks:

1. **Universal Dependencies**: CC BY-SA 4.0 (attribute UD project)
2. **Your annotations**: Choose your license
3. **Generated rules**: Derivative of source data license

To share publicly:
1. Document treebank source
2. Include LICENSE file
3. Attribute original annotators
4. Consider contributing to UD if useful

## Troubleshooting

### Grammar file not loading
- Check file path: `resources/nlp_grammar.json`
- Validate JSON syntax
- Check logs for parse errors

### Low confidence scores
- Increase weights for important patterns
- Add more synonyms
- Expand sentence templates

### Missing categories
- Add verbs to `pipeline_rules` mapping
- Regenerate with `extract_grammar.py`

### Performance issues
- Use FlatBuffer binary format
- Reduce template count (filter by frequency)
- Cache compiled regexes

## Files in This System

```
nlp/
├── grammar_rules.fbs          # FlatBuffer schema
├── grammar_parser.hpp/cpp     # Grammar parser implementation
└── GrammarRules/              # Generated FlatBuffer code (after compilation)

scripts/
├── extract_grammar.py         # Treebank → JSON extractor
└── compile_grammar_fb.py      # JSON → FlatBuffer compiler

resources/
├── nlp_grammar.json          # Main grammar file (JSON)
└── grammar_rules.fb          # Compiled binary (optional)
```

## Next Steps

1. ✅ Basic grammar loaded and working
2. ⏭️ Download UD treebank for production-quality rules
3. ⏭️ Tune weights based on user feedback
4. ⏭️ Add domain-specific verbs/objects
5. ⏭️ Implement online learning from corrections
6. ⏭️ Compile to FlatBuffer for performance

## References

- Universal Dependencies: https://universaldependencies.org/
- FlatBuffers: https://google.github.io/flatbuffers/
- CoNLL-U Format: https://universaldependencies.org/format.html
- Stanza (annotation tool): https://stanfordnlp.github.io/stanza/

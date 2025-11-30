# GRIM Data Collection & Verification System

## Overview

The GRIM Data Collection system is responsible for gathering, verifying, and preprocessing training data for the GRIM-text language model. It implements a multi-stage pipeline that ensures high-quality, diverse, and reliable training data.

## Architecture

```
Raw Data Sources → Collection → Verification → Preprocessing → Training Data
                      ↓              ↓              ↓              ↓
                   .jsonl        verified/      split/         .grmt
                                              train/val/test
```

## Components

### 1. Data Collection (`web_collector.hpp`)
- **Web Scraping**: Collects text from various online sources
- **Source Types**: Academic papers, technical docs, Wikipedia, GitHub, etc.
- **Output Format**: JSONL (JSON Lines) with metadata

### 2. Verification Layer (`verifier.hpp`, `verifier.cpp`)
The verification layer implements multiple quality filters to ensure training data quality.

#### **Quality Filters (In Order of Execution)**

##### **A. Duplicate Detection** ✅ NEW
- **Method**: Hash-based content deduplication
- **Algorithm**: Normalizes first 500 characters, removes special chars, converts to lowercase, then hashes
- **Purpose**: Prevents the model from memorizing duplicate content
- **Stats Tracked**: `duplicate_rejected`

```cpp
// Example: Same content with different formatting is detected as duplicate
"Hello World!" → hash(hello world) = 12345
"  hello   world  " → hash(hello world) = 12345  // DUPLICATE
```

##### **B. Language Detection** ✅ NEW
- **Method**: Common English word frequency analysis
- **Threshold**: Must contain at least 4 common English words in first 500 chars
- **Common Words Checked**: the, is, at, which, on, and, are, for, was, with, this, that, from, have, has
- **Purpose**: Filter out non-English content that would confuse the model
- **Stats Tracked**: `quality_rejected` (reason: `non_english_content`)

```cpp
// Example: English text passes
"The quick brown fox jumps over the lazy dog" → PASS (contains: the, over)

// Example: Non-English text fails
"Le chat mange le poisson dans la cuisine" → REJECT (no common English words)
```

##### **C. Low-Quality Content Filter** ✅ NEW
Detects and rejects multiple types of problematic content:

**Spam Patterns**:
- "buy now", "click here", "limited time", "act now"
- "offer expires", "subscribe now", "sign up today", "free trial"

**Excessive Repetition**:
- Same character repeated 10+ times (e.g., "!!!!!!!!!!!!", "zzzzzzzzzzz")

**UI/Navigation Artifacts**:
- "enable javascript", "404", "page not found"
- "home | about | contact", "search...", "loading..."
- "menu", "login", "sign in", "register", "copyright ©"

**Encoding Errors**:
- Malformed UTF-8: `â€™` (should be `'`), `â€œ` (should be `"`), `Ã©` (should be `é`)

**Purpose**: Remove web scraping artifacts and spam that won't help language learning
**Stats Tracked**: `quality_rejected` (reason: `low_quality_patterns`)

##### **D. Domain Validation**
- **Method**: Domain whitelist (optional)
- **Purpose**: Restrict sources to trusted domains
- **Configuration**: `domain_whitelist` in Config
- **Stats Tracked**: `domain_rejected`

##### **E. Length Validation**
- **Min Length**: 75 characters (configurable)
- **Max Length**: 75,000 characters (configurable)
- **Behavior**: Progressive filtering mode available
  - Short content gets penalized score (50-100% based on ratio)
  - Long content gets slight penalty (90% of score)
- **Stats Tracked**: `quality_rejected` (reason: `too_short` or `too_long`)

##### **F. Content Quality Assessment** ✅ NEW
Advanced scoring based on multiple factors:

**Sentence Structure** (±15% score):
- Expects ~1 sentence per 100 characters
- Proper sentence endings (`.`, `?`, `!`)
- Balanced sentence length

**Capitalization** (±10% score):
- Mixed case = natural text (PASS)
- All lowercase or all uppercase = suspicious (PENALTY)

**Special Characters** (±20% score):
- Clean text (<5% special chars) = BONUS
- Excessive special chars (>15%) = PENALTY
- Excludes common punctuation (`.`, `,`, `!`, `?`)

**Code Blocks** (+5% score):
- Presence of triple backticks ` ``` ` or 4-space indentation
- Indicates technical/educational content

**URL Density** (−15% score):
- More than 5 URLs = likely scraped link list

**Final Score Calculation**:
```cpp
reliability_score = (source_weight * 0.7) + (content_quality * 0.3)
```

**Stats Tracked**: `failed_verification` (reason: `low_reliability_score`)

##### **G. Quality Tier Classification**
Accepted entries are classified into quality tiers:

| Tier | Score Range | Use Case |
|------|-------------|----------|
| **High** | ≥0.8 | Premium training data, core examples |
| **Medium** | 0.6-0.79 | Standard training data |
| **Low** | 0.4-0.59 | Supplementary data, augmentation |

**Stats Tracked**: `high_quality_count`, `medium_quality_count`, `low_quality_count`

##### **H. Semantic Consistency Filter (DeBERTa-v3 MNLI)** ✅ NEW
- **Model**: `microsoft/deberta-v3-base-mnli` exported to ONNX
- **Runtime**: `onnxruntime-gpu` (falls back to CPU if CUDA unavailable)
- **Tokenizer**: HuggingFace SentencePiece model (`*.spm`) via `sentencepiece`
- **Method**: Zero-shot NLI. Premise = collected text, hypotheses describe *high quality* vs *low quality/spam* writing.
- **Score**: Average entailment probability of positive prompts, penalized by entailment of negative prompts.
- **Usage**:
  - Hard filter (`semantic_hard_filter=true`): reject when semantic score `< semantic_min_score`
  - Soft blend: weight semantic score into final `reliability_score` via `semantic_quality_weight`
- **Stats Tracked**: `semantic_rejected`, rejection reason `semantic_low_confidence`
- **Files needed** (default lookup path relative to repo root):
  - `resources/models/GRIM-text/quality/deberta-v3-base-mnli.onnx`
  - `resources/models/GRIM-text/quality/deberta-v3-base-mnli.spm`
- **Config knobs** (see `Config` struct):
  - `enable_semantic_model`, `semantic_model_path`, `semantic_tokenizer_path`
  - `semantic_min_score`, `semantic_quality_weight`, `semantic_positive_prompts`, `semantic_negative_prompts`
  - `semantic_max_seq_length` (default 512) and `semantic_use_gpu`

> **Tip:** Install tokenizer dependency with `vcpkg install sentencepiece:x64-windows`. Export the ONNX + tokenizer files from HuggingFace, drop them into the quality folder above, and the verifier auto-discovers them.

#### **Source Type Weights**

Different sources have different inherent reliability:

| Source Type | Weight | Examples |
|-------------|--------|----------|
| Academic | 1.0 | JSTOR, ArXiv papers |
| Philosophy/Classical | 0.95 | Philosophical texts, classics |
| Technical Docs | 0.9 | Tech documentation, erudite writing |
| Linguistics/Grammar | 0.9 | Language resources, grammar guides |
| Logic/Theoretical | 0.95 | Formal logic, theoretical science |
| Wikipedia | 0.8 | Wikipedia articles |
| GitHub | 0.85 | Code, documentation |
| StackOverflow | 0.75 | Q&A content |
| News | 0.7 | News articles |
| Reddit | 0.6 | Social content |
| Unknown | 0.7 | Default for unclassified |

#### **Configuration Options**

```cpp
struct Config {
    float reliability_threshold = 0.6f;      // Minimum score to accept
    int min_cross_references = 2;            // (Future: cross-validation)
    bool enable_cross_check = true;          // (Future: fact-checking)
    size_t min_length = 75;                  // Minimum content length
    size_t max_length = 75000;               // Maximum content length
    bool progressive_filtering = true;        // Gradual penalties vs hard cutoffs
    bool save_rejected = false;              // Save rejected entries for analysis
    bool verbose_logging = false;            // Log rejection reasons
    
    // Quality thresholds
    float high_quality_threshold = 0.8f;
    float medium_quality_threshold = 0.6f;
    float low_quality_threshold = 0.4f;

    // Semantic ONNX verifier
    bool enable_semantic_model = false;
    bool semantic_use_gpu = true;
    bool semantic_hard_filter = true;
    int semantic_max_seq_length = 512;
    float semantic_min_score = 0.55f;
    float semantic_quality_weight = 0.35f;
    std::string semantic_model_path;
    std::string semantic_tokenizer_path;
    std::vector<std::string> semantic_positive_prompts;
    std::vector<std::string> semantic_negative_prompts;
};
```

#### **Statistics & Reporting**

The verifier tracks comprehensive statistics:

```cpp
struct Stats {
    size_t total_processed;          // Total entries processed
    size_t passed_verification;      // Successfully verified
    size_t failed_verification;      // Failed reliability threshold
    size_t domain_rejected;          // Rejected by domain filter
    size_t quality_rejected;         // Rejected by quality filters
    size_t duplicate_rejected;       // Rejected as duplicates
    size_t semantic_rejected;        // Rejected by semantic ONNX filter
    
    // Quality tiers
    size_t high_quality_count;       // High-quality entries
    size_t medium_quality_count;     // Medium-quality entries
    size_t low_quality_count;        // Low-quality entries
    
    // Detailed rejection tracking
    std::unordered_map<std::string, size_t> rejection_reasons;
};
```

**Output Formats**:
1. **Human-readable log**: `logs/verification_stats.log`
2. **FlatBuffer binary**: `resources/models/GRIM-text/training/data/verification_stats.bin` (for UI)
3. **JSON**: `resources/models/GRIM-text/training/data/verification_stats.json` (backward compatibility)

### 3. Data Preprocessing (`data_preprocessor.hpp`)
- **Tokenization**: BPE tokenization with trained vocabulary
- **Normalization**: Text cleaning and formatting
- **Chunking**: Splits long texts into model-friendly chunks

### 4. Data Splitting (`data_splitter.hpp`)
- **Train/Val/Test Split**: Typically 80/10/10 or 90/5/5
- **Random Shuffling**: Ensures data diversity
- **Stratified Sampling**: Maintains source distribution across splits

## Usage

### Running the Full Pipeline

```bash
# Via UI Training Panel
1. Click "Run Data Pipeline" button
2. Monitor progress in the UI

# Via Command Line
cd DataCollection
./grim_data_pipeline.exe
```

### Running Individual Components

```bash
# Collection only
./main_data_collection.exe

# Verification only
./main_verifier.exe

# Merge checkpoints
./merge_checkpoints.exe
```

## File Formats

### Input: Raw Data (JSONL)
```json
{
  "content": "The actual text content...",
  "source_url": "https://example.com/article",
  "source_type": "academic",
  "author": "John Doe",
  "metadata": "{\"date\": \"2025-11-13\"}"
}
```

### Output: Verified Data (JSONL)
```json
{
  "content": "The actual text content...",
  "source_url": "https://example.com/article",
  "source_type": "academic",
  "author": "John Doe",
  "metadata": "{\"date\": \"2025-11-13\"}",
  "reliability_score": 0.85,
  "verification_time": 1699900800
}
```

### Final: Training Data (GRMT)
Binary format optimized for training, contains:
- Tokenized sequences
- Source metadata
- Quality scores
- Split information (train/val/test)

## Performance Considerations

### Memory Usage
- **Duplicate Detection**: O(n) memory for content hashes
- **Batch Processing**: Processes entries in chunks
- **Streaming**: Large files processed line-by-line

### Speed Optimization
- **Parallel Processing**: Multi-threaded verification (future)
- **Hash-based Dedup**: O(1) lookup time
- **Early Rejection**: Fails fast on low-quality content

## Quality Metrics

### Expected Pass Rates
- **Academic Sources**: 80-90% pass rate
- **Technical Docs**: 70-85% pass rate
- **Wikipedia**: 60-75% pass rate
- **Social Media**: 30-50% pass rate

### Rejection Breakdown (Typical)
- Duplicates: 10-20%
- Language: 5-10%
- Quality Filters: 15-25%
- Length: 5-10%
- Reliability Score: 10-20%

## Future Enhancements

### Planned Features
- [ ] Cross-reference validation (fact-checking)
- [ ] Semantic similarity detection (better deduplication)
- [ ] Multi-language support
- [ ] Active learning (user feedback integration)
- [ ] Real-time collection monitoring dashboard

### Experimental
- [ ] GPT-based quality assessment
- [ ] Adversarial content detection
- [ ] Domain-specific filtering rules
- [ ] Privacy-sensitive data detection (PII removal)

## Troubleshooting

### Common Issues

**Issue**: Too many duplicates rejected
- **Solution**: Check if multiple sources are providing same content. Review source URLs.

**Issue**: Low pass rate
- **Solution**: Review `rejection_reasons` in stats. Adjust thresholds in Config.

**Issue**: Non-English content passing
- **Solution**: Lower `reliability_threshold` or add more common words to detection.

**Issue**: Quality content rejected
- **Solution**: Enable `save_rejected` to analyze false negatives. Adjust quality weights.

## Version History

### v1.1.0 (November 13, 2025)
- ✅ Added hash-based duplicate detection
- ✅ Added English language detection
- ✅ Added content quality assessment scoring
- ✅ Added low-quality content filters (spam, UI, malformed)
- ✅ Added quality tier classification
- ✅ Enhanced statistics tracking with rejection reasons
- ✅ Improved progressive filtering mode

### v1.0.0 (November 2025)
- Initial release with basic verification
- Source type weighting
- Length validation
- Domain whitelisting

## Contact

For issues or questions about the data collection system, see the main GRIM project documentation.

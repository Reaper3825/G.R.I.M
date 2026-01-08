#include <torch/torch.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>

namespace fs = std::filesystem;
using json = nlohmann::json;

//======================================================//
//  Training Configuration (matches GRIM-text style)
//======================================================//

struct TrainConfig {
    std::string data_path = "resources/models/GRIM-text/training/data/merged_verified_cache.jsonl";
    std::string field = "content";
    std::string prompt = "Hello world";
    std::string device = "cuda";
    
    // GRMT data format support (full GRIM-text vocab)
    bool use_grmt = false;
    std::string grmt_path = "resources/models/GRIM-text/training/data/training_data.grmt";
    std::string grmt_val_path = "resources/models/GRIM-text/training/data/validation_data.grmt";
    std::string vocab_path = "resources/models/GRIM-text/training/data/vocab.bin";

    // Sequence and batching
    int64_t seq_len = 256;
    int64_t batch_size = 8;
    int64_t max_lines = 2000;
    int64_t max_tokens = 2 * 1000 * 1000;
    int64_t seed = 1;

    // Training schedule (GRIM-text style: epoch-based)
    int64_t epochs = 3;
    int64_t log_interval = 10;
    int64_t sample_interval = 200;
    int64_t sample_tokens = 120;
    int64_t validation_interval = 100;  // Run validation every N batches
    
    // Gradient accumulation (matches GRIM-text)
    int64_t accumulation_steps = 1;
    
    // Learning rate schedule
    int64_t warmup_steps = 100;
    double lr = 3e-4;
    double lr_min = 1e-5;
    double weight_decay = 0.1;
    double clip_grad = 1.0;
    
    // Model architecture
    int64_t n_layer = 6;
    int64_t n_head = 8;
    int64_t n_kv_head = 8;  // For GQA: set to < n_head (e.g., 4 for 8:4 ratio)
    int64_t n_embd = 512;
    double dropout = 0.0;
    
    bool use_rmsnorm = true;
    bool tie_weights = false;
    
    //======================================================//
    //  TOGGLEABLE FEATURES (matching GRIM-text)
    //======================================================//
    
    // Gradient accumulation: accumulate gradients over N micro-batches
    bool enable_grad_accumulation = true;
    
    // Learning rate warmup: linear warmup for first N steps
    bool enable_lr_warmup = true;
    
    // Cosine LR decay: decay LR from base to min using cosine schedule
    bool enable_cosine_decay = true;
    
    // Gradient clipping: clip gradients by global norm
    bool enable_grad_clipping = true;
    
    // Gradient metrics: log per-component gradient norms (emb, attn, ffn, etc.)
    bool enable_grad_metrics = true;
    
    // Validation: run validation pass periodically
    bool enable_validation = true;
    
    // Sample generation: generate text samples during training
    bool enable_sampling = true;
    
    // Focal loss: down-weight easy examples (gamma > 0)
    bool enable_focal_loss = false;
    double focal_gamma = 2.0;
    double focal_alpha = 1.0;
    
    // Label smoothing: smooth target distribution
    bool enable_label_smoothing = false;
    double label_smoothing = 0.1;
    
    // Auto-stop: stop training on plateau
    bool enable_auto_stop = false;
    int64_t auto_stop_patience = 10;  // Batches without improvement
    double auto_stop_min_delta = 0.001;  // Minimum improvement threshold
    
    // Telemetry: detailed per-batch logging
    bool enable_telemetry = true;
    
    // Weight stats: log weight statistics periodically
    bool enable_weight_stats = false;
    
    // Optimizer state logging: log AdamW m/v statistics
    bool enable_optimizer_stats = false;
    
    // Xavier initialization: use Xavier/Glorot normal for weight init
    bool use_xavier_init = true;
};

struct GPTConfig {
    int64_t vocab_size = 257;
    int64_t seq_len = 256;
    int64_t n_layer = 6;
    int64_t n_head = 8;
    int64_t n_kv_head = 8;  // For GQA: num KV heads (n_kv_head < n_head)
    int64_t n_embd = 512;
    double dropout = 0.0;
};

static constexpr int64_t kEosToken = 256;

static void replace_all(std::string & value, const std::string & from, const std::string & to) {
    if (from.empty()) {
        return;
    }
    std::size_t start = 0;
    while ((start = value.find(from, start)) != std::string::npos) {
        value.replace(start, from.size(), to);
        start += to.size();
    }
}

static std::string sanitize_text(std::string text) {
    replace_all(text, "<s>", "");
    replace_all(text, "</s>", "");
    return text;
}

static void append_bytes(const std::string & text, std::vector<int64_t> & out, int64_t eos_id) {
    for (unsigned char c : text) {
        out.push_back(static_cast<int64_t>(c));
    }
    out.push_back(eos_id);
}

static std::vector<int64_t> load_corpus(const TrainConfig & cfg) {
    std::ifstream file(cfg.data_path);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open dataset: " + cfg.data_path);
    }

    std::vector<int64_t> tokens;
    tokens.reserve(static_cast<std::size_t>(cfg.max_tokens > 0 ? cfg.max_tokens : 1024));

    std::string line;
    int64_t lines = 0;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        json j = json::parse(line, nullptr, false);
        if (j.is_discarded()) {
            continue;
        }

        auto it = j.find(cfg.field);
        if (it == j.end() || !it->is_string()) {
            continue;
        }

        std::string text = sanitize_text(it->get<std::string>());
        append_bytes(text, tokens, kEosToken);

        lines++;
        if (cfg.max_lines > 0 && lines >= cfg.max_lines) {
            break;
        }
        if (cfg.max_tokens > 0 && static_cast<int64_t>(tokens.size()) >= cfg.max_tokens) {
            break;
        }
    }

    return tokens;
}

//======================================================//
//  GRMT Binary Format Loader (GRIM-text full vocab)
//======================================================//

struct GRMTHeader {
    uint32_t magic;        // 0x474D5254 "GRMT"
    uint32_t version;      // 4
    uint32_t num_sequences;
    uint32_t vocab_size;
};

// Returns: pair of (flat token sequence, vocab_size)
static std::pair<std::vector<int64_t>, uint32_t> load_grmt(const std::string& path, int64_t max_tokens = -1) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open GRMT file: " + path);
    }
    
    GRMTHeader header;
    file.read(reinterpret_cast<char*>(&header.magic), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&header.version), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&header.num_sequences), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&header.vocab_size), sizeof(uint32_t));
    
    // Validate header
    constexpr uint32_t kGRMTMagic = 0x474D5254;  // "GRMT"
    if (header.magic != kGRMTMagic) {
        throw std::runtime_error("invalid GRMT magic: " + std::to_string(header.magic));
    }
    if (header.version != 4) {
        throw std::runtime_error("unsupported GRMT version: " + std::to_string(header.version));
    }
    
    std::cout << "[GRMT] Loading " << path << "\n";
    std::cout << "[GRMT] version=" << header.version 
              << " sequences=" << header.num_sequences 
              << " vocab_size=" << header.vocab_size << "\n";
    
    std::vector<int64_t> all_tokens;
    all_tokens.reserve(1024 * 1024);  // Pre-allocate 1M tokens
    
    constexpr size_t kTextFeatureDim = 16;  // Must match GRIM::Tokenizer::kTextFeatureDim
    
    // Read sequences
    for (uint32_t seq_idx = 0; seq_idx < header.num_sequences; ++seq_idx) {
        uint32_t seq_len;
        file.read(reinterpret_cast<char*>(&seq_len), sizeof(uint32_t));
        
        if (!file.good() || seq_len == 0 || seq_len > 100000) {
            std::cerr << "[GRMT] Warning: invalid seq_len=" << seq_len << " at seq " << seq_idx << "\n";
            break;
        }
        
        // Read token IDs
        std::vector<uint32_t> token_ids(seq_len);
        file.read(reinterpret_cast<char*>(token_ids.data()), seq_len * sizeof(uint32_t));
        
        // Skip numeric_values (float array)
        file.seekg(seq_len * sizeof(float), std::ios::cur);
        
        // Skip numeric_mask (uint8 array)
        file.seekg(seq_len * sizeof(uint8_t), std::ios::cur);
        
        // Skip text_features (uint16 array, 8 features per token)
        file.seekg(seq_len * kTextFeatureDim * sizeof(uint16_t), std::ios::cur);
        
        // Skip text_mask (uint8 array)
        file.seekg(seq_len * sizeof(uint8_t), std::ios::cur);
        
        // Append tokens to flat sequence
        for (uint32_t tid : token_ids) {
            all_tokens.push_back(static_cast<int64_t>(tid));
        }
        
        // Check max_tokens limit
        if (max_tokens > 0 && static_cast<int64_t>(all_tokens.size()) >= max_tokens) {
            std::cout << "[GRMT] Reached max_tokens limit (" << max_tokens << ")\n";
            break;
        }
        
        // Progress logging
        if ((seq_idx + 1) % 1000 == 0) {
            std::cout << "[GRMT] Loaded " << (seq_idx + 1) << "/" << header.num_sequences 
                      << " sequences, " << all_tokens.size() << " tokens\n";
        }
    }
    
    std::cout << "[GRMT] Done: " << all_tokens.size() << " tokens loaded\n";
    return {all_tokens, header.vocab_size};
}

//======================================================//
//  GRIM Vocabulary Loader (UnigramLM format)
//======================================================//

class GRIMVocab {
public:
    static constexpr uint32_t BYTE_TOKEN_BASE = 0;      // Bytes: 0-255
    static constexpr uint32_t ATOM_TOKEN_BASE = 256;    // Atoms: 256-511  
    static constexpr uint32_t UNIGRAM_TOKEN_BASE = 512; // Unigram: 512+
    
    struct Piece {
        std::string text;
        float score;
        bool is_special;
    };
    
    std::vector<Piece> pieces_;           // Unigram pieces (index in file order)
    std::unordered_map<std::string, uint32_t> piece_to_id_;  // For encoding
    uint32_t total_vocab_size_ = 0;
    
    bool load(const std::string& vocab_path) {
        std::ifstream file(vocab_path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[GRIMVocab] Failed to open: " << vocab_path << "\n";
            return false;
        }
        
        // Read magic: KTMG (4 bytes)
        char magic[4];
        file.read(magic, 4);
        if (magic[0] != 'K' || magic[1] != 'T' || magic[2] != 'M' || magic[3] != 'G') {
            std::cerr << "[GRIMVocab] Invalid magic header\n";
            return false;
        }
        
        // Read version (2 bytes)
        uint16_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 2) {
            std::cerr << "[GRIMVocab] Unsupported version: " << version << "\n";
            return false;
        }
        
        // Skip checksum (4 bytes)
        uint32_t checksum;
        file.read(reinterpret_cast<char*>(&checksum), sizeof(checksum));
        
        // Read config vocab_size (4 bytes) - number of unigram pieces
        uint32_t unigram_count;
        file.read(reinterpret_cast<char*>(&unigram_count), sizeof(unigram_count));
        
        // Skip max_length (4 bytes)
        uint32_t max_length;
        file.read(reinterpret_cast<char*>(&max_length), sizeof(max_length));
        
        // Skip flags (3 bytes)
        char flags[3];
        file.read(flags, 3);
        
        // Read total vocab size (4 bytes) - includes bytes + atoms + unigram
        file.read(reinterpret_cast<char*>(&total_vocab_size_), sizeof(total_vocab_size_));
        
        std::cout << "[GRIMVocab] Loading " << vocab_path << "\n";
        std::cout << "[GRIMVocab] unigram_pieces=" << unigram_count 
                  << " total_vocab_size=" << total_vocab_size_ << "\n";
        
        // Read pieces: length (4 bytes) + text + score (4 bytes float)
        pieces_.clear();
        pieces_.reserve(unigram_count);
        
        for (uint32_t i = 0; i < unigram_count; ++i) {
            uint32_t len;
            file.read(reinterpret_cast<char*>(&len), sizeof(len));
            
            if (len > 1024) {
                std::cerr << "[GRIMVocab] Invalid piece length: " << len << " at " << i << "\n";
                return false;
            }
            
            std::string text(len, '\0');
            file.read(&text[0], len);
            
            float score;
            file.read(reinterpret_cast<char*>(&score), sizeof(score));
            
            bool is_special = !text.empty() && text[0] == '<';
            pieces_.push_back({text, score, is_special});
            
            // Map piece to its global token ID (offset by UNIGRAM_TOKEN_BASE)
            piece_to_id_[text] = UNIGRAM_TOKEN_BASE + i;
        }
        
        std::cout << "[GRIMVocab] Loaded " << pieces_.size() << " unigram pieces\n";
        return true;
    }
    
    // Simple greedy encode (not Viterbi - good enough for prompts)
    std::vector<int64_t> encode(const std::string& text) const {
        std::vector<int64_t> tokens;
        tokens.reserve(text.size());
        
        size_t pos = 0;
        while (pos < text.size()) {
            // Try to find longest matching piece
            bool found = false;
            for (size_t len = std::min(text.size() - pos, size_t(32)); len > 0; --len) {
                std::string substr = text.substr(pos, len);
                auto it = piece_to_id_.find(substr);
                if (it != piece_to_id_.end()) {
                    tokens.push_back(static_cast<int64_t>(it->second));
                    pos += len;
                    found = true;
                    break;
                }
            }
            
            // Fallback to byte token
            if (!found) {
                tokens.push_back(static_cast<int64_t>(static_cast<unsigned char>(text[pos])));
                pos++;
            }
        }
        
        return tokens;
    }
    
    // Decode tokens back to text
    std::string decode(const std::vector<int64_t>& tokens) const {
        std::string out;
        out.reserve(tokens.size() * 4);  // Estimate
        
        for (int64_t t : tokens) {
            uint32_t id = static_cast<uint32_t>(t);
            
            if (id < 256) {
                // Byte token
                out.push_back(static_cast<char>(id));
            } else if (id >= UNIGRAM_TOKEN_BASE && (id - UNIGRAM_TOKEN_BASE) < pieces_.size()) {
                // Unigram piece
                out.append(pieces_[id - UNIGRAM_TOKEN_BASE].text);
            } else if (id >= ATOM_TOKEN_BASE && id < UNIGRAM_TOKEN_BASE) {
                // Atom placeholder - skip or show placeholder
                out.append("[ATOM:" + std::to_string(id - ATOM_TOKEN_BASE) + "]");
            } else {
                // Unknown token
                out.append("[UNK:" + std::to_string(id) + "]");
            }
        }
        
        return out;
    }
};

// Global vocab instance (set when use_grmt is enabled)
static std::unique_ptr<GRIMVocab> g_vocab;

static std::vector<int64_t> encode_prompt(const std::string & text) {
    // Use GRIM vocab if available, otherwise byte-level encoding
    if (g_vocab) {
        return g_vocab->encode(text);
    }
    
    std::vector<int64_t> tokens;
    tokens.reserve(text.size());
    for (unsigned char c : text) {
        tokens.push_back(static_cast<int64_t>(c));
    }
    return tokens;
}

static std::string decode_tokens(const std::vector<int64_t> & tokens) {
    // Use GRIM vocab if available, otherwise byte-level decoding
    if (g_vocab) {
        return g_vocab->decode(tokens);
    }
    
    std::string out;
    out.reserve(tokens.size());
    for (int64_t t : tokens) {
        if (t == kEosToken) {
            break;
        }
        // Byte tokens: 0-255
        if (t >= 0 && t <= 255) {
            out.push_back(static_cast<char>(t));
        } else {
            // For GRMT mode: print placeholder for subword tokens
            out.append("[" + std::to_string(t) + "]");
        }
    }
    return out;
}

struct CausalSelfAttentionImpl : torch::nn::Module {
    explicit CausalSelfAttentionImpl(const GPTConfig & cfg)
        : n_head(cfg.n_head),
          n_kv_head(cfg.n_kv_head),
          n_embd(cfg.n_embd),
          head_dim(cfg.n_embd / cfg.n_head) {
        
        // GQA: Q has n_head, K/V have n_kv_head
        const int64_t qkv_dim = cfg.n_embd + 2 * (cfg.n_kv_head * head_dim);
        c_attn = register_module("c_attn", torch::nn::Linear(cfg.n_embd, qkv_dim));
        c_proj = register_module("c_proj", torch::nn::Linear(cfg.n_embd, cfg.n_embd));
        
        mask = torch::tril(torch::ones({cfg.seq_len, cfg.seq_len}, torch::kBool));
        register_buffer("mask", mask);
    }

    torch::Tensor forward(torch::Tensor x) {
        const auto B = x.size(0);
        const auto T = x.size(1);
        const auto C = x.size(2);

        auto qkv = c_attn->forward(x);
        
        // Split Q, K, V with GQA dimensions
        const int64_t kv_dim = n_kv_head * head_dim;
        auto q = qkv.slice(2, 0, n_embd);
        auto k = qkv.slice(2, n_embd, n_embd + kv_dim);
        auto v = qkv.slice(2, n_embd + kv_dim, n_embd + 2 * kv_dim);

        // Reshape Q: [B, T, n_head, head_dim] -> [B, n_head, T, head_dim]
        q = q.view({B, T, n_head, head_dim}).transpose(1, 2);
        
        // Reshape K, V: [B, T, n_kv_head, head_dim] -> [B, n_kv_head, T, head_dim]
        k = k.view({B, T, n_kv_head, head_dim}).transpose(1, 2);
        v = v.view({B, T, n_kv_head, head_dim}).transpose(1, 2);
        
        // GQA: Repeat K/V across Q heads
        // Each KV head is shared by (n_head / n_kv_head) Q heads
        if (n_kv_head < n_head) {
            const int64_t n_rep = n_head / n_kv_head;
            k = k.repeat_interleave(n_rep, 1);  // [B, n_head, T, head_dim]
            v = v.repeat_interleave(n_rep, 1);  // [B, n_head, T, head_dim]
        }

        auto att = torch::matmul(q, k.transpose(-2, -1));
        att = att * (1.0 / std::sqrt(static_cast<double>(head_dim)));

        auto attn_mask = mask.index({torch::indexing::Slice(0, T),
                                     torch::indexing::Slice(0, T)})
                             .unsqueeze(0)
                             .unsqueeze(0);
        att = att.masked_fill(attn_mask.logical_not(), -1e4);

        att = torch::softmax(att, -1);
        auto y = torch::matmul(att, v);
        y = y.transpose(1, 2).contiguous().view({B, T, C});
        return c_proj->forward(y);
    }

    int64_t n_head;
    int64_t n_kv_head;
    int64_t n_embd;
    int64_t head_dim;
    torch::Tensor mask;
    torch::nn::Linear c_attn{nullptr};
    torch::nn::Linear c_proj{nullptr};
};
TORCH_MODULE(CausalSelfAttention);

struct MLPImpl : torch::nn::Module {
    explicit MLPImpl(int64_t n_embd)
        : fc(register_module("fc", torch::nn::Linear(n_embd, 4 * n_embd))),
          proj(register_module("proj", torch::nn::Linear(4 * n_embd, n_embd))) {}

    torch::Tensor forward(torch::Tensor x) {
        x = torch::gelu(fc->forward(x));
        return proj->forward(x);
    }

    torch::nn::Linear fc{nullptr};
    torch::nn::Linear proj{nullptr};
};
TORCH_MODULE(MLP);

// RMSNorm implementation (like GRIM-text)
struct RMSNormImpl : torch::nn::Module {
    explicit RMSNormImpl(int64_t dim, double eps = 1e-6)
        : eps_(eps) {
        weight = register_parameter("weight", torch::ones({dim}));
    }

    torch::Tensor forward(torch::Tensor x) {
        // RMS = sqrt(mean(x^2) + eps)
        auto rms = torch::sqrt(torch::mean(x * x, -1, true) + eps_);
        return weight * x / rms;
    }

    torch::Tensor weight;
    double eps_;
};
TORCH_MODULE(RMSNorm);

struct BlockImpl : torch::nn::Module {
    explicit BlockImpl(const GPTConfig & cfg, bool use_rmsnorm = false)
        : use_rmsnorm_(use_rmsnorm),
          attn(register_module("attn", CausalSelfAttention(cfg))),
          mlp(register_module("mlp", MLP(cfg.n_embd))) {
        if (use_rmsnorm_) {
            ln1_rms = register_module("ln1", RMSNorm(cfg.n_embd));
            ln2_rms = register_module("ln2", RMSNorm(cfg.n_embd));
        } else {
            ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({cfg.n_embd})));
            ln2 = register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({cfg.n_embd})));
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        if (use_rmsnorm_) {
            x = x + attn->forward(ln1_rms->forward(x));
            x = x + mlp->forward(ln2_rms->forward(x));
        } else {
            x = x + attn->forward(ln1->forward(x));
            x = x + mlp->forward(ln2->forward(x));
        }
        return x;
    }

    bool use_rmsnorm_;
    torch::nn::LayerNorm ln1{nullptr};
    torch::nn::LayerNorm ln2{nullptr};
    RMSNorm ln1_rms{nullptr};
    RMSNorm ln2_rms{nullptr};
    CausalSelfAttention attn{nullptr};
    MLP mlp{nullptr};
};
TORCH_MODULE(Block);

struct GPTImpl : torch::nn::Module {
    explicit GPTImpl(const GPTConfig & cfg, bool use_rmsnorm = false, bool tie_weights = false)
        : config(cfg),
          use_rmsnorm_(use_rmsnorm),
          tie_weights_(tie_weights),
          tok_emb(register_module("tok_emb", torch::nn::Embedding(cfg.vocab_size, cfg.n_embd))),
          pos_emb(register_module("pos_emb", torch::nn::Embedding(cfg.seq_len, cfg.n_embd))) {
        
        if (use_rmsnorm_) {
            ln_f_rms = register_module("ln_f", RMSNorm(cfg.n_embd));
        } else {
            ln_f = register_module("ln_f", torch::nn::LayerNorm(torch::nn::LayerNormOptions({cfg.n_embd})));
        }
        
        // CRITICAL FIX: Weight tying must be done properly!
        // Only create head module if NOT tying weights
        // When tie_weights=true, we'll manually apply embedding.weight.T in forward()
        if (!tie_weights_) {
            head = register_module("head", torch::nn::Linear(torch::nn::LinearOptions(cfg.n_embd, cfg.vocab_size).bias(false)));
        }
        
        for (int64_t i = 0; i < cfg.n_layer; ++i) {
            blocks->push_back(Block(cfg, use_rmsnorm_));
        }
        register_module("blocks", blocks);
    }
    
    void initialize_weights(bool use_xavier) {
        if (use_xavier) {
            // Xavier/Glorot normal initialization for all parameters
            for (const auto& p : named_parameters()) {
                if (p.value().dim() >= 2) {
                    // Weight matrices: use Xavier normal
                    torch::nn::init::xavier_normal_(p.value());
                } else {
                    // Biases and 1D parameters: zero init
                    torch::nn::init::zeros_(p.value());
                }
            }
        }
        // If use_xavier=false, keep PyTorch's default initialization (Kaiming uniform)
    }

    torch::Tensor forward(torch::Tensor idx) {
        auto B = idx.size(0);
        auto T = idx.size(1);
        auto pos = torch::arange(0, T, torch::TensorOptions().dtype(torch::kLong).device(idx.device()))
                       .unsqueeze(0)
                       .expand({B, T});
        auto x = tok_emb->forward(idx) + pos_emb->forward(pos);
        x = blocks->forward(x);
        
        if (use_rmsnorm_) {
            x = ln_f_rms->forward(x);
        } else {
            x = ln_f->forward(x);
        }
        
        // CRITICAL FIX: Proper weight tying
        if (tie_weights_) {
            // Apply logits = x @ embedding.weight.T (tie with embedding)
            // Shape: [B, T, d_embd] @ [vocab_size, d_embd].T = [B, T, vocab_size]
            return torch::matmul(x, tok_emb->weight.transpose(0, 1));
        } else {
            return head->forward(x);
        }
    }

    GPTConfig config;
    bool use_rmsnorm_;
    bool tie_weights_;
    torch::nn::Embedding tok_emb{nullptr};
    torch::nn::Embedding pos_emb{nullptr};
    torch::nn::Sequential blocks;
    torch::nn::LayerNorm ln_f{nullptr};
    RMSNorm ln_f_rms{nullptr};
    torch::nn::Linear head{nullptr};
};
TORCH_MODULE(GPT);

static std::unordered_map<std::string, std::string> parse_args(int argc, char ** argv) {
    std::unordered_map<std::string, std::string> out;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--", 0) != 0) {
            continue;
        }
        key = key.substr(2);
        if (i + 1 < argc) {
            out[key] = argv[++i];
        } else {
            out[key] = "1";
        }
    }
    return out;
}

static void apply_args(const std::unordered_map<std::string, std::string> & args, TrainConfig & cfg) {
    auto get = [&](const std::string & key) -> const std::string * {
        auto it = args.find(key);
        if (it == args.end()) {
            return nullptr;
        }
        return &it->second;
    };

    if (auto v = get("data")) cfg.data_path = *v;
    if (auto v = get("field")) cfg.field = *v;
    if (auto v = get("prompt")) cfg.prompt = *v;
    if (auto v = get("device")) cfg.device = *v;
    
    // GRMT data format
    auto parse_bool = [](const std::string& s) { return s == "1" || s == "true"; };
    if (auto v = get("use_grmt")) cfg.use_grmt = parse_bool(*v);
    if (auto v = get("grmt_path")) cfg.grmt_path = *v;
    if (auto v = get("grmt_val_path")) cfg.grmt_val_path = *v;
    if (auto v = get("vocab_path")) cfg.vocab_path = *v;

    if (auto v = get("seq_len")) cfg.seq_len = std::stoll(*v);
    if (auto v = get("batch_size")) cfg.batch_size = std::stoll(*v);
    if (auto v = get("epochs")) cfg.epochs = std::stoll(*v);
    if (auto v = get("log_interval")) cfg.log_interval = std::stoll(*v);
    if (auto v = get("sample_interval")) cfg.sample_interval = std::stoll(*v);
    if (auto v = get("sample_tokens")) cfg.sample_tokens = std::stoll(*v);
    if (auto v = get("validation_interval")) cfg.validation_interval = std::stoll(*v);
    if (auto v = get("max_lines")) cfg.max_lines = std::stoll(*v);
    if (auto v = get("max_tokens")) cfg.max_tokens = std::stoll(*v);
    if (auto v = get("seed")) cfg.seed = std::stoll(*v);
    
    if (auto v = get("accumulation_steps")) cfg.accumulation_steps = std::stoll(*v);
    if (auto v = get("warmup_steps")) cfg.warmup_steps = std::stoll(*v);

    if (auto v = get("n_layer")) cfg.n_layer = std::stoll(*v);
    if (auto v = get("n_head")) cfg.n_head = std::stoll(*v);
    if (auto v = get("n_kv_head")) cfg.n_kv_head = std::stoll(*v);
    if (auto v = get("n_embd")) cfg.n_embd = std::stoll(*v);
    if (auto v = get("dropout")) cfg.dropout = std::stod(*v);

    if (auto v = get("lr")) cfg.lr = std::stod(*v);
    if (auto v = get("lr_min")) cfg.lr_min = std::stod(*v);
    if (auto v = get("weight_decay")) cfg.weight_decay = std::stod(*v);
    if (auto v = get("clip_grad")) cfg.clip_grad = std::stod(*v);
    
    if (auto v = get("rmsnorm")) cfg.use_rmsnorm = (*v == "1" || *v == "true");
    if (auto v = get("tie_weights")) cfg.tie_weights = (*v == "1" || *v == "true");
    
    // Toggleable features
    if (auto v = get("grad_accumulation")) cfg.enable_grad_accumulation = parse_bool(*v);
    if (auto v = get("lr_warmup")) cfg.enable_lr_warmup = parse_bool(*v);
    if (auto v = get("cosine_decay")) cfg.enable_cosine_decay = parse_bool(*v);
    if (auto v = get("grad_clipping")) cfg.enable_grad_clipping = parse_bool(*v);
    if (auto v = get("grad_metrics")) cfg.enable_grad_metrics = parse_bool(*v);
    if (auto v = get("validation")) cfg.enable_validation = parse_bool(*v);
    if (auto v = get("sampling")) cfg.enable_sampling = parse_bool(*v);
    if (auto v = get("focal_loss")) cfg.enable_focal_loss = parse_bool(*v);
    if (auto v = get("focal_gamma")) cfg.focal_gamma = std::stod(*v);
    if (auto v = get("focal_alpha")) cfg.focal_alpha = std::stod(*v);
    if (auto v = get("label_smoothing")) cfg.enable_label_smoothing = parse_bool(*v);
    if (auto v = get("smoothing_eps")) cfg.label_smoothing = std::stod(*v);
    if (auto v = get("auto_stop")) cfg.enable_auto_stop = parse_bool(*v);
    if (auto v = get("auto_stop_patience")) cfg.auto_stop_patience = std::stoll(*v);
    if (auto v = get("auto_stop_min_delta")) cfg.auto_stop_min_delta = std::stod(*v);
    if (auto v = get("telemetry")) cfg.enable_telemetry = parse_bool(*v);
    if (auto v = get("weight_stats")) cfg.enable_weight_stats = parse_bool(*v);
    if (auto v = get("optimizer_stats")) cfg.enable_optimizer_stats = parse_bool(*v);
    if (auto v = get("xavier_init")) cfg.use_xavier_init = parse_bool(*v);
}

//======================================================//
//  Focal Loss (matches GRIM-text UnifiedLoss)
//======================================================//

static torch::Tensor compute_focal_loss(
    torch::Tensor logits,  // [B*T, vocab]
    torch::Tensor targets, // [B*T]
    double gamma,
    double alpha,
    bool label_smoothing_enabled,
    double smoothing_eps,
    int64_t vocab_size) {
    
    // Compute log probabilities
    auto log_probs = torch::log_softmax(logits, -1);
    
    // Get log probability of target class
    auto target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1);
    
    // Compute pt = exp(log_pt)
    auto pt = target_log_probs.exp();
    
    // Focal weight: (1 - pt)^gamma
    auto focal_weight = (1.0 - pt).pow(gamma);
    
    // Base cross-entropy loss
    torch::Tensor ce_loss;
    if (label_smoothing_enabled && smoothing_eps > 0.0) {
        // Label smoothing: mix one-hot with uniform
        auto smooth_targets = torch::zeros_like(logits).fill_(smoothing_eps / (vocab_size - 1));
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing_eps);
        ce_loss = -(smooth_targets * log_probs).sum(-1);
    } else {
        ce_loss = -target_log_probs;
    }
    
    // Apply focal weight and alpha
    auto loss = alpha * focal_weight * ce_loss;
    
    return loss.mean();
}

//======================================================//
//  Weight Statistics (matches GRIM-text sampleWeightStats)
//======================================================//

struct WeightStats {
    double mean = 0.0;
    double std = 0.0;
    double min = 0.0;
    double max = 0.0;
    double rms = 0.0;
};

static WeightStats compute_weight_stats(GPT& model) {
    WeightStats stats;
    std::vector<double> all_values;
    
    for (const auto& p : model->parameters()) {
        auto flat = p.flatten().to(torch::kCPU).to(torch::kFloat64);
        auto* data = flat.data_ptr<double>();
        for (int64_t i = 0; i < flat.numel(); ++i) {
            all_values.push_back(data[i]);
        }
    }
    
    if (all_values.empty()) return stats;
    
    double sum = 0.0, sq_sum = 0.0;
    stats.min = all_values[0];
    stats.max = all_values[0];
    
    for (double v : all_values) {
        sum += v;
        sq_sum += v * v;
        stats.min = std::min(stats.min, v);
        stats.max = std::max(stats.max, v);
    }
    
    stats.mean = sum / all_values.size();
    stats.rms = std::sqrt(sq_sum / all_values.size());
    
    double var_sum = 0.0;
    for (double v : all_values) {
        var_sum += (v - stats.mean) * (v - stats.mean);
    }
    stats.std = std::sqrt(var_sum / all_values.size());
    
    return stats;
}

//======================================================//
//  Batch Result (matches GRIM-text BatchResult)
//======================================================//

struct BatchResult {
    int batch_idx = 0;
    float loss = 0.0f;
    float grad_norm = 0.0f;
    float learning_rate = 0.0f;
    int64_t tokens_processed = 0;
    bool gradient_clipped = false;
    bool skipped = false;
    std::string skip_reason;
};

//======================================================//
//  Epoch Result (matches GRIM-text EpochResult)
//======================================================//

struct EpochResult {
    int batches_processed = 0;
    int batches_skipped = 0;
    float avg_loss = 0.0f;
    float best_batch_loss = std::numeric_limits<float>::max();
    float worst_batch_loss = 0.0f;
    float validation_loss = 0.0f;
    float validation_perplexity = 0.0f;
    std::chrono::milliseconds duration{0};
};

//======================================================//
//  Gradient Metrics (matches GRIM-text GradientMetrics)
//======================================================//

struct GradientMetrics {
    double total_norm = 0.0;
    double embedding_norm = 0.0;
    double lm_head_norm = 0.0;
    double attention_norm = 0.0;
    double ffn_norm = 0.0;
    double layernorm_norm = 0.0;
};

static GradientMetrics compute_gradient_metrics(GPT& model) {
    GradientMetrics metrics;
    double total_sq = 0.0;
    double emb_sq = 0.0;
    double head_sq = 0.0;
    double attn_sq = 0.0;
    double ffn_sq = 0.0;
    double ln_sq = 0.0;
    
    auto params = model->named_parameters();
    for (const auto& p : params) {
        if (!p.value().grad().defined()) continue;
        
        double norm_sq = p.value().grad().pow(2).sum().item<double>();
        total_sq += norm_sq;
        
        const std::string& name = p.key();
        if (name.find("tok_emb") != std::string::npos || name.find("pos_emb") != std::string::npos) {
            emb_sq += norm_sq;
        } else if (name.find("head") != std::string::npos) {
            head_sq += norm_sq;
        } else if (name.find("c_attn") != std::string::npos || name.find("c_proj") != std::string::npos) {
            attn_sq += norm_sq;
        } else if (name.find("fc") != std::string::npos || name.find("proj") != std::string::npos) {
            ffn_sq += norm_sq;
        } else if (name.find("ln") != std::string::npos || name.find("weight") != std::string::npos) {
            ln_sq += norm_sq;
        }
    }
    
    metrics.total_norm = std::sqrt(total_sq);
    metrics.embedding_norm = std::sqrt(emb_sq);
    metrics.lm_head_norm = std::sqrt(head_sq);
    metrics.attention_norm = std::sqrt(attn_sq);
    metrics.ffn_norm = std::sqrt(ffn_sq);
    metrics.layernorm_norm = std::sqrt(ln_sq);
    
    return metrics;
}

//======================================================//
//  Learning Rate Schedule (matches GRIM-text)
//======================================================//

static double get_scheduled_learning_rate(
    int64_t step,
    double base_lr,
    int64_t warmup_steps,
    double min_lr,
    int64_t total_steps) {
    
    if (step < warmup_steps) {
        // Linear warmup
        return base_lr * static_cast<double>(step + 1) / static_cast<double>(warmup_steps);
    }
    
    // Cosine annealing after warmup
    int64_t decay_steps = total_steps - warmup_steps;
    int64_t current_decay_step = step - warmup_steps;
    double progress = static_cast<double>(current_decay_step) / static_cast<double>(decay_steps);
    progress = std::clamp(progress, 0.0, 1.0);
    
    // Cosine decay from base_lr to min_lr
    double decay = 0.5 * (1.0 + std::cos(M_PI * progress));
    return min_lr + (base_lr - min_lr) * decay;
}

//======================================================//
//  Batch Structure and Creation
//======================================================//

struct Batch {
    torch::Tensor x;
    torch::Tensor y;
};

static Batch make_batch(const std::vector<int64_t> & data,
                        int64_t batch_size,
                        int64_t seq_len,
                        torch::Device device,
                        std::mt19937 & rng) {
    const int64_t max_offset = static_cast<int64_t>(data.size()) - seq_len - 1;
    if (max_offset <= 0) {
        throw std::runtime_error("dataset too small for seq_len");
    }

    std::uniform_int_distribution<int64_t> dist(0, max_offset);
    std::vector<int64_t> xb(batch_size * seq_len);
    std::vector<int64_t> yb(batch_size * seq_len);

    for (int64_t i = 0; i < batch_size; ++i) {
        int64_t offset = dist(rng);
        for (int64_t t = 0; t < seq_len; ++t) {
            xb[i * seq_len + t] = data[offset + t];
            yb[i * seq_len + t] = data[offset + t + 1];
        }
    }

    auto x = torch::from_blob(xb.data(), {batch_size, seq_len}, torch::kInt64).clone().to(device);
    auto y = torch::from_blob(yb.data(), {batch_size, seq_len}, torch::kInt64).clone().to(device);
    return {x, y};
}

//======================================================//
//  Validation (matches GRIM-text runValidation)
//======================================================//

static std::pair<float, float> run_validation(
    GPT& model,
    const std::vector<int64_t>& val_data,
    const TrainConfig& cfg,
    const GPTConfig& model_cfg,
    torch::Device device,
    std::mt19937& rng) {
    
    torch::NoGradGuard guard;
    model->eval();
    
    const int64_t num_val_batches = 10;  // Fixed validation size
    double total_loss = 0.0;
    int64_t total_tokens = 0;
    
    for (int64_t i = 0; i < num_val_batches; ++i) {
        auto batch = make_batch(val_data, cfg.batch_size, cfg.seq_len, device, rng);
        auto logits = model->forward(batch.x);
        auto loss = torch::nn::functional::cross_entropy(
            logits.view({-1, model_cfg.vocab_size}),
            batch.y.view({-1}),
            torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kSum));
        
        total_loss += loss.item<double>();
        total_tokens += cfg.batch_size * cfg.seq_len;
    }
    
    model->train();
    
    float avg_loss = static_cast<float>(total_loss / total_tokens);
    float perplexity = std::exp(avg_loss);
    return {avg_loss, perplexity};
}

//======================================================//
//  Generation
//======================================================//

static std::vector<int64_t> generate(GPT & model,
                                     const std::vector<int64_t> & prompt,
                                     int64_t max_new_tokens,
                                     int64_t seq_len,
                                     torch::Device device) {
    std::vector<int64_t> tokens = prompt;
    tokens.reserve(prompt.size() + max_new_tokens);

    torch::NoGradGuard guard;
    for (int64_t i = 0; i < max_new_tokens; ++i) {
        int64_t start = 0;
        if (static_cast<int64_t>(tokens.size()) > seq_len) {
            start = static_cast<int64_t>(tokens.size()) - seq_len;
        }

        std::vector<int64_t> window(tokens.begin() + start, tokens.end());
        auto idx = torch::from_blob(window.data(), {1, static_cast<int64_t>(window.size())}, torch::kInt64)
                       .clone()
                       .to(device);
        auto logits = model->forward(idx);
        auto next_logits = logits.index({0, logits.size(1) - 1});
        auto next_token = std::get<1>(next_logits.max(-1));
        int64_t next_id = next_token.item<int64_t>();
        tokens.push_back(next_id);
        if (next_id == kEosToken) {
            break;
        }
    }

    return tokens;
}

//======================================================//
//  Main Training Entry Point
//======================================================//

int main(int argc, char ** argv) {
    TrainConfig cfg;
    auto args = parse_args(argc, argv);
    apply_args(args, cfg);

    //======================================================//
    //  Configuration Summary
    //======================================================//
    std::cout << "\n========================================\n";
    std::cout << "LIBTORCH BASELINE (GRIM-text Feature Parity)\n";
    std::cout << "========================================\n";
    std::cout << "[Config] dataset: " << cfg.data_path << "\n";
    std::cout << "[Config] seq_len=" << cfg.seq_len
              << " batch_size=" << cfg.batch_size
              << " epochs=" << cfg.epochs << "\n";
    
    // Feature toggles
    std::cout << "\n[Features] Toggleable Settings:\n";
    std::cout << "  grad_accumulation: " << (cfg.enable_grad_accumulation ? "ON" : "OFF") 
              << " (steps=" << cfg.accumulation_steps << ")\n";
    std::cout << "  lr_warmup: " << (cfg.enable_lr_warmup ? "ON" : "OFF")
              << " (steps=" << cfg.warmup_steps << ")\n";
    std::cout << "  cosine_decay: " << (cfg.enable_cosine_decay ? "ON" : "OFF") << "\n";
    std::cout << "  grad_clipping: " << (cfg.enable_grad_clipping ? "ON" : "OFF")
              << " (max_norm=" << cfg.clip_grad << ")\n";
    std::cout << "  grad_metrics: " << (cfg.enable_grad_metrics ? "ON" : "OFF") << "\n";
    std::cout << "  validation: " << (cfg.enable_validation ? "ON" : "OFF")
              << " (interval=" << cfg.validation_interval << ")\n";
    std::cout << "  sampling: " << (cfg.enable_sampling ? "ON" : "OFF")
              << " (interval=" << cfg.sample_interval << ")\n";
    std::cout << "  focal_loss: " << (cfg.enable_focal_loss ? "ON" : "OFF")
              << " (gamma=" << cfg.focal_gamma << ", alpha=" << cfg.focal_alpha << ")\n";
    std::cout << "  label_smoothing: " << (cfg.enable_label_smoothing ? "ON" : "OFF")
              << " (eps=" << cfg.label_smoothing << ")\n";
    std::cout << "  auto_stop: " << (cfg.enable_auto_stop ? "ON" : "OFF")
              << " (patience=" << cfg.auto_stop_patience << ")\n";
    std::cout << "  telemetry: " << (cfg.enable_telemetry ? "ON" : "OFF") << "\n";
    std::cout << "  weight_stats: " << (cfg.enable_weight_stats ? "ON" : "OFF") << "\n";
    std::cout << "  optimizer_stats: " << (cfg.enable_optimizer_stats ? "ON" : "OFF") << "\n";
    std::cout << "  xavier_init: " << (cfg.use_xavier_init ? "ON" : "OFF") << "\n";
    std::cout << "  use_grmt: " << (cfg.use_grmt ? "ON" : "OFF") << "\n";

    // Determine vocab_size - either 257 (byte level) or from GRMT header
    uint32_t vocab_size = 257;  // Default byte-level
    std::vector<int64_t> data;
    
    if (cfg.use_grmt) {
        // Load vocab.bin first (for encoding prompts and decoding output)
        if (!fs::exists(cfg.vocab_path)) {
            std::cerr << "[ERROR] missing vocab file: " << cfg.vocab_path << "\n";
            return 1;
        }
        g_vocab = std::make_unique<GRIMVocab>();
        if (!g_vocab->load(cfg.vocab_path)) {
            std::cerr << "[ERROR] failed to load vocab file\n";
            return 1;
        }
        
        // Load from GRMT binary format (full GRIM-text vocab)
        if (!fs::exists(cfg.grmt_path)) {
            std::cerr << "[ERROR] missing GRMT dataset: " << cfg.grmt_path << "\n";
            return 1;
        }
        auto [grmt_data, grmt_vocab] = load_grmt(cfg.grmt_path, cfg.max_tokens);
        data = std::move(grmt_data);
        vocab_size = grmt_vocab;
    } else {
        // Load from JSONL (byte-level encoding)
        if (!fs::exists(cfg.data_path)) {
            std::cerr << "[ERROR] missing dataset: " << cfg.data_path << "\n";
            return 1;
        }
        data = load_corpus(cfg);
    }
    
    torch::manual_seed(cfg.seed);
    std::mt19937 rng(static_cast<uint32_t>(cfg.seed));
    
    if (data.size() < static_cast<std::size_t>(cfg.seq_len + 1)) {
        std::cerr << "[ERROR] dataset too small for seq_len\n";
        return 1;
    }
    std::cout << "\n[Data] tokens loaded: " << data.size() << "\n";

    torch::Device device(torch::kCPU);
    if (cfg.device == "cuda" && torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
    }
    std::cout << "[Model] device: " << device.str() << "\n";
    std::cout << "[Model] normalization: " << (cfg.use_rmsnorm ? "RMSNorm" : "LayerNorm") << "\n";
    std::cout << "[Model] weight_tying: " << (cfg.tie_weights ? "enabled" : "disabled") << "\n";
    std::cout << "[Model] attention: " << cfg.n_head << " Q heads, " << cfg.n_kv_head << " KV heads";
    if (cfg.n_kv_head < cfg.n_head) {
        std::cout << " (GQA " << (cfg.n_head / cfg.n_kv_head) << ":1)";
    } else {
        std::cout << " (MHA)";
    }
    std::cout << "\n";

    GPTConfig model_cfg;
    model_cfg.vocab_size = static_cast<int64_t>(vocab_size);  // Dynamic vocab_size from GRMT or default 257
    model_cfg.seq_len = cfg.seq_len;
    model_cfg.n_layer = cfg.n_layer;
    model_cfg.n_head = cfg.n_head;
    model_cfg.n_kv_head = cfg.n_kv_head;
    model_cfg.n_embd = cfg.n_embd;
    model_cfg.dropout = cfg.dropout;
    
    std::cout << "[Model] vocab_size: " << model_cfg.vocab_size << "\n";

    GPT model(model_cfg, cfg.use_rmsnorm, cfg.tie_weights);
    model->to(device);
    
    // Initialize weights (Xavier or default PyTorch)
    model->initialize_weights(cfg.use_xavier_init);
    
    // Count parameters
    int64_t total_params = 0;
    for (const auto& p : model->parameters()) {
        total_params += p.numel();
    }
    std::cout << "[Model] total_params: " << total_params << "\n";

    torch::optim::AdamW optimizer(model->parameters(),
                                  torch::optim::AdamWOptions(cfg.lr)
                                      .betas(std::make_tuple(0.9, 0.95))
                                      .weight_decay(cfg.weight_decay));

    std::cout << "========================================\n\n";

    //======================================================//
    //  Training Loop (GRIM-text style)
    //======================================================//
    
    model->train();
    
    // Training state
    int64_t global_step = 0;
    int64_t micro_step = 0;  // For gradient accumulation
    float best_loss = std::numeric_limits<float>::max();
    int64_t steps_without_improvement = 0;
    bool auto_stopped = false;
    
    // Estimate total steps for LR schedule
    const int64_t batches_per_epoch = static_cast<int64_t>(data.size()) / (cfg.batch_size * cfg.seq_len);
    const int64_t total_steps = cfg.epochs * batches_per_epoch / 
                                 (cfg.enable_grad_accumulation ? cfg.accumulation_steps : 1);
    
    std::cout << "[Training] Starting training...\n";
    std::cout << "[Training] batches_per_epoch=" << batches_per_epoch 
              << " total_steps=" << total_steps << "\n\n";
    
    auto training_start = std::chrono::steady_clock::now();
    
    for (int64_t epoch = 0; epoch < cfg.epochs && !auto_stopped; ++epoch) {
        std::cout << "========================================\n";
        std::cout << "Epoch " << (epoch + 1) << "/" << cfg.epochs << "\n";
        std::cout << "========================================\n";
        
        auto epoch_start = std::chrono::steady_clock::now();
        float epoch_loss = 0.0f;
        int64_t epoch_batches = 0;
        
        for (int64_t batch_idx = 0; batch_idx < batches_per_epoch && !auto_stopped; ++batch_idx) {
            auto batch = make_batch(data, cfg.batch_size, cfg.seq_len, device, rng);
            
            //======================================================//
            //  Forward Pass
            //======================================================//
            auto logits = model->forward(batch.x);
            
            //======================================================//
            //  Loss Computation (with toggleable features)
            //======================================================//
            torch::Tensor loss;
            if (cfg.enable_focal_loss || cfg.enable_label_smoothing) {
                loss = compute_focal_loss(
                    logits.view({-1, model_cfg.vocab_size}),
                    batch.y.view({-1}),
                    cfg.enable_focal_loss ? cfg.focal_gamma : 0.0,
                    cfg.enable_focal_loss ? cfg.focal_alpha : 1.0,
                    cfg.enable_label_smoothing,
                    cfg.label_smoothing,
                    model_cfg.vocab_size);
            } else {
                loss = torch::nn::functional::cross_entropy(
                    logits.view({-1, model_cfg.vocab_size}),
                    batch.y.view({-1}),
                    torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kMean));
            }
            
            //======================================================//
            //  Gradient Accumulation
            //======================================================//
            if (cfg.enable_grad_accumulation && cfg.accumulation_steps > 1) {
                // Scale loss for accumulation
                loss = loss / static_cast<float>(cfg.accumulation_steps);
            }
            
            loss.backward();
            micro_step++;
            
            // Only update weights after accumulation_steps micro-batches
            bool should_step = !cfg.enable_grad_accumulation || 
                               (micro_step % cfg.accumulation_steps == 0);
            
            if (should_step) {
                //======================================================//
                //  Gradient Metrics (toggleable)
                //======================================================//
                float grad_norm_before_clip = 0.0f;
                GradientMetrics grad_metrics;
                
                if (cfg.enable_grad_metrics) {
                    grad_metrics = compute_gradient_metrics(model);
                    grad_norm_before_clip = static_cast<float>(grad_metrics.total_norm);
                }
                
                //======================================================//
                //  Gradient Clipping (toggleable)
                //======================================================//
                bool was_clipped = false;
                if (cfg.enable_grad_clipping && cfg.clip_grad > 0.0) {
                    double clipped_norm = torch::nn::utils::clip_grad_norm_(
                        model->parameters(), cfg.clip_grad);
                    was_clipped = (clipped_norm > cfg.clip_grad);
                }
                
                //======================================================//
                //  Learning Rate Schedule (toggleable)
                //======================================================//
                double current_lr = cfg.lr;
                if (cfg.enable_lr_warmup || cfg.enable_cosine_decay) {
                    current_lr = get_scheduled_learning_rate(
                        global_step,
                        cfg.lr,
                        cfg.enable_lr_warmup ? cfg.warmup_steps : 0,
                        cfg.enable_cosine_decay ? cfg.lr_min : cfg.lr,
                        total_steps);
                    
                    // Update optimizer LR
                    for (auto& group : optimizer.param_groups()) {
                        group.options().set_lr(current_lr);
                    }
                }
                
                //======================================================//
                //  Optimizer Step
                //======================================================//
                optimizer.step();
                optimizer.zero_grad();
                global_step++;
                
                float batch_loss = loss.item<float>() * 
                    (cfg.enable_grad_accumulation ? cfg.accumulation_steps : 1);
                epoch_loss += batch_loss;
                epoch_batches++;
                
                //======================================================//
                //  Telemetry Logging (toggleable)
                //======================================================//
                if (cfg.enable_telemetry && global_step % cfg.log_interval == 0) {
                    std::cout << "[Step " << global_step << "] "
                              << "loss=" << std::fixed << std::setprecision(4) << batch_loss
                              << " lr=" << std::scientific << std::setprecision(2) << current_lr;
                    
                    if (cfg.enable_grad_metrics) {
                        std::cout << " grad_norm=" << std::fixed << std::setprecision(4) << grad_norm_before_clip;
                        if (was_clipped) std::cout << " (clipped)";
                    }
                    std::cout << "\n";
                    
                    // Detailed gradient breakdown
                    if (cfg.enable_grad_metrics && global_step % (cfg.log_interval * 10) == 0) {
                        std::cout << "[GradTrace] COMPONENTS: "
                                  << "emb=" << std::fixed << std::setprecision(4) << grad_metrics.embedding_norm
                                  << " head=" << grad_metrics.lm_head_norm
                                  << " attn=" << grad_metrics.attention_norm
                                  << " ffn=" << grad_metrics.ffn_norm
                                  << " ln=" << grad_metrics.layernorm_norm << "\n";
                    }
                }
                
                //======================================================//
                //  Weight Stats (toggleable)
                //======================================================//
                if (cfg.enable_weight_stats && global_step % (cfg.log_interval * 10) == 0) {
                    auto stats = compute_weight_stats(model);
                    std::cout << "[WeightStats] mean=" << std::scientific << stats.mean
                              << " std=" << stats.std
                              << " rms=" << stats.rms
                              << " range=[" << stats.min << ", " << stats.max << "]\n";
                }
                
                //======================================================//
                //  Validation (toggleable)
                //======================================================//
                if (cfg.enable_validation && global_step % cfg.validation_interval == 0) {
                    auto [val_loss, val_ppl] = run_validation(model, data, cfg, model_cfg, device, rng);
                    std::cout << "[Validation] loss=" << std::fixed << std::setprecision(4) << val_loss
                              << " ppl=" << std::setprecision(2) << val_ppl << "\n";
                    
                    // Auto-stop check
                    if (cfg.enable_auto_stop) {
                        if (val_loss < best_loss - cfg.auto_stop_min_delta) {
                            best_loss = val_loss;
                            steps_without_improvement = 0;
                        } else {
                            steps_without_improvement++;
                            if (steps_without_improvement >= cfg.auto_stop_patience) {
                                std::cout << "[AutoStop] No improvement for " << cfg.auto_stop_patience 
                                          << " validations. Stopping.\n";
                                auto_stopped = true;
                            }
                        }
                    }
                }
                
                //======================================================//
                //  Sample Generation (toggleable)
                //======================================================//
                if (cfg.enable_sampling && cfg.sample_interval > 0 && 
                    global_step % cfg.sample_interval == 0) {
                    model->eval();
                    auto prompt_tokens = encode_prompt(cfg.prompt);
                    auto out_tokens = generate(model, prompt_tokens, cfg.sample_tokens, 
                                               cfg.seq_len, device);
                    std::cout << "[Sample] \"" << decode_tokens(out_tokens) << "\"\n";
                    model->train();
                }
            }
        }
        
        // Epoch summary
        auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - epoch_start);
        float avg_epoch_loss = epoch_loss / std::max(epoch_batches, int64_t(1));
        std::cout << "\n[Epoch " << (epoch + 1) << " Complete] "
                  << "avg_loss=" << std::fixed << std::setprecision(4) << avg_epoch_loss
                  << " duration=" << epoch_duration.count() << "s\n\n";
    }
    
    //======================================================//
    //  Training Summary
    //======================================================//
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - training_start);
    
    std::cout << "========================================\n";
    std::cout << "Training Complete\n";
    std::cout << "========================================\n";
    std::cout << "Total steps: " << global_step << "\n";
    std::cout << "Total time: " << total_duration.count() << "s\n";
    if (auto_stopped) {
        std::cout << "Stopped early: auto_stop triggered\n";
    }
    
    // Final sample
    if (cfg.enable_sampling) {
        model->eval();
        auto prompt_tokens = encode_prompt(cfg.prompt);
        auto out_tokens = generate(model, prompt_tokens, cfg.sample_tokens, cfg.seq_len, device);
        std::cout << "\n[Final Sample]\n" << decode_tokens(out_tokens) << "\n";
    }
    
    return 0;
}

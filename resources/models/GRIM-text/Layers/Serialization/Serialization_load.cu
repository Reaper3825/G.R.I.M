#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Shared/GRMT/GrmtFormat.hpp"
#include "../../Common/grim_model_serialization_version.hpp"
#include "Serialization_GPU.hpp"

namespace {

constexpr auto kLogModule = GRIM::Logging::ModuleId::Checkpoint;

template <typename... Args>
std::string Msg(Args&&... args) {
    std::ostringstream oss;
    (oss << ... << args);
    return oss.str();
}

void dumpCudaDeviceState() {
    auto emit = [](const std::string& msg) {
        GRIM::Logging::EmitModuleError(kLogModule, msg);
    };
    emit("[DUMP] --- CUDA device state ---");
    std::size_t free_mem = 0, total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
        emit(Msg("[DUMP]   GPU memory: free=", free_mem / (1024*1024), "MB total=",
                 total_mem / (1024*1024), "MB used=", (total_mem - free_mem) / (1024*1024), "MB"));
    } else {
        emit("[DUMP]   GPU memory query FAILED");
    }
    int dev = -1;
    if (cudaGetDevice(&dev) == cudaSuccess) {
        emit(Msg("[DUMP]   active CUDA device: ", dev));
    }
    cudaError_t sticky = cudaPeekAtLastError();
    if (sticky != cudaSuccess) {
        emit(Msg("[DUMP]   CUDA sticky error: ", cudaGetErrorString(sticky)));
    }
}

void dumpBinaryAnalysis(const std::vector<uint8_t>& buffer, std::size_t file_size, const std::string& path) {
    auto emit = [](const std::string& msg) {
        GRIM::Logging::EmitModuleError(kLogModule, msg);
    };

    emit("========== CHECKPOINT LOAD FAILURE — FULL BINARY DUMP ==========");
    emit(Msg("[DUMP] path: ", path));
    emit(Msg("[DUMP] file_size: ", file_size, " bytes (", file_size / (1024*1024), " MB, ",
             file_size % (1024*1024), " remainder bytes)"));

    // --- Hex dump first 256 bytes with ASCII sidebar ---
    auto hexDumpRange = [&](std::size_t start, std::size_t end, const char* label) {
        emit(Msg("[DUMP] --- ", label, " (offset ", start, " to ", end - 1, ") ---"));
        for (std::size_t row = start; row < end; row += 16) {
            std::ostringstream hex;
            std::string ascii;
            hex << "[DUMP]   " << std::hex << std::setw(8) << std::setfill('0') << row << ": ";
            for (std::size_t col = 0; col < 16 && (row + col) < end; ++col) {
                uint8_t b = buffer[row + col];
                hex << std::setw(2) << std::setfill('0') << static_cast<int>(b) << " ";
                ascii += (b >= 32 && b < 127 ? static_cast<char>(b) : '.');
            }
            emit(hex.str() + " |" + ascii + "|");
        }
    };

    std::size_t head_len = std::min<std::size_t>(256, file_size);
    if (head_len > 0) hexDumpRange(0, head_len, "First 256 bytes");
    if (file_size > 512) {
        std::size_t tail_start = file_size - 256;
        hexDumpRange(tail_start, file_size, "Last 256 bytes");
    }

    // --- Zero-run scan (entire file, report runs >= 1KB) ---
    emit("[DUMP] --- Zero-region scan (runs >= 1KB) ---");
    std::size_t zero_run_start = 0;
    bool in_zero_run = false;
    std::size_t total_zero_bytes = 0;
    int zero_regions = 0;
    for (std::size_t i = 0; i < file_size; ++i) {
        if (buffer[i] == 0) {
            if (!in_zero_run) { zero_run_start = i; in_zero_run = true; }
            ++total_zero_bytes;
        } else {
            if (in_zero_run) {
                std::size_t run_len = i - zero_run_start;
                if (run_len >= 1024) {
                    emit(Msg("[DUMP]   ZERO RUN: offset=", zero_run_start, " length=", run_len,
                             " bytes (", run_len / 1024, " KB)"));
                    ++zero_regions;
                    if (zero_regions >= 50) { emit("[DUMP]   ... truncated (>50 zero regions)"); break; }
                }
            }
            in_zero_run = false;
        }
    }
    if (in_zero_run) {
        std::size_t run_len = file_size - zero_run_start;
        if (run_len >= 1024) {
            emit(Msg("[DUMP]   ZERO RUN (tail): offset=", zero_run_start, " length=", run_len,
                     " bytes — likely unflushed filesystem write"));
            ++zero_regions;
        }
    }
    emit(Msg("[DUMP]   Total zero bytes: ", total_zero_bytes, " / ", file_size,
             " (", (file_size > 0 ? total_zero_bytes * 100 / file_size : 0), "%)"));
    emit(Msg("[DUMP]   Zero regions found (>=1KB): ", zero_regions));
    if (file_size > 0 && total_zero_bytes * 100 / file_size > 50) {
        emit("[DUMP]   WARNING: >50% zero bytes — file is likely truncated or unflushed");
    }

    // --- FlatBuffer structure analysis ---
    emit("[DUMP] --- FlatBuffer structure analysis ---");
    if (file_size >= 4) {
        uint32_t root_off = 0;
        std::memcpy(&root_off, buffer.data(), 4);
        emit(Msg("[DUMP]   root_offset (bytes 0-3): ", root_off,
                 " valid=", (root_off < file_size ? "yes" : "NO — outside buffer")));
        if (root_off < file_size && root_off + 4 <= file_size) {
            int32_t vtable_soff = 0;
            std::memcpy(&vtable_soff, buffer.data() + root_off, 4);
            int64_t vtable_pos = static_cast<int64_t>(root_off) - vtable_soff;
            emit(Msg("[DUMP]   vtable signed_offset=", vtable_soff, " → vtable_pos=", vtable_pos));
            if (vtable_pos >= 0 && static_cast<std::size_t>(vtable_pos) + 4 <= file_size) {
                uint16_t vt_size = 0, obj_size = 0;
                std::memcpy(&vt_size, buffer.data() + vtable_pos, 2);
                std::memcpy(&obj_size, buffer.data() + vtable_pos + 2, 2);
                emit(Msg("[DUMP]   vtable_size=", vt_size, " object_inline_size=", obj_size,
                         " vtable_fields=", (vt_size > 4 ? (vt_size - 4) / 2 : 0)));
            } else {
                emit(Msg("[DUMP]   vtable_pos INVALID — outside buffer bounds"));
            }
        }
    }
    if (file_size >= 8) {
        char id[5] = {};
        std::memcpy(id, buffer.data() + 4, 4);
        uint32_t id_raw = 0;
        std::memcpy(&id_raw, buffer.data() + 4, 4);
        std::ostringstream id_hex;
        id_hex << "0x" << std::hex << std::setw(8) << std::setfill('0') << id_raw;
        emit("[DUMP]   file_identifier=\"" + std::string(id) + "\" (" + id_hex.str() + ") expected=\"GRMT\"");
    }

    // --- Byte frequency distribution (top 10 non-zero byte values) ---
    emit("[DUMP] --- Byte frequency (top 10 non-zero) ---");
    std::size_t freq[256] = {};
    for (std::size_t i = 0; i < file_size; ++i) ++freq[buffer[i]];
    std::vector<std::pair<int, std::size_t>> sorted;
    for (int i = 1; i < 256; ++i) {
        if (freq[i] > 0) sorted.push_back({i, freq[i]});
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    for (int i = 0; i < std::min<int>(10, static_cast<int>(sorted.size())); ++i) {
        std::ostringstream oss;
        oss << "[DUMP]   byte 0x" << std::hex << std::setw(2) << std::setfill('0') << sorted[i].first
            << std::dec << " ('" << (sorted[i].first >= 32 && sorted[i].first < 127
                                     ? static_cast<char>(sorted[i].first) : '.')
            << "'): " << sorted[i].second << " occurrences";
        emit(oss.str());
    }
    emit(Msg("[DUMP]   unique non-zero byte values: ", sorted.size(), " / 255"));

    // --- CUDA device state ---
    dumpCudaDeviceState();

    emit("========== END CHECKPOINT LOAD FAILURE DUMP ==========");
}

} // namespace

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                                                             \
    do {                                                                                                             \
        cudaError_t err__ = (call);                                                                                  \
        if (err__ != cudaSuccess) {                                                                                  \
            GRIM::Logging::EmitModuleError(kLogModule, Msg("[CUDA] ", __FILE__, ':', __LINE__, ": ",                 \
                                         cudaGetErrorString(err__)));                                                \
            throw std::runtime_error("CUDA failure");                                                                 \
        }                                                                                                            \
    } while (0)
#endif

namespace GRIM {

SerializationLayer::SerializationLayer(SerializationConfig config) : config_(std::move(config)) {}

void SerializationLayer::setConfig(const SerializationConfig& config) {
    config_ = config;
}

bool SerializationLayer::load(SerializationLoadRequest& request) {
    const auto& cfg = request.config;
    const auto& req = request.capabilities;

    if (cfg.vocab_size <= 0 || cfg.d_model <= 0 || cfg.num_layers <= 0) {
        Logging::EmitModuleError(kLogModule, "[load] Invalid model dimensions");
        return false;
    }
    if (request.encoder_layers.size() != static_cast<std::size_t>(cfg.num_layers)) {
        Logging::EmitModuleError(kLogModule, Msg("[load] Encoder layer mismatch (expected ",
                                            cfg.num_layers, ", got ", request.encoder_layers.size(), ")"));
        return false;
    }

    // ─── Step 1: Read file ───
    std::ifstream file(request.path, std::ios::binary | std::ios::ate);
    if (!file) {
        Logging::EmitModuleError(kLogModule, Msg("[load] Failed to open: ", request.path));
        return false;
    }
    const auto file_size = static_cast<std::size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(file_size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), file_size)) {
        Logging::EmitModuleError(kLogModule, "[load] Read failed");
        return false;
    }
    file.close();

    // ─── Step 2: Verify FlatBuffer ───
    auto emitFlatBufferLoadDiag = [&](const std::string& issue) {
        Logging::EmitModuleError(kLogModule, "[load] FlatBuffer: " + issue);
    };
    std::vector<std::string> issues;

    if (file_size < 8) {
        issues.push_back("buffer too small (need >= 8 bytes), got " + std::to_string(file_size));
    }
    if (file_size >= 4) {
        uint32_t first4 = 0;
        std::memcpy(&first4, buffer.data(), 4);
        if (first4 == GRIM::GRMT::kMagic) {
            issues.push_back("first 4 bytes = " + GRIM::GRMT::hex32(GRIM::GRMT::kMagic) +
                             " (GRMT magic) — looks like training data .grmt, not a checkpoint");
        }
        if (first4 >= file_size) {
            issues.push_back("root_offset=" + std::to_string(first4) +
                             " points outside buffer (size=" + std::to_string(file_size) + ")");
        }
    }
    if (file_size >= 8) {
        const char* expected_id = "GRMT";
        const char* file_id = reinterpret_cast<const char*>(buffer.data()) + 4;
        if (std::memcmp(file_id, expected_id, 4) != 0) {
            std::ostringstream ss;
            char observed_id[5] = { file_id[0], file_id[1], file_id[2], file_id[3], '\0' };
            uint32_t observed_raw = 0;
            std::memcpy(&observed_raw, file_id, 4);
            ss << "file_identifier wrong: expected \"GRMT\", got \""
               << observed_id << "\" (0x" << std::hex << observed_raw << std::dec << ")";
            issues.push_back(ss.str());
        }
    }

    flatbuffers::Verifier verifier(buffer.data(), buffer.size());
    if (!GRIMTransformer::VerifyTransformerModelBuffer(verifier)) {
        emitFlatBufferLoadDiag("verification failed — running component-level diagnostics...");

        // Bypass verifier and read raw — identify WHICH component fails
        const auto* raw_model = GRIMTransformer::GetTransformerModel(buffer.data());
        if (!raw_model) {
            emitFlatBufferLoadDiag("DIAG: GetTransformerModel returned NULL — root table broken");
        } else {
            Logging::EmitModuleError(kLogModule, Msg("[load] DIAG: version=", raw_model->version()));

            // Config
            {
                const auto* c = raw_model->config();
                if (!c) { emitFlatBufferLoadDiag("DIAG: config = NULL (required!)"); }
                else { flatbuffers::Verifier v(buffer.data(), buffer.size()); Logging::EmitModuleError(kLogModule, Msg("[load] DIAG: config verify=", c->Verify(v) ? "OK" : "FAIL")); }
            }
            // Embeddings
            {
                const auto* e = raw_model->embeddings();
                if (!e) { emitFlatBufferLoadDiag("DIAG: embeddings = NULL (required!)"); }
                else { flatbuffers::Verifier v(buffer.data(), buffer.size()); Logging::EmitModuleError(kLogModule, Msg("[load] DIAG: embeddings verify=", e->Verify(v) ? "OK" : "FAIL")); }
            }
            // Encoder layers
            if (raw_model->encoder_layers()) {
                Logging::EmitModuleError(kLogModule, Msg("[load] DIAG: encoder_layers count=", raw_model->encoder_layers()->size()));
                for (uint32_t i = 0; i < raw_model->encoder_layers()->size(); i++) {
                    const auto* layer = raw_model->encoder_layers()->Get(i);
                    if (!layer) {
                        Logging::EmitModuleError(kLogModule, Msg("[load] DIAG: encoder_layer[", i, "] = NULL"));
                        continue;
                    }
                    flatbuffers::Verifier vl(buffer.data(), buffer.size());
                    bool layer_ok = layer->Verify(vl);
                    if (!layer_ok) {
                        Logging::EmitModuleError(kLogModule, Msg("[load] DIAG: encoder_layer[", i, "] verify=FAIL"));
                        const auto* attn = layer->attention();
                        const auto* ffn = layer->ffn();
                        const auto* r1 = layer->rms1();
                        const auto* r2 = layer->rms2();
                        if (attn) { flatbuffers::Verifier va(buffer.data(), buffer.size()); Logging::EmitModuleError(kLogModule, Msg("[load] DIAG:   layer[", i, "].attention verify=", attn->Verify(va) ? "OK" : "FAIL")); }
                        else { Logging::EmitModuleError(kLogModule, Msg("[load] DIAG:   layer[", i, "].attention = NULL")); }
                        if (ffn) { flatbuffers::Verifier vf(buffer.data(), buffer.size()); Logging::EmitModuleError(kLogModule, Msg("[load] DIAG:   layer[", i, "].ffn verify=", ffn->Verify(vf) ? "OK" : "FAIL")); }
                        else { Logging::EmitModuleError(kLogModule, Msg("[load] DIAG:   layer[", i, "].ffn = NULL")); }
                        if (r1) { flatbuffers::Verifier vr1(buffer.data(), buffer.size()); Logging::EmitModuleError(kLogModule, Msg("[load] DIAG:   layer[", i, "].rms1 verify=", r1->Verify(vr1) ? "OK" : "FAIL")); }
                        if (r2) { flatbuffers::Verifier vr2(buffer.data(), buffer.size()); Logging::EmitModuleError(kLogModule, Msg("[load] DIAG:   layer[", i, "].rms2 verify=", r2->Verify(vr2) ? "OK" : "FAIL")); }
                    } else {
                        Logging::EmitModuleError(kLogModule, Msg("[load] DIAG: encoder_layer[", i, "] verify=OK"));
                    }
                }
            } else {
                emitFlatBufferLoadDiag("DIAG: encoder_layers = NULL (required!)");
            }
            // LM head
            {
                const auto* lm = raw_model->lm_head();
                if (!lm) { emitFlatBufferLoadDiag("DIAG: lm_head = NULL (required!)"); }
                else { flatbuffers::Verifier v(buffer.data(), buffer.size()); Logging::EmitModuleError(kLogModule, Msg("[load] DIAG: lm_head verify=", lm->Verify(v) ? "OK" : "FAIL")); }
            }
            // Training metadata
            {
                const auto* tm = raw_model->training_metadata();
                if (!tm) { emitFlatBufferLoadDiag("DIAG: training_metadata = NULL"); }
                else { flatbuffers::Verifier v(buffer.data(), buffer.size()); Logging::EmitModuleError(kLogModule, Msg("[load] DIAG: training_metadata verify=", tm->Verify(v) ? "OK" : "FAIL")); }
            }
            // Final RMS gamma
            if (raw_model->final_rms_gamma()) {
                Logging::EmitModuleError(kLogModule, Msg("[load] DIAG: final_rms_gamma size=", raw_model->final_rms_gamma()->size()));
            } else {
                emitFlatBufferLoadDiag("DIAG: final_rms_gamma = NULL");
            }
            {
                const auto* ltp = raw_model->latent_trajectory_preset();
                if (!ltp) { emitFlatBufferLoadDiag("DIAG: latent_trajectory_preset = NULL"); }
                else { flatbuffers::Verifier v(buffer.data(), buffer.size()); Logging::EmitModuleError(kLogModule, Msg("[load] DIAG: latent_trajectory_preset verify=", ltp->Verify(v) ? "OK" : "FAIL")); }
            }

            // Test with relaxed limits (higher max_depth + max_tables)
            flatbuffers::Verifier relaxed(buffer.data(), buffer.size(), 128, 10000000);
            bool relaxed_ok = GRIMTransformer::VerifyTransformerModelBuffer(relaxed);
            Logging::EmitModuleError(kLogModule, Msg("[load] DIAG: relaxed verifier (depth=128, tables=10M) = ", relaxed_ok ? "PASS" : "FAIL"));
        }

        for (const auto& s : issues) emitFlatBufferLoadDiag(s);
        dumpBinaryAnalysis(buffer, file_size, request.path);
        return false;
    }

    const auto* model_fb = GRIMTransformer::GetTransformerModel(buffer.data());
    if (!model_fb) {
        Logging::EmitModuleError(kLogModule, "[load] Failed to parse FlatBuffer — GetTransformerModel returned null");
        dumpBinaryAnalysis(buffer, file_size, request.path);
        return false;
    }

    // ─── Step 3: Version check ───
    if (model_fb->version() != GRIM_MODEL_VERSION) {
        Logging::EmitModuleError(kLogModule, Msg("[load] Version mismatch: checkpoint_version=",
                                            model_fb->version(), " expected=", GRIM_MODEL_VERSION));
        const auto* fb_cfg = model_fb->config();
        if (fb_cfg) {
            Logging::EmitModuleError(kLogModule, Msg("[load]   checkpoint config: vocab=",
                fb_cfg->vocab_size(), " d_model=", fb_cfg->d_model(),
                " layers=", fb_cfg->num_layers(), " heads=", fb_cfg->num_heads(),
                " kv_heads=", fb_cfg->num_kv_heads()));
        }
        Logging::EmitModuleError(kLogModule, Msg("[load]   model config: vocab=",
            cfg.vocab_size, " d_model=", cfg.d_model,
            " layers=", cfg.num_layers, " heads=", cfg.num_heads,
            " kv_heads=", cfg.num_kv_heads));
        Logging::EmitModuleError(kLogModule, Msg("[load]   path=", request.path,
            " file_size=", file_size));
        return false;
    }

    // ─── Step 4: VALIDATE — before any GPU writes ───
    if (!validate_checkpoint_capabilities(model_fb, cfg, req, request)) {
        return false;
    }

    Logging::EmitModuleInfo(kLogModule, Msg("[load] GQA: num_heads=",
        cfg.num_heads, " num_kv_heads=", cfg.num_kv_heads));

    // ─── Step 5 would be: if false → return (handled above) ───

    // ─── Step 6: LOAD — deterministic upload, gated by requires_* only ───

    auto upload_device_vector = [](const std::vector<float>& host,
                                   const DeviceWriteView& view,
                                   const char* label) -> bool {
        if (host.empty()) return true;
        if (!view.ptr) {
            Logging::EmitModuleError(kLogModule, Msg("[load] UPLOAD FAIL — null destination for ", label));
            Logging::EmitModuleError(kLogModule, Msg("[load]   host_elements=", host.size(),
                " host_bytes=", host.size() * sizeof(float),
                " view.count=", view.count, " view.ptr=NULL"));
            dumpCudaDeviceState();
            return false;
        }
        if (view.count != host.size()) {
            Logging::EmitModuleError(kLogModule, Msg("[load] UPLOAD FAIL — size mismatch for ", label));
            Logging::EmitModuleError(kLogModule, Msg("[load]   dest_count=", view.count,
                " src_size=", host.size(),
                " diff=", static_cast<int64_t>(view.count) - static_cast<int64_t>(host.size()),
                " dest_bytes=", view.count * sizeof(float),
                " src_bytes=", host.size() * sizeof(float)));
            dumpCudaDeviceState();
            return false;
        }
        cudaError_t copy_err = cudaMemcpy(view.ptr, host.data(),
                                          host.size() * sizeof(float), cudaMemcpyHostToDevice);
        if (copy_err != cudaSuccess) {
            Logging::EmitModuleError(kLogModule, Msg("[load] UPLOAD FAIL — cudaMemcpy failed for ", label,
                ": ", cudaGetErrorString(copy_err)));
            Logging::EmitModuleError(kLogModule, Msg("[load]   elements=", host.size(),
                " bytes=", host.size() * sizeof(float)));
            dumpCudaDeviceState();
            throw std::runtime_error("CUDA failure");
        }
        return true;
    };

    // ─── Embeddings ───
    const auto* fb_embeddings = model_fb->embeddings();
    const auto* fb_token_vec = fb_embeddings->token_embeddings();
    std::vector<float> token_host(fb_token_vec->begin(), fb_token_vec->end());
    const int vocab_size = static_cast<int>(fb_embeddings->vocab_size());
    const int d_model = static_cast<int>(fb_embeddings->d_model());

    if (cfg.use_gpu) {
        if (!request.gpu_embedding.token_embeddings.ptr) {
            Logging::EmitModuleError(kLogModule, "[load] GPU embedder not initialized");
            return false;
        }
        if (!upload_device_vector(token_host, request.gpu_embedding.token_embeddings, "token embeddings"))
            return false;
    }

    // ─── Encoder layers ───
    const auto* fb_layers = model_fb->encoder_layers();
    for (int layer_idx = 0; layer_idx < cfg.num_layers; ++layer_idx) {
        const auto* fb_layer = fb_layers->Get(layer_idx);
        const auto& layer_view = request.encoder_layers[layer_idx];

        const auto* fb_attn = fb_layer->attention();
        std::vector<float> h_W_qkv(fb_attn->w_qkv_data()->begin(), fb_attn->w_qkv_data()->end());
        std::vector<float> h_b_qkv;
        if (fb_attn->b_qkv_data())
            h_b_qkv.assign(fb_attn->b_qkv_data()->begin(), fb_attn->b_qkv_data()->end());
        std::vector<float> h_W_o(fb_attn->w_o_data()->begin(), fb_attn->w_o_data()->end());
        std::vector<float> h_b_o;
        if (fb_attn->b_o_data())
            h_b_o.assign(fb_attn->b_o_data()->begin(), fb_attn->b_o_data()->end());
        if (!upload_device_vector(h_W_qkv, layer_view.attn_w_qkv, "attn.W_qkv") ||
            !upload_device_vector(h_b_qkv, layer_view.attn_b_qkv, "attn.b_qkv") ||
            !upload_device_vector(h_W_o, layer_view.attn_w_o, "attn.W_o") ||
            !upload_device_vector(h_b_o, layer_view.attn_b_o, "attn.b_o"))
            return false;

        const auto* fb_ffn = fb_layer->ffn();
        std::vector<float> h_W_gate;
        if (fb_ffn->w_gate_data())
            h_W_gate.assign(fb_ffn->w_gate_data()->begin(), fb_ffn->w_gate_data()->end());
        std::vector<float> h_W1(fb_ffn->w1_data()->begin(), fb_ffn->w1_data()->end());
        std::vector<float> h_W2(fb_ffn->w2_data()->begin(), fb_ffn->w2_data()->end());
        std::vector<float> h_b2;
        if (fb_ffn->b2_data())
            h_b2.assign(fb_ffn->b2_data()->begin(), fb_ffn->b2_data()->end());
        if ((!h_W_gate.empty() && !upload_device_vector(h_W_gate, layer_view.ffn_w_gate, "ffn.W_gate")) ||
            !upload_device_vector(h_W1, layer_view.ffn_w1, "ffn.W1") ||
            !upload_device_vector(h_W2, layer_view.ffn_w2, "ffn.W2") ||
            !upload_device_vector(h_b2, layer_view.ffn_b2, "ffn.b2"))
            return false;

        std::vector<float> h_rms1(fb_layer->rms1()->gamma()->begin(), fb_layer->rms1()->gamma()->end());
        std::vector<float> h_rms2(fb_layer->rms2()->gamma()->begin(), fb_layer->rms2()->gamma()->end());
        if (layer_view.rms1_gamma.ptr && !upload_device_vector(h_rms1, layer_view.rms1_gamma, "rms1.gamma"))
            return false;
        if (layer_view.rms2_gamma.ptr && !upload_device_vector(h_rms2, layer_view.rms2_gamma, "rms2.gamma"))
            return false;

        if (layer_view.layer_scale1.ptr) {
            std::vector<float> h_ls1(fb_layer->layer_scale1()->begin(), fb_layer->layer_scale1()->end());
            if (!upload_device_vector(h_ls1, layer_view.layer_scale1, "layer_scale1"))
                return false;
        }
        if (layer_view.layer_scale2.ptr) {
            std::vector<float> h_ls2(fb_layer->layer_scale2()->begin(), fb_layer->layer_scale2()->end());
            if (!upload_device_vector(h_ls2, layer_view.layer_scale2, "layer_scale2"))
                return false;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // ─── LM head ───
    const auto* fb_lm_head = model_fb->lm_head();
    if (fb_lm_head) {
        if (request.lm_head.projection.ptr && fb_lm_head->projection_data()) {
            std::vector<float> lm_proj_host(fb_lm_head->projection_data()->begin(), fb_lm_head->projection_data()->end());
            if (!upload_device_vector(lm_proj_host, request.lm_head.projection, "LM head projection"))
                return false;
        } else if (!cfg.tie_embeddings) {
            Logging::EmitModuleError(kLogModule, "[load] LM head projection missing (tie_embeddings=false)");
            return false;
        }

        // Presence-driven: load the bias whenever the model allocated a bias
        // destination (covers use_bias and the dedicated unigram bias).
        if (request.lm_head.bias.ptr && fb_lm_head->bias_data()) {
            std::vector<float> lm_bias_host(fb_lm_head->bias_data()->begin(), fb_lm_head->bias_data()->end());
            if (!upload_device_vector(lm_bias_host, request.lm_head.bias, "LM head bias"))
                return false;
        } else if (request.lm_head.expect_bias) {
            Logging::EmitModuleError(kLogModule, "[load] LM head bias missing (model allocated a bias destination)");
            return false;
        }

        // Head-side residual SwiGLU adapter (config.lm_head_mlp_enabled).
        // Same presence-driven contract as the bias: a model that allocated the
        // adapter must find it in the checkpoint (never silently re-initialize
        // trained adapter weights); a checkpoint carrying an adapter the model
        // dropped is skipped with a notice.
        const bool ckpt_has_mlp = fb_lm_head->mlp_enabled()
                               && fb_lm_head->mlp_w_gate_data()
                               && fb_lm_head->mlp_w_up_data()
                               && fb_lm_head->mlp_w_down_data();
        if (request.lm_head.expect_mlp) {
            if (!ckpt_has_mlp) {
                Logging::EmitModuleError(kLogModule,
                    "[load] LM head residual SwiGLU adapter missing in checkpoint "
                    "(model allocated adapter destinations — lm_head_mlp_enabled)");
                return false;
            }
            std::vector<float> mlp_gate_host(fb_lm_head->mlp_w_gate_data()->begin(), fb_lm_head->mlp_w_gate_data()->end());
            if (!upload_device_vector(mlp_gate_host, request.lm_head.mlp_w_gate, "LM head mlp_W_gate"))
                return false;
            std::vector<float> mlp_up_host(fb_lm_head->mlp_w_up_data()->begin(), fb_lm_head->mlp_w_up_data()->end());
            if (!upload_device_vector(mlp_up_host, request.lm_head.mlp_w_up, "LM head mlp_W_up"))
                return false;
            std::vector<float> mlp_down_host(fb_lm_head->mlp_w_down_data()->begin(), fb_lm_head->mlp_w_down_data()->end());
            if (!upload_device_vector(mlp_down_host, request.lm_head.mlp_w_down, "LM head mlp_W_down"))
                return false;
        } else if (ckpt_has_mlp) {
            Logging::EmitModuleInfo(kLogModule,
                "[load] Checkpoint carries an LM-head SwiGLU adapter but lm_head_mlp_enabled "
                "is off — adapter weights skipped");
        }
    }

    // ─── NumberEncoder (gated by requires_number_encoder) ───
    if (req.requires_number_encoder) {
        const auto* fb_ne = model_fb->number_encoder();
        auto ul = [&](const flatbuffers::Vector<float>* src, const DeviceWriteView& dst, const char* name) -> bool {
            if (!src) return false;
            std::vector<float> buf(src->begin(), src->end());
            return upload_device_vector(buf, dst, name);
        };
        // Optional loader for use_bias-gated hidden biases. Tolerates the four
        // combinations of {checkpoint has bias} × {model allocated a bias dest}:
        // both present → load; both absent → skip; model wants a bias the
        // checkpoint lacks → fail; checkpoint has a bias the model dropped
        // (e.g. resuming a pre-gate checkpoint with use_bias=false) → skip.
        auto ul_opt = [&](const flatbuffers::Vector<float>* src, const DeviceWriteView& dst, const char* name) -> bool {
            const bool have_src = (src != nullptr);
            const bool have_dst = (dst.ptr != nullptr);
            if (have_src && have_dst) return ul(src, dst, name);
            if (!have_src && !have_dst) return true;
            if (!have_src && have_dst) {
                Logging::EmitModuleError(kLogModule, Msg("[load] NumberEncoder bias destination present but missing in checkpoint: ", name));
                return false;
            }
            Logging::EmitModuleInfo(kLogModule, Msg("[load] skipping NumberEncoder bias absent in model (use_bias gated off): ", name));
            return true;
        };
        const auto& ne = request.number_encoder;
        bool ne_ok = true;
        ne_ok = ne_ok && ul(fb_ne->digit_emb_data(), ne.digit_emb, "NE digit_emb");
        ne_ok = ne_ok && ul(fb_ne->pow10_emb_data(), ne.pow10_emb, "NE pow10_emb");
        ne_ok = ne_ok && ul(fb_ne->w_c1_data(), ne.W_c1, "NE W_c1");
        ne_ok = ne_ok && ul_opt(fb_ne->b_c1_data(), ne.b_c1, "NE b_c1");
        ne_ok = ne_ok && ul(fb_ne->w_c2_data(), ne.W_c2, "NE W_c2");
        ne_ok = ne_ok && ul(fb_ne->w_g1_data(), ne.W_g1, "NE W_g1");
        ne_ok = ne_ok && ul_opt(fb_ne->b_g1_data(), ne.b_g1, "NE b_g1");
        ne_ok = ne_ok && ul(fb_ne->w_g2_data(), ne.W_g2, "NE W_g2");
        if (!ne_ok) return false;
        request.report.number_encoder_loaded = true;
        Logging::EmitModuleInfo(kLogModule, "[load] NumberEncoder weights loaded");
    }

    // ─── Arg/option selector (gated by requires_arg_selector) ───
    if (req.requires_arg_selector) {
        const auto* fb_sel = model_fb->arg_selector();
        auto ul = [&](const flatbuffers::Vector<float>* src, const DeviceWriteView& dst, const char* name) -> bool {
            if (!src) return false;
            std::vector<float> buf(src->begin(), src->end());
            return upload_device_vector(buf, dst, name);
        };
        const auto& sel = request.arg_selector;
        if (!fb_sel || !ul(fb_sel->w_q_data(), sel.W_q, "SEL W_q")) {
            Logging::EmitModuleError(kLogModule, "[load] FATAL: arg_selector required but missing/invalid in checkpoint");
            return false;
        }
        request.report.arg_selector_loaded = true;
        Logging::EmitModuleInfo(kLogModule, "[load] ArgSelector weights loaded");
    }

    // ─── ExecutionBlock (gated by requires_execution_block) ───
    if (req.requires_execution_block) {
        const auto* fb_eb = model_fb->execution_block();
        auto ul = [&](const flatbuffers::Vector<float>* src, const DeviceWriteView& dst, const char* name) -> bool {
            if (!src) return true;  // Field absent in checkpoint — skip (e.g., older schema)
            std::vector<float> buf(src->begin(), src->end());
            return upload_device_vector(buf, dst, name);
        };
        const auto& eb = request.execution_block;
        bool eb_ok = true;
        eb_ok = eb_ok && ul(fb_eb->w_decode_1_data(), eb.w_decode_1, "EB w_decode_1");
        eb_ok = eb_ok && ul(fb_eb->b_decode_1_data(), eb.b_decode_1, "EB b_decode_1");
        eb_ok = eb_ok && ul(fb_eb->w_decode_2_data(), eb.w_decode_2, "EB w_decode_2");
        eb_ok = eb_ok && ul(fb_eb->w_arg1_select_data(), eb.w_arg1_select, "EB w_arg1_select");
        eb_ok = eb_ok && ul(fb_eb->w_arg2_select_data(), eb.w_arg2_select, "EB w_arg2_select");
        eb_ok = eb_ok && ul(fb_eb->w_op_select_data(), eb.W_op_select, "EB W_op_select");
        eb_ok = eb_ok && ul(fb_eb->w_key_proj_data(), eb.W_key_proj, "EB W_key_proj");
        eb_ok = eb_ok && ul(fb_eb->w_write_query_data(), eb.W_write_query, "EB W_write_query");
        eb_ok = eb_ok && ul(fb_eb->w_write_key_data(), eb.W_write_key, "EB W_write_key");
        eb_ok = eb_ok && ul(fb_eb->alpha_data(), eb.alpha, "EB alpha");
        eb_ok = eb_ok && ul(fb_eb->beta_data(), eb.beta, "EB beta");
        eb_ok = eb_ok && ul(fb_eb->step_embeddings_data(), eb.step_embeddings, "EB step_embeddings");
        eb_ok = eb_ok && ul(fb_eb->type_num_embed_data(), eb.type_num_embed, "EB type_num_embed");
        eb_ok = eb_ok && ul(fb_eb->w_value_to_emb_data(), eb.W_value_to_emb, "EB W_value_to_emb");
        eb_ok = eb_ok && ul(fb_eb->b_value_to_emb_data(), eb.b_value_to_emb, "EB b_value_to_emb");
        eb_ok = eb_ok && ul(fb_eb->w_inject_gate_data(), eb.w_inject_gate, "EB w_inject_gate");
        eb_ok = eb_ok && ul(fb_eb->w_q_read_data(), eb.W_Q_read, "EB W_Q_read");
        eb_ok = eb_ok && ul(fb_eb->w_k_read_data(), eb.W_K_read, "EB W_K_read");
        eb_ok = eb_ok && ul(fb_eb->w_v_read_data(), eb.W_V_read, "EB W_V_read");
        eb_ok = eb_ok && ul(fb_eb->w_o_read_data(), eb.W_O_read, "EB W_O_read");
        eb_ok = eb_ok && ul(fb_eb->w_gate_read_data(), eb.W_gate_read, "EB W_gate_read");
        eb_ok = eb_ok && ul(fb_eb->tau_data(), eb.tau, "EB tau");
        eb_ok = eb_ok && ul(fb_eb->e_slot_data(), eb.E_slot, "EB E_slot");
        eb_ok = eb_ok && ul(fb_eb->e_op_data(), eb.E_op, "EB E_op");
        eb_ok = eb_ok && ul(fb_eb->w_scal_data(), eb.W_scal, "EB W_scal");
        eb_ok = eb_ok && ul(fb_eb->b_scal_data(), eb.b_scal, "EB b_scal");
        eb_ok = eb_ok && ul(fb_eb->w_trace_data(), eb.W_trace, "EB W_trace");
        eb_ok = eb_ok && ul(fb_eb->b_trace_data(), eb.b_trace, "EB b_trace");
        eb_ok = eb_ok && ul(fb_eb->w_reason_gate_data(), eb.W_reason_gate, "EB W_reason_gate");
        // W_trace_gate is optional for backward compatibility with pre-Fix#5 checkpoints
        if (fb_eb->w_trace_gate_data()) {
            eb_ok = eb_ok && ul(fb_eb->w_trace_gate_data(), eb.W_trace_gate, "EB W_trace_gate");
        }
        if (!eb_ok) return false;
        request.report.execution_block_loaded = true;
        Logging::EmitModuleInfo(kLogModule, "[load] ExecutionBlock v2 weights loaded");
    }

    // ─── LatentTrajectoryPreset (gated by requires_latent_trajectory_preset) ───
    if (req.requires_latent_trajectory_preset) {
        const auto* fb_ltp = model_fb->latent_trajectory_preset();
        if (!fb_ltp) {
            Logging::EmitModuleError(kLogModule, "[load] FATAL: LatentTrajectoryPreset required but missing in checkpoint");
            return false;
        }
        auto ul = [&](const flatbuffers::Vector<float>* src, const DeviceWriteView& dst, const char* name) -> bool {
            if (!src) return false;
            std::vector<float> buf(src->begin(), src->end());
            return upload_device_vector(buf, dst, name);
        };
        const auto& ltp = request.latent_trajectory_preset;
        bool ltp_ok = true;
        ltp_ok = ltp_ok && ul(fb_ltp->w_hidden_traj_data(), ltp.W_hidden_traj, "LTP W_hidden_traj");
        ltp_ok = ltp_ok && ul(fb_ltp->b_hidden_traj_data(), ltp.b_hidden_traj, "LTP b_hidden_traj");
        ltp_ok = ltp_ok && ul(fb_ltp->w_fuse_data(), ltp.W_fuse, "LTP W_fuse");
        ltp_ok = ltp_ok && ul(fb_ltp->b_fuse_data(), ltp.b_fuse, "LTP b_fuse");
        ltp_ok = ltp_ok && ul(fb_ltp->w_down_data(), ltp.W_down, "LTP W_down");
        ltp_ok = ltp_ok && ul(fb_ltp->b_down_data(), ltp.b_down, "LTP b_down");
        ltp_ok = ltp_ok && ul(fb_ltp->w_up_data(), ltp.W_up, "LTP W_up");
        ltp_ok = ltp_ok && ul(fb_ltp->b_up_data(), ltp.b_up, "LTP b_up");
        ltp_ok = ltp_ok && ul(fb_ltp->w_gate_data(), ltp.W_gate, "LTP W_gate");
        ltp_ok = ltp_ok && ul(fb_ltp->b_gate_data(), ltp.b_gate, "LTP b_gate");
        ltp_ok = ltp_ok && ul(fb_ltp->fuse_norm_gamma_data(), ltp.fuse_norm_gamma, "LTP fuse_norm_gamma");
        ltp_ok = ltp_ok && ul(fb_ltp->preset_norm_gamma_data(), ltp.preset_norm_gamma, "LTP preset_norm_gamma");
        if (!ltp_ok) return false;
        if (fb_ltp->codebook_data()) {
            if (!ul(fb_ltp->codebook_data(), ltp.codebook, "LTP codebook") ||
                !ul(fb_ltp->w_slots_data(), ltp.W_slots, "LTP W_slots")) {
                return false;
            }
            Logging::EmitModuleInfo(kLogModule, "[load] LatentTrajectoryPreset codebook and slot decoder loaded");
        } else {
            Logging::EmitModuleInfo(kLogModule,
                "[load] LatentTrajectoryPreset codebook absent; fresh initialization retained");
        }
        request.report.latent_trajectory_preset_loaded = true;
        Logging::EmitModuleInfo(kLogModule, "[load] LatentTrajectoryPreset weights loaded");
    }

    // ─── final_rms_gamma (gated by requires_final_rms_gamma) ───
    if (req.requires_final_rms_gamma) {
        const auto* fb_frg = model_fb->final_rms_gamma();
        std::vector<float> frg_data(fb_frg->begin(), fb_frg->end());
        if (!upload_device_vector(frg_data, request.final_rms_gamma, "final_rms_gamma"))
            return false;
        Logging::EmitModuleInfo(kLogModule, Msg("[load] final_rms_gamma: size=", frg_data.size()));
    }

    // ─── Step 7: Final load verification (safety) ───
    if (req.requires_number_encoder && !request.report.number_encoder_loaded) {
        Logging::EmitModuleError(kLogModule, "[load] FATAL: NumberEncoder required but not loaded");
        return false;
    }
    if (req.requires_arg_selector && !request.report.arg_selector_loaded) {
        Logging::EmitModuleError(kLogModule, "[load] FATAL: ArgSelector required but not loaded");
        return false;
    }
    if (req.requires_execution_block && !request.report.execution_block_loaded) {
        Logging::EmitModuleError(kLogModule, "[load] FATAL: ExecutionBlock required but not loaded");
        return false;
    }
    if (req.requires_latent_trajectory_preset && !request.report.latent_trajectory_preset_loaded) {
        Logging::EmitModuleError(kLogModule, "[load] FATAL: LatentTrajectoryPreset required but not loaded");
        return false;
    }

    Logging::EmitModuleInfo(kLogModule, "[load] Model loaded successfully");
    return true;
}

} // namespace GRIM

/**
 * PyTorchVerify.hpp - Side-by-side PyTorch verification for GRIM-text
 * 
 * When GRIM_PYTORCH_VERIFY is defined, this module runs PyTorch reference
 * implementations alongside CUDA computations and logs both for comparison.
 * 
 * Usage:
 *   #define GRIM_PYTORCH_VERIFY  // Enable before including EquationLogging.hpp
 *   
 * The verification works by:
 *   1. Writing tensor data to a temp file
 *   2. Running a Python subprocess with pytorch_verify.py
 *   3. Reading back PyTorch's result
 *   4. Logging comparison in equation format
 * 
 * This is SLOW but gives ground-truth comparison for debugging.
 */

#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <cuda_runtime.h>  // For cudaMemcpy in verification

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace GRIM {
namespace PyTorchVerify {

// ============================================================================
// CONFIGURATION
// ============================================================================

// Path to the PyTorch verification script (relative to GRIM root)
constexpr const char* PYTORCH_VERIFY_SCRIPT = "resources/models/GRIM-text/Shared/EquationLogging/pytorch_verify.py";

// Temp directory for tensor exchange
constexpr const char* TEMP_DIR = "temp/pytorch_verify";

// Tolerance for floating-point comparison
constexpr float RTOL = 1e-4f;  // Relative tolerance
constexpr float ATOL = 1e-6f;  // Absolute tolerance

// ============================================================================
// HELPER: Write tensor to binary file (handles CUDA device pointers)
// ============================================================================

inline bool writeTensorToFile(const std::string& filepath, 
                               const float* device_data, 
                               size_t num_elements,
                               const std::vector<int>& shape) {
    // Allocate host buffer and copy from device
    std::vector<float> host_data(num_elements);
    cudaError_t err = cudaMemcpy(host_data.data(), device_data, 
                                  num_elements * sizeof(float), 
                                  cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "[PyTorchVerify] cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    std::ofstream f(filepath, std::ios::binary);
    if (!f.is_open()) return false;
    
    // Write shape
    int ndim = static_cast<int>(shape.size());
    f.write(reinterpret_cast<const char*>(&ndim), sizeof(int));
    for (int dim : shape) {
        f.write(reinterpret_cast<const char*>(&dim), sizeof(int));
    }
    
    // Write data (from host buffer)
    f.write(reinterpret_cast<const char*>(host_data.data()), num_elements * sizeof(float));
    return f.good();
}

// ============================================================================
// HELPER: Copy device tensor to host vector (for comparison)
// ============================================================================

inline std::vector<float> copyDeviceToHost(const float* device_data, size_t num_elements) {
    std::vector<float> host_data(num_elements);
    cudaError_t err = cudaMemcpy(host_data.data(), device_data, 
                                  num_elements * sizeof(float), 
                                  cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "[PyTorchVerify] cudaMemcpy failed in copyDeviceToHost: " << cudaGetErrorString(err) << std::endl;
        return std::vector<float>();  // Return empty on error
    }
    return host_data;
}

// ============================================================================
// HELPER: Read tensor from binary file
// ============================================================================

inline bool readTensorFromFile(const std::string& filepath,
                                std::vector<float>& data,
                                std::vector<int>& shape) {
    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open()) return false;
    
    // Read shape
    int ndim;
    f.read(reinterpret_cast<char*>(&ndim), sizeof(int));
    shape.resize(ndim);
    size_t num_elements = 1;
    for (int i = 0; i < ndim; i++) {
        f.read(reinterpret_cast<char*>(&shape[i]), sizeof(int));
        num_elements *= shape[i];
    }
    
    // Read data
    data.resize(num_elements);
    f.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(float));
    return f.good();
}

// ============================================================================
// HELPER: Run Python subprocess
// ============================================================================

inline std::string runPythonCommand(const std::string& cmd) {
    std::string result;
    
#ifdef _WIN32
    FILE* pipe = _popen(cmd.c_str(), "r");
#else
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    
    if (!pipe) return "";
    
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    
#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    
    return result;
}

// ============================================================================
// HELPER: Ensure temp directory exists
// ============================================================================

inline void ensureTempDir() {
    std::filesystem::create_directories(TEMP_DIR);
}

// ============================================================================
// HELPER: Compute tensor statistics
// ============================================================================

struct TensorStats {
    float min = 0.0f;
    float max = 0.0f;
    float mean = 0.0f;
    float rms = 0.0f;
    size_t count = 0;
    bool has_nan = false;
    bool has_inf = false;
};

inline TensorStats computeStats(const float* data, size_t n) {
    TensorStats s;
    s.count = n;
    if (n == 0) return s;
    
    s.min = data[0];
    s.max = data[0];
    double sum = 0.0, sum_sq = 0.0;
    
    for (size_t i = 0; i < n; i++) {
        float v = data[i];
        if (std::isnan(v)) { s.has_nan = true; continue; }
        if (std::isinf(v)) { s.has_inf = true; continue; }
        s.min = std::min(s.min, v);
        s.max = std::max(s.max, v);
        sum += v;
        sum_sq += v * v;
    }
    
    s.mean = static_cast<float>(sum / n);
    s.rms = static_cast<float>(std::sqrt(sum_sq / n));
    return s;
}

// ============================================================================
// HELPER: Compare tensors with tolerance
// ============================================================================

struct CompareResult {
    bool match = true;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    size_t mismatch_count = 0;
    size_t first_mismatch_idx = 0;
};

inline CompareResult compareTensors(const float* cuda_data, 
                                     const float* pytorch_data, 
                                     size_t n,
                                     float rtol = RTOL, 
                                     float atol = ATOL) {
    CompareResult r;
    
    for (size_t i = 0; i < n; i++) {
        float c = cuda_data[i];
        float p = pytorch_data[i];
        
        float abs_diff = std::abs(c - p);
        float rel_diff = abs_diff / (std::abs(p) + 1e-8f);
        
        r.max_abs_diff = std::max(r.max_abs_diff, abs_diff);
        r.max_rel_diff = std::max(r.max_rel_diff, rel_diff);
        
        // Check if within tolerance: |c - p| <= atol + rtol * |p|
        bool within_tol = abs_diff <= atol + rtol * std::abs(p);
        
        if (!within_tol) {
            if (r.mismatch_count == 0) {
                r.first_mismatch_idx = i;
            }
            r.mismatch_count++;
            r.match = false;
        }
    }
    
    return r;
}

// ============================================================================
// FORMAT HELPERS
// ============================================================================

inline std::string formatStats(const TensorStats& s, const std::string& name) {
    std::ostringstream oss;
    oss << name << ": min=" << s.min << " max=" << s.max 
        << " mean=" << s.mean << " rms=" << s.rms;
    if (s.has_nan) oss << " [HAS_NAN]";
    if (s.has_inf) oss << " [HAS_INF]";
    return oss.str();
}

inline std::string formatCompare(const CompareResult& r) {
    std::ostringstream oss;
    if (r.match) {
        oss << "[MATCH] max_abs_diff=" << r.max_abs_diff << " max_rel_diff=" << r.max_rel_diff;
    } else {
        oss << "[MISMATCH] " << r.mismatch_count << " elements differ, "
            << "max_abs=" << r.max_abs_diff << " max_rel=" << r.max_rel_diff
            << " first_at=" << r.first_mismatch_idx;
    }
    return oss.str();
}

// ============================================================================
// VERIFICATION CLASS
// ============================================================================

class PyTorchVerifier {
public:
    bool enabled_ = false;
    std::string python_path_ = "python";  // Or full path to venv Python
    std::string script_path_;
    std::string temp_dir_;
    int call_count_ = 0;
    int mismatch_count_ = 0;
    
    // Constructor
    PyTorchVerifier() = default;
    
    // Initialize verifier
    bool initialize(const std::string& grim_root = ".") {
        temp_dir_ = std::string(grim_root) + "/" + TEMP_DIR;
        script_path_ = std::string(grim_root) + "/" + PYTORCH_VERIFY_SCRIPT;
        
        // Check if script exists
        if (!std::filesystem::exists(script_path_)) {
            std::cerr << "[PyTorchVerify] Script not found: " << script_path_ << std::endl;
            return false;
        }
        
        // Create temp directory and clear any stale files from previous runs
        std::filesystem::create_directories(temp_dir_);
        for (const auto& entry : std::filesystem::directory_iterator(temp_dir_)) {
            if (entry.is_regular_file() && entry.path().extension() == ".bin") {
                std::filesystem::remove(entry.path());
            }
        }
        
        // Quick Python test
        std::string test_cmd = python_path_ + " -c \"import torch; print('PyTorch', torch.__version__)\"";
        std::string result = runPythonCommand(test_cmd);
        if (result.empty() || result.find("PyTorch") == std::string::npos) {
            std::cerr << "[PyTorchVerify] PyTorch not available" << std::endl;
            return false;
        }
        
        std::cout << "[PyTorchVerify] Initialized with " << result;
        enabled_ = true;
        return true;
    }
    
    // ========================================================================
    // VERIFICATION METHODS - Each matches an equation phase
    // ========================================================================
    
    // Verify RMSNorm: y = x * gamma / sqrt(mean(x²) + eps)
    std::string verifyRMSNorm(
        const float* x_data, const float* gamma_data, const float* cuda_output,
        int batch_size, int hidden_dim, float eps,
        int batch_idx, int layer_idx, int step_idx
    ) {
        if (!enabled_) return "";
        call_count_++;
        
        size_t n = batch_size * hidden_dim;
        
        // Write inputs
        std::string x_path = temp_dir_ + "/rmsnorm_x.bin";
        std::string gamma_path = temp_dir_ + "/rmsnorm_gamma.bin";
        std::string out_path = temp_dir_ + "/rmsnorm_out.bin";
        
        writeTensorToFile(x_path, x_data, n, {batch_size, hidden_dim});
        writeTensorToFile(gamma_path, gamma_data, hidden_dim, {hidden_dim});
        
        // Run PyTorch
        std::ostringstream cmd;
        cmd << python_path_ << " \"" << script_path_ << "\" rmsnorm "
            << "\"" << x_path << "\" \"" << gamma_path << "\" \"" << out_path << "\" "
            << eps;
        
        std::string py_result = runPythonCommand(cmd.str());
        
        // Read PyTorch output
        std::vector<float> pytorch_out;
        std::vector<int> out_shape;
        if (!readTensorFromFile(out_path, pytorch_out, out_shape)) {
            return "[PYTORCH_VERIFY] Failed to read PyTorch output";
        }
        
        // Copy CUDA output to host for comparison
        std::vector<float> cuda_out_host = copyDeviceToHost(cuda_output, n);
        if (cuda_out_host.empty()) {
            return "[PYTORCH_VERIFY] Failed to copy CUDA output to host";
        }
        
        // Compare (both are now host pointers)
        CompareResult cmp = compareTensors(cuda_out_host.data(), pytorch_out.data(), n);
        if (!cmp.match) mismatch_count_++;
        
        // Format result
        std::ostringstream result;
        result << "[RMSNORM_VERIFY] batch=" << batch_idx << " layer=" << layer_idx 
               << " step=" << step_idx << "\n";
        result << "  CUDA: " << formatStats(computeStats(cuda_out_host.data(), n), "output") << "\n";
        result << "  PyTorch: " << formatStats(computeStats(pytorch_out.data(), n), "output") << "\n";
        result << "  " << formatCompare(cmp) << "\n";
        
        return result.str();
    }
    
    // Verify Embedding lookup with scale: y = embedding[tokens] * scale
    std::string verifyEmbedding(
        const float* weight_data, const int* tokens,
        const float* cuda_output,
        int vocab_size, int hidden_dim, int seq_len,
        float scale,
        int batch_idx, int step_idx
    ) {
        if (!enabled_) return "";
        call_count_++;
        
        // Write inputs
        std::string weight_path = temp_dir_ + "/emb_weight.bin";
        std::string tokens_path = temp_dir_ + "/emb_tokens.bin";
        std::string out_path = temp_dir_ + "/emb_out.bin";
        
        writeTensorToFile(weight_path, weight_data, vocab_size * hidden_dim, {vocab_size, hidden_dim});
        
        // Copy tokens from device to host
        std::vector<int> tokens_host(seq_len);
        cudaError_t err = cudaMemcpy(tokens_host.data(), tokens, 
                                      seq_len * sizeof(int), 
                                      cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            return "[PYTORCH_VERIFY] Failed to copy tokens to host";
        }
        
        // Write tokens as int
        std::ofstream tf(tokens_path, std::ios::binary);
        int ndim = 1;
        tf.write(reinterpret_cast<const char*>(&ndim), sizeof(int));
        tf.write(reinterpret_cast<const char*>(&seq_len), sizeof(int));
        tf.write(reinterpret_cast<const char*>(tokens_host.data()), seq_len * sizeof(int));
        tf.close();
        
        // Run PyTorch
        std::ostringstream cmd;
        cmd << python_path_ << " \"" << script_path_ << "\" embedding "
            << "\"" << weight_path << "\" \"" << tokens_path << "\" \"" << out_path << "\" "
            << scale;
        
        std::string py_result = runPythonCommand(cmd.str());
        
        // Read PyTorch output
        std::vector<float> pytorch_out;
        std::vector<int> out_shape;
        if (!readTensorFromFile(out_path, pytorch_out, out_shape)) {
            return "[PYTORCH_VERIFY] Failed to read PyTorch output";
        }
        
        size_t n = seq_len * hidden_dim;
        
        // Copy CUDA output to host for comparison
        std::vector<float> cuda_out_host = copyDeviceToHost(cuda_output, n);
        if (cuda_out_host.empty()) {
            return "[PYTORCH_VERIFY] Failed to copy CUDA output to host";
        }
        
        CompareResult cmp = compareTensors(cuda_out_host.data(), pytorch_out.data(), n);
        if (!cmp.match) mismatch_count_++;
        
        std::ostringstream result;
        result << "[EMBEDDING_VERIFY] batch=" << batch_idx << " step=" << step_idx 
               << " scale=" << scale << "\n";
        result << "  CUDA: " << formatStats(computeStats(cuda_out_host.data(), n), "output") << "\n";
        result << "  PyTorch: " << formatStats(computeStats(pytorch_out.data(), n), "output") << "\n";
        result << "  " << formatCompare(cmp) << "\n";
        
        return result.str();
    }
    
    // Verify MatMul: C = A @ B (or C = A @ B^T if transpose_b)
    std::string verifyMatMul(
        const float* A_data, const float* B_data, const float* cuda_output,
        int M, int K, int N,
        const std::string& op_name,
        int batch_idx, int layer_idx, int step_idx,
        bool transpose_b = false
    ) {
        if (!enabled_) return "";
        call_count_++;
        
        // Use call_count_ in filenames to prevent reuse between calls (Issue: PyTorch reads stale file)
        std::string suffix = "_" + std::to_string(call_count_);
        std::string a_path = temp_dir_ + "/matmul_a" + suffix + ".bin";
        std::string b_path = temp_dir_ + "/matmul_b" + suffix + ".bin";
        std::string out_path = temp_dir_ + "/matmul_out" + suffix + ".bin";
        
        writeTensorToFile(a_path, A_data, M * K, {M, K});
        
        // When transpose_b=true, B is stored as [N, K] (GRIM uses row-major [out_dim, in_dim])
        // When transpose_b=false, B is stored as [K, N]
        if (transpose_b) {
            writeTensorToFile(b_path, B_data, N * K, {N, K});
        } else {
            writeTensorToFile(b_path, B_data, K * N, {K, N});
        }
        
        std::ostringstream cmd;
        cmd << python_path_ << " \"" << script_path_ << "\" matmul "
            << "\"" << a_path << "\" \"" << b_path << "\" \"" << out_path << "\" "
            << (transpose_b ? "1" : "0");
        
        runPythonCommand(cmd.str());
        
        std::vector<float> pytorch_out;
        std::vector<int> out_shape;
        if (!readTensorFromFile(out_path, pytorch_out, out_shape)) {
            return "[PYTORCH_VERIFY] Failed to read PyTorch output";
        }
        
        // Clean up temp files to prevent disk space bloat
        std::remove(a_path.c_str());
        std::remove(b_path.c_str());
        std::remove(out_path.c_str());
        
        size_t n = M * N;
        
        // Validate PyTorch output size matches expected
        if (pytorch_out.size() != n) {
            std::ostringstream err;
            err << "[PYTORCH_VERIFY] Size mismatch! Expected " << n 
                << " (M=" << M << " * N=" << N << "), got " << pytorch_out.size()
                << " from PyTorch";
            return err.str();
        }
        
        // Copy CUDA output to host for comparison
        std::vector<float> cuda_out_host = copyDeviceToHost(cuda_output, n);
        if (cuda_out_host.empty()) {
            return "[PYTORCH_VERIFY] Failed to copy CUDA output to host";
        }
        
        CompareResult cmp = compareTensors(cuda_out_host.data(), pytorch_out.data(), n);
        if (!cmp.match) mismatch_count_++;
        
        std::ostringstream result;
        const char* b_shape_str = transpose_b ? "^T" : "";
        result << "[" << op_name << "_VERIFY] batch=" << batch_idx 
               << " layer=" << layer_idx << " step=" << step_idx
               << " dims=[" << M << "x" << K << "]@[" << K << "x" << N << "]" << b_shape_str << "\n";
        result << "  CUDA: " << formatStats(computeStats(cuda_out_host.data(), n), "output") << "\n";
        result << "  PyTorch: " << formatStats(computeStats(pytorch_out.data(), n), "output") << "\n";
        result << "  " << formatCompare(cmp) << "\n";
        
        return result.str();
    }
    
    // Verify Softmax Cross-Entropy Loss: L = -log(softmax(logits)[target])
    std::string verifyCrossEntropyLoss(
        const float* logits, const int* targets, const float* cuda_loss,
        int batch_size, int vocab_size,
        int batch_idx, int step_idx
    ) {
        if (!enabled_) return "";
        call_count_++;
        
        std::string logits_path = temp_dir_ + "/ce_logits.bin";
        std::string targets_path = temp_dir_ + "/ce_targets.bin";
        std::string out_path = temp_dir_ + "/ce_loss.bin";
        
        writeTensorToFile(logits_path, logits, batch_size * vocab_size, {batch_size, vocab_size});
        
        // Copy targets from device to host
        std::vector<int> targets_host(batch_size);
        cudaError_t err = cudaMemcpy(targets_host.data(), targets, 
                                      batch_size * sizeof(int), 
                                      cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            return "[PYTORCH_VERIFY] Failed to copy targets to host";
        }
        
        std::ofstream tf(targets_path, std::ios::binary);
        int ndim = 1;
        tf.write(reinterpret_cast<const char*>(&ndim), sizeof(int));
        tf.write(reinterpret_cast<const char*>(&batch_size), sizeof(int));
        tf.write(reinterpret_cast<const char*>(targets_host.data()), batch_size * sizeof(int));
        tf.close();
        
        std::ostringstream cmd;
        cmd << python_path_ << " \"" << script_path_ << "\" cross_entropy "
            << "\"" << logits_path << "\" \"" << targets_path << "\" \"" << out_path << "\"";
        
        runPythonCommand(cmd.str());
        
        std::vector<float> pytorch_loss;
        std::vector<int> out_shape;
        if (!readTensorFromFile(out_path, pytorch_loss, out_shape)) {
            return "[PYTORCH_VERIFY] Failed to read PyTorch loss";
        }
        
        // Loss is scalar or per-token, compare appropriately
        // Note: cuda_loss is already a HOST pointer (passed as &mean_loss from caller)
        size_t n = pytorch_loss.size();
        CompareResult cmp = compareTensors(cuda_loss, pytorch_loss.data(), n);
        if (!cmp.match) mismatch_count_++;
        
        std::ostringstream result;
        result << "[CE_LOSS_VERIFY] batch=" << batch_idx << " step=" << step_idx << "\n";
        result << "  CUDA loss: " << *cuda_loss << "\n";
        result << "  PyTorch loss: " << pytorch_loss[0] << "\n";
        result << "  " << formatCompare(cmp) << "\n";
        
        return result.str();
    }
    
    // Verify GELU activation: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))
    std::string verifyGELU(
        const float* input, const float* cuda_output,
        int num_elements,
        int batch_idx, int layer_idx, int step_idx
    ) {
        if (!enabled_) return "";
        call_count_++;
        
        std::string in_path = temp_dir_ + "/gelu_in.bin";
        std::string out_path = temp_dir_ + "/gelu_out.bin";
        
        writeTensorToFile(in_path, input, num_elements, {num_elements});
        
        std::ostringstream cmd;
        cmd << python_path_ << " \"" << script_path_ << "\" gelu "
            << "\"" << in_path << "\" \"" << out_path << "\"";
        
        runPythonCommand(cmd.str());
        
        std::vector<float> pytorch_out;
        std::vector<int> out_shape;
        if (!readTensorFromFile(out_path, pytorch_out, out_shape)) {
            return "[PYTORCH_VERIFY] Failed to read PyTorch output";
        }
        
        // Copy CUDA output to host for comparison
        std::vector<float> cuda_out_host = copyDeviceToHost(cuda_output, num_elements);
        if (cuda_out_host.empty()) {
            return "[PYTORCH_VERIFY] Failed to copy CUDA output to host";
        }
        
        CompareResult cmp = compareTensors(cuda_out_host.data(), pytorch_out.data(), num_elements);
        if (!cmp.match) mismatch_count_++;
        
        std::ostringstream result;
        result << "[GELU_VERIFY] batch=" << batch_idx << " layer=" << layer_idx 
               << " step=" << step_idx << " n=" << num_elements << "\n";
        result << "  CUDA: " << formatStats(computeStats(cuda_out_host.data(), num_elements), "output") << "\n";
        result << "  PyTorch: " << formatStats(computeStats(pytorch_out.data(), num_elements), "output") << "\n";
        result << "  " << formatCompare(cmp) << "\n";
        
        return result.str();
    }
    
    // Verify AdamW step: w = w - lr * (m_hat / (sqrt(v_hat) + eps) + wd * w)
    std::string verifyAdamWStep(
        const float* weights, const float* grads,
        const float* m, const float* v,
        const float* cuda_weights_after,
        int num_params,
        float lr, float beta1, float beta2, float eps, float weight_decay,
        int step, // For bias correction
        int batch_idx, int step_idx
    ) {
        if (!enabled_) return "";
        call_count_++;
        
        std::string w_path = temp_dir_ + "/adamw_w.bin";
        std::string g_path = temp_dir_ + "/adamw_g.bin";
        std::string m_path = temp_dir_ + "/adamw_m.bin";
        std::string v_path = temp_dir_ + "/adamw_v.bin";
        std::string out_path = temp_dir_ + "/adamw_out.bin";
        
        writeTensorToFile(w_path, weights, num_params, {num_params});
        writeTensorToFile(g_path, grads, num_params, {num_params});
        writeTensorToFile(m_path, m, num_params, {num_params});
        writeTensorToFile(v_path, v, num_params, {num_params});
        
        std::ostringstream cmd;
        cmd << python_path_ << " \"" << script_path_ << "\" adamw "
            << "\"" << w_path << "\" \"" << g_path << "\" "
            << "\"" << m_path << "\" \"" << v_path << "\" \"" << out_path << "\" "
            << lr << " " << beta1 << " " << beta2 << " " << eps << " " << weight_decay << " " << step;
        
        runPythonCommand(cmd.str());
        
        std::vector<float> pytorch_out;
        std::vector<int> out_shape;
        if (!readTensorFromFile(out_path, pytorch_out, out_shape)) {
            return "[PYTORCH_VERIFY] Failed to read PyTorch output";
        }
        
        // Copy CUDA output to host for comparison
        std::vector<float> cuda_out_host = copyDeviceToHost(cuda_weights_after, num_params);
        if (cuda_out_host.empty()) {
            return "[PYTORCH_VERIFY] Failed to copy CUDA output to host";
        }
        
        CompareResult cmp = compareTensors(cuda_out_host.data(), pytorch_out.data(), num_params);
        if (!cmp.match) mismatch_count_++;
        
        std::ostringstream result;
        result << "[ADAMW_VERIFY] batch=" << batch_idx << " step=" << step_idx 
               << " n_params=" << num_params << "\n";
        result << "  CUDA: " << formatStats(computeStats(cuda_out_host.data(), num_params), "w_after") << "\n";
        result << "  PyTorch: " << formatStats(computeStats(pytorch_out.data(), num_params), "w_after") << "\n";
        result << "  " << formatCompare(cmp) << "\n";
        
        return result.str();
    }
    
    // Verify Scaled Dot-Product Attention (softmax(Q @ K^T / sqrt(d)) @ V)
    std::string verifyScaledDotProductAttention(
        const float* Q, const float* K, const float* V,
        const float* cuda_output,
        int batch_size, int num_heads, int seq_len, int head_dim,
        float scale,
        int batch_idx, int layer_idx, int step_idx
    ) {
        if (!enabled_) return "";
        call_count_++;
        
        size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
        
        std::string q_path = temp_dir_ + "/sdpa_q.bin";
        std::string k_path = temp_dir_ + "/sdpa_k.bin";
        std::string v_path = temp_dir_ + "/sdpa_v.bin";
        std::string out_path = temp_dir_ + "/sdpa_out.bin";
        
        writeTensorToFile(q_path, Q, qkv_size, {batch_size, num_heads, seq_len, head_dim});
        writeTensorToFile(k_path, K, qkv_size, {batch_size, num_heads, seq_len, head_dim});
        writeTensorToFile(v_path, V, qkv_size, {batch_size, num_heads, seq_len, head_dim});
        
        std::ostringstream cmd;
        cmd << python_path_ << " \"" << script_path_ << "\" sdpa "
            << "\"" << q_path << "\" \"" << k_path << "\" \"" << v_path << "\" "
            << "\"" << out_path << "\" " << scale;
        
        runPythonCommand(cmd.str());
        
        std::vector<float> pytorch_out;
        std::vector<int> out_shape;
        if (!readTensorFromFile(out_path, pytorch_out, out_shape)) {
            return "[PYTORCH_VERIFY] SDPA failed to read output";
        }
        
        // Copy CUDA output to host for comparison
        std::vector<float> cuda_out_host = copyDeviceToHost(cuda_output, qkv_size);
        if (cuda_out_host.empty()) {
            return "[PYTORCH_VERIFY] Failed to copy CUDA output to host";
        }
        
        CompareResult cmp = compareTensors(cuda_out_host.data(), pytorch_out.data(), qkv_size);
        if (!cmp.match) mismatch_count_++;
        
        std::ostringstream result;
        result << "[SDPA_VERIFY] batch=" << batch_idx << " layer=" << layer_idx 
               << " heads=" << num_heads << " seq=" << seq_len << " d=" << head_dim << "\n";
        result << "  CUDA: " << formatStats(computeStats(cuda_out_host.data(), qkv_size), "output") << "\n";
        result << "  PyTorch: " << formatStats(computeStats(pytorch_out.data(), qkv_size), "output") << "\n";
        result << "  " << formatCompare(cmp) << "\n";
        
        return result.str();
    }
    
    // ========================================================================
    // EQUATION-BASED DIAGNOSTIC LOGGING (Rule 21)
    // These call Python functions that log detailed gradient/hidden state analysis
    // ========================================================================
    
    // Log weight gradient equation for a specific token
    void logWeightGradientEquation(
        const float* W_data, const float* grad_W_data,
        int vocab_size, int d_model, int token_idx,
        const int* targets, int n_targets, float lr
    ) {
        if (!enabled_) return;
        
        std::string w_path = temp_dir_ + "/eq_weights.bin";
        std::string grad_path = temp_dir_ + "/eq_grad_w.bin";
        std::string targets_path = temp_dir_ + "/eq_targets.bin";
        
        writeTensorToFile(w_path, W_data, vocab_size * d_model, {vocab_size, d_model});
        writeTensorToFile(grad_path, grad_W_data, vocab_size * d_model, {vocab_size, d_model});
        
        // Write targets as int tensor
        std::ofstream tf(targets_path, std::ios::binary);
        int ndim = 1;
        tf.write(reinterpret_cast<const char*>(&ndim), sizeof(int));
        tf.write(reinterpret_cast<const char*>(&n_targets), sizeof(int));
        tf.write(reinterpret_cast<const char*>(targets), n_targets * sizeof(int));
        tf.close();
        
        std::ostringstream cmd;
        cmd << python_path_ << " \"" << script_path_ << "\" weight_gradient_equation "
            << "\"" << w_path << "\" \"" << grad_path << "\" " << token_idx << " "
            << "\"" << targets_path << "\" " << lr;
        
        runPythonCommand(cmd.str());
    }
    
    // Log hidden state equation for gradient analysis
    void logHiddenStateEquation(
        const float* hidden_data, const float* grad_logits_data,
        int n_tokens, int d_model, int vocab_size,
        const int* targets, int token_idx
    ) {
        if (!enabled_) return;
        
        std::string hidden_path = temp_dir_ + "/eq_hidden.bin";
        std::string grad_path = temp_dir_ + "/eq_grad_logits.bin";
        std::string targets_path = temp_dir_ + "/eq_targets.bin";
        
        writeTensorToFile(hidden_path, hidden_data, n_tokens * d_model, {n_tokens, d_model});
        writeTensorToFile(grad_path, grad_logits_data, n_tokens * vocab_size, {n_tokens, vocab_size});
        
        // Write targets
        std::ofstream tf(targets_path, std::ios::binary);
        int ndim = 1;
        tf.write(reinterpret_cast<const char*>(&ndim), sizeof(int));
        tf.write(reinterpret_cast<const char*>(&n_tokens), sizeof(int));
        tf.write(reinterpret_cast<const char*>(targets), n_tokens * sizeof(int));
        tf.close();
        
        std::ostringstream cmd;
        cmd << python_path_ << " \"" << script_path_ << "\" hidden_state_equation "
            << "\"" << hidden_path << "\" \"" << grad_path << "\" \"" << targets_path << "\" " << token_idx;
        
        runPythonCommand(cmd.str());
    }
    
    // Log feedback loop equation for mode collapse tracking
    void logFeedbackLoopEquation(
        const float* hidden_data, const float* W_data,
        int n_tokens, int d_model, int vocab_size, int token_idx,
        float prev_h_norm = 0, float prev_w_norm = 0, float prev_cos = 0
    ) {
        if (!enabled_) return;
        
        std::string hidden_path = temp_dir_ + "/eq_hidden.bin";
        std::string w_path = temp_dir_ + "/eq_weights.bin";
        
        writeTensorToFile(hidden_path, hidden_data, n_tokens * d_model, {n_tokens, d_model});
        writeTensorToFile(w_path, W_data, vocab_size * d_model, {vocab_size, d_model});
        
        std::ostringstream cmd;
        cmd << python_path_ << " \"" << script_path_ << "\" feedback_loop_equation "
            << "\"" << hidden_path << "\" \"" << w_path << "\" " << token_idx;
        
        if (prev_h_norm > 0) {
            cmd << " " << prev_h_norm << " " << prev_w_norm << " " << prev_cos;
        }
        
        runPythonCommand(cmd.str());
    }
    
    // Print summary statistics
    void printSummary() const {
        std::cout << "[PyTorchVerify] Summary: " << call_count_ << " verifications, "
                  << mismatch_count_ << " mismatches" << std::endl;
    }
    
    // Shutdown
    void shutdown() {
        if (enabled_) {
            printSummary();
            enabled_ = false;
        }
    }
};

// ============================================================================
// GLOBAL VERIFIER INSTANCE
// ============================================================================

inline PyTorchVerifier& getPyTorchVerifier() {
    static PyTorchVerifier instance;
    return instance;
}

} // namespace PyTorchVerify
} // namespace GRIM

// ============================================================================
// CONVENIENCE MACROS
// ============================================================================

#ifdef GRIM_PYTORCH_VERIFY

#define PYTORCH_VERIFY_INIT(grim_root) \
    GRIM::PyTorchVerify::getPyTorchVerifier().initialize(grim_root)

#define PYTORCH_VERIFY_RMSNORM(x, gamma, output, batch, hidden, eps, batch_idx, layer_idx, step_idx) \
    do { \
        auto& _pv_verifier_ = GRIM::PyTorchVerify::getPyTorchVerifier(); \
        if (_pv_verifier_.enabled_) { \
            std::string _pv_result_ = _pv_verifier_.verifyRMSNorm(x, gamma, output, batch, hidden, eps, batch_idx, layer_idx, step_idx); \
            if (!_pv_result_.empty()) std::cerr << _pv_result_; \
        } \
    } while(0)

#define PYTORCH_VERIFY_MATMUL(A, B, C, M, K, N, name, batch_idx, layer_idx, step_idx, transpose_b) \
    do { \
        auto& _pv_verifier_ = GRIM::PyTorchVerify::getPyTorchVerifier(); \
        if (_pv_verifier_.enabled_) { \
            std::string _pv_result_ = _pv_verifier_.verifyMatMul(A, B, C, M, K, N, name, batch_idx, layer_idx, step_idx, transpose_b); \
            if (!_pv_result_.empty()) std::cerr << _pv_result_; \
        } \
    } while(0)

#define PYTORCH_VERIFY_GELU(input, output, n, batch_idx, layer_idx, step_idx) \
    do { \
        auto& _pv_verifier_ = GRIM::PyTorchVerify::getPyTorchVerifier(); \
        if (_pv_verifier_.enabled_) { \
            std::string _pv_result_ = _pv_verifier_.verifyGELU(input, output, n, batch_idx, layer_idx, step_idx); \
            if (!_pv_result_.empty()) std::cerr << _pv_result_; \
        } \
    } while(0)

#define PYTORCH_VERIFY_CE_LOSS(logits, targets, loss, batch, vocab, batch_idx, step_idx) \
    do { \
        auto& _pv_verifier_ = GRIM::PyTorchVerify::getPyTorchVerifier(); \
        if (_pv_verifier_.enabled_) { \
            std::string _pv_result_ = _pv_verifier_.verifyCrossEntropyLoss(logits, targets, loss, batch, vocab, batch_idx, step_idx); \
            if (!_pv_result_.empty()) std::cerr << _pv_result_; \
        } \
    } while(0)

#define PYTORCH_VERIFY_SDPA(Q, K, V, out, b, h, s, d, scale, batch_idx, layer_idx, step_idx) \
    do { \
        auto& _pv_verifier_ = GRIM::PyTorchVerify::getPyTorchVerifier(); \
        if (_pv_verifier_.enabled_) { \
            std::string _pv_result_ = _pv_verifier_.verifyScaledDotProductAttention(Q, K, V, out, b, h, s, d, scale, batch_idx, layer_idx, step_idx); \
            if (!_pv_result_.empty()) std::cerr << _pv_result_; \
        } \
    } while(0)

#define PYTORCH_VERIFY_SHUTDOWN() \
    GRIM::PyTorchVerify::getPyTorchVerifier().shutdown()

// Equation-based diagnostic logging macros (Rule 21)
#define PYTORCH_LOG_WEIGHT_GRADIENT_EQUATION(W, grad_W, vocab, d_model, token_idx, targets, n_targets, lr) \
    do { \
        auto& _pv_verifier_ = GRIM::PyTorchVerify::getPyTorchVerifier(); \
        if (_pv_verifier_.enabled_) { \
            _pv_verifier_.logWeightGradientEquation(W, grad_W, vocab, d_model, token_idx, targets, n_targets, lr); \
        } \
    } while(0)

#define PYTORCH_LOG_HIDDEN_STATE_EQUATION(hidden, grad_logits, n_tokens, d_model, vocab, targets, token_idx) \
    do { \
        auto& _pv_verifier_ = GRIM::PyTorchVerify::getPyTorchVerifier(); \
        if (_pv_verifier_.enabled_) { \
            _pv_verifier_.logHiddenStateEquation(hidden, grad_logits, n_tokens, d_model, vocab, targets, token_idx); \
        } \
    } while(0)

#define PYTORCH_LOG_FEEDBACK_LOOP_EQUATION(hidden, W, n_tokens, d_model, vocab, token_idx, prev_h, prev_w, prev_cos) \
    do { \
        auto& _pv_verifier_ = GRIM::PyTorchVerify::getPyTorchVerifier(); \
        if (_pv_verifier_.enabled_) { \
            _pv_verifier_.logFeedbackLoopEquation(hidden, W, n_tokens, d_model, vocab, token_idx, prev_h, prev_w, prev_cos); \
        } \
    } while(0)

#else // GRIM_PYTORCH_VERIFY not defined - no-op macros

#define PYTORCH_VERIFY_INIT(grim_root) ((void)0)
#define PYTORCH_VERIFY_RMSNORM(x, gamma, output, batch, hidden, eps, batch_idx, layer_idx, step_idx) ((void)0)
#define PYTORCH_VERIFY_MATMUL(A, B, C, M, K, N, name, batch_idx, layer_idx, step_idx, transpose_b) ((void)0)
#define PYTORCH_VERIFY_GELU(input, output, n, batch_idx, layer_idx, step_idx) ((void)0)
#define PYTORCH_VERIFY_CE_LOSS(logits, targets, loss, batch, vocab, batch_idx, step_idx) ((void)0)
#define PYTORCH_VERIFY_SDPA(Q, K, V, out, b, h, s, d, scale, batch_idx, layer_idx, step_idx) ((void)0)
#define PYTORCH_VERIFY_SHUTDOWN() ((void)0)
#define PYTORCH_LOG_WEIGHT_GRADIENT_EQUATION(W, grad_W, vocab, d_model, token_idx, targets, n_targets, lr) ((void)0)
#define PYTORCH_LOG_HIDDEN_STATE_EQUATION(hidden, grad_logits, n_tokens, d_model, vocab, targets, token_idx) ((void)0)
#define PYTORCH_LOG_FEEDBACK_LOOP_EQUATION(hidden, W, n_tokens, d_model, vocab, token_idx, prev_h, prev_w, prev_cos) ((void)0)

#endif // GRIM_PYTORCH_VERIFY

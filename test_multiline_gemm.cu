// Test file for multi-line GEMM call parsing

#include <cublas_v2.h>

void test_multiline_gemm() {
    cublasHandle_t handle;
    int M = 768;
    int N = 2048;
    int K = 768;
    float alpha = 1.0f;
    float beta = 0.0f;
    
    float *A, *B, *C;
    
    // Test 1: Multi-line with comments
    cublasSgemm(
        handle,
        CUBLAS_OP_N,  // No transpose for A
        CUBLAS_OP_N,  // No transpose for B
        M, N, K,
        &alpha,
        A, M,  // Leading dimension = M
        B, K,  // Leading dimension = K
        &beta,
        C, M   // Leading dimension = M
    );
    
    // Test 2: Macro wrapper
    #define MY_GEMM(...) cublasSgemm(__VA_ARGS__)
    
    MY_GEMM(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        M, N, K,
        &alpha,
        A, K,  /* lda = K for transpose */
        B, K,
        &beta,
        C, M
    );
    
    // Test 3: Block comments between args
    cublasSgemm(handle, /* comment */
                CUBLAS_OP_N, /* another comment */
                CUBLAS_OP_T, M, N, K, &alpha,
                A, M, B, N, /* more comments */
                &beta, C, M);
    
    // Test 4: Nested function call in argument
    cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        M, N, K,
        &alpha,
        A, M,
        get_matrix_pointer(B, 0),  // Nested call
        K,
        &beta,
        C, M
    );
}

// Focused CUDA test: real production kernels and CUDA storage, no test shims.
#include "../Shared/TensorConversion/TensorConversion.cu"
#include "../Shared/PBM/PositionalBiasMethod.cu"
#include <cassert>
#include <cstdio>
#include <vector>

static void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
static void close(float got, float expected) {
    if (std::abs(got - expected) > 2e-5f) {
        throw std::runtime_error("gradient mismatch: " + std::to_string(got) + " vs " + std::to_string(expected));
    }
}
static void split_test(cudaStream_t stream, int dim) {
    const int batch=2, heads=4, kv_heads=2, seq=3;
    const int width=(heads+2*kv_heads)*dim;
    std::vector<float> expected(batch*seq*width, 7.0f), actual(expected.size());
    float* dst=nullptr;
    check(cudaMalloc(&dst, expected.size()*sizeof(float)));
    check(cudaMemcpyAsync(dst, expected.data(), expected.size()*sizeof(float), cudaMemcpyHostToDevice, stream));
    // Different output arrival order, repeated slice, and untouched slices.
    for (int output : {2, 0, 0, 1}) {
        const int hcount=output==0 ? heads : kv_heads;
        const int offset=output==0 ? 0 : (heads+(output-1)*kv_heads)*dim;
        std::vector<float> input(batch*hcount*seq*dim);
        for (size_t i=0;i<input.size();++i) input[i]=float(i%19)-9.0f;
        float* src=nullptr;
        check(cudaMalloc(&src,input.size()*sizeof(float)));
        check(cudaMemcpyAsync(src,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
        TensorConversion::accumulate_qkv_grad_gqa(src,dst,batch,heads,kv_heads,seq,dim,output,stream);
        for(int b=0;b<batch;++b) for(int h=0;h<hcount;++h) for(int s=0;s<seq;++s) for(int d=0;d<dim;++d)
            expected[(b*seq+s)*width+offset+h*dim+d] += input[((b*hcount+h)*seq+s)*dim+d];
        check(cudaMemcpyAsync(actual.data(),dst,actual.size()*sizeof(float),cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream));
        for(size_t i=0;i<actual.size();++i) close(actual[i],expected[i]);
        check(cudaFree(src));
    }
    check(cudaFree(dst));
}
static void rope_test(cudaStream_t stream, bool is_q) {
    const int batch=2, qheads=4, kheads=2, seq=3, dim=6, rotary=4, offset=5;
    const int heads=is_q?qheads:kheads;
    std::vector<float> input(batch*heads*seq*dim), actual(input.size()), expected(input.size(),3.0f);
    for(size_t i=0;i<input.size();++i) input[i]=float(i%11)*0.1f;
    const float frequencies[]={0.3f,0.07f};
    float *src=nullptr,*dst=nullptr,*freq=nullptr;
    check(cudaMalloc(&src,input.size()*sizeof(float)));
    check(cudaMalloc(&dst,input.size()*sizeof(float)));
    check(cudaMalloc(&freq,sizeof(frequencies)));
    check(cudaMemcpyAsync(src,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
    check(cudaMemcpyAsync(dst,expected.data(),expected.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
    check(cudaMemcpyAsync(freq,frequencies,sizeof(frequencies),cudaMemcpyHostToDevice,stream));
    for(int repeat=0;repeat<2;++repeat) {
        GRIM::PBM::ropeRotationGQABackwardKernel<<<dim3(1,heads,batch),256,0,stream>>>(
            is_q?dst:nullptr,is_q?nullptr:dst,freq,batch,qheads,kheads,seq,dim,rotary,is_q,offset,src);
        for(int b=0;b<batch;++b) for(int h=0;h<heads;++h) for(int s=0;s<seq;++s) {
            int base=((b*heads+h)*seq+s)*dim;
            for(int d=0;d<rotary;d+=2) {
                float angle=(s+offset)*frequencies[d/2],c=std::cos(angle),sn=std::sin(angle);
                expected[base+d]+=input[base+d]*c+input[base+d+1]*sn;
                expected[base+d+1]+=-input[base+d]*sn+input[base+d+1]*c;
            }
            for(int d=rotary;d<dim;++d) expected[base+d]+=input[base+d];
        }
    }
    check(cudaMemcpyAsync(actual.data(),dst,actual.size()*sizeof(float),cudaMemcpyDeviceToHost,stream));
    check(cudaStreamSynchronize(stream));
    for(size_t i=0;i<actual.size();++i) close(actual[i],expected[i]);
    check(cudaMemcpy(actual.data(),src,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
    for(size_t i=0;i<actual.size();++i) close(actual[i],input[i]);
    check(cudaFree(src)); check(cudaFree(dst)); check(cudaFree(freq));
}
int main() {
    cudaStream_t stream=nullptr;
    check(cudaStreamCreate(&stream));
    split_test(stream,4); split_test(stream,5);
    rope_test(stream,true); rope_test(stream,false);
    check(cudaStreamDestroy(stream));
    std::puts("Attention destination CUDA kernel tests passed");
}

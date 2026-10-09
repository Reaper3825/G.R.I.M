// CPU execution shim for the production loss kernels, assembled by Python.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
using std::isfinite;
constexpr int CUDA_BLOCK_SIZE_STANDARD = 256;
namespace HyperParameters {
// LOSS_CONFIG
}
struct Index { int x = 0; };
thread_local Index threadIdx;
Index blockIdx, blockDim;
constexpr int warpSize = 32;
class Barrier {
    std::mutex mutex;
    std::condition_variable cv;
    int arrived = 0, generation = 0;
public:
    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        const int previous = generation;
        if (++arrived == blockDim.x) { arrived = 0; ++generation; cv.notify_all(); }
        else cv.wait(lock, [&] { return generation != previous; });
    }
} barrier;
float shuffle_values[1024];
float __shfl_down_sync(unsigned, float value, int offset) {
    shuffle_values[threadIdx.x] = value;
    barrier.wait();
    const int lane = threadIdx.x % 32;
    const float result = lane + offset < 32 ? shuffle_values[threadIdx.x + offset] : value;
    barrier.wait();
    return result;
}
void __syncthreads() { barrier.wait(); }
std::mutex atomic_mutex;
template<class T> void atomicAdd(T* dst, T value) { std::lock_guard<std::mutex> guard(atomic_mutex); *dst += value; }
void trapInvalidCrossEntropyInput() { std::abort(); }
#define __device__
#define __global__
#define __forceinline__ inline
#define __shared__ static
#define __restrict__
// LOSS_KERNELS

template<class Fn> void launch(int threads, Fn fn) {
    blockDim.x = threads;
    std::vector<std::thread> workers;
    for (int i=0; i<threads; ++i) workers.emplace_back([&, i] { threadIdx.x=i; fn(); });
    for (auto& worker : workers) worker.join();
}
double reference(const std::vector<double>& z, int target, const HyperParameters::LossConfigHP& c) {
    const double m = *std::max_element(z.begin(), z.end());
    double sum = 0;
    for (double x : z) sum += std::exp(x-m);
    const double lse = m + std::log(sum);
    double ce=0, entropy=0;
    for (size_t i=0; i<z.size(); ++i) {
        const double q = c.smoothing_enabled && z.size()>1
            ? (i==target ? 1-c.smoothing_epsilon : c.smoothing_epsilon/(z.size()-1)) : double(i==target);
        ce -= q*(z[i]-lse);
        entropy += std::exp(z[i]-lse)*(z[i]-lse);
    }
    if (c.focal_enabled) ce *= c.focal_alpha * std::pow(std::max(0.0, 1-std::exp(z[target]-lse)), c.focal_gamma);
    return ce + (c.entropy_reg_enabled ? c.entropy_reg_lambda*entropy : 0);
}
void check(int vocab, int threads, int flags, float gamma, int extreme=0, float upstream=-.37f) {
    HyperParameters::LossConfigHP c;
    c.focal_enabled=flags&1; c.smoothing_enabled=flags&2;
    c.entropy_reg_enabled=flags&4; c.class_balanced_enabled=flags&8;
    c.focal_alpha=.7f; c.focal_gamma=gamma; c.smoothing_epsilon=.2f;
    c.entropy_reg_lambda=c.entropy_reg_enabled ? .13f : 0;
    std::vector<double> z(vocab);
    for(int i=0;i<vocab;++i) z[i]=std::sin(double(i+1))*3;
    const int target=vocab-1;
    if (extreme) { std::fill(z.begin(),z.end(),-1000); z[extreme==1?target:0]=0; }
    double m=*std::max_element(z.begin(),z.end()), sum=0;
    for(double x:z) sum+=std::exp(x-m);
    std::vector<float> lp(vocab), weights(vocab,1.7f), dx(vocab, .25f);
    for(int i=0;i<vocab;++i) lp[i]=float(z[i]-m-std::log(sum));
    const float cw=c.class_balanced_enabled?weights[target]:1;
    const float* weights_ptr=c.class_balanced_enabled?weights.data():nullptr;
    float loss=0, weight_sum=0; int count=0;
    blockIdx.x=0;
    launch(threads,[&] { kernelCrossEntropyNLLForward(lp.data(), &target, &loss, &count, &weight_sum, 1, vocab,c,weights_ptr); });
    assert(count==1 && std::abs(weight_sum-cw)<1e-6);
    assert(std::abs(loss-cw*reference(z,target,c)) < 2e-4*(1+std::abs(loss)));
    const float scale=upstream/(c.class_balanced_enabled?4.2f:3.f);
    for(int pass=1;pass<=2;++pass) {
        launch(threads,[&] { kernelCrossEntropyLogitsBackward(lp.data(),&target,dx.data(),1,vocab,scale,c,weights_ptr); });
        double row_sum=0;
        for(int i=0;i<vocab;++i) {
            assert(isfinite(dx[i]));
            // Sample finite differences at vocabulary scale to avoid O(V^2)
            // reference work; still check finiteness and row sum everywhere.
            if(vocab<1024 || i<4 || i==target || i==vocab/2) {
                auto plus=z, minus=z; plus[i]+=1e-4; minus[i]-=1e-4;
                const double expected=cw*scale*(reference(plus,target,c)-reference(minus,target,c))/2e-4;
                assert(std::abs(dx[i]-(.25+pass*expected)) < 3e-5);
            }
            row_sum+=dx[i]-.25;
        }
        assert(std::abs(row_sum)<2e-4);
    }
    // Masked rows must leave a populated destination untouched, including when
    // their forward values are unusable. This also catches accidental zeroing.
    int masked=-1; auto before=dx;
    std::fill(lp.begin(),lp.end(),NAN);
    launch(threads,[&] { kernelCrossEntropyLogitsBackward(lp.data(),&masked,dx.data(),1,vocab,scale,c,weights_ptr); });
    assert(dx==before);
}
int main() {
    int cases=0;
    for(int flags=0;flags<16;++flags)
        for(float gamma : {0.f,.5f,2.f})
            for(int vocab : {1,7,67}) { check(vocab,32,flags,gamma); ++cases; }
    for(int threads : {64,256}) { check(513,threads,15,2); ++cases; }
    for(int extreme : {1,2}) for(float gamma : {0.f,.5f,2.f}) { check(7,32,15,gamma,extreme); ++cases; }
    for(float upstream : {0.f,1.f}) { check(67,64,15,2,0,upstream); ++cases; }
    check(50376,256,15,2); ++cases;
    std::cout << cases << " production loss kernel host cases passed (forward, finite differences, accumulation, masking)\n";
}

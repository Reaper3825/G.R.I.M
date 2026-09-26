// Included after the actual production kernel bodies by test_head_gate_cuda.py.
namespace {
void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void near(float a,float b,float tolerance=3e-5f) {
    if(std::abs(a-b)>tolerance*(1+std::abs(b)))
        throw std::runtime_error("head gate result/gradient mismatch");
}
struct Buffer {
    float* data=nullptr;size_t size;
    explicit Buffer(size_t n):size(n) { check(cudaMalloc(&data,n*sizeof(float))); }
    ~Buffer() { cudaFree(data); }
    void put(const std::vector<float>& v) { check(cudaMemcpy(data,v.data(),size*sizeof(float),cudaMemcpyHostToDevice)); }
    std::vector<float> get() const {
        std::vector<float> v(size);check(cudaMemcpy(v.data(),data,size*sizeof(float),cudaMemcpyDeviceToHost));return v;
    }
};
void test(cudaStream_t stream,int B,int H,int S,int D) {
    const size_t n=size_t(B)*H*S*D, ng=size_t(B)*S*H;
    std::vector<float> x(n),g(ng),dy(n),dx(n,3),dg(ng,5),expected(n);
    for(size_t i=0;i<n;++i) { x[i]=float(int(i%17)-8)*0.125f;dy[i]=float(int(i%11)-5)*0.125f; }
    for(size_t i=0;i<ng;++i) g[i]=float(i%13)*0.125f; // distinct gates within each KV group
    Buffer bx(n),bg(ng),by(n),bdy(n),bdx(n),bdg(ng);
    bx.put(x);bg.put(g);bdy.put(dy);bdx.put(dx);bdg.put(dg);
    TensorConversion::head_gate_BHSD_to_BSM(bx.data,bg.data,by.data,B,H,S,D,stream);
    check(cudaGetLastError());check(cudaStreamSynchronize(stream));
    auto y=by.get();
    for(int b=0;b<B;++b) for(int h=0;h<H;++h) for(int s=0;s<S;++s) for(int d=0;d<D;++d) {
        size_t j=((b*H+h)*S+s)*D+d,k=(b*S+s)*H+h,i=k*D+d;
        expected[i]=x[j]*g[k];near(y[i],expected[i]);
        dx[j]+=2*dy[i]*g[k];dg[k]+=2*dy[i]*x[j];
    }
    // Two deliveries must accumulate, including tails/non-power-of-two D.
    for(int pass=0;pass<2;++pass)
        TensorConversion::head_gate_BHSD_to_BSM_backward(bdy.data,bx.data,bg.data,bdx.data,bdg.data,B,H,S,D,stream);
    check(cudaGetLastError());check(cudaStreamSynchronize(stream));
    auto actual_dx=bdx.get(),actual_dg=bdg.get();
    for(size_t i=0;i<n;++i) near(actual_dx[i],dx[i]);
    for(size_t i=0;i<ng;++i) near(actual_dg[i],dg[i]);
    // Frozen destinations must be independently optional.
    TensorConversion::head_gate_BHSD_to_BSM_backward(bdy.data,bx.data,bg.data,nullptr,bdg.data,B,H,S,D,stream);
    TensorConversion::head_gate_BHSD_to_BSM_backward(bdy.data,bx.data,bg.data,bdx.data,nullptr,B,H,S,D,stream);
    check(cudaStreamSynchronize(stream));
    auto frozen_dx=bdx.get(),frozen_dg=bdg.get();
    for(size_t i=0;i<n;++i) near(frozen_dx[i],3+1.5f*(dx[i]-3));
    for(size_t i=0;i<ng;++i) near(frozen_dg[i],5+1.5f*(dg[i]-5));
    // Numeric gradient against the GPU forward, for both gate and raw attention.
    auto loss=[&]() {
        TensorConversion::head_gate_BHSD_to_BSM(bx.data,bg.data,by.data,B,H,S,D,stream);
        check(cudaStreamSynchronize(stream));auto v=by.get();double sum=0;
        for(size_t i=0;i<n;++i) sum+=double(v[i])*dy[i];return sum;
    };
    const float eps=0.015625f;
    for(bool gate:{false,true}) for(size_t idx:{size_t(0),(gate?ng:n)/2,(gate?ng:n)-1}) {
        auto& v=gate?g:x;auto& buf=gate?bg:bx;float old=v[idx];
        v[idx]=old+eps;buf.put(v);double plus=loss();
        v[idx]=old-eps;buf.put(v);double minus=loss();v[idx]=old;buf.put(v);
        float analytic=gate?(dg[idx]-5)/2:(dx[idx]-3)/2;
        near(float((plus-minus)/(2*eps)),analytic,2e-4f);
    }
    // Zero logits -> multiplier one: exact identity through the layout change.
    std::fill(g.begin(),g.end(),1.0f);bg.put(g);
    TensorConversion::head_gate_BHSD_to_BSM(bx.data,bg.data,by.data,B,H,S,D,stream);
    check(cudaStreamSynchronize(stream));y=by.get();
    for(int b=0;b<B;++b) for(int s=0;s<S;++s) for(int h=0;h<H;++h) for(int d=0;d<D;++d)
        near(y[((b*S+s)*H+h)*D+d],x[((b*H+h)*S+s)*D+d],0);
    // Input caches were never modified.
    auto retained=bx.get();for(size_t i=0;i<n;++i) near(retained[i],x[i],0);
}
}
int main() {
    try {
        cudaStream_t stream;check(cudaStreamCreate(&stream));
        for(int d:{1,7,32,64,129}) test(stream,2,12,3,d);
        test(stream,1,12,1,64); // decode
        test(stream,1,12,17,64); // prefill
        check(cudaStreamDestroy(stream));
        std::puts("Head gate CUDA forward/backward/finite-difference tests passed");
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what());return 1; }
}

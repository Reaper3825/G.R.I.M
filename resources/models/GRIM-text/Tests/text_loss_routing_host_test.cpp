// Host boundary for production TextLossGradFn, assembled by Python.
#include <cmath>
#include <cstring>
#include <sstream>
#define EQ_LOG(...) do {} while(0)
namespace GRIM::VerboseLogging { constexpr bool ENABLE_LOSS_BACKWARD_SAMPLING = true; }
#define __host__
#define AG_TRACE(...) do {} while(0)
constexpr int cudaMemcpyDeviceToHost=1;
inline int cudaMemcpyAsync(void* dst,const void* src,size_t n,int,cudaStream_t stream) {
    assert(stream); std::memcpy(dst,src,n); return 0;
}
inline void cudaMallocOrThrow(void** ptr,size_t n,const char*) { *ptr=std::malloc(n); }
inline void queueForDeferredCleanup(void* ptr) { std::free(ptr); }
namespace GRIM::HyperParameters { struct LossConfigHP {}; }
namespace GRIM::autograd {
struct CrossEntropyTargetSelection { static CrossEntropyTargetSelection primaryLm() { return {}; } };
struct CrossEntropyForwardWorkspace { float* loss; int* count; float* weight; size_t a,b,c; cudaStream_t stream; };
struct CrossEntropyForwardResult { float mean_loss, weight_sum; };
inline int forward_calls=0, backward_calls=0;
inline float* last_destination=nullptr;
inline Tensor log_softmax(const Tensor& x,cudaStream_t stream) {
    assert(!x.requires_grad && !x.owns_data && !x.grad_fn);
    ++forward_calls;
    return Tensor::zeros(x.shape,false,stream,"saved");
}
inline CrossEntropyForwardResult computeCrossEntropyForwardFromLogProbs(
    const float*,const Batching::BatchPayload&,const Batching::BatchDeviceBindings&,
    const CrossEntropyTargetSelection&,const CrossEntropyForwardWorkspace&,
    const HyperParameters::LossConfigHP&,const float*,cudaStream_t) { return {2,1}; }
inline void computeCrossEntropyBackwardToLogits(const float*,const Batching::BatchPayload& p,
    const Batching::BatchDeviceBindings&,const CrossEntropyTargetSelection&,float* dst,int,float,
    const HyperParameters::LossConfigHP&,const float*,float scale,cudaStream_t) {
    ++backward_calls; last_destination=dst;
    for(int i=0;i<p.vocab_size;++i) dst[i]+=scale*(i+1);
}
}
// PRODUCTION_NODE
namespace {
using namespace GRIM;
using namespace GRIM::autograd;
cudaStream_t stream=reinterpret_cast<void*>(1);
Batching::BatchPayload payload;
Batching::BatchDeviceBindings bindings;
Tensor tensor(size_t n=3) { return Tensor::zeros({n},true,stream,"test"); }
struct Producer : GradFn {
    Tensor* leaf; int calls=0;
    explicit Producer(Tensor& x):leaf(x.grad_.get()) {}
    void apply_impl(const Tensor& g,cudaStream_t s,const Batching::BatchPayload*,const Batching::BatchDeviceBindings*) override {
        ++calls; accumulate_grad(*leaf,g,1,s,"producer");
    }
};
Tensor loss(Tensor& logits) {
    auto n=std::make_shared<TextLossGradFn>();
    const auto before=allocations;
    n->capture_inputs(logits,payload,bindings,CrossEntropyTargetSelection::primaryLm(),{},nullptr,stream);
    if(!logits.is_leaf) assert(allocations==before+1); // Only saved forward data.
    assert(n->saved_log_probs->owns_data && !n->saved_log_probs->grad_fn);
    Tensor out=tensor(1); out.is_leaf=false; out.grad_fn=n; return out;
}
void backward(Tensor& out,float scale,bool engine=true) {
    Tensor seed=tensor(1); seed.data[0]=scale;
    if(engine) {
        out.grad_fn->receive_gradient(seed,stream);
        AutogradEngine scheduler(stream,&payload,&bindings); scheduler.run(out.grad_fn.get());
    } else out.grad_fn->apply(seed,stream,&payload,&bindings);
}
void expect(Tensor& leaf,float baseline,float scale) {
    for(int i=0;i<3;++i) assert(leaf.grad_->data[i]==baseline+scale*(i+1));
}
}
int main() {
    for(bool engine : {true,false}) {
        Tensor leaf=tensor(); leaf.ensure_grad(); std::fill_n(leaf.grad_->data,3,5.f);
        for(int pass=1;pass<=2;++pass) {
            Tensor out=loss(leaf); backward(out,.5f,engine);
            assert(last_destination==leaf.grad_->data); expect(leaf,5,.5f*pass);
            auto n=std::static_pointer_cast<TextLossGradFn>(out.grad_fn);
            std::weak_ptr<Tensor> saved=n->saved_log_probs;
            n->release_saved(); n->release_saved();
            assert(saved.expired() && !n->input_producer && !n->leaf_input_gradient);
        }
        assert(leaf.grad_->deliveries==2);
    }
    for(bool engine : {true,false}) {
        Tensor leaf=tensor(); leaf.ensure_grad();
        auto producer=std::make_shared<Producer>(leaf);
        Tensor logits=tensor(); logits.is_leaf=false; logits.grad_fn=producer;
        Tensor out=loss(logits); assert(!producer->pending_gradient_);
        backward(out,-.5f,engine); expect(leaf,0,-.5f); assert(producer->calls==1);
        backward(out,1,false); expect(leaf,0,-.5f); assert(producer->calls==1);
    }
    // Shared producer waits for both scalar objectives and runs exactly once.
    {
        Tensor leaf=tensor(); leaf.ensure_grad();
        auto producer=std::make_shared<Producer>(leaf);
        Tensor logits=tensor(); logits.is_leaf=false; logits.grad_fn=producer;
        Tensor a=loss(logits), b=loss(logits), root=add(a,b,stream);
        backward(root,.5f); expect(leaf,0,1); assert(producer->calls==1);
    }
    // Evaluation capture makes no input edge or gradient allocation.
    {
        Tensor logits=tensor(); logits.requires_grad=false;
        auto n=std::make_shared<TextLossGradFn>();
        n->capture_inputs(logits,payload,bindings,CrossEntropyTargetSelection::primaryLm(),{},nullptr,stream);
        assert(!logits.grad_ && !n->input_producer && !n->leaf_input_gradient && n->engine_inputs_.empty());
        std::weak_ptr<Tensor> saved=n->saved_log_probs; n.reset(); assert(saved.expired());
    }
    // Fail before delivery on malformed scalar, payload, bindings or geometry.
    for(int bad=0;bad<5;++bad) {
        Tensor leaf=tensor(), out=loss(leaf), seed=tensor(bad==0?2:1);
        seed.data[0]=bad==4?NAN:1;
        auto wrong=payload; wrong.vocab_size=4;
        const int before=backward_calls; bool rejected=false;
        try { out.grad_fn->apply(seed,stream,bad==1?nullptr:(bad==3?&wrong:&payload),bad==2?nullptr:&bindings); }
        catch(const std::runtime_error&) { rejected=true; }
        assert(rejected && backward_calls==before && leaf.grad_->deliveries==0);
    }
    std::cout << "TextLossGradFn production host routing cases passed\n";
}

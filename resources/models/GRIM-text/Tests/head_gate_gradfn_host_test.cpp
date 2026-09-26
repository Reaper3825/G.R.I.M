// Production HeadGateGradFn and engine run against host storage in the runner.
namespace {
using namespace GRIM;
using namespace GRIM::autograd;
using TensorContract::TensorShape;
cudaStream_t stream = reinterpret_cast<void*>(1);
void near(float a, float b) { assert(std::abs(a-b) < 2e-4f); }
Tensor make(TensorShape shape, bool grad = true) {
    auto t = Tensor::zeros(shape, grad, stream, "test");
    if (grad) t.ensure_grad();
    for(size_t i=0;i<t.numel();++i) t.data[i] = float(int(i%13)-6)*0.1f;
    return t;
}
struct Identity : GradFn {
    Tensor* leaf;
    int calls = 0;
    explicit Identity(Tensor& t) : leaf(t.grad_.get()) {}
    void apply_impl(const Tensor& dy,cudaStream_t s,const Batching::BatchPayload*,const Batching::BatchDeviceBindings*) override {
        ++calls; autograd::accumulate_grad(*leaf,dy,1,s,"test identity");
    }
};
Tensor produced(Tensor leaf, std::shared_ptr<Identity>& producer) {
    producer=std::make_shared<Identity>(leaf);
    leaf.is_leaf=false; leaf.grad_fn=producer; return leaf;
}
void run(Tensor& y,const Tensor& dy,bool engine) {
    if(engine) {
        y.grad_fn->receive_gradient(dy,stream);
        AutogradEngine scheduler(stream,nullptr,nullptr); scheduler.run(y.grad_fn.get());
    } else y.grad_fn->apply(dy,stream,nullptr,nullptr);
}
void routing() {
    const int B=2,H=12,S=3,D=5;
    for(bool engine:{false,true}) for(bool gx:{false,true}) for(bool gg:{false,true})
    for(bool px:{false,true}) for(bool pg:{false,true}) {
        auto lx=make(TensorShape::make_BHSD(B,H,S,D));
        auto lg=make(TensorShape::make_BSM(B*S,H));
        std::shared_ptr<Identity> nx,ng;
        auto x=px?produced(lx,nx):lx, g=pg?produced(lg,ng):lg;
        x.requires_grad=gx;g.requires_grad=gg;
        std::fill_n(lx.grad_->data,lx.numel(),3.0f);
        std::fill_n(lg.grad_->data,lg.numel(),5.0f);
        // A new graph each pass, same leaf gradient owners (microbatch accumulation).
        for(int pass=1;pass<=2;++pass) {
            if(nx) { nx->applied=false;nx->pending_gradient_.reset(); }
            if(ng) { ng->applied=false;ng->pending_gradient_.reset(); }
            auto y=head_gate_bhsd_to_flat(g,x,stream);
            assert(y.requires_grad==(gx||gg));assert(bool(y.grad_fn)==(gx||gg));
            auto dy=make(y.shape,false);
            if(gx||gg) run(y,dy,engine);
            for(int b=0;b<B;++b) for(int s=0;s<S;++s) for(int h=0;h<H;++h) {
                float sum=0; int k=(b*S+s)*H+h;
                for(int d=0;d<D;++d) {
                    int i=k*D+d,j=((b*H+h)*S+s)*D+d;
                    near(y.data[i],x.data[j]*g.data[k]);
                    near(lx.grad_->data[j],3+(gx?pass*dy.data[i]*g.data[k]:0));
                    sum+=dy.data[i]*x.data[j];
                }
                near(lg.grad_->data[k],5+(gg?pass*sum:0));
            }
            if(nx) assert(nx->calls==pass*int(gx));
            if(ng) assert(ng->calls==pass*int(gg));
            if(y.grad_fn) {
                auto n=std::static_pointer_cast<HeadGateGradFn>(y.grad_fn);
                n->release_saved();
                assert(!n->cached_x&&!n->cached_scale&&!n->x_grad_fn&&!n->scale_grad_fn);
                assert(!n->leaf_x_gradient&&!n->leaf_scale_gradient&&n->engine_inputs_.empty());
            }
        }
    }
}
void shared_producer() {
    // Two branches from one ancestor, as with ln1_out -> QKV and gate matmul.
    // D=1 lets the host shape-adapter share storage without a data conversion.
    struct ShapeAdapter : GradFn {
        std::shared_ptr<GradFn> source;
        TensorShape source_shape;
        ShapeAdapter(std::shared_ptr<GradFn> p,TensorShape shape):source(p),source_shape(shape) { register_input(p); }
        void apply_impl(const Tensor& dy,cudaStream_t s,const Batching::BatchPayload*,const Batching::BatchDeviceBindings*) override {
            auto& dst=source->gradient_destination(source_shape,s);
            autograd::accumulate_grad(dst,dy,1,s,"adapter");
            AutogradEngine::active()->contribute(source.get());
        }
    };
    auto leaf=make(TensorShape::make_BSM(2,12));
    std::shared_ptr<Identity> p;auto g=produced(leaf,p);auto x=g;
    x.shape=TensorShape::make_BHSD(1,12,2,1);
    x.grad_fn=std::make_shared<ShapeAdapter>(p,leaf.shape);
    g.grad_fn=std::make_shared<ShapeAdapter>(p,leaf.shape);
    auto y=head_gate_bhsd_to_flat(g,x,stream),dy=make(y.shape,false);
    std::vector<float> want(leaf.numel(),0);
    TensorConversion::head_gate_BHSD_to_BSM_backward(dy.data,x.data,g.data,
        want.data(),want.data(),1,12,2,1,stream);
    run(y,dy,true);assert(p->calls==1);
    for(size_t i=0;i<want.size();++i) near(leaf.grad_->data[i],want[i]);
}
void identity_and_decode() {
    auto x=make(TensorShape::make_BHSD(1,12,3,7),false);
    auto g=make(TensorShape::make_BSM(3,12),false);
    std::fill_n(g.data,g.numel(),1.0f);
    auto y=head_gate_bhsd_to_flat(g,x,stream);
    for(int s=0;s<3;++s) {
        auto one=make(TensorShape::make_BHSD(1,12,1,7),false);
        auto gate=make(TensorShape::make_BSM(1,12),false);
        for(int h=0;h<12;++h) {
            gate.data[h]=g.data[s*12+h];
            for(int d=0;d<7;++d) one.data[h*7+d]=x.data[(h*3+s)*7+d];
        }
        auto decode=head_gate_bhsd_to_flat(gate,one,stream);
        for(size_t i=0;i<decode.numel();++i) near(decode.data[i],y.data[s*84+i]);
    }
    // Four KV-head gates cannot substitute for twelve query-head gates.
    auto bad=make(TensorShape::make_BSM(3,4),false); bool rejected=false;
    try { head_gate_bhsd_to_flat(bad,x,stream); } catch(const std::runtime_error&) { rejected=true; }
    assert(rejected);
    // Same number of output elements, wrong geometry must also fail in backward.
    x.requires_grad=true;x.ensure_grad();auto out=head_gate_bhsd_to_flat(g,x,stream);
    auto wrong=make(TensorShape::make_BSM(1,252),false);rejected=false;
    try { out.grad_fn->apply(wrong,stream,nullptr,nullptr); } catch(const std::runtime_error&) { rejected=true; }
    assert(rejected&&!out.grad_fn->applied);
}
}
int main() {
    try {
        routing();shared_producer();identity_and_decode();
        std::cout<<"Head gate ownership/routing/geometry tests passed\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}

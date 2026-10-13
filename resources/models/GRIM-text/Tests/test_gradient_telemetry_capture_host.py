"""CPU checks of real stream registration, names, summary math and publication.
No CUDA/training target is built. CUDA allocation uses host shims in this test.
"""
from pathlib import Path
from test_add_gradfn_host import compile_and_run
from test_unary_gradfn_host import method


def main():
    root = Path(__file__).resolve().parent.parent
    capture = (root / 'training/Diagnostics/GradientTelemetryCapture.cu').read_text(encoding='utf-8')
    lattice = (root / 'Shared/Telemetry/TelemetryLattice_GPU.cu').read_text(encoding='utf-8')
    header = (root / 'Shared/Telemetry/TelemetryLattice_GPU.hpp').read_text(encoding='utf-8')
    logger = (root / 'Shared/Telemetry/TelemetryCsvLogger.cu').read_text(encoding='utf-8')
    assert 'cudaStreamSynchronize' not in capture and 'cudaDeviceSynchronize' not in capture
    assert 'cudaEventSynchronize' not in capture and 'logCapture' not in logger
    assert 'stream_idx >= 94 && isnan(x_t)' in lattice
    assert '!std::isfinite(raw_obs[s])' in logger
    assert 'lattice.readSnapshot(snapshot)' in logger
    assert 'lattice.readState(' not in logger
    loop = (root / 'training/Phases/Phase2_TrainingLoop.cu').read_text(encoding='utf-8')
    window = method(loop, 'void runOptimizerWindowFromEpoch(')
    assert window.index('captureGradientWindowTelemetry(') < window.index('launchOptimizerUpdate(')
    assert window.index('cudaStreamSynchronize(') < window.index('collectGradientTelemetry(')
    assert window.index('collectGradientTelemetry(') < window.index('updateTelemetryObservations(')
    assert loop.index('captureGateForwardTelemetry(') < loop.index('auto loss_result =')
    assert 'bindings.d_sequence_lengths' in capture
    write_header = method(logger, 'void TelemetryCsvLogger::writeHeader(')
    assert 'record_kind' not in write_header
    enum = method(header, 'enum class MetricStream') + ';'
    active = method(header, '__host__ __device__ constexpr bool isActiveMetricStream(').replace('__host__ __device__ ', '')
    names = method(lattice, 'const char* getMetricStreamName(')
    stats = method(capture, 'struct Stats') + ';'
    merge = method(capture, '__device__ Stats merge(').replace('__device__ ', '')
    constants = capture[capture.index('using Telemetry::MetricStream;'):capture.index('struct Stats {')]
    state = method(capture, 'struct GradientTelemetryCaptureState') + ';'
    registration = method(capture, 'void registerGradientMetricStreams(')
    publication = method(capture, 'void publishStats(')
    collection = method(capture, 'void collectGradientTelemetry(')
    source = r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using cudaStream_t = void*;
using cudaError_t = int;
constexpr int cudaSuccess = 0;
const char* cudaGetErrorString(int) {return "host mock";}
int cudaMallocAsync(void** p,size_t n,void*) {*p=std::malloc(n);return 0;}
int cudaMallocHost(void** p,size_t n) {*p=std::malloc(n);return 0;}
int cudaFree(void* p) {std::free(p);return 0;}
int cudaFreeHost(void* p) {std::free(p);return 0;}
int cudaMemsetAsync(void* p,int v,size_t n,void*) {std::memset(p,v,n);return 0;}
namespace GRIM {
namespace HyperParameters {
constexpr int TELEMETRY_MAX_STREAMS = 4096;
struct HP {int num_layers;bool attention_residual_gate_enabled;};
HP encoderLayerConstructionHP(int config) {return {config==2?200:2,config!=0};}
}
namespace Telemetry {
struct LatticeConfig {int num_streams=94;void* stream=nullptr;std::vector<std::string> metric_stream_names;};
""" + enum + active + names + r"""
}
namespace Diagnostics {
""" + constants + stats + merge + state + r"""
}
}
struct Group {std::string name;};
struct Registry {
    std::vector<Group> groups={{"layer0_wo_weight"},{"layer0_attention_residual_gate_bias"}};
    const std::vector<Group>& requireParameterGroups(const char*) {return groups;}
};
namespace GRIMText::Training {
struct TrainingContext {
    int config=1;
    Registry parameter_registry;
    struct {
        float last_obs[4096]={};
        std::shared_ptr<GRIM::Diagnostics::GradientTelemetryCaptureState> gradient_capture;
    } telemetry;
};
}
namespace GRIM::Diagnostics {
""" + registration + publication + collection + r"""
Stats point(double x) {
    Stats s{};s.count=1;s.mean=s.min=s.max=x;s.sum_sq=x*x;
    s.slope_sum=x*(1-.5*x);s.below_one=x<1;s.low=x<=.1;s.high=x>=1.9;
    return s;
}
}
int main() {
    using namespace GRIM;using namespace GRIM::Diagnostics;using namespace GRIM::Telemetry;
    Stats total{};
    for(double x:{0.,1.,2.}) total=merge(total,point(x));
    assert(total.count==3 && total.mean==1 && total.m2==2 && total.sum_sq==5);
    assert(total.min==0 && total.max==2 && total.low==1 && total.high==1);
    assert(std::abs(merge(point(1.-1e-7),point(1.+1e-7)).m2-2e-14)<1e-22);
    Stats bad{};bad.nonfinite=1;assert(merge(bad,point(1)).nonfinite);
    for(int i=14;i<27;++i) assert(!isActiveMetricStream(i));
    for(int i=63;i<94;++i) assert(!isActiveMetricStream(i));
    assert(isActiveMetricStream(94) && !isActiveMetricStream(4096));
    GRIMText::Training::TrainingContext ctx;
    LatticeConfig config;registerGradientMetricStreams(ctx,config);
    assert(config.num_streams==142+2*34+2*5);
    for(int i=94;i<config.num_streams;++i) {
        assert(std::isnan(ctx.telemetry.last_obs[i]));
        assert(std::string(getMetricStreamName(static_cast<MetricStream>(i),&config))!="unknown");
    }
    assert(std::string(getMetricStreamName(MetricStream::LOSS,nullptr))=="loss");
    assert(std::string(getMetricStreamName(MetricStream::TOP_ENERGY_GROUP_BASE,nullptr))=="top_energy_1_group_index");
    assert(config.metric_stream_names[142]=="layer0_attention_residual_gate_multiplier_mean");
    assert(config.metric_stream_names[142+34+29]=="layer1_attention_residual_gate_bias_grad_postclip_mean");
    assert(config.metric_stream_names[210]=="layer0_wo_weight_grad_rms_pre");
    auto& state=*ctx.telemetry.gradient_capture;
    state.forward_pending=true;state.readback_pending=true;state.forward_tokens=3;state.forward_step=17;
    for(int layer=0;layer<2;++layer) {
        state.host_stats[2*layer]=total;state.host_stats[2*layer+1]=total;
        for(int part=0;part<4;++part) {
            state.window_valid[4*layer+part]=true;
            state.host_stats[4+4*layer+part]=point(part==3?-.013469:1.);
        }
    }
    collectGradientTelemetry(ctx);
    assert(std::abs(ctx.telemetry.last_obs[142+29]+.013469f)<1e-8f);
    assert(ctx.telemetry.last_obs[142]==1.f && ctx.telemetry.last_obs[99]==17.f);
    assert(!state.forward_pending && !state.readback_pending);
    GRIMText::Training::TrainingContext disabled;disabled.config=0;
    LatticeConfig disabled_config;registerGradientMetricStreams(disabled,disabled_config);
    assert(disabled_config.num_streams==152 && disabled.telemetry.gradient_capture->layers==0);
    GRIMText::Training::TrainingContext too_large;too_large.config=2;
    LatticeConfig large_config;bool rejected=false;
    try {registerGradientMetricStreams(too_large,large_config);} catch(const std::runtime_error&) {rejected=true;}
    assert(rejected);
    std::cout<<"MetricStream registration, names, missing samples and summary publication passed\n";
}
"""
    compile_and_run(source)
    print('Capture timing, retired-ID preservation and existing CSV schema checks passed')


if __name__ == '__main__':
    main()

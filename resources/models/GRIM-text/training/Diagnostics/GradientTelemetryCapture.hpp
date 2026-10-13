#pragma once

#include <cuda_runtime.h>

namespace GRIMText::Training { struct TrainingContext; }
namespace GRIM::Batching { struct BatchPayload; }
namespace GRIM::Batching { struct BatchDeviceBindings; }
namespace GRIM::Telemetry { struct LatticeConfig; }
namespace GRIM::Forward { struct ModelForwardOutputs; }
namespace GRIM::GradClip { struct ClipResult; struct ClipScratch; }

namespace GRIM::Diagnostics {
struct GradientTelemetryCaptureState;
// Register named MetricStreams and allocate reusable summary buffers at startup.
void registerGradientMetricStreams(GRIMText::Training::TrainingContext& ctx,
                                  GRIM::Telemetry::LatticeConfig& config);
// Consume pinned summaries AFTER the optimizer boundary's existing stream sync.
void collectGradientTelemetry(GRIMText::Training::TrainingContext& ctx);
// Samples real (non-padding) tokens at the configured log_interval, before backward.
void captureGateForwardTelemetry(GRIMText::Training::TrainingContext& ctx,
    const GRIM::Forward::ModelForwardOutputs& outputs,
    const GRIM::Batching::BatchPayload& payload,
    const GRIM::Batching::BatchDeviceBindings& bindings,
    int batch_idx, cudaStream_t stream);

// Every completed accumulation window: energy measurements are pre-clip;
// signed gate gradients are read post-clip, before optimizer mutation/zeroing.
void captureGradientWindowTelemetry(GRIMText::Training::TrainingContext& ctx,
    const GRIM::GradClip::ClipResult& clip, const GRIM::GradClip::ClipScratch& scratch,
    int optimizer_step, cudaStream_t stream);
}

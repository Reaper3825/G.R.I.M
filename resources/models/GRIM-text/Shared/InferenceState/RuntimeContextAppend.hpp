#pragma once

#include <cstddef>

namespace GRIM::Tokenizer { struct UniByteResult; }
namespace GRIM::HyperParameters { struct ModelHP; }
namespace GRIM::Inference {
struct InferenceGenerationSession;

// Append already-encoded permitted context to a span-paused trajectory. Does
// not acknowledge the boundary, run forward, score tokens, or emit span events.
// Atom addresses are remapped into session-owned tables. Failure leaves the
// trajectory unchanged. Leave one context position available for continuation.
std::size_t appendRuntimeContext(InferenceGenerationSession& session,
    const Tokenizer::UniByteResult& encoded, const HyperParameters::ModelHP& model_hp,
    int vocab_size);
}

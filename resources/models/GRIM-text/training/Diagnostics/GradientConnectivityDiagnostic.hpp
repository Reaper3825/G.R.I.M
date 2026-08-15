//======================================================//
//  GradientConnectivityDiagnostic.hpp
//  Backward gradient connectivity diagnostics
//======================================================//

#pragma once

#include <memory>

namespace GRIM {
namespace Autograd {

struct AutogradContext;

namespace Diagnostics {

class GradientVerificationSession {
public:
    GradientVerificationSession(
        AutogradContext& ctx,
        bool require_current_microbatch_delta);
    ~GradientVerificationSession();

    GradientVerificationSession(const GradientVerificationSession&) = delete;
    GradientVerificationSession& operator=(const GradientVerificationSession&) = delete;
    GradientVerificationSession(GradientVerificationSession&&) noexcept;
    GradientVerificationSession& operator=(GradientVerificationSession&&) noexcept;

    bool verify(AutogradContext& ctx) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

void logPostBackwardGradientSamples(AutogradContext& ctx, bool accumulate);
bool verifyGradientsAreConnected(AutogradContext& ctx);

}  // namespace Diagnostics
}  // namespace Autograd
}  // namespace GRIM
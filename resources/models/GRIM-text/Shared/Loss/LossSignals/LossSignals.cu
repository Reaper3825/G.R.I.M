//======================================================//
//  LossSignals.cu — validation-loss policy detector.
//  See LossSignals.hpp for design / Rule 20 notes.
//======================================================//

#include "LossSignals.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace GRIM::Loss {

namespace {
inline bool isFinite(float v) { return std::isfinite(static_cast<double>(v)); }
}

LossSignalBus::LossSignalBus(const LossSignalConfig& cfg) : cfg_(cfg) {
    // Rule 20: validate construction-time invariants loudly.
    if (!isFinite(cfg_.validation_high_threshold)) {
        throw std::runtime_error(
            "LossSignalBus: validation_high_threshold must be finite (got " +
            std::to_string(cfg_.validation_high_threshold) + ") at " +
            std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (cfg_.validation_high_patience < 0) {
        throw std::runtime_error(
            "LossSignalBus: validation_high_patience must be >= 0 (got " +
            std::to_string(cfg_.validation_high_patience) + ")");
    }
}

const LossSignals& LossSignalBus::recordValidation(int epoch, float val_loss) {
    (void)epoch;

    if (!isFinite(val_loss)) {
        throw std::runtime_error(
            "LossSignalBus::recordValidation got non-finite val_loss (" +
            std::to_string(val_loss) + ") at " +
            std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    // Reset per-epoch booleans.
    latest_.validation_high = false;

    // ValidationHigh patience counter.
    if (val_loss >= cfg_.validation_high_threshold) {
        consecutive_validation_high_++;
    } else {
        consecutive_validation_high_ = 0;
    }
    if (cfg_.validation_high_patience > 0 &&
        consecutive_validation_high_ >= cfg_.validation_high_patience) {
        latest_.validation_high = true;
    }

    latest_.last_validation_loss       = val_loss;
    latest_.consecutive_validation_high = consecutive_validation_high_;
    return latest_;
}

} // namespace GRIM::Loss

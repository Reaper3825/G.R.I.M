#pragma once

#include "SuccessCriteria.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace GRIM {

struct CriterionEvidenceSpanView {
    GoalTokenSpan criterion;
    GoalTokenSpan evidence;
};

// Read-only forward-boundary view over low-level Goal identification data.
// Goal remains passive storage; row-aware owners construct this view. The view
// remains valid only while its owning BatchPayload/ModelForwardOutputs metadata
// is alive and unchanged.
class GoalSpanView {
public:
    GoalSpanView() = default;
    GoalSpanView(const GoalTokenSpan* target_state,
                 const SuccessCriteria* success_criteria) noexcept
        : target_state_(target_state), success_criteria_(success_criteria) {}

    bool hasGoal() const noexcept {
        return target_state_ != nullptr || success_criteria_ != nullptr;
    }
    bool hasTargetState() const noexcept { return target_state_ != nullptr; }
    bool hasCriteria() const noexcept { return success_criteria_ != nullptr; }

    const GoalTokenSpan& targetStateSpan() const {
        if (!hasTargetState()) {
            throw std::runtime_error(
                "GoalSpanView::targetStateSpan: target_state is absent");
        }
        return *target_state_;
    }

    const GoalTokenSpan& criteriaSpan() const {
        if (!hasCriteria()) {
            throw std::runtime_error(
                "GoalSpanView::criteriaSpan: success_criteria is absent");
        }
        return success_criteria_->span;
    }

    std::size_t criterionCount() const noexcept {
        return hasCriteria() ? success_criteria_->entries.size() : 0;
    }

    CriterionEvidenceSpanView criterionEvidenceSpans(std::size_t index) const {
        if (!hasCriteria()) {
            throw std::runtime_error(
                "GoalSpanView::criterionEvidenceSpans: success_criteria is absent");
        }
        const auto& entries = success_criteria_->entries;
        if (index >= entries.size()) {
            throw std::out_of_range(
                "GoalSpanView::criterionEvidenceSpans: index=" +
                std::to_string(index) +
                " is outside criterionCount=" +
                std::to_string(entries.size()));
        }
        return CriterionEvidenceSpanView{
            entries[index].criterion_span,
            entries[index].evidence_span};
    }

private:
    const GoalTokenSpan* target_state_ = nullptr;
    const SuccessCriteria* success_criteria_ = nullptr;
};

} // namespace GRIM

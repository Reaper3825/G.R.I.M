#pragma once

#include "ConceptBlockSpans.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace GRIM {

// Read-only row view over top-level ConceptBlock known/unknown spans.
class ConceptBlockSpanView {
public:
    ConceptBlockSpanView() = default;
    explicit ConceptBlockSpanView(const ConceptBlockSpans* spans) noexcept
        : spans_(spans) {}

    bool hasConceptBlockSpans() const noexcept { return spans_ != nullptr; }
    bool hasKnowns() const noexcept {
        return spans_ != nullptr && !spans_->knowns.empty();
    }
    bool hasUnknowns() const noexcept {
        return spans_ != nullptr && !spans_->unknowns.empty();
    }

    std::size_t knownCount() const noexcept {
        return hasKnowns() ? spans_->knowns.size() : 0;
    }
    std::size_t unknownCount() const noexcept {
        return hasUnknowns() ? spans_->unknowns.size() : 0;
    }

    const GoalTokenSpan& knownSpan(std::size_t index) const {
        return entrySpan(spans_ ? &spans_->knowns : nullptr, index, "known");
    }
    const GoalTokenSpan& unknownSpan(std::size_t index) const {
        return entrySpan(spans_ ? &spans_->unknowns : nullptr, index, "unknown");
    }

private:
    static const GoalTokenSpan& entrySpan(
        const std::vector<ConceptBlockSpanEntry>* entries,
        std::size_t index,
        const char* field) {
        if (!entries) {
            throw std::runtime_error(
                std::string("ConceptBlockSpanView::") + field +
                "Span: concept-block spans are absent");
        }
        if (index >= entries->size()) {
            throw std::out_of_range(
                std::string("ConceptBlockSpanView::") + field +
                "Span: index=" + std::to_string(index) +
                " is outside count=" + std::to_string(entries->size()));
        }
        return (*entries)[index].span;
    }

    const ConceptBlockSpans* spans_ = nullptr;
};

} // namespace GRIM

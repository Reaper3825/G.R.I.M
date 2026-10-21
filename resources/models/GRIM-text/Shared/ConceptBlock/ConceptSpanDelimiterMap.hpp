#pragma once

#include "NamedConceptSpans.hpp"

#include <unordered_map>
#include <utility>

namespace GRIM {

enum class ConceptSpanMarkerKind : std::uint8_t { Open, Close };

// Definition indices, rather than names, identify nodes across different parents.
struct ConceptSpanDelimiterDefinition {
    std::string name;
    std::uint32_t parent = kNoNamedConceptSpanEntry;
    int open_id = -1;
    int close_id = -1;
};

struct ConceptSpanMarker {
    std::uint32_t definition;
    ConceptSpanMarkerKind kind;
};

// Derived host-only state for one resolved configuration/vocabulary pairing.
// Build completely before sharing as const. Never serialize this lookup.
class ConceptSpanDelimiterMap {
public:
    explicit ConceptSpanDelimiterMap(const NamedConceptSpanDefinitions& definitions)
        : definitions_(definitions) {
        validateConceptSpanDefinitions(definitions_);
        std::function<void(const NamedConceptSpanDefinitions&, std::uint32_t)> visit;
        visit = [&](const auto& nodes, std::uint32_t parent) {
            for (const auto& d : nodes) {
                // Transparent groups do not introduce a runtime delimiter scope.
                if (d.open_delimiter.empty()) {
                    visit(d.children, parent);
                    continue;
                }
                if (d.open_delimiter_id < 0 || d.close_delimiter_id < 0)
                    throw std::runtime_error("Unresolved concept span marker: " + d.name);
                const auto index = static_cast<std::uint32_t>(nodes_.size());
                nodes_.push_back({d.name, parent, d.open_delimiter_id, d.close_delimiter_id});
                add(d.open_delimiter_id, {index, ConceptSpanMarkerKind::Open});
                add(d.close_delimiter_id, {index, ConceptSpanMarkerKind::Close});
                visit(d.children, index);
            }
        };
        visit(definitions_, kNoNamedConceptSpanEntry);
    }

    const NamedConceptSpanDefinitions& definitions() const { return definitions_; }
    const std::vector<ConceptSpanDelimiterDefinition>& nodes() const { return nodes_; }
    const std::vector<ConceptSpanMarker>* find(int token_id) const {
        const auto it = markers_.find(token_id);
        return it == markers_.end() ? nullptr : &it->second;
    }

private:
    void add(int id, ConceptSpanMarker marker) {
        auto& entries = markers_[id];
        for (const auto& previous : entries) {
            if (previous.kind != marker.kind ||
                nodes_[previous.definition].parent == nodes_[marker.definition].parent)
                throw std::runtime_error("Ambiguous concept span delimiter ID: " + std::to_string(id));
        }
        entries.push_back(marker);
    }
    NamedConceptSpanDefinitions definitions_;
    std::vector<ConceptSpanDelimiterDefinition> nodes_;
    std::unordered_map<int, std::vector<ConceptSpanMarker>> markers_;
};

enum class GeneratedSpanEventKind : std::uint8_t { Open, Close, InvalidMarker };

struct GeneratedSpanEvent {
    GeneratedSpanEventKind kind;
    std::uint32_t definition;
    std::size_t token_position;
    // For Close, [begin, token_position + 1) includes both delimiters.
    std::size_t begin;
};

// Per-session parser. Ordinary tokens are ignored; malformed markers are
// retained as events and never mistaken for completed spans or EOS.
class ConceptSpanTracker {
public:
    explicit ConceptSpanTracker(std::shared_ptr<const ConceptSpanDelimiterMap> map)
        : map_(std::move(map)) {}

    void observe(int token_id, std::size_t position, std::vector<GeneratedSpanEvent>& events) {
        if (!map_) return;
        const auto* markers = map_->find(token_id);
        if (!markers) return;
        const auto kind = markers->front().kind;
        if (kind == ConceptSpanMarkerKind::Close) {
            if (!active_.empty()) {
                for (const auto& marker : *markers) {
                    if (marker.definition == active_.back().definition) {
                        events.push_back({GeneratedSpanEventKind::Close, marker.definition,
                                          position, active_.back().begin});
                        active_.pop_back();
                        return;
                    }
                }
            }
        } else {
            const auto parent = active_.empty() ? kNoNamedConceptSpanEntry : active_.back().definition;
            for (const auto& marker : *markers) {
                if (map_->nodes()[marker.definition].parent == parent) {
                    active_.push_back({marker.definition, position});
                    events.push_back({GeneratedSpanEventKind::Open, marker.definition, position, position});
                    return;
                }
            }
        }
        events.push_back({GeneratedSpanEventKind::InvalidMarker, kNoNamedConceptSpanEntry,
                          position, position});
    }

private:
    struct ActiveSpan { std::uint32_t definition; std::size_t begin; };
    std::shared_ptr<const ConceptSpanDelimiterMap> map_;
    std::vector<ActiveSpan> active_;
};

} // namespace GRIM

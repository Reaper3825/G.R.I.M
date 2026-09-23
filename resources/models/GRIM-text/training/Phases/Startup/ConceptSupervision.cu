#include "ConceptSupervision.hpp"
#include "../../../Shared/TokenizerArtifacts/GrmtSequence.hpp"
#include <algorithm>
#include <stdexcept>
#include <type_traits>

namespace GRIMText::Training {
bool projectConceptSupervision(
    GRIM::TokenizerArtifacts::GrmtSequence& sequence,
    const GRIM::NamedConceptSpanDefinitions& definitions,
    const std::string& source) {
    using Policy = GRIM::ConceptSpanSupervision;
    GRIM::validateConceptSpanDefinitions(definitions);
    if (sequence.conceptSpanLayoutIdentity() != GRIM::conceptSpanLayout(definitions))
        throw std::runtime_error(source + ": concept span layout changed; regenerate GRMT");
    if (!sequence.named_concept_spans) return false;
    const auto& spans = *sequence.named_concept_spans;
    const auto n = sequence.token_ids.size();
    GRIM::validateNamedConceptSpans(spans, n);
    const auto policies = GRIM::conceptSpanPolicies(spans.entries, definitions);
    std::vector<Policy> token_policy(n, Policy::Context);
    for (std::size_t i = 0; i < spans.size(); ++i) {
        const auto& entry = spans.entries[i];
        std::fill(token_policy.begin() + entry.span.begin, token_policy.begin() + entry.span.end, policies[i]);
    }
    if (std::find(token_policy.begin(), token_policy.end(), Policy::Supervised) == token_policy.end())
        return false;
    std::vector<int32_t> map(n + 1, 0);
    std::vector<Policy> kept_policy;
    for (std::size_t i = 0; i < n; ++i) {
        const bool keep = token_policy[i] != Policy::Ignore;
        map[i + 1] = map[i] + (keep ? 1 : 0);
        if (keep) kept_policy.push_back(token_policy[i]);
    }
    const auto first = std::find(kept_policy.begin(), kept_policy.end(), Policy::Supervised);
    const auto prefix = static_cast<int32_t>(first - kept_policy.begin());
    if (prefix == 0)
        throw std::runtime_error(source + ": first supervised span needs a preceding causal token; enable BOS");
    for (auto& binding : sequence.compiled_bootstrap_bindings) {
        const auto pos = static_cast<std::size_t>(binding.token_pos);
        if (pos >= n || map[pos] == map[pos + 1])
            throw std::runtime_error(source + ": ignored span contains an execution bootstrap binding");
        binding.token_pos = map[pos];
    }
    const auto select = [&](auto& values) {
        if (values.empty()) return;
        if (values.size() != n) throw std::runtime_error(source + ": unaligned token metadata");
        std::size_t out = 0;
        for (std::size_t i = 0; i < n; ++i)
            if (map[i] != map[i + 1]) values[out++] = values[i];
        values.resize(out);
    };
    select(sequence.token_ids);
    select(sequence.token_numeric_values);
    select(sequence.token_atom_mask);
    select(sequence.token_atom_flags);
    select(sequence.atom_entry_ids);
    select(sequence.token_local_atom_indices);
    select(sequence.token_atom_aux_target_mask);
    select(sequence.token_exec_slot_indices);
    sequence.targets.assign(sequence.token_ids.size(), -1);
    for (std::size_t i = 1; i < kept_policy.size(); ++i)
        if (kept_policy[i] == Policy::Supervised) sequence.targets[i - 1] = sequence.token_ids[i];
    sequence.named_concept_spans = GRIM::remapNamedConceptSpans(sequence.named_concept_spans, map);
    sequence.prompt_length = prefix;
    sequence.prompt_end_pos = prefix - 1;
    return true;
}
}

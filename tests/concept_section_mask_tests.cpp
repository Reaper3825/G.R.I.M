#include "../DataCollection/concept_block_canonical.hpp"
#include "../resources/models/GRIM-text/training/Phases/Startup/SlidingWindow.hpp"
#include "../resources/models/GRIM-text/Shared/ConceptBlock/ConceptBlockSpans.hpp"
#include "../resources/models/GRIM-text/Shared/UnigramByte/TokenLayout.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

int main() {
    const nlohmann::json row{
        {"prompt", "Add 10 and 2."},
        {"determine", "10 + 2 = sum"},
        {"execute", "<TOOL>(10 + 2)</TOOL> -> sum"},
        {"answer", "The sum is 12."},
    };
    const auto rendered = GRIM::ConceptCanonical::render(row);
    GRIM::TokenizerArtifacts::GrmtSequence sequence;
    sequence.concept_block_id = "section-mask-test";
    for (const unsigned char byte : rendered.text) {
        sequence.token_ids.push_back(GRIM::Tokenizer::BYTE_TOKEN_OFFSET + byte);
    }
    const auto length = sequence.token_ids.size();
    sequence.targets.assign(length, -1);
    sequence.token_numeric_values.assign(length, 0.0f);
    sequence.token_atom_mask.assign(length, 0);
    sequence.token_atom_flags.assign(length, 0);
    sequence.atom_entry_ids.assign(length, GRIM::Tokenizer::kAtomEntryNone);
    sequence.token_local_atom_indices.assign(
        length, GRIM::Tokenizer::kLocalAtomIndexNone);
    sequence.token_exec_slot_indices.assign(length, -1);
    sequence.prompt_length = static_cast<std::int32_t>(rendered.prompt_byte_end);
    sequence.prompt_end_pos = sequence.prompt_length - 1;
    auto spans = std::make_shared<GRIM::ConceptBlockSpans>();
    const auto entry = [&](const GRIM::ConceptCanonical::LogicalByteSpan& span) {
        GRIM::ConceptBlockSpanEntry result;
        result.span = GRIM::GoalTokenSpan{
            static_cast<std::int32_t>(span.begin),
            static_cast<std::int32_t>(span.end)};
        result.token_ids.assign(
            sequence.token_ids.begin() + span.begin,
            sequence.token_ids.begin() + span.end);
        return result;
    };
    spans->determine = entry(rendered.determine);
    spans->execute = entry(rendered.execute);
    sequence.concept_block_spans = spans;
    sequence.answer_span = GRIM::GoalTokenSpan{
        static_cast<std::int32_t>(rendered.answer.begin),
        static_cast<std::int32_t>(rendered.answer.end)};

    const auto log_path = std::filesystem::path("build") /
        "training_concept_section_mask_tests.log";
    {
        TrainingLogger logger("build", "concept_section_mask_tests");
        std::vector<GRIM::TokenizerArtifacts::GrmtSequence> rows{sequence};
        GRIMText::Training::applySlidingWindows(
            rows, "section-mask", GRIM::HyperParameters::TrainingStage::SFT,
            {"determine", "execute", "answer"}, {"prompt"},
            1024, 768, 1, false, false, logger);
        assert(rows.size() == 1);
        const auto& projected = rows.front();
        assert(projected.prompt_end_pos ==
               static_cast<std::int32_t>(rendered.determine.begin - 1));
        for (std::size_t position = 1; position < length; ++position) {
            const bool supervised =
                (position >= rendered.determine.begin &&
                 position < rendered.determine.end) ||
                (position >= rendered.execute.begin &&
                 position < rendered.execute.end) ||
                (position >= rendered.answer.begin &&
                 position < rendered.answer.end);
            assert(projected.targets[position - 1] ==
                   (supervised ? projected.token_ids[position] : -1));
        }
        assert(projected.targets.back() == -1);
    }
    std::filesystem::remove(log_path);
}

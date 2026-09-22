#include "../DataCollection/concept_block_canonical.hpp"
#include "../resources/models/GRIM-text/training/Phases/Startup/SlidingWindow.hpp"
#include "../resources/models/GRIM-text/Shared/ConceptBlock/ConceptBlockSpans.hpp"
#include "../resources/models/GRIM-text/Shared/ConceptBlock/NamedConceptSpans.hpp"
#include "../resources/models/GRIM-text/Shared/UnigramByte/TokenLayout.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

int main() {
    const nlohmann::json row{
        {"prompt", "Add 10 and 2."},
        {"determine", "Compute the total by adding the two supplied quantities."},
        {"define", "Let the supplied quantities be addends; sum is their total."},
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
    auto named_spans = std::make_shared<GRIM::NamedConceptSpans>();
    for (const auto& rendered_span : rendered.named_spans) {
        GRIM::NamedConceptSpan result;
        result.name = rendered_span.name;
        result.span = GRIM::GoalTokenSpan{
            static_cast<std::int32_t>(rendered_span.span.begin),
            static_cast<std::int32_t>(rendered_span.span.end)};
        result.token_ids.assign(
            sequence.token_ids.begin() + rendered_span.span.begin,
            sequence.token_ids.begin() + rendered_span.span.end);
        named_spans->entries.push_back(std::move(result));
    }
    sequence.named_concept_spans = std::move(named_spans);

    const auto log_path = std::filesystem::path("build") /
        "training_concept_section_mask_tests.log";
    {
        TrainingLogger logger("build", "concept_section_mask_tests");
        std::vector<GRIM::TokenizerArtifacts::GrmtSequence> rows{sequence};
        GRIMText::Training::applySlidingWindows(
            rows, "section-mask", GRIM::HyperParameters::TrainingStage::SFT,
            {"determine", "define", "execute", "answer"}, {"prompt"},
            1024, 768, 1, false, false, logger);
        assert(rows.size() == 1);
        const auto& projected = rows.front();
        assert(projected.prompt_end_pos ==
               static_cast<std::int32_t>(rendered.determine.begin - 1));
        for (std::size_t position = 1; position < length; ++position) {
            const bool supervised =
                (position >= rendered.determine.begin &&
                 position < rendered.determine.end) ||
                (position >= rendered.define.begin &&
                 position < rendered.define.end) ||
                (position >= rendered.execute.begin &&
                 position < rendered.execute.end) ||
                (position >= rendered.answer.begin &&
                 position < rendered.answer.end);
            assert(projected.targets[position - 1] ==
                   (supervised ? projected.token_ids[position] : -1));
        }
        assert(projected.targets.back() == -1);

        const auto make_collection_row = [](bool known) {
            GRIM::TokenizerArtifacts::GrmtSequence result;
            result.concept_block_id = known
                ? "knowns-only-section-mask-test"
                : "unknowns-only-section-mask-test";
            result.token_ids = {100, 101, 102, 103, 104, 105};
            result.targets.assign(result.token_ids.size(), -1);
            result.token_numeric_values.assign(result.token_ids.size(), 0.0f);
            result.token_atom_mask.assign(result.token_ids.size(), 0);
            result.token_atom_flags.assign(result.token_ids.size(), 0);
            result.atom_entry_ids.assign(
                result.token_ids.size(), GRIM::Tokenizer::kAtomEntryNone);
            result.token_local_atom_indices.assign(
                result.token_ids.size(), GRIM::Tokenizer::kLocalAtomIndexNone);
            result.token_exec_slot_indices.assign(result.token_ids.size(), -1);
            result.prompt_length = 2;
            result.prompt_end_pos = 1;

            auto collection_spans = std::make_shared<GRIM::ConceptBlockSpans>();
            GRIM::ConceptBlockSpanEntry collection_entry{
                {result.token_ids[2], result.token_ids[3]},
                GRIM::GoalTokenSpan{2, 4}};
            if (known) {
                collection_spans->knowns.push_back(std::move(collection_entry));
            } else {
                collection_spans->unknowns.push_back(std::move(collection_entry));
            }
            result.concept_block_spans = std::move(collection_spans);
            return result;
        };

        for (const bool known : {true, false}) {
            std::vector<GRIM::TokenizerArtifacts::GrmtSequence> collection_rows{
                make_collection_row(known)};
            GRIMText::Training::applySlidingWindows(
                collection_rows,
                known ? "knowns-only-mask" : "unknowns-only-mask",
                GRIM::HyperParameters::TrainingStage::SFT,
                {known ? "knowns" : "unknowns"},
                {"prompt"},
                1024, 768, 1, false, false, logger);
            assert(collection_rows.size() == 1);
            const auto& collection = collection_rows.front();
            assert(collection.token_ids.size() == 4);
            assert(collection.targets[0] == -1);
            assert(collection.targets[1] == collection.token_ids[2]);
            assert(collection.targets[2] == collection.token_ids[3]);
            assert(collection.targets[3] == -1);
        }
    }
    std::filesystem::remove(log_path);
}

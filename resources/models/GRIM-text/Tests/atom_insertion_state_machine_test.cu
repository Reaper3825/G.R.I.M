//======================================================//
//  atom_insertion_state_machine_test.cu
//  Focused OPEN-type + EXIT supervision/decoder checks
//======================================================//

#include "../Shared/AtomInsertion/AtomInsertionData.hpp"
#include "../Shared/AtomInsertion/AtomInsertionDecode.hpp"
#include "../Shared/AtomInsertion/AtomInsertionDecisionLayout.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<float> blankDecisionLogits(std::size_t gap_count) {
    return std::vector<float>(
        gap_count * GRIM::AtomInsertion::kAtomDecisionClassCount,
        -10.0f);
}

void setDecision(std::vector<float>& logits,
                 std::size_t gap,
                 int decision_class,
                 float value = 10.0f) {
    logits[gap * GRIM::AtomInsertion::kAtomDecisionClassCount +
           static_cast<std::size_t>(decision_class)] = value;
}

void testAuthoredTargetsUseGenericExit() {
    const auto example = GRIM::AtomInsertion::buildAtomInsertionExample(
        "<INT>42</INT><FLOAT>3.5</FLOAT><STRING>x</STRING><BOOL>true</BOOL>",
        true,
        "testAuthoredTargetsUseGenericExit");
    require(example.spans.size() == 4, "expected four authored atom spans");

    for (const auto& span : example.spans) {
        const int open_class =
            GRIM::AtomInsertion::openDecisionClassIndexOrThrow(
                span.type, "testAuthoredTargetsUseGenericExit");
        require(example.hasDecisionTarget(span.begin_gap, open_class),
                "typed OPEN target is missing");
        require(example.hasDecisionTarget(
                    span.end_gap,
                    GRIM::AtomInsertion::kExitDecisionClassIndex),
                "generic EXIT target is missing");
    }

    const auto& first = example.spans[0];
    const auto& second = example.spans[1];
    require(first.end_gap == second.begin_gap,
            "first two spans should be adjacent");
    require(example.hasDecisionTarget(
                first.end_gap,
                GRIM::AtomInsertion::kExitDecisionClassIndex),
            "adjacent gap must close the active span");
    require(example.hasDecisionTarget(
                second.begin_gap,
                GRIM::AtomInsertion::openDecisionClassIndexOrThrow(
                    second.type, "testAuthoredTargetsUseGenericExit")),
            "adjacent gap must open the next typed span");

    const auto payload = GRIM::AtomInsertion::buildAtomInsertionBatchPayload(
        {example},
        GRIM::Tokenizer::UNIGRAM_VOCAB_OFFSET,
        static_cast<int>(example.transformerInputSize()),
        true,
        "testAuthoredTargetsUseGenericExit");
    require(payload.atom_insertion_gap_targets.size() ==
                static_cast<std::size_t>(payload.atomInsertionGapRowCount()) *
                    GRIM::AtomInsertion::kAtomDecisionClassCount,
            "payload decision rectangle width is not five");
    require(payload.atom_insertion_positive_label_count == 8,
            "each span must author exactly one OPEN and one EXIT target");
}

void testExitRendersPersistedType() {
    const std::string plain_text = "3.14";
    std::vector<std::uint8_t> valid_gaps(plain_text.size() + 1, 1);
    auto logits = blankDecisionLogits(valid_gaps.size());
    setDecision(
        logits,
        0,
        GRIM::AtomInsertion::openDecisionClassIndexOrThrow(
            GRIM::Tokenizer::AtomType::ATOM_FLOAT,
            "testExitRendersPersistedType"));
    setDecision(
        logits,
        plain_text.size(),
        GRIM::AtomInsertion::kExitDecisionClassIndex);

    const std::string decoded =
        GRIM::AtomInsertion::decodeAtomDecisionPredictions(
            plain_text, valid_gaps, logits, 0.0f);
    require(decoded == "<FLOAT>3.14</FLOAT>",
            "EXIT did not render the close token from the persisted FLOAT type");
}

void testAdjacentExitThenOpen() {
    const std::string plain_text = "12.0";
    std::vector<std::uint8_t> valid_gaps(plain_text.size() + 1, 1);
    auto logits = blankDecisionLogits(valid_gaps.size());
    setDecision(
        logits,
        0,
        GRIM::AtomInsertion::openDecisionClassIndexOrThrow(
            GRIM::Tokenizer::AtomType::ATOM_INT,
            "testAdjacentExitThenOpen"));
    setDecision(logits, 1, GRIM::AtomInsertion::kExitDecisionClassIndex);
    setDecision(
        logits,
        1,
        GRIM::AtomInsertion::openDecisionClassIndexOrThrow(
            GRIM::Tokenizer::AtomType::ATOM_FLOAT,
            "testAdjacentExitThenOpen"));
    setDecision(
        logits,
        plain_text.size(),
        GRIM::AtomInsertion::kExitDecisionClassIndex);

    const std::string decoded =
        GRIM::AtomInsertion::decodeAtomDecisionPredictions(
            plain_text, valid_gaps, logits, 0.0f);
    require(decoded == "<INT>1</INT><FLOAT>2.0</FLOAT>",
            "shared EXIT/OPEN gap did not preserve both span types");
}

} // namespace

int main() {
    try {
        static_assert(GRIM::AtomInsertion::kAtomDecisionClassCount == 5);
        static_assert(GRIM::AtomInsertion::kExitDecisionClassIndex == 4);
        testAuthoredTargetsUseGenericExit();
        testExitRendersPersistedType();
        testAdjacentExitThenOpen();
        std::cout << "atom_insertion_state_machine_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "atom_insertion_state_machine_test: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}

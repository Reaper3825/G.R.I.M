#include "resources/models/GRIM-text/Shared/UnigramByte/TokenLayout.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>

using namespace GRIM::Tokenizer;

static void check(bool condition) {
    if (!condition) throw std::runtime_error("Manual vocabulary budget regression failed");
}

static void rejects(int ordinary, std::size_t manual) {
    try {
        (void)unigramPieceBudgetWithManualAllowanceOrThrow(ordinary, manual);
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error("Invalid vocabulary budget was accepted");
}

int main() {
    check(unigramPieceBudgetWithManualAllowanceOrThrow(10000, 0) == 10000);
    const int combined = unigramPieceBudgetWithManualAllowanceOrThrow(10000, 587);
    const auto layout = tokenLayoutFromActualVocabOrThrow(
        UNIGRAM_VOCAB_OFFSET + combined, "manual vocab budget test");
    check(layout.num_unigram == 10587 && layout.total_vocab() == 10906);
    check(layout.num_unigram - 587 == 10000);
    check(layout.isUnigram(tokenIdForIndex(0)));
    check(layout.isUnigram(tokenIdForIndex(combined - 1)));
    check(!layout.isUnigram(layout.total_vocab()));
    check(layout.isByte(BYTE_TOKEN_OFFSET) && layout.isNumeric(NUMERIC_TOKEN_OFFSET));
    check(layout.isAtom(ATOM_TOKEN_OFFSET) && layout.isNewline(NEWLINE_TOKEN_ID));
    // Authored entries may outnumber the ordinary budget.
    check(unigramPieceBudgetWithManualAllowanceOrThrow(1, 587) == 588);

    const int max_pieces = std::numeric_limits<int>::max() - UNIGRAM_VOCAB_OFFSET;
    check(unigramPieceBudgetWithManualAllowanceOrThrow(max_pieces - 1, 1) == max_pieces);
    rejects(0, 0);
    rejects(-1, 587);
    rejects(std::numeric_limits<int>::min(), 0);
    rejects(std::numeric_limits<int>::max(), 0);
    rejects(max_pieces, 1);
    rejects(10000, std::numeric_limits<std::size_t>::max());
    std::cout << "Manual vocabulary budget tests passed\n";
}

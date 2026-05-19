#pragma once
//======================================================//
//  TokenTypeGate.hpp
//  Fixed hard token-layout gate for embedding / LM-head rows.
//
//  Gate classes are derived from TokenLayout token-id ranges. The gate is
//  deterministic, non-trainable, and intentionally symmetric for tied
//  embeddings: input embedding lookup and LM-head effective weights use the
//  same row-id -> active-dimension range contract.
//======================================================//

#include "../UnigramByte/TokenLayout.hpp"

#include <cstdint>

#ifdef __CUDACC__
#define GRIM_TOKEN_GATE_HD __host__ __device__
#else
#define GRIM_TOKEN_GATE_HD
#endif

namespace GRIM::TensorContract {

constexpr int TOKEN_TYPE_GATE_UNIGRAM_OFFSET =
    GRIM::Tokenizer::ATOM_TOKEN_OFFSET + GRIM::Tokenizer::kAtomTypeCount;

enum class TokenTypeGateClass : int {
    SPECIAL = 0,
    BYTE = 1,
    ATOM = 2,
    UNIGRAM = 3,
    INVALID = 4
};

struct TokenTypeGateRange {
    int start = -1;
    int end = -1;
    int width = 0;
    TokenTypeGateClass type = TokenTypeGateClass::INVALID;
};

GRIM_TOKEN_GATE_HD inline TokenTypeGateClass tokenTypeGateClassForTokenId(
    int token_id,
    int vocab_size)
{
    if (token_id < 0 || token_id >= vocab_size) {
        return TokenTypeGateClass::INVALID;
    }
    if (token_id >= GRIM::Tokenizer::SPECIAL_TOKEN_OFFSET &&
        token_id < GRIM::Tokenizer::SPECIAL_TOKEN_OFFSET + GRIM::Tokenizer::NUM_SPECIAL_TOKENS) {
        return TokenTypeGateClass::SPECIAL;
    }
    if (token_id >= GRIM::Tokenizer::BYTE_TOKEN_OFFSET &&
        token_id < GRIM::Tokenizer::BYTE_TOKEN_OFFSET + GRIM::Tokenizer::BYTE_VOCAB_SIZE) {
        return TokenTypeGateClass::BYTE;
    }
    if (token_id >= GRIM::Tokenizer::ATOM_TOKEN_OFFSET &&
        token_id < TOKEN_TYPE_GATE_UNIGRAM_OFFSET) {
        return TokenTypeGateClass::ATOM;
    }
    if (token_id >= TOKEN_TYPE_GATE_UNIGRAM_OFFSET && token_id < vocab_size) {
        return TokenTypeGateClass::UNIGRAM;
    }
    return TokenTypeGateClass::INVALID;
}

GRIM_TOKEN_GATE_HD inline TokenTypeGateRange tokenTypeGateRangeForClass(
    TokenTypeGateClass gate_class,
    int d_model)
{
    if (d_model < 4) {
        return TokenTypeGateRange{};
    }

    const int q = d_model / 4;
    TokenTypeGateRange range;
    range.type = gate_class;

    switch (gate_class) {
        case TokenTypeGateClass::SPECIAL:
            range.start = 0;
            range.end = q;
            break;
        case TokenTypeGateClass::BYTE:
            range.start = q;
            range.end = 2 * q;
            break;
        case TokenTypeGateClass::ATOM:
            range.start = 2 * q;
            range.end = 3 * q;
            break;
        case TokenTypeGateClass::UNIGRAM:
            range.start = 3 * q;
            range.end = d_model;
            break;
        case TokenTypeGateClass::INVALID:
            range.start = -1;
            range.end = -1;
            break;
    }

    range.width = range.end - range.start;
    if (range.start < 0 || range.end > d_model || range.width <= 0) {
        return TokenTypeGateRange{};
    }
    return range;
}

GRIM_TOKEN_GATE_HD inline TokenTypeGateRange tokenTypeGateRangeForTokenId(
    int token_id,
    int d_model,
    int vocab_size)
{
    return tokenTypeGateRangeForClass(
        tokenTypeGateClassForTokenId(token_id, vocab_size),
        d_model);
}

GRIM_TOKEN_GATE_HD inline bool tokenTypeGateAllowsDimension(
    int token_id,
    int dim,
    int d_model,
    int vocab_size)
{
    const TokenTypeGateRange range = tokenTypeGateRangeForTokenId(token_id, d_model, vocab_size);
    return dim >= range.start && dim < range.end;
}

}  // namespace GRIM::TensorContract

#undef GRIM_TOKEN_GATE_HD

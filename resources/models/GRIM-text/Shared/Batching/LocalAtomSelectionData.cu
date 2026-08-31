//======================================================//
//  LocalAtomSelectionData.cu
//======================================================//

#include "LocalAtomSelectionData.hpp"

#include "BatchPayload.hpp"
#include "../UnigramByte/SequenceLocalAtomTable.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace GRIM {
namespace Batching {
namespace {

struct FirstOccurrence {
    bool complete = false;
    int close_position = -1;
    std::vector<int> content_positions;
};

std::string prefix(const char* caller) {
    if (!caller || caller[0] == '\0') {
        throw std::runtime_error(
            "materializeLocalAtomSelectionMetadata requires a non-empty caller label");
    }
    return std::string(caller) + ": local atom selection metadata";
}

} // namespace

void materializeLocalAtomSelectionMetadata(
    BatchPayload& payload,
    const char* caller) {
    const std::string error_prefix = prefix(caller);

    payload.local_atom_query_positions.clear();
    payload.local_atom_query_types.clear();
    payload.local_atom_query_targets.clear();
    payload.local_atom_candidate_first_close_positions.clear();
    payload.local_atom_candidate_content_offsets.clear();
    payload.local_atom_candidate_content_positions.clear();
    payload.local_atom_row_type_candidate_offsets.clear();
    payload.local_atom_reference_target_count = 0;

    if (!payload.ownsHostInputData() || payload.isInferenceDecode()) {
        return;
    }
    if (payload.batch_size <= 0 || payload.max_seq_len <= 0 ||
        payload.total_tokens != payload.batch_size * payload.max_seq_len) {
        throw std::runtime_error(error_prefix + ": invalid batch geometry");
    }
    if (payload.input_ids.size() != static_cast<std::size_t>(payload.total_tokens) ||
        payload.token_local_atom_indices.size() !=
            static_cast<std::size_t>(payload.total_tokens)) {
        throw std::runtime_error(
            error_prefix + ": token/local-index arrays do not match total_tokens");
    }
    if (payload.seq_lengths.size() != static_cast<std::size_t>(payload.batch_size) ||
        payload.seq_local_atom_tables.size() !=
            static_cast<std::size_t>(payload.batch_size)) {
        throw std::runtime_error(
            error_prefix + ": row metadata does not match batch_size");
    }

    payload.local_atom_candidate_content_offsets.push_back(0);
    payload.local_atom_row_type_candidate_offsets.push_back(0);

    for (int row = 0; row < payload.batch_size; ++row) {
        const int row_length = payload.seq_lengths[static_cast<std::size_t>(row)];
        if (row_length < 0 || row_length > payload.max_seq_len) {
            throw std::runtime_error(
                error_prefix + ": invalid sequence length for row=" +
                std::to_string(row));
        }
        const auto& table = payload.seq_local_atom_tables[static_cast<std::size_t>(row)];
        if (!table) {
            throw std::runtime_error(
                error_prefix + ": sequence-local atom table is null for row=" +
                std::to_string(row));
        }

        std::array<std::vector<FirstOccurrence>,
                   static_cast<std::size_t>(GRIM::Tokenizer::kAtomTypeCount)>
            first_occurrences;
        for (int type_index = 0;
             type_index < GRIM::Tokenizer::kAtomTypeCount;
             ++type_index) {
            const auto type = static_cast<GRIM::Tokenizer::AtomType>(type_index);
            first_occurrences[static_cast<std::size_t>(type_index)].resize(
                table->size(type));
        }

        bool inside_atom = false;
        GRIM::Tokenizer::AtomType open_type =
            GRIM::Tokenizer::AtomType::ATOM_INT;
        std::uint32_t open_local_index = GRIM::Tokenizer::kLocalAtomIndexNone;
        int open_position = -1;

        const int row_offset = row * payload.max_seq_len;
        for (int token = 0; token < payload.max_seq_len; ++token) {
            const int flat_position = row_offset + token;
            const std::uint32_t local_index =
                payload.token_local_atom_indices[static_cast<std::size_t>(flat_position)];
            if (token >= row_length) {
                if (local_index != GRIM::Tokenizer::kLocalAtomIndexNone) {
                    throw std::runtime_error(
                        error_prefix + ": local atom index is present on padding at flat position=" +
                        std::to_string(flat_position));
                }
                continue;
            }

            const int token_id = payload.input_ids[static_cast<std::size_t>(flat_position)];
            if (GRIM::Tokenizer::isAtomOpenTokenId(token_id)) {
                if (inside_atom) {
                    throw std::runtime_error(
                        error_prefix + ": nested atom opening at flat position=" +
                        std::to_string(flat_position));
                }
                if (local_index == GRIM::Tokenizer::kLocalAtomIndexNone) {
                    throw std::runtime_error(
                        error_prefix + ": atom opening has no local index at flat position=" +
                        std::to_string(flat_position));
                }
                open_type = GRIM::Tokenizer::tokenIdToAtomType(token_id);
                const std::size_t type_index = static_cast<std::size_t>(
                    GRIM::Tokenizer::atomTypeIndexOrThrow(
                        open_type,
                        "materializeLocalAtomSelectionMetadata"));
                auto& candidates = first_occurrences[type_index];
                if (static_cast<std::size_t>(local_index) >= candidates.size() ||
                    !table->contains(open_type, local_index)) {
                    throw std::runtime_error(
                        error_prefix + ": invalid local address (" +
                        std::string(GRIM::Tokenizer::atomTypeName(open_type)) + ", " +
                        std::to_string(local_index) + ") at flat position=" +
                        std::to_string(flat_position));
                }
                if (local_index >=
                    static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
                    throw std::runtime_error(
                        error_prefix + ": local index cannot be encoded as selector target");
                }

                payload.local_atom_query_positions.push_back(flat_position);
                payload.local_atom_query_types.push_back(static_cast<int>(open_type));
                if (candidates[static_cast<std::size_t>(local_index)].complete) {
                    payload.local_atom_query_targets.push_back(
                        static_cast<int>(local_index) + 1);
                    ++payload.local_atom_reference_target_count;
                } else {
                    payload.local_atom_query_targets.push_back(
                        kLocalAtomNoReferenceTarget);
                }

                inside_atom = true;
                open_local_index = local_index;
                open_position = flat_position;
                continue;
            }

            if (local_index != GRIM::Tokenizer::kLocalAtomIndexNone) {
                throw std::runtime_error(
                    error_prefix +
                    ": local atom index is present outside an opening boundary at flat position=" +
                    std::to_string(flat_position));
            }

            if (!GRIM::Tokenizer::isAtomCloseTokenId(token_id)) {
                continue;
            }
            if (!inside_atom ||
                GRIM::Tokenizer::tokenIdToAtomType(token_id) != open_type) {
                throw std::runtime_error(
                    error_prefix + ": unmatched or mismatched atom close at flat position=" +
                    std::to_string(flat_position));
            }

            const std::size_t type_index = static_cast<std::size_t>(
                GRIM::Tokenizer::atomTypeIndexOrThrow(
                    open_type,
                    "materializeLocalAtomSelectionMetadata"));
            FirstOccurrence& candidate =
                first_occurrences[type_index][static_cast<std::size_t>(open_local_index)];
            if (!candidate.complete) {
                candidate.complete = true;
                candidate.close_position = flat_position;
                candidate.content_positions.reserve(
                    static_cast<std::size_t>(flat_position - open_position - 1));
                for (int content_position = open_position + 1;
                     content_position < flat_position;
                     ++content_position) {
                    candidate.content_positions.push_back(content_position);
                }
            }

            inside_atom = false;
            open_local_index = GRIM::Tokenizer::kLocalAtomIndexNone;
            open_position = -1;
        }

        if (inside_atom) {
            throw std::runtime_error(
                error_prefix + ": unterminated atom span in row=" +
                std::to_string(row));
        }

        for (int type_index = 0;
             type_index < GRIM::Tokenizer::kAtomTypeCount;
             ++type_index) {
            const auto type = static_cast<GRIM::Tokenizer::AtomType>(type_index);
            const auto& candidates =
                first_occurrences[static_cast<std::size_t>(type_index)];
            for (std::size_t local_index = 0;
                 local_index < candidates.size();
                 ++local_index) {
                const FirstOccurrence& candidate = candidates[local_index];
                if (!candidate.complete) {
                    throw std::runtime_error(
                        error_prefix + ": local table entry (" +
                        std::string(GRIM::Tokenizer::atomTypeName(type)) + ", " +
                        std::to_string(local_index) +
                        ") has no complete first occurrence in row=" +
                        std::to_string(row));
                }
                payload.local_atom_candidate_first_close_positions.push_back(
                    candidate.close_position);
                payload.local_atom_candidate_content_positions.insert(
                    payload.local_atom_candidate_content_positions.end(),
                    candidate.content_positions.begin(),
                    candidate.content_positions.end());
                if (payload.local_atom_candidate_content_positions.size() >
                    static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                    throw std::runtime_error(
                        error_prefix + ": candidate content position count exceeds int range");
                }
                payload.local_atom_candidate_content_offsets.push_back(
                    static_cast<int>(
                        payload.local_atom_candidate_content_positions.size()));
            }
            if (payload.local_atom_candidate_first_close_positions.size() >
                static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error(
                    error_prefix + ": candidate count exceeds int range");
            }
            payload.local_atom_row_type_candidate_offsets.push_back(
                static_cast<int>(
                    payload.local_atom_candidate_first_close_positions.size()));
        }
    }
}

} // namespace Batching
} // namespace GRIM


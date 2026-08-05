#include "concept_block_io_flatbuffer.hpp"

#include "../concept_block_generated.h"

#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/verifier.h>

#include <algorithm>
#include <fstream>
#include <limits>
#include <system_error>

#ifdef _WIN32
#include <Windows.h>
#endif

namespace GRIM::ConceptBlockIO {
namespace {

std::string stringValue(const flatbuffers::String* value) {
    return value ? value->str() : std::string{};
}

template <typename T>
std::vector<T> scalarVectorValue(const flatbuffers::Vector<T>* values) {
    if (!values) return {};
    return std::vector<T>(values->begin(), values->end());
}

std::vector<std::string> stringVectorValue(
    const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>* values) {
    std::vector<std::string> result;
    if (!values) return result;
    result.reserve(values->size());
    for (const auto* value : *values) result.push_back(stringValue(value));
    return result;
}

ConceptBlock fromFlatBuffer(const GRIMConcept::ConceptBlock& source) {
    ConceptBlock block;
    block.id = stringValue(source.id());
    block.name = stringValue(source.name());
    block.prompt = stringValue(source.prompt());
    block.intermediates = stringVectorValue(source.intermediates());
    block.answer = stringValue(source.answer());
    block.execution_gate_target =
        static_cast<ConceptExecutionGateTarget>(source.execution_gate_target());
    block.explanation = stringVectorValue(source.explanation());

    if (const auto* state0 = source.state_0()) {
        block.state_0.atoms = scalarVectorValue(state0->atoms());
        block.state_0.type = stringValue(state0->type());
    }

    if (const auto* steps = source.execution()) {
        block.execution.reserve(steps->size());
        for (const auto* source_step : *steps) {
            if (!source_step) continue;
            ConceptExecutionStep step;
            step.op = stringValue(source_step->op());
            step.args = scalarVectorValue(source_step->args());
            step.arg_slots = scalarVectorValue(source_step->arg_slots());
            step.result = source_step->result();
            block.execution.push_back(std::move(step));
        }
    }

    if (const auto* state1 = source.state_1()) {
        block.state_1.result = state1->result();
        block.state_1.has_result = true;
    }

    block.intermediate_count = source.intermediate_count();
    block.step_index = scalarVectorValue(source.step_index());
    block.format_type = stringValue(source.format_type());
    block.source_sequence_id = stringValue(source.source_sequence_id());
    block.timestamp = source.timestamp();

    // Derived fields are normalized on load so older/additive schema versions
    // cannot leave the editor with stale counts or step indices.
    block.recomputeDerived();
    return block;
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>>
createStringVector(flatbuffers::FlatBufferBuilder& builder,
                   const std::vector<std::string>& values) {
    std::vector<flatbuffers::Offset<flatbuffers::String>> offsets;
    offsets.reserve(values.size());
    for (const auto& value : values) offsets.push_back(builder.CreateString(value));
    return builder.CreateVector(offsets);
}

flatbuffers::Offset<GRIMConcept::ConceptBlock>
toFlatBuffer(flatbuffers::FlatBufferBuilder& builder, const ConceptBlock& block) {
    const auto id = builder.CreateString(block.id);
    const auto name = builder.CreateString(block.name);
    const auto prompt = builder.CreateString(block.prompt);
    const auto intermediates = createStringVector(builder, block.intermediates);
    const auto answer = builder.CreateString(block.answer);
    const auto explanation = createStringVector(builder, block.explanation);

    flatbuffers::Offset<GRIMConcept::ConceptBlockState0> state0;
    if (!block.state_0.type.empty() || !block.state_0.atoms.empty()) {
        state0 = GRIMConcept::CreateConceptBlockState0(
            builder,
            builder.CreateVector(block.state_0.atoms),
            builder.CreateString(block.state_0.type));
    }

    std::vector<flatbuffers::Offset<GRIMConcept::ConceptExecutionStep>> steps;
    steps.reserve(block.execution.size());
    for (const auto& step : block.execution) {
        steps.push_back(GRIMConcept::CreateConceptExecutionStep(
            builder,
            builder.CreateString(step.op),
            builder.CreateVector(step.args),
            builder.CreateVector(step.arg_slots),
            step.result));
    }
    const auto execution = builder.CreateVector(steps);

    flatbuffers::Offset<GRIMConcept::ConceptBlockState1> state1;
    if (block.state_1.has_result) {
        state1 = GRIMConcept::CreateConceptBlockState1(builder, block.state_1.result);
    }

    const auto step_index = builder.CreateVector(block.step_index);
    const auto format_type = builder.CreateString(block.format_type);
    const auto source_sequence_id = builder.CreateString(block.source_sequence_id);

    return GRIMConcept::CreateConceptBlock(
        builder,
        id,
        name,
        prompt,
        intermediates,
        answer,
        static_cast<GRIMConcept::ExecutionGateTarget>(block.execution_gate_target),
        explanation,
        state0,
        execution,
        state1,
        block.intermediate_count,
        step_index,
        format_type,
        source_sequence_id,
        block.timestamp);
}

size_t estimatedBufferSize(const std::vector<ConceptBlock>& blocks) {
    // A close initial allocation avoids repeated multi-hundred-MiB reallocations
    // for the production corpus. This intentionally overestimates table/vtable
    // metadata while remaining below FlatBuffers' 2 GiB default buffer limit.
    size_t estimate = 1024;
    const auto add = [&estimate](size_t amount) {
        if (amount > std::numeric_limits<size_t>::max() - estimate) {
            estimate = std::numeric_limits<size_t>::max();
        } else {
            estimate += amount;
        }
    };
    for (const auto& block : blocks) {
        add(256 + block.id.size() + block.name.size() + block.prompt.size()
            + block.answer.size() + block.format_type.size()
            + block.source_sequence_id.size());
        for (const auto& text : block.intermediates) add(8 + text.size());
        for (const auto& text : block.explanation) add(8 + text.size());
        add(block.state_0.atoms.size() * sizeof(double) + block.state_0.type.size());
        add(block.step_index.size() * sizeof(int32_t));
        for (const auto& step : block.execution) {
            add(96 + step.op.size() + step.args.size() * sizeof(double)
                + step.arg_slots.size() * sizeof(int32_t));
        }
    }
    constexpr size_t kMaxInitialSize = 1024ull * 1024ull * 1024ull;
    return std::max<size_t>(1024, std::min(estimate, kMaxInitialSize));
}

bool replaceFile(const std::filesystem::path& temporary,
                 const std::filesystem::path& destination,
                 std::string* error) {
#ifdef _WIN32
    if (::MoveFileExW(temporary.c_str(), destination.c_str(),
                      MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        return true;
    }
    if (error) {
        *error = "failed to replace destination (Win32 error "
            + std::to_string(::GetLastError()) + ")";
    }
    return false;
#else
    std::error_code ec;
    std::filesystem::rename(temporary, destination, ec);
    if (!ec) return true;
    if (error) *error = "failed to replace destination: " + ec.message();
    return false;
#endif
}

} // namespace

bool loadFlatBuffer(const std::filesystem::path& path,
                    std::vector<ConceptBlock>& blocks,
                    std::string* error) {
    blocks.clear();
    try {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input.is_open()) {
            if (error) *error = "cannot open " + path.string();
            return false;
        }
        const std::streamsize size = input.tellg();
        if (size <= 0) {
            if (error) *error = "file is empty";
            return false;
        }
        input.seekg(0, std::ios::beg);
        std::vector<uint8_t> buffer(static_cast<size_t>(size));
        if (!input.read(reinterpret_cast<char*>(buffer.data()), size)) {
            if (error) *error = "failed to read the complete file";
            return false;
        }

        flatbuffers::Verifier verifier(buffer.data(), buffer.size());
        if (!GRIMConcept::VerifyConceptBlockDatasetBuffer(verifier)) {
            if (error) *error = "FlatBuffer verification failed";
            return false;
        }

        const auto* dataset = GRIMConcept::GetConceptBlockDataset(buffer.data());
        if (!dataset || dataset->schema_version() > kSchemaVersion) {
            if (error) {
                *error = "unsupported schema version "
                    + std::to_string(dataset ? dataset->schema_version() : 0);
            }
            return false;
        }

        if (const auto* source_blocks = dataset->blocks()) {
            blocks.reserve(source_blocks->size());
            for (const auto* source : *source_blocks) {
                if (source) blocks.push_back(fromFlatBuffer(*source));
            }
        }
        return true;
    } catch (const std::exception& ex) {
        blocks.clear();
        if (error) *error = ex.what();
        return false;
    }
}

bool saveFlatBuffer(const std::filesystem::path& path,
                    const std::vector<ConceptBlock>& blocks,
                    std::string* error) {
    try {
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
        if (ec) {
            if (error) *error = "cannot create dataset directory: " + ec.message();
            return false;
        }

        flatbuffers::FlatBufferBuilder builder(estimatedBufferSize(blocks));
        std::vector<flatbuffers::Offset<GRIMConcept::ConceptBlock>> offsets;
        offsets.reserve(blocks.size());
        for (const auto& block : blocks) offsets.push_back(toFlatBuffer(builder, block));

        const auto root = GRIMConcept::CreateConceptBlockDataset(
            builder, kSchemaVersion, builder.CreateVector(offsets));
        GRIMConcept::FinishConceptBlockDatasetBuffer(builder, root);

        std::filesystem::path temporary = path;
        temporary += ".tmp";
        {
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            if (!output.is_open()) {
                if (error) *error = "cannot open temporary file " + temporary.string();
                return false;
            }
            output.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                         static_cast<std::streamsize>(builder.GetSize()));
            output.flush();
            if (!output.good()) {
                if (error) *error = "failed while writing temporary dataset";
                return false;
            }
        }
        return replaceFile(temporary, path, error);
    } catch (const std::exception& ex) {
        if (error) *error = ex.what();
        return false;
    }
}

} // namespace GRIM::ConceptBlockIO

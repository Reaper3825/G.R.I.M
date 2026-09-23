#include "ConceptBlockCorpusReader.hpp"

#include "CurriculumRegistry.hpp"
#include "../../../../../DataCollection/concept_block_generated.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace GRIM {

using json = nlohmann::json;

std::string fbString(const flatbuffers::String* value) {
	return value ? value->str() : std::string{};
}

json conceptBlockFlatBufferToJson(const GRIMConcept::ConceptBlock& source) {
	json j;
	j["id"] = fbString(source.id());
	j["name"] = fbString(source.name());
	j["prompt"] = fbString(source.prompt());
	j["determine"] = fbString(source.determine());
	j["define"] = fbString(source.define());
	j["execute"] = fbString(source.execute());
	j["update"] = fbString(source.update());
	j["answer"] = fbString(source.answer());
	j["raw"] = fbString(source.raw());
	j["format_type"] = fbString(source.format_type());
	j["source_sequence_id"] = fbString(source.source_sequence_id());
	j["timestamp"] = source.timestamp();
	j["intermediate_count"] = source.intermediate_count();
	j["knowns"] = json::array();
	if (const auto* values = source.knowns()) {
		for (const auto* value : *values) j["knowns"].push_back(fbString(value));
	}
	j["unknowns"] = json::array();
	if (const auto* values = source.unknowns()) {
		for (const auto* value : *values) j["unknowns"].push_back(fbString(value));
	}
	if (const auto* goal = source.goal()) {
		json goal_json{{"target_state", fbString(goal->target_state())}};
		goal_json["success_criteria"] = json::array();
		if (const auto* criteria = goal->success_criteria()) {
			for (const auto* entry : *criteria) {
				if (!entry) continue;
				goal_json["success_criteria"].push_back(json{
					{"criterion", fbString(entry->criterion())},
					{"evidence", fbString(entry->evidence())}
				});
			}
		}
		goal_json["constraints"] = json::array();
		if (const auto* constraints = goal->constraints()) {
			for (const auto* constraint : *constraints) {
				goal_json["constraints"].push_back(fbString(constraint));
			}
		}
		j["goal"] = std::move(goal_json);
	}

	j["intermediates"] = json::array();
	if (const auto* values = source.intermediates()) {
		for (const auto* value : *values) j["intermediates"].push_back(fbString(value));
	}
	j["explanation"] = json::array();
	if (const auto* values = source.explanation()) {
		for (const auto* value : *values) j["explanation"].push_back(fbString(value));
	}
	// Normalize the legacy source alias here; span traversal never branches on names.
	if (j["explanation"].empty()) j["explanation"] = j["intermediates"];
	j["step_index"] = json::array();
	if (const auto* values = source.step_index()) {
		for (const auto value : *values) j["step_index"].push_back(value);
	}

	if (const auto* steps = source.execution()) {
		j["execution"] = json::array();
		for (const auto* source_step : *steps) {
			if (!source_step) continue;
			json step;
			step["op"] = fbString(source_step->op());
			step["result"] = source_step->result();
			step["args"] = json::array();
			if (const auto* args = source_step->args()) {
				for (const auto value : *args) step["args"].push_back(value);
			}
			step["arg_slots"] = json::array();
			if (const auto* slots = source_step->arg_slots()) {
				for (const auto value : *slots) step["arg_slots"].push_back(value);
			}
			j["execution"].push_back(std::move(step));
		}
	}

	return j;
}

bool selectedByCurriculum(const std::string& id, const CurriculumMetadata& metadata) {
	return metadata.concept_block_ids.find(id) != metadata.concept_block_ids.end();
}

void loadConceptBlocks(const fs::path& cache_dir,
                       std::vector<json>& out,
                       CurriculumMetadata& out_metadata,
                       const std::string& curriculum_name) {
	const fs::path flatbuffer_path = cache_dir / "concept_blocks.fb";
	const fs::path legacy_path = cache_dir / "concept_blocks.jsonl";
	out_metadata = LoadCurriculumMetadataFromRegistry(cache_dir, curriculum_name);

	if (fs::exists(flatbuffer_path)) {
		if (fs::exists(legacy_path)) {
			std::error_code fb_time_error;
			std::error_code legacy_time_error;
			const auto fb_time = fs::last_write_time(flatbuffer_path, fb_time_error);
			const auto legacy_time = fs::last_write_time(legacy_path, legacy_time_error);
			if (!fb_time_error && !legacy_time_error && legacy_time > fb_time) {
				throw std::runtime_error(
					"[DataLoader] FATAL: legacy concept_blocks.jsonl is newer than "
					"concept_blocks.fb. Open the dataset in DataHub to refresh the "
					"FlatBuffer before training.");
			}
		}
		std::ifstream input(flatbuffer_path, std::ios::binary | std::ios::ate);
		if (!input.is_open()) {
			throw std::runtime_error(
				"[DataLoader] FATAL: cannot open " + flatbuffer_path.string());
		}
		const std::streamsize size = input.tellg();
		if (size <= 0) {
			throw std::runtime_error(
				"[DataLoader] FATAL: concept-block FlatBuffer is empty: " +
				flatbuffer_path.string());
		}
		input.seekg(0, std::ios::beg);
		std::vector<uint8_t> buffer(static_cast<size_t>(size));
		if (!input.read(reinterpret_cast<char*>(buffer.data()), size)) {
			throw std::runtime_error(
				"[DataLoader] FATAL: failed to read " + flatbuffer_path.string());
		}

		flatbuffers::Verifier::Options verifier_options;
		// Large curricula can contain more than FlatBuffers' default limit of
		// one million nested tables even when the file is valid. Keep structural
		// verification enabled, but give concept-block datasets a bounded budget
		// appropriate for the corpus size.
		verifier_options.max_tables = 2'000'000;
		flatbuffers::Verifier verifier(buffer.data(), buffer.size(), verifier_options);
		if (!GRIMConcept::VerifyConceptBlockDatasetBuffer(verifier)) {
			throw std::runtime_error(
				"[DataLoader] FATAL: FlatBuffer verification failed for " +
				flatbuffer_path.string());
		}
		const auto* dataset = GRIMConcept::GetConceptBlockDataset(buffer.data());
		constexpr uint32_t supported_schema_version = 2;
		if (!dataset || dataset->schema_version() > supported_schema_version) {
			throw std::runtime_error(
				"[DataLoader] FATAL: unsupported concept-block schema version " +
				std::to_string(dataset ? dataset->schema_version() : 0));
		}

		size_t total = 0;
		size_t accepted = 0;
		if (const auto* blocks = dataset->blocks()) {
			out.reserve(blocks->size());
			for (const auto* block : *blocks) {
				if (!block) continue;
				++total;
				const std::string id = fbString(block->id());
				if (!selectedByCurriculum(id, out_metadata)) continue;
				out.push_back(conceptBlockFlatBufferToJson(*block));
				++accepted;
			}
		}
		std::cout << "[DataLoader] Loaded " << accepted;
		if (!out_metadata.concept_block_ids.empty()) std::cout << "/" << total;
		std::cout << " concept blocks from " << flatbuffer_path.string() << std::endl;
		return;
	}

	// Rollout fallback only. Once the DataHub migration has produced the FB,
	// a corrupt FB never silently falls back to a stale JSONL dataset.
	std::ifstream input(legacy_path);
	if (!input.is_open()) {
		std::cout << "[DataLoader] No concept_blocks.fb or legacy JSONL at "
		          << cache_dir.string() << "\n";
		return;
	}
	std::cout << "[DataLoader] WARNING: using legacy concept_blocks.jsonl; "
	          << "open the dataset in DataHub to migrate it.\n";

	std::string line;
	size_t total = 0;
	size_t accepted = 0;
	while (std::getline(input, line)) {
		if (line.empty()) continue;
		try {
			auto j = json::parse(line);
			++total;
			if (!selectedByCurriculum(j.value("id", std::string()), out_metadata)) continue;
			out.push_back(std::move(j));
			++accepted;
		} catch (const std::exception& e) {
			std::cerr << "[DataLoader] concept_blocks.jsonl skip line: " << e.what() << "\n";
		}
	}
	std::cout << "[DataLoader] Loaded " << accepted;
	if (!out_metadata.concept_block_ids.empty()) std::cout << "/" << total;
	std::cout << " legacy concept blocks from " << legacy_path.string() << std::endl;
}

} // namespace GRIM

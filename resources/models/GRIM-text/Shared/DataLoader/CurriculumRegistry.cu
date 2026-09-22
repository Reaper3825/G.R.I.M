#include "CurriculumRegistry.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

namespace GRIM {
namespace {

using json = nlohmann::json;

// Course membership is authoritative. The curriculum-level compatibility list
// is deliberately ignored, so stale flattened data cannot select extra blocks.
void addCourseMembership(const json& registry, const json& source, CurriculumMetadata& metadata) {
    if (!source.contains("course_ids") || !source["course_ids"].is_array()) {
        throw std::runtime_error("[DataLoader] FATAL: curriculum '" + metadata.name +
                                 "' is missing the course_ids array; migrate the registry first");
    }
    if (!registry.contains("courses") || !registry["courses"].is_array()) {
        throw std::runtime_error("[DataLoader] FATAL: curriculum registry is missing the courses array");
    }
    std::unordered_map<std::string, const json*> courses;
    for (const auto& course : registry["courses"]) {
        if (!course.is_object() || !course.contains("id") || !course["id"].is_string() ||
            course["id"].get_ref<const std::string&>().empty()) {
            throw std::runtime_error("[DataLoader] FATAL: registry contains an invalid course ID");
        }
        const auto& id = course["id"].get_ref<const std::string&>();
        if (!courses.emplace(id, &course).second)
            throw std::runtime_error("[DataLoader] FATAL: duplicate course ID '" + id + "'");
    }
    std::unordered_set<std::string> assigned;
    for (const auto& id : source["course_ids"]) {
        if (!id.is_string() || id.get_ref<const std::string&>().empty())
            throw std::runtime_error("[DataLoader] FATAL: curriculum '" + metadata.name + "' contains an invalid course_id");
        const auto& course_id = id.get_ref<const std::string&>();
        if (!assigned.insert(course_id).second)
            throw std::runtime_error("[DataLoader] FATAL: duplicate course assignment '" + course_id + "'");
        const auto found = courses.find(course_id);
        if (found == courses.end())
            throw std::runtime_error("[DataLoader] FATAL: curriculum '" + metadata.name + "' references missing course '" + course_id + "'");
        const auto& course = *found->second;
        if (!course.contains("concept_block_ids") || !course["concept_block_ids"].is_array())
            throw std::runtime_error("[DataLoader] FATAL: course '" + course_id + "' is missing the concept_block_ids array");
        CourseMetadata course_metadata;
        course_metadata.id = course_id;
        for (const auto& block_id : course["concept_block_ids"]) {
            if (!block_id.is_string() || block_id.get_ref<const std::string&>().empty())
                throw std::runtime_error("[DataLoader] FATAL: course '" + course_id + "' contains an invalid concept_block_id");
            metadata.concept_block_ids.insert(block_id.get<std::string>());
            course_metadata.concept_block_ids.push_back(block_id.get<std::string>());
        }
        metadata.courses.push_back(std::move(course_metadata));
    }
}

void readCurriculumMetadata(const json& registry, const json& source,
	                        CurriculumMetadata& metadata,
	                        const std::string& expected_name) {
	metadata.id = source.value("id", std::string{});
	metadata.name = source.value("name", std::string{});
	metadata.training_stage = source.value("training_stage", std::string{});
	if (metadata.id.empty() || metadata.name.empty() || metadata.name != expected_name) {
		throw std::runtime_error(
			"[DataLoader] FATAL: curriculum registry entry has invalid identity metadata");
	}
	if (metadata.training_stage != "pt" && metadata.training_stage != "sft" &&
		metadata.training_stage != "dpo" && metadata.training_stage != "rlhf") {
		throw std::runtime_error(
			"[DataLoader] FATAL: curriculum '" + metadata.name +
			"' has invalid training_stage '" + metadata.training_stage + "'");
	}
	metadata.randomize_course_order = source.value("randomize_course_order", false);
	metadata.randomize_concept_block_order = source.value("randomize_concept_block_order", false);
	addCourseMembership(registry, source, metadata);
}

// curriculum_registry.json is the sole source of curriculum metadata.
CurriculumMetadata loadCurriculumMetadata(const fs::path& dir, const std::string& curriculum_name) {
	if (curriculum_name.empty()) {
		throw std::runtime_error(
			"[DataLoader] FATAL: curriculum name is required for registry lookup");
	}

	const fs::path registry_path = dir / "curriculum_registry.json";
	std::ifstream registry_input(registry_path);
	if (!registry_input.is_open()) {
		throw std::runtime_error(
			"[DataLoader] FATAL: cannot open curriculum registry: " + registry_path.string());
	}

	json registry;
	try {
		registry = json::parse(registry_input);
	} catch (const json::exception& error) {
		throw std::runtime_error(
			"[DataLoader] FATAL: failed to parse " + registry_path.string() +
			": " + error.what());
	}
	if (!registry.contains("curriculums") || !registry["curriculums"].is_array()) {
		throw std::runtime_error(
			"[DataLoader] FATAL: curriculum registry is missing the curriculums array");
	}

	for (const auto& source : registry["curriculums"]) {
		if (!source.is_object() || source.value("name", std::string{}) != curriculum_name) continue;
		CurriculumMetadata metadata;
		readCurriculumMetadata(registry, source, metadata, curriculum_name);
		if (metadata.concept_block_ids.empty()) {
			throw std::runtime_error(
				"[DataLoader] FATAL: curriculum '" + curriculum_name +
				"' has no concept blocks in its assigned courses");
		}
		std::cout << "[DataLoader] Curriculum '" << metadata.name
		          << "' loaded from registry: " << registry_path.string()
		          << ", curriculum_id=" << metadata.id
		          << ", training_stage=" << metadata.training_stage
		          << ", format_as_concept=" << (metadata.formatAsConcept() ? "true" : "false")
		          << ", courses=" << source["course_ids"].size()
		          << ", course concept_block_ids=" << metadata.concept_block_ids.size()
		          << std::endl;
		return metadata;
	}

	throw std::runtime_error(
		"[DataLoader] FATAL: curriculum '" + curriculum_name +
		"' was not found in " + registry_path.string());
}

} // namespace

CurriculumMetadata LoadCurriculumMetadataFromRegistry(
	const fs::path& directory,
	const std::string& curriculum_name)
{
	return loadCurriculumMetadata(directory, curriculum_name);
}

} // namespace GRIM

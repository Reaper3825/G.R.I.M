#include "../concept_block.hpp"
#include "../io/concept_block_io_flatbuffer.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr std::string_view kBlockPrefix = "cb_atom_ident_v1_";
constexpr std::string_view kSourceId = "atom_identification_v1";
constexpr std::string_view kCurriculumId = "curr_atom_identification_v1";
constexpr std::string_view kCurriculumName = "Atom Identification v1";
constexpr int kSinglePerType = 6000;
constexpr int kMixedPairPerCombination = 400;
constexpr int kMixedTriplePerCombination = 600;
constexpr int kMixedAllFour = 1200;
constexpr int kExpectedBlockCount = 30000;
constexpr int kExpectedOccurrencesPerType = 10200;

enum class ValueType : std::size_t {
    Int = 0,
    Float = 1,
    String = 2,
    Bool = 3,
};

constexpr std::array<ValueType, 4> kTypes{
    ValueType::Int,
    ValueType::Float,
    ValueType::String,
    ValueType::Bool,
};

constexpr std::array<std::array<ValueType, 2>, 6> kPairs{{
    {ValueType::Int, ValueType::Float},
    {ValueType::Int, ValueType::String},
    {ValueType::Int, ValueType::Bool},
    {ValueType::Float, ValueType::String},
    {ValueType::Float, ValueType::Bool},
    {ValueType::String, ValueType::Bool},
}};

constexpr std::array<std::array<ValueType, 3>, 4> kTriples{{
    {ValueType::Int, ValueType::Float, ValueType::String},
    {ValueType::Int, ValueType::Float, ValueType::Bool},
    {ValueType::Int, ValueType::String, ValueType::Bool},
    {ValueType::Float, ValueType::String, ValueType::Bool},
}};

const char* typeName(ValueType type) {
    switch (type) {
        case ValueType::Int: return "INT";
        case ValueType::Float: return "FLOAT";
        case ValueType::String: return "STRING";
        case ValueType::Bool: return "BOOL";
    }
    throw std::runtime_error("unknown ValueType");
}

const char* typeSlug(ValueType type) {
    switch (type) {
        case ValueType::Int: return "int";
        case ValueType::Float: return "float";
        case ValueType::String: return "string";
        case ValueType::Bool: return "bool";
    }
    throw std::runtime_error("unknown ValueType");
}

std::string zeroPadded(int value, int width = 6) {
    std::ostringstream out;
    out << std::setw(width) << std::setfill('0') << value;
    return out.str();
}

std::string base36(std::uint64_t value) {
    constexpr char digits[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    std::string result;
    do {
        result.push_back(digits[value % 36]);
        value /= 36;
    } while (value != 0);
    std::reverse(result.begin(), result.end());
    return result;
}

template <std::size_t N>
const std::string& choose(const std::array<std::string, N>& values,
                          std::size_t index) {
    return values[index % N];
}

const std::array<std::string, 16> kSubjects{
    "deployment", "calibration job", "archive request", "message route",
    "worker pool", "search task", "backup plan", "render pass",
    "import operation", "service profile", "analysis run", "cache policy",
    "notification rule", "batch request", "simulation", "release candidate",
};

const std::array<std::string, 16> kActions{
    "starting the run", "sending the request", "opening the queue",
    "committing the profile", "launching the workers", "saving the record",
    "rebuilding the cache", "publishing the result", "validating the input",
    "handing off the task", "scheduling execution", "closing the session",
    "applying the policy", "creating the snapshot", "running verification",
    "finalizing the configuration",
};

const std::array<std::string, 16> kStages{
    "intake", "validation", "scheduling", "handoff",
    "rollout", "archival", "verification", "publication",
    "calibration", "import", "rendering", "backup",
    "analysis", "review", "cache refresh", "completion",
};

const std::array<std::string, 16> kIntFields{
    "retry_limit", "worker_count", "batch_size", "page_limit",
    "shard_count", "sample_count", "queue_depth", "max_attempts",
    "tile_count", "port_number", "epoch_count", "buffer_capacity",
    "copy_count", "priority_level", "window_size", "record_limit",
};

const std::array<std::string, 16> kFloatFields{
    "threshold", "temperature", "learning_rate", "opacity",
    "mix_ratio", "confidence_floor", "scale_factor", "decay_rate",
    "sampling_weight", "timeout_seconds", "gain", "tolerance",
    "blend_amount", "compression_ratio", "noise_level", "score_cutoff",
};

const std::array<std::string, 16> kStringFields{
    "queue_name", "profile_name", "region", "output_label",
    "cache_key", "worker_alias", "project_code", "route_name",
    "dataset_name", "release_channel", "storage_class", "theme_name",
    "owner_tag", "namespace", "policy_name", "destination",
};

const std::array<std::string, 16> kBoolFields{
    "enabled", "dry_run", "allow_cache", "strict_mode",
    "send_alerts", "use_backup", "preserve_order", "overwrite_existing",
    "include_metadata", "verify_checksum", "compress_output", "keep_history",
    "auto_retry", "require_review", "stream_results", "lock_record",
};

const std::string& fieldFor(ValueType type, std::size_t index) {
    switch (type) {
        case ValueType::Int: return choose(kIntFields, index);
        case ValueType::Float: return choose(kFloatFields, index);
        case ValueType::String: return choose(kStringFields, index);
        case ValueType::Bool: return choose(kBoolFields, index);
    }
    throw std::runtime_error("unknown ValueType");
}

std::string integerValue(std::size_t index) {
    const std::uint64_t magnitude = ((index + 1) * 7919ULL) % 900000000ULL + 1ULL;
    if (index % 41 == 0) return "0";
    if (index % 3 == 0) return "-" + std::to_string(magnitude);
    if (index % 11 == 0) return "+" + std::to_string(magnitude);
    return std::to_string(magnitude);
}

std::string floatValue(std::size_t index) {
    const std::uint64_t mixed = ((index + 1) * 104729ULL) % 900000000ULL + 1001ULL;
    const bool negative = index % 3 == 0;
    const bool explicit_positive = !negative && index % 17 == 0;
    const std::string sign = negative ? "-" : (explicit_positive ? "+" : "");
    const std::uint64_t whole = (mixed / 1000ULL) % 100000ULL;
    const std::uint64_t fraction = mixed % 1000ULL;
    std::ostringstream out;
    switch (index % 6) {
        case 0:
        case 1:
            out << sign << whole << '.' << std::setw(3) << std::setfill('0')
                << fraction;
            break;
        case 2:
            out << sign << "0." << std::setw(6) << std::setfill('0')
                << (mixed % 1000000ULL);
            break;
        case 3:
            out << sign << ((mixed % 900ULL) + 100ULL) / 100.0
                << 'e' << (static_cast<int>(index % 19) - 9);
            break;
        case 4:
            out << sign << ((mixed % 9000ULL) + 1000ULL) / 1000.0
                << 'E' << (static_cast<int>(index % 23) - 11);
            break;
        default:
            out << sign << '.' << std::setw(5) << std::setfill('0')
                << ((mixed % 99999ULL) + 1ULL);
            break;
    }
    return out.str();
}

std::string stringValue(std::size_t index) {
    static const std::array<std::string, 20> prefixes{
        "north", "amber", "quiet", "lunar", "rapid", "cedar", "velvet",
        "silver", "orbit", "harbor", "billing", "research", "München",
        "Québec", "東京", "Δelta", "résumé", "naïve", "crimson", "winter",
    };
    static const std::array<std::string, 20> suffixes{
        "star", "priority", "queue", "review", "cache", "worker", "route",
        "archive", "channel", "profile", "snapshot", "garden", "bridge",
        "signal", "catalog", "matrix", "studio", "field", "harvest", "lane",
    };
    static const std::array<std::string, 6> separators{
        "-", "_", "/", ".", " ", "::",
    };
    return prefixes[index % prefixes.size()] +
        separators[(index / prefixes.size()) % separators.size()] +
        suffixes[(index / (prefixes.size() * separators.size())) % suffixes.size()] +
        "-" + base36(index + 1000);
}

std::string rawValue(ValueType type, std::size_t index) {
    switch (type) {
        case ValueType::Int: return integerValue(index);
        case ValueType::Float: return floatValue(index);
        case ValueType::String: return stringValue(index);
        case ValueType::Bool: return index % 2 == 0 ? "true" : "false";
    }
    throw std::runtime_error("unknown ValueType");
}

std::string taggedValue(ValueType type, std::size_t index) {
    const std::string tag = typeName(type);
    return "<" + tag + ">" + rawValue(type, index) + "</" + tag + ">";
}

std::string makeSingleRaw(ValueType type, std::size_t index) {
    std::size_t quotient = index;
    const std::size_t pattern = quotient % 20;
    quotient /= 20;
    const std::string& field = fieldFor(type, quotient % 16);
    quotient /= 16;
    const std::string& subject = choose(kSubjects, quotient % 16);
    quotient /= 16;
    const std::string& stage = choose(kStages, quotient % 16);
    const std::string& action = choose(kActions, index * 13 + pattern);
    const std::string atom = taggedValue(type, index);
    const auto finalize = [&](std::string text) {
        return text + " This setting is scoped to the " + subject + " during " +
            stage + ".";
    };

    switch (pattern) {
        case 0: return finalize("Set " + field + " to " + atom + " before " + action + ".");
        case 1: return finalize("Before " + action + ", assign " + atom + " to " + field + ".");
        case 2: return finalize("The " + subject + " request uses " + field + "=" + atom + ".");
        case 3: return finalize("For the " + subject + ", the " + field + " value is " + atom + ".");
        case 4: return finalize("Please keep " + field + " at " + atom + " while " + action + ".");
        case 5: return finalize("Confirm that " + field + " should be " + atom + " for the " + subject + ".");
        case 6: return finalize("Update the " + subject + " so " + field + " becomes " + atom + ".");
        case 7: return finalize("Use " + atom + " as " + field + " when " + action + ".");
        case 8: return finalize(field + ": " + atom + ". Apply this setting to the " + subject + ".");
        case 9: return finalize("When preparing the " + subject + ", choose " + atom + " for " + field + ".");
        case 10: return finalize("The value " + atom + " belongs in " + field + " for the " + subject + ".");
        case 11: return finalize("Should the " + subject + " use " + atom + " as its " + field + " value?");
        case 12: return finalize("Record " + field + " as " + atom + ", then continue the workflow.");
        case 13: return finalize("On the " + subject + ", replace the current " + field + " with " + atom + ".");
        case 14: return finalize("The requested " + field + " for the " + subject + " is " + atom + ".");
        case 15: return finalize("For the next " + subject + " operation, set " + field + "=" + atom + ".");
        case 16: return finalize("Assign " + field + " the value " + atom + " in the " + subject + " configuration.");
        case 17: return finalize("Keep the " + subject + " unchanged except for " + field + ", which must be " + atom + ".");
        case 18: return finalize("While " + action + ", read " + field + " as " + atom + ".");
        default: return finalize("Apply this " + subject + " option: " + field + "=" + atom + ".");
    }
}

struct MixedField {
    ValueType type;
    std::string field;
    std::string atom;
};

std::string join(const std::vector<std::string>& parts,
                 std::string_view separator,
                 std::string_view final_separator = {}) {
    if (parts.empty()) return {};
    if (parts.size() == 1) return parts.front();
    std::ostringstream out;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) {
            std::string_view selected =
                i + 1 == parts.size() && !final_separator.empty()
                    ? final_separator : separator;
            if (parts.size() == 2 && selected == ", and ") selected = " and ";
            if (parts.size() == 2 && selected == "; and ") selected = " and ";
            if (parts.size() == 2 && selected == ", while setting ") {
                selected = " while setting ";
            }
            out << selected;
        }
        out << parts[i];
    }
    return out.str();
}

std::string makeMixedRaw(std::vector<ValueType> types, std::size_t index) {
    const std::size_t rotation = index % types.size();
    std::rotate(types.begin(), types.begin() + rotation, types.end());
    if (index % 2 != 0) std::reverse(types.begin(), types.end());

    std::vector<MixedField> fields;
    fields.reserve(types.size());
    for (std::size_t position = 0; position < types.size(); ++position) {
        const std::size_t value_index = 100000 + index * 7 + position * 100003;
        fields.push_back(MixedField{
            types[position],
            fieldFor(types[position], index + position * 5),
            taggedValue(types[position], value_index),
        });
    }

    std::vector<std::string> equals;
    std::vector<std::string> natural;
    std::vector<std::string> reversed;
    std::vector<std::string> colon;
    for (const auto& field : fields) {
        equals.push_back(field.field + "=" + field.atom);
        natural.push_back(field.field + " to " + field.atom);
        reversed.push_back(field.atom + " for " + field.field);
        colon.push_back(field.field + ": " + field.atom);
    }

    const std::string& subject = choose(kSubjects, index * 5 + types.size());
    const std::string& action = choose(kActions, index * 11 + types.size());
    switch (index % 12) {
        case 0:
            return "Configure the " + subject + " with " +
                join(equals, ", ", ", and ") + " before " + action + ".";
        case 1:
            return "For the " + subject + ", set " +
                join(natural, "; ", "; and ") + ", then continue the workflow.";
        case 2:
            return "The " + subject + " request carries " +
                join(natural, ", ", ", and ") + ".";
        case 3:
            return "While " + action + ", use these values: " +
                join(equals, " | ") + ".";
        case 4:
            return "Should the " + subject + " use " +
                join(equals, ", ", ", and ") + "?";
        case 5:
            return "Update the " + subject + ": " +
                join(natural, ", ", ", while setting ") + ".";
        case 6:
            return "Before " + action + ", apply " +
                join(colon, "; ", "; and ") + " to the " + subject + ".";
        case 7:
            return "Use " + join(reversed, ", ", ", and ") +
                " when handling the " + subject + ".";
        case 8:
            return "The configuration for the " + subject + " is {" +
                join(equals, ", ") + "}.";
        case 9:
            return "Assign the following " + subject + " values—" +
                join(equals, "; ") + "—before " + action + ".";
        case 10:
            return "On the " + subject + ", keep " +
                join(natural, ", ", ", and ") + ".";
        default:
            return "Prepare the " + subject + " with " +
                join(equals, "; ", "; and ") + "; afterward, continue the workflow.";
    }
}

GRIM::ConceptBlock makeBlock(std::string id,
                             std::string name,
                             std::string raw) {
    GRIM::ConceptBlock block;
    block.id = std::move(id);
    block.name = std::move(name);
    block.raw = std::move(raw);
    block.format_type = "raw";
    block.source_sequence_id = std::string(kSourceId);
    block.timestamp = 0;
    block.recomputeDerived();
    return block;
}

struct GeneratedCurriculum {
    std::vector<GRIM::ConceptBlock> blocks;
    std::array<int, 4> occurrences{};
    std::array<int, 4> singles{};
    int mixed_pairs = 0;
    int mixed_triples = 0;
    int mixed_all_four = 0;
};

void addOccurrences(GeneratedCurriculum& generated,
                    const std::vector<ValueType>& types) {
    for (const ValueType type : types) {
        ++generated.occurrences[static_cast<std::size_t>(type)];
    }
}

GeneratedCurriculum generateCurriculum() {
    GeneratedCurriculum generated;
    generated.blocks.reserve(kExpectedBlockCount);

    for (const ValueType type : kTypes) {
        for (int index = 0; index < kSinglePerType; ++index) {
            generated.blocks.push_back(makeBlock(
                std::string(kBlockPrefix) + typeSlug(type) + "_" + zeroPadded(index + 1),
                std::string("Atom identification single ") + typeName(type) + " " +
                    zeroPadded(index + 1),
                makeSingleRaw(type, static_cast<std::size_t>(index))));
            ++generated.singles[static_cast<std::size_t>(type)];
            ++generated.occurrences[static_cast<std::size_t>(type)];
        }
    }

    int mixed_id = 0;
    std::size_t mixed_index = 0;
    for (const auto& combination : kPairs) {
        const std::vector<ValueType> types(combination.begin(), combination.end());
        for (int index = 0; index < kMixedPairPerCombination; ++index) {
            ++mixed_id;
            generated.blocks.push_back(makeBlock(
                std::string(kBlockPrefix) + "mixed_" + zeroPadded(mixed_id),
                "Atom identification mixed pair " + zeroPadded(mixed_id),
                makeMixedRaw(types, mixed_index++)));
            addOccurrences(generated, types);
            ++generated.mixed_pairs;
        }
    }
    for (const auto& combination : kTriples) {
        const std::vector<ValueType> types(combination.begin(), combination.end());
        for (int index = 0; index < kMixedTriplePerCombination; ++index) {
            ++mixed_id;
            generated.blocks.push_back(makeBlock(
                std::string(kBlockPrefix) + "mixed_" + zeroPadded(mixed_id),
                "Atom identification mixed triple " + zeroPadded(mixed_id),
                makeMixedRaw(types, mixed_index++)));
            addOccurrences(generated, types);
            ++generated.mixed_triples;
        }
    }
    const std::vector<ValueType> all_types(kTypes.begin(), kTypes.end());
    for (int index = 0; index < kMixedAllFour; ++index) {
        ++mixed_id;
        generated.blocks.push_back(makeBlock(
            std::string(kBlockPrefix) + "mixed_" + zeroPadded(mixed_id),
            "Atom identification mixed all types " + zeroPadded(mixed_id),
            makeMixedRaw(all_types, mixed_index++)));
        addOccurrences(generated, all_types);
        ++generated.mixed_all_four;
    }
    return generated;
}

bool startsWith(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

void validateInteger(std::string_view content, const std::string& id) {
    if (content.empty()) throw std::runtime_error(id + ": empty INT");
    std::size_t begin = content.front() == '+' ? 1 : 0;
    if (begin == content.size()) throw std::runtime_error(id + ": invalid INT sign");
    std::int64_t value = 0;
    const auto parsed = std::from_chars(
        content.data() + begin, content.data() + content.size(), value, 10);
    if (parsed.ec != std::errc{} || parsed.ptr != content.data() + content.size()) {
        throw std::runtime_error(id + ": invalid INT content " + std::string(content));
    }
}

void validateFloat(std::string_view content, const std::string& id) {
    static const std::regex grammar(
        R"(^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$)");
    const std::string text(content);
    if (!std::regex_match(text, grammar) ||
        text.find('.') == std::string::npos &&
            text.find('e') == std::string::npos && text.find('E') == std::string::npos) {
        throw std::runtime_error(id + ": invalid or ambiguous FLOAT content " + text);
    }
    char* end = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    if (!end || end != text.c_str() + text.size() || !std::isfinite(value)) {
        throw std::runtime_error(id + ": non-finite FLOAT content " + text);
    }
}

std::array<int, 4> validateAnnotatedRaw(const GRIM::ConceptBlock& block) {
    std::array<int, 4> counts{};
    std::size_t cursor = 0;
    while (true) {
        const std::size_t open_begin = block.raw.find('<', cursor);
        if (open_begin == std::string::npos) break;

        bool matched = false;
        for (const ValueType type : kTypes) {
            const std::string tag = typeName(type);
            const std::string open = "<" + tag + ">";
            const std::string close = "</" + tag + ">";
            if (block.raw.compare(open_begin, open.size(), open) != 0) continue;
            const std::size_t content_begin = open_begin + open.size();
            const std::size_t close_begin = block.raw.find(close, content_begin);
            if (close_begin == std::string::npos) {
                throw std::runtime_error(block.id + ": missing " + close);
            }
            if (block.raw.find('<', content_begin) < close_begin) {
                throw std::runtime_error(block.id + ": nested or mismatched atom delimiter");
            }
            const std::string_view content(
                block.raw.data() + content_begin, close_begin - content_begin);
            switch (type) {
                case ValueType::Int: validateInteger(content, block.id); break;
                case ValueType::Float: validateFloat(content, block.id); break;
                case ValueType::String:
                    if (content.empty()) throw std::runtime_error(block.id + ": empty STRING");
                    break;
                case ValueType::Bool:
                    if (content != "true" && content != "false") {
                        throw std::runtime_error(block.id + ": invalid BOOL content");
                    }
                    break;
            }
            ++counts[static_cast<std::size_t>(type)];
            cursor = close_begin + close.size();
            matched = true;
            break;
        }
        if (!matched) {
            throw std::runtime_error(block.id + ": unexpected angle-bracket text");
        }
    }
    if (block.raw.find('>', cursor) != std::string::npos) {
        throw std::runtime_error(block.id + ": unmatched closing angle bracket");
    }
    return counts;
}

void auditGenerated(const GeneratedCurriculum& generated) {
    if (generated.blocks.size() != kExpectedBlockCount) {
        throw std::runtime_error("generated block count is not 30000");
    }
    if (generated.mixed_pairs != 2400 || generated.mixed_triples != 2400 ||
        generated.mixed_all_four != 1200) {
        throw std::runtime_error("mixed distribution is incorrect");
    }

    std::unordered_set<std::string> ids;
    std::unordered_set<std::string> raws;
    ids.reserve(generated.blocks.size() * 2);
    raws.reserve(generated.blocks.size() * 2);
    std::array<int, 4> audited_occurrences{};
    for (const auto& block : generated.blocks) {
        if (!ids.insert(block.id).second) {
            throw std::runtime_error("duplicate block id " + block.id);
        }
        if (!raws.insert(block.raw).second) {
            throw std::runtime_error("duplicate authored text in " + block.id);
        }
        if (block.raw.size() < 40 || block.raw.size() > 512) {
            throw std::runtime_error(block.id + ": authored text length outside [40,512]");
        }
        const auto counts = validateAnnotatedRaw(block);
        for (std::size_t type = 0; type < counts.size(); ++type) {
            audited_occurrences[type] += counts[type];
        }
    }

    for (std::size_t type = 0; type < kTypes.size(); ++type) {
        if (generated.singles[type] != kSinglePerType ||
            generated.occurrences[type] != kExpectedOccurrencesPerType ||
            audited_occurrences[type] != kExpectedOccurrencesPerType) {
            throw std::runtime_error(
                std::string("type balance failed for ") + typeName(kTypes[type]));
        }
    }
}

json blockJson(const GRIM::ConceptBlock& block) {
    return json{
        {"id", block.id},
        {"name", block.name},
        {"prompt", block.prompt},
        {"intermediates", block.intermediates},
        {"explanation", block.explanation},
        {"answer", block.answer},
        {"raw", block.raw},
        {"execution", json::array()},
        {"intermediate_count", block.intermediate_count},
        {"step_index", block.step_index},
        {"format_type", block.format_type},
        {"source_sequence_id", block.source_sequence_id},
        {"timestamp", block.timestamp},
    };
}

void replaceFile(const fs::path& temporary, const fs::path& destination) {
#ifdef _WIN32
    if (!::MoveFileExW(temporary.c_str(), destination.c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        throw std::runtime_error(
            "failed to replace " + destination.string() + " (Win32 error " +
            std::to_string(::GetLastError()) + ")");
    }
#else
    std::error_code error;
    fs::rename(temporary, destination, error);
    if (error) throw std::runtime_error("failed to replace " + destination.string());
#endif
}

void writeJsonAtomic(const fs::path& path, const json& document) {
    fs::path temporary = path;
    temporary += ".atom_curriculum.tmp";
    {
        std::ofstream output(temporary, std::ios::trunc);
        if (!output.is_open()) throw std::runtime_error("cannot write " + temporary.string());
        output << document.dump(2) << '\n';
        if (!output.good()) throw std::runtime_error("failed while writing " + temporary.string());
    }
    replaceFile(temporary, path);
}

void writeJsonl(const fs::path& path,
                const std::vector<GRIM::ConceptBlock>& blocks) {
    fs::path temporary = path;
    temporary += ".tmp";
    {
        std::ofstream output(temporary, std::ios::trunc);
        if (!output.is_open()) throw std::runtime_error("cannot write " + temporary.string());
        for (const auto& block : blocks) output << blockJson(block).dump() << '\n';
        if (!output.good()) throw std::runtime_error("failed while writing " + temporary.string());
    }
    replaceFile(temporary, path);
}

void updateRegistry(const fs::path& path,
                    const std::vector<GRIM::ConceptBlock>& blocks) {
    std::ifstream input(path);
    if (!input.is_open()) throw std::runtime_error("cannot open " + path.string());
    json registry = json::parse(input);
    input.close();
    if (!registry.contains("curriculums") || !registry["curriculums"].is_array()) {
        throw std::runtime_error("curriculum registry lacks curriculums array");
    }

    json ids = json::array();
    for (const auto& block : blocks) ids.push_back(block.id);
    json curriculum{
        {"id", kCurriculumId},
        {"name", kCurriculumName},
        {"training_stage", "pt"},
        {"timestamp", 0},
        {"format_as_concept", false},
        {"concept_block_ids", std::move(ids)},
    };

    bool updated = false;
    for (auto& existing : registry["curriculums"]) {
        if (existing.is_object() && existing.value("id", std::string{}) == kCurriculumId) {
            existing = curriculum;
            updated = true;
            break;
        }
    }
    if (!updated) registry["curriculums"].push_back(std::move(curriculum));
    writeJsonAtomic(path, registry);
}

json makeManifest(const GeneratedCurriculum& generated) {
    json occurrence_counts;
    json single_counts;
    for (std::size_t index = 0; index < kTypes.size(); ++index) {
        occurrence_counts[typeName(kTypes[index])] = generated.occurrences[index];
        single_counts[typeName(kTypes[index])] = generated.singles[index];
    }
    return json{
        {"curriculum_id", kCurriculumId},
        {"curriculum_name", kCurriculumName},
        {"generation_version", 1},
        {"deterministic", true},
        {"total_blocks", generated.blocks.size()},
        {"single_blocks", 24000},
        {"single_blocks_by_type", std::move(single_counts)},
        {"mixed_blocks", 6000},
        {"mixed_pair_blocks", generated.mixed_pairs},
        {"mixed_triple_blocks", generated.mixed_triples},
        {"mixed_all_four_blocks", generated.mixed_all_four},
        {"atom_occurrences_by_type", std::move(occurrence_counts)},
        {"format_type", "raw"},
        {"training_stage", "pt"},
    };
}

fs::path parseDataDirectory(int argc, char** argv) {
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) == "--data-dir" && index + 1 < argc) {
            return fs::absolute(argv[index + 1]).lexically_normal();
        }
    }
    throw std::runtime_error("usage: atom_curriculum_builder --data-dir <training-data-directory>");
}

} // namespace

int main(int argc, char** argv) {
    try {
        const fs::path data_directory = parseDataDirectory(argc, argv);
        const fs::path flatbuffer_path = data_directory / "concept_blocks.fb";
        const fs::path registry_path = data_directory / "curriculum_registry.json";
        if (!fs::is_regular_file(flatbuffer_path) || !fs::is_regular_file(registry_path)) {
            throw std::runtime_error("data directory lacks concept_blocks.fb or curriculum_registry.json");
        }

        GeneratedCurriculum generated = generateCurriculum();
        auditGenerated(generated);
        std::cout << "Generated and audited " << generated.blocks.size()
                  << " atom-identification blocks\n";

        std::vector<GRIM::ConceptBlock> existing;
        std::string io_error;
        if (!GRIM::ConceptBlockIO::loadFlatBuffer(flatbuffer_path, existing, &io_error)) {
            throw std::runtime_error("failed to load concept blocks: " + io_error);
        }
        const std::size_t before = existing.size();
        existing.erase(
            std::remove_if(existing.begin(), existing.end(), [](const auto& block) {
                return startsWith(block.id, kBlockPrefix) || block.source_sequence_id == kSourceId;
            }),
            existing.end());
        const std::size_t replaced = before - existing.size();
        existing.insert(existing.end(), generated.blocks.begin(), generated.blocks.end());

        if (!GRIM::ConceptBlockIO::saveFlatBuffer(flatbuffer_path, existing, &io_error)) {
            throw std::runtime_error("failed to save concept blocks: " + io_error);
        }
        updateRegistry(registry_path, generated.blocks);
        writeJsonl(data_directory / "atom_identification_v1.jsonl", generated.blocks);
        writeJsonAtomic(
            data_directory / "atom_identification_v1_manifest.json",
            makeManifest(generated));

        std::cout << "Replaced prior curriculum blocks: " << replaced << '\n'
                  << "Concept-block total: " << existing.size() << '\n'
                  << "Curriculum: " << kCurriculumName << " (" << kCurriculumId << ")\n"
                  << "Per-type occurrences: " << kExpectedOccurrencesPerType << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "atom_curriculum_builder: " << error.what() << '\n';
        return 1;
    }
}

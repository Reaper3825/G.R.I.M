#include "../concept_block.hpp"
#include "../io/concept_block_io_flatbuffer.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
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
constexpr int kMixedAllTypes = 1200;
// v5 contrast families. They exist to teach what is *not* an atom: ordinary
// noun phrases, interrogative clauses and role descriptions occupying the same
// syntactic slots the tagged values occupy.
constexpr int kContrastSamePerType = 700;
constexpr int kContrastPairPerCombination = 150;
constexpr int kNegativeOnlyBlocks = 1000;
constexpr int kExpectedEntityStringSameSurfaceContrasts = 850;

enum class ValueType : std::size_t {
    Int = 0,
    Float = 1,
    String = 2,
    Bool = 3,
    Entity = 4,
};

constexpr std::array<ValueType, 5> kTypes{
    ValueType::Int,
    ValueType::Float,
    ValueType::String,
    ValueType::Bool,
    ValueType::Entity,
};

constexpr std::array<std::array<ValueType, 2>, 10> kPairs{{
    {ValueType::Int, ValueType::Float},
    {ValueType::Int, ValueType::String},
    {ValueType::Int, ValueType::Bool},
    {ValueType::Float, ValueType::String},
    {ValueType::Float, ValueType::Bool},
    {ValueType::String, ValueType::Bool},
    {ValueType::Int, ValueType::Entity},
    {ValueType::Float, ValueType::Entity},
    {ValueType::String, ValueType::Entity},
    {ValueType::Bool, ValueType::Entity},
}};

constexpr std::array<std::array<ValueType, 3>, 10> kTriples{{
    {ValueType::Int, ValueType::Float, ValueType::String},
    {ValueType::Int, ValueType::Float, ValueType::Bool},
    {ValueType::Int, ValueType::String, ValueType::Bool},
    {ValueType::Float, ValueType::String, ValueType::Bool},
    {ValueType::Int, ValueType::Float, ValueType::Entity},
    {ValueType::Int, ValueType::String, ValueType::Entity},
    {ValueType::Int, ValueType::Bool, ValueType::Entity},
    {ValueType::Float, ValueType::String, ValueType::Entity},
    {ValueType::Float, ValueType::Bool, ValueType::Entity},
    {ValueType::String, ValueType::Bool, ValueType::Entity},
}};

constexpr int kExpectedBlockCount =
    kSinglePerType * static_cast<int>(kTypes.size()) +
    kMixedPairPerCombination * static_cast<int>(kPairs.size()) +
    kMixedTriplePerCombination * static_cast<int>(kTriples.size()) +
    kMixedAllTypes +
    kContrastSamePerType * static_cast<int>(kTypes.size()) +
    kContrastPairPerCombination * static_cast<int>(kPairs.size()) +
    kNegativeOnlyBlocks;

// Every type sits in exactly four pair combinations and six triples, so the
// per-type occurrence total stays identical across types by construction.
// Derived rather than hardcoded so the balance audit tracks the counts above.
constexpr int expectedOccurrencesPerType(ValueType type) {
    int total = kSinglePerType + kMixedAllTypes + 2 * kContrastSamePerType;
    for (const auto& pair : kPairs) {
        for (const ValueType member : pair) {
            if (member == type) {
                total += kMixedPairPerCombination + kContrastPairPerCombination;
            }
        }
    }
    for (const auto& triple : kTriples) {
        for (const ValueType member : triple) {
            if (member == type) total += kMixedTriplePerCombination;
        }
    }
    return total;
}

const char* typeName(ValueType type) {
    switch (type) {
        case ValueType::Int: return "INT";
        case ValueType::Float: return "FLOAT";
        case ValueType::String: return "STRING";
        case ValueType::Bool: return "BOOL";
        case ValueType::Entity: return "ENTITY";
    }
    throw std::runtime_error("unknown ValueType");
}

const char* typeSlug(ValueType type) {
    switch (type) {
        case ValueType::Int: return "int";
        case ValueType::Float: return "float";
        case ValueType::String: return "string";
        case ValueType::Bool: return "bool";
        case ValueType::Entity: return "entity";
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

std::uint64_t mixIndex(std::size_t index, std::uint64_t salt) {
    std::uint64_t value = static_cast<std::uint64_t>(index) + salt +
        0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
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

const std::array<std::string, 16> kEntityFields{
    "owner", "assignee", "organization", "contact_name",
    "office", "project", "customer", "reviewer",
    "vendor", "team", "location", "account",
    "service", "repository", "document", "workspace",
};

// ---------------------------------------------------------------------------
// v5 negative inventory.
//
// v4 only ever put multi-word English inside <ENTITY>: INT/FLOAT are numerals,
// STRING is a slug and BOOL is true/false, so "a run of English words" was a
// perfectly consistent rule for ENTITY and the model began tagging the question
// clause of a word problem ("How many liters were used"). Everything below is
// text that must stay OUTSIDE every delimiter -- interrogative clauses,
// definite descriptions and bare common nouns sitting in exactly the slots a
// tagged value would occupy.
// ---------------------------------------------------------------------------

const std::array<std::string, 20> kCountNouns{
    "liters", "crates", "boxes", "tickets", "pages", "sacks", "bottles",
    "packets", "seats", "panels", "rolls", "cables", "bricks", "trays",
    "envelopes", "batteries", "tiles", "jars", "cartons", "spools",
};

// FLOAT quantities read as measurements, not counts of discrete objects.
const std::array<std::string, 20> kMeasureNouns{
    "liters", "kilograms", "meters", "hours", "kilometers", "grams",
    "milliliters", "minutes", "degrees", "megabytes", "watts", "acres",
    "gallons", "pounds", "seconds", "millimeters", "tons", "yards",
    "ounces", "centimeters",
};

const std::array<std::string, 16> kContainers{
    "tank", "warehouse", "shelf", "pallet", "storage bin", "cabinet",
    "delivery truck", "storeroom", "locker", "rack", "drawer",
    "shipping container", "toolbox", "cooler", "trailer", "loading dock",
};

// Definite descriptions naming a participant by role instead of by name.
const std::array<std::string, 16> kPersonRoles{
    "the on-call engineer", "the shift supervisor", "the reviewing analyst",
    "the duty manager", "the next available reviewer", "the current maintainer",
    "the vendor representative", "the account holder of record",
    "the person who signed off", "the auditor on record", "the default assignee",
    "whoever closes the batch", "the previous owner", "an unnamed contributor",
    "the customer who filed the ticket", "the requesting team",
};

const std::array<std::string, 16> kThingRoles{
    "the largest available region", "the originating service",
    "the upstream caller", "the team that owns the queue",
    "the same account as before", "the department that raised it",
    "the group listed on the ticket", "whichever worker is idle",
    "the queue with the shortest backlog", "the office nearest the depot",
    "the repository that failed last night", "the document under review",
    "the workspace shared with the vendor", "the service that timed out",
    "the project nobody has renamed", "the account flagged during intake",
};

// Clauses stating that a slot has no name yet. Every ENTITY field word appears
// here inside a sentence where nothing at all should be tagged.
const std::array<std::string, 16> kUnnamedClauses{
    "no owner has been recorded yet",
    "the reviewer has not been named",
    "nobody is listed on the request",
    "the team is still unassigned",
    "the destination is not decided",
    "the account remains anonymous",
    "the contact is left blank",
    "no organization is attached to this record",
    "the location is described but never named",
    "the vendor is referred to only by role",
    "the customer stays unidentified in this note",
    "the maintainer is listed as pending",
    "the workspace has no name yet",
    "the service is referenced generically",
    "the document carries no title",
    "the project is described without a name",
};

// Interrogative clauses. The object in question is the thing being asked about,
// never an entity, so none of these may ever be wrapped in a delimiter.
const std::array<std::string, 16> kOpenQuestions{
    "Which setting changed during the last run",
    "Who approved the change",
    "What should happen when the queue drains",
    "Which step failed first",
    "How was the decision recorded",
    "What is the object in question here",
    "Which record does this line describe",
    "Who is responsible once the batch closes",
    "What remains to be verified",
    "Which value should be trusted",
    "How long should the record be kept",
    "What was requested in the original ticket",
    "Which part of the workflow repeats",
    "Who receives the summary",
    "What happens to the leftover work",
    "Which option was chosen in the end",
};

std::string countQuestion(const std::string& noun, std::size_t index) {
    switch (index % 16) {
        case 0: return "How many " + noun + " were used";
        case 1: return "How many " + noun + " remain";
        case 2: return "How many more " + noun + " are needed";
        case 3: return "What is the total number of " + noun;
        case 4: return "How many " + noun + " were added";
        case 5: return "How many " + noun + " are left over";
        case 6: return "How many " + noun + " were returned";
        case 7: return "How many " + noun + " did the second delivery bring";
        case 8: return "How many " + noun + " were counted at the end";
        case 9: return "By how many " + noun + " did the count fall short";
        case 10: return "How many " + noun + " must still be packed";
        case 11: return "What was the starting number of " + noun;
        case 12: return "How many " + noun + " were removed during the check";
        case 13: return "How many " + noun + " fit in the remaining space";
        case 14: return "What is the difference in " + noun;
        default: return "How many " + noun + " were unaccounted for";
    }
}

// Picks a clause that differs from the one already used in the body, so a
// block never states the same thing twice.
const std::string& otherUnnamedClause(const std::string& used,
                                      std::uint64_t seed) {
    const std::size_t start = static_cast<std::size_t>(seed % kUnnamedClauses.size());
    for (std::size_t step = 0; step < kUnnamedClauses.size(); ++step) {
        const std::string& candidate =
            kUnnamedClauses[(start + step) % kUnnamedClauses.size()];
        if (candidate != used) return candidate;
    }
    return kUnnamedClauses.front();
}

std::string capitalize(std::string text) {
    if (!text.empty()) {
        text[0] = static_cast<char>(
            std::toupper(static_cast<unsigned char>(text[0])));
    }
    return text;
}

const std::string& fieldFor(ValueType type, std::size_t index) {
    switch (type) {
        case ValueType::Int: return choose(kIntFields, index);
        case ValueType::Float: return choose(kFloatFields, index);
        case ValueType::String: return choose(kStringFields, index);
        case ValueType::Bool: return choose(kBoolFields, index);
        case ValueType::Entity: return choose(kEntityFields, index);
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
        "silver", "orbit", "harbor", "billing", "research", "summit",
        "meadow", "timber", "delta", "notebook", "simple", "crimson", "winter",
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

std::size_t entityCategory(std::size_t index) {
    return static_cast<std::size_t>(
        mixIndex(index, 0x454e544954595f31ULL) % 6ULL);
}

std::string entityValue(std::size_t index) {
    // These are literal entity surface forms only. Role/category words belong
    // to the surrounding field (for example, project=<ENTITY>Atlas</ENTITY>)
    // unless they are genuinely part of the entity's proper name.
    static const std::array<std::string, 16> given_names{
        "Ada", "Grace", "Maya", "Amelia", "Ren", "Sofia", "Amina", "Luca",
        "Noor", "Diego", "Imani", "Hiro", "Leah", "Nia", "Pavel", "Samira",
    };
    static const std::array<std::string, 16> family_names{
        "Lovelace", "Hopper", "Chen", "O'Connor-Smith", "Tanaka", "Rossi",
        "Okafor", "Miller", "Dubois", "Garcia", "Patel", "Kim", "Silva",
        "Kowalski", "Hassan", "Wang",
    };
    static const std::array<std::string, 16> organization_stems{
        "Northwind", "Acme", "Lumen", "Cedar", "Bluebird", "Nimbus",
        "Solstice", "Orion", "Aurora", "Meridian", "Harbor", "Atlas",
        "Juniper", "Mosaic", "Lantern", "Summit",
    };
    static const std::array<std::string, 16> organization_names{
        "Labs", "Systems", "Works", "Analytics", "Dynamics", "Studio",
        "Collective", "Network", "Research", "Technologies", "Partners",
        "Foundation", "Digital", "Industries", "Ventures", "AI",
    };
    static const std::array<std::string, 16> place_roots{
        "Boston", "Seattle", "Denver", "Austin", "Phoenix", "Portland",
        "Detroit", "Atlanta", "Raleigh", "Tampa", "Madison", "Dublin",
        "Bristol", "Oxford", "York", "Cambridge",
    };
    static const std::array<std::string, 16> place_names{
        "Central", "North", "South", "East", "West", "Harbor", "Heights",
        "Old Town", "Riverside", "Market", "Garden", "Station", "Quarter",
        "Park", "Point", "Crossing",
    };
    static const std::array<std::string, 16> named_roots{
        "Atlas", "Aurora", "Orion", "Cedar", "Nimbus", "Solstice", "Delta",
        "Lumen", "Mosaic", "Juniper", "Lantern", "Harbor", "Meridian",
        "Bluebird", "Northstar", "Summit",
    };
    static const std::array<std::string, 16> named_variants{
        "Prime", "Nova", "One", "Blue", "Green", "Gold", "Silver", "Dawn",
        "Echo", "Pulse", "Wave", "Core", "Edge", "Light", "Field", "Bridge",
    };
    static const std::array<std::string, 16> compact_names{
        "OpenAI", "NASA", "CERN", "IBM", "GitHub", "GRIM-text", "R2-D2",
        "B-612", "ACME-7", "NVIDIA", "UNESCO", "DeepMind", "SpaceX",
        "Q4-Atlas", "K-2", "X.Org",
    };
    static const std::array<std::string, 16> additional_place_names{
        "Albany", "Baltimore", "Charlotte", "Columbus", "Dallas", "Hartford",
        "Houston", "Jacksonville", "Louisville", "Memphis", "Nashville",
        "Pittsburgh", "Richmond", "Savannah", "Spokane", "Wilmington",
    };

    const std::uint64_t mixed = mixIndex(index, 0x454e544954595f31ULL);
    const std::size_t category = entityCategory(index);
    const std::size_t first = static_cast<std::size_t>((mixed >> 8) % 16ULL);
    const std::size_t second = static_cast<std::size_t>((mixed >> 24) % 16ULL);
    const bool compact = ((mixed >> 40) & 1ULL) != 0;
    switch (category) {
        case 0:
            if (!compact && ((mixed >> 41) & 1ULL) != 0) {
                const char middle_initial = static_cast<char>('A' + ((mixed >> 42) % 26ULL));
                return given_names[first] + " " + middle_initial + ". " + family_names[second];
            }
            return given_names[first] + " " + family_names[second];
        case 1: return organization_stems[first] + " " + organization_names[second];
        case 2:
            return compact ? place_roots[first]
                           : place_roots[first] + " " + place_names[second];
        case 3:
            return compact ? named_roots[first]
                           : named_roots[first] + " " + named_variants[second];
        case 4: return compact_names[first];
        default: return additional_place_names[first];
    }
}

const std::string& entityFieldFor(std::size_t entity_index,
                                  std::size_t variation_index) {
    static const std::array<std::string, 4> person_fields{
        "owner", "assignee", "contact_name", "reviewer",
    };
    static const std::array<std::string, 4> organization_fields{
        "organization", "customer", "vendor", "account",
    };
    static const std::array<std::string, 4> place_fields{
        "office", "location", "destination", "workspace",
    };
    static const std::array<std::string, 5> named_fields{
        "project", "service", "repository", "document", "workspace",
    };
    static const std::array<std::string, 5> compact_fields{
        "organization", "project", "service", "repository", "account",
    };

    const std::uint64_t mixed = mixIndex(
        variation_index,
        0x454e544954595f46ULL + static_cast<std::uint64_t>(entity_index));
    switch (entityCategory(entity_index)) {
        case 0: return choose(person_fields, mixed);
        case 1: return choose(organization_fields, mixed);
        case 2: return choose(place_fields, mixed);
        case 3: return choose(named_fields, mixed);
        case 4: return choose(compact_fields, mixed);
        default: return choose(place_fields, mixed);
    }
}

// ---------------------------------------------------------------------------
// v5 ENTITY surface pool.
//
// Deliberately separate from entityValue(): the mixed blocks are frozen for
// this revision, so widening their surface pool would change their bytes. Only
// the regenerated ENTITY singles and the contrast families draw from here.
// ---------------------------------------------------------------------------

std::size_t entitySingleCategory(std::size_t index) { return index % 10; }

std::string entitySingleSurface(std::size_t index) {
    static const std::array<std::string, 24> given_names{
        "Ada", "Grace", "Maya", "Amelia", "Ren", "Sofia", "Amina", "Luca",
        "Noor", "Diego", "Imani", "Hiro", "Leah", "Nia", "Pavel", "Samira",
        "Jonas", "Priya", "Kwame", "Elena", "Tomas", "Yara", "Oskar", "Mei",
    };
    static const std::array<std::string, 24> family_names{
        "Lovelace", "Hopper", "Chen", "O'Connor", "Tanaka", "Rossi",
        "Okafor", "Miller", "Dubois", "Garcia", "Patel", "Kim", "Silva",
        "Kowalski", "Hassan", "Wang", "Nakamura", "Ferreira", "Bergstrom",
        "Adeyemi", "Ibarra", "Novak", "Whitfield", "Larsen",
    };
    static const std::array<std::string, 24> organization_stems{
        "Northwind", "Acme", "Lumen", "Cedar", "Bluebird", "Nimbus",
        "Solstice", "Orion", "Aurora", "Meridian", "Harbor", "Atlas",
        "Juniper", "Mosaic", "Lantern", "Summit", "Ironwood", "Kestrel",
        "Marlowe", "Pinnacle", "Redstone", "Thistle", "Vantage", "Wexler",
    };
    static const std::array<std::string, 24> organization_tails{
        "Labs", "Systems", "Works", "Analytics", "Dynamics", "Studio",
        "Collective", "Network", "Research", "Technologies", "Partners",
        "Foundation", "Digital", "Industries", "Ventures", "AI", "Logistics",
        "Robotics", "Instruments", "Holdings", "Group", "Micro", "Optics",
        "Freight",
    };
    static const std::array<std::string, 24> place_roots{
        "Boston", "Seattle", "Denver", "Austin", "Phoenix", "Portland",
        "Detroit", "Atlanta", "Raleigh", "Tampa", "Madison", "Dublin",
        "Bristol", "Oxford", "York", "Cambridge", "Albany", "Baltimore",
        "Charlotte", "Columbus", "Hartford", "Nashville", "Savannah", "Spokane",
    };
    static const std::array<std::string, 16> place_qualifiers{
        "Central", "North", "South", "East", "West", "Harbor", "Heights",
        "Old Town", "Riverside", "Market", "Garden", "Station", "Quarter",
        "Park", "Point", "Crossing",
    };
    static const std::array<std::string, 24> product_roots{
        "Atlas", "Aurora", "Orion", "Cedar", "Nimbus", "Solstice", "Delta",
        "Lumen", "Mosaic", "Juniper", "Lantern", "Harbor", "Meridian",
        "Bluebird", "Northstar", "Summit", "Halcyon", "Quill", "Tessera",
        "Vireo", "Zephyr", "Anvil", "Beacon", "Cinder",
    };
    static const std::array<std::string, 16> product_tails{
        "Prime", "Nova", "One", "Mk II", "Mk III", "Core", "Edge", "Light",
        "Field", "Bridge", "Echo", "Pulse", "Wave", "Dawn", "Relay", "Arc",
    };
    static const std::array<std::string, 24> compact_names{
        "OpenAI", "NASA", "CERN", "IBM", "GitHub", "GRIM-text", "R2-D2",
        "B-612", "ACME-7", "NVIDIA", "UNESCO", "DeepMind", "SpaceX",
        "Q4-Atlas", "K-2", "X.Org", "NOAA", "ESA-9", "T-1000", "V-2",
        "MIT", "CSIRO", "JAXA", "BBC",
    };

    const std::uint64_t mixed = mixIndex(index, 0x454e545f53474cULL);
    const std::size_t first = static_cast<std::size_t>((mixed >> 8) % 24ULL);
    const std::size_t second = static_cast<std::size_t>((mixed >> 24) % 24ULL);
    const std::size_t small = static_cast<std::size_t>((mixed >> 36) % 16ULL);
    // Offset keeps the two halves of a compound distinct without a retry loop.
    const std::size_t other = (first + 1 + (second % 23)) % 24;
    switch (entitySingleCategory(index)) {
        case 0:
            return given_names[first] + " " + family_names[second];
        case 1: {
            const char initial = static_cast<char>('A' + ((mixed >> 44) % 26ULL));
            return given_names[first] + " " + initial + ". " + family_names[second];
        }
        case 2:
            return given_names[first] + " " + family_names[second] + "-" +
                family_names[other];
        case 3:
            return given_names[first];
        case 4:
            return organization_stems[first] + " " + organization_tails[second];
        case 5:
            return organization_stems[first] + " & " + organization_stems[other];
        case 6:
            return ((mixed >> 40) & 1ULL) != 0
                ? place_roots[first]
                : place_roots[first] + " " + place_qualifiers[small];
        case 7:
            return product_roots[first] + " " + product_tails[small];
        case 8:
            return compact_names[first];
        default:
            return organization_stems[first] + "-" + organization_stems[other];
    }
}

const std::string& entitySingleField(std::size_t index,
                                     std::size_t variation_index) {
    static const std::array<std::string, 4> person_fields{
        "owner", "assignee", "contact_name", "reviewer",
    };
    static const std::array<std::string, 4> organization_fields{
        "organization", "customer", "vendor", "account",
    };
    static const std::array<std::string, 4> place_fields{
        "office", "location", "destination", "workspace",
    };
    static const std::array<std::string, 5> named_fields{
        "project", "service", "repository", "document", "workspace",
    };
    static const std::array<std::string, 5> compact_fields{
        "organization", "project", "service", "repository", "account",
    };

    const std::uint64_t mixed = mixIndex(
        variation_index,
        0x454e545f464c44ULL + static_cast<std::uint64_t>(index));
    switch (entitySingleCategory(index)) {
        case 0:
        case 1:
        case 2:
        case 3: return choose(person_fields, mixed);
        case 4:
        case 5:
        case 9: return choose(organization_fields, mixed);
        case 6: return choose(place_fields, mixed);
        case 7: return choose(named_fields, mixed);
        default: return choose(compact_fields, mixed);
    }
}

std::string rawValue(ValueType type, std::size_t index) {
    switch (type) {
        case ValueType::Int: return integerValue(index);
        case ValueType::Float: return floatValue(index);
        case ValueType::String: return stringValue(index);
        case ValueType::Bool: return index % 2 == 0 ? "true" : "false";
        case ValueType::Entity: return entityValue(index);
    }
    throw std::runtime_error("unknown ValueType");
}

std::string taggedValue(ValueType type, std::size_t index) {
    const std::string tag = typeName(type);
    return "<" + tag + ">" + rawValue(type, index) + "</" + tag + ">";
}

// v5 ENTITY singles. Two changes from v4: a wider template and surface pool,
// and role descriptions / interrogative clauses woven through the templates so
// every block carries an in-place contrast between the name that belongs inside
// the delimiter and the noun phrase beside it that does not.
//
// The (pattern, subject, stage) triple is drawn through a multiplier coprime
// with 48*16*16, so those three slots differ for every index below 6000 and the
// authored text is unique by construction.
std::string makeEntitySingleRaw(std::size_t index) {
    const std::size_t spread = (index * 7) % 12288;
    const std::size_t pattern = spread % 48;
    const std::string& subject = kSubjects[(spread / 48) % 16];
    const std::string& stage = kStages[(spread / 768) % 16];

    const std::uint64_t mixed = mixIndex(index, 0x454e544954595f32ULL);
    const std::size_t context_index = index / 48;
    const std::string& field = entitySingleField(index, context_index);
    const std::string& action = choose(kActions, mixed >> 24);
    const std::string& person = choose(kPersonRoles, mixed >> 8);
    const std::string& thing = choose(kThingRoles, mixed >> 16);
    const std::string& open_question = choose(kOpenQuestions, mixed >> 32);
    const std::string entity =
        "<ENTITY>" + entitySingleSurface(index) + "</ENTITY>";

    const auto finalize = [&](std::string text) {
        switch ((mixed >> 48) % 4ULL) {
            case 0:
                return text + " The " + subject + " remains in " + stage +
                    " before " + action + ".";
            case 1:
                return text + " On this " + subject + ", " + person +
                    " names a role rather than a value, so " + stage +
                    " continues unchanged.";
            case 2:
                return text + " Everything else on the " + subject +
                    " still points at " + thing + " during " + stage + ".";
            default:
                return text + " The " + subject + " stays in " + stage +
                    " while " + person + " reviews the entry.";
        }
    };

    switch (pattern) {
        case 0: return finalize("Set " + field + " to " + entity + " before " + action + ".");
        case 1: return finalize("Before " + action + ", assign " + entity + " to " + field + ".");
        case 2: return finalize("The " + subject + " request uses " + field + "=" + entity + ".");
        case 3: return finalize("For the " + subject + ", the " + field + " value is " + entity + ".");
        case 4: return finalize("Please keep " + field + " at " + entity + " while " + action + ".");
        case 5: return finalize("Confirm that " + field + " should be " + entity + " for the " + subject + ".");
        case 6: return finalize("Update the " + subject + " so " + field + " becomes " + entity + ".");
        case 7: return finalize("Use " + entity + " as " + field + " when " + action + ".");
        case 8: return finalize(field + ": " + entity + ". Apply this setting to the " + subject + ".");
        case 9: return finalize("When preparing the " + subject + ", choose " + entity + " for " + field + ".");
        case 10: return finalize("The value " + entity + " belongs in " + field + " for the " + subject + ".");
        case 11: return finalize("Should the " + subject + " use " + entity + " as its " + field + " value?");
        case 12: return finalize("Record " + field + " as " + entity + ", then continue the workflow.");
        case 13: return finalize("On the " + subject + ", replace the current " + field + " with " + entity + ".");
        case 14: return finalize("The requested " + field + " for the " + subject + " is " + entity + ".");
        case 15: return finalize("For the next " + subject + " operation, set " + field + "=" + entity + ".");
        case 16: return finalize("Assign " + field + " the value " + entity + " in the " + subject + " configuration.");
        case 17: return finalize("Keep the " + subject + " unchanged except for " + field + ", which must be " + entity + ".");
        case 18: return finalize("While " + action + ", read " + field + " as " + entity + ".");
        case 19: return finalize("Apply this " + subject + " option: " + field + "=" + entity + ".");
        case 20: return finalize(entity + " is recorded under " + field + " for the " + subject + ".");
        case 21: return finalize("For " + field + " on the " + subject + ", use " + entity + ".");
        case 22: return finalize("\"" + entity + "\"" + " is listed in the " + field + " field.");
        case 23: return finalize("(" + entity + ") is the current value of " + field + ".");
        case 24: return finalize(subject + " | " + field + " maps to " + entity + ".");
        case 25: return finalize(entity + " is assigned from " + field + " for the " + subject + ".");
        case 26: return finalize("During " + stage + ", the " + subject + " associates " + field + " with " + entity + ".");
        case 27: return finalize("Keep " + field + " unchanged unless it should point to " + entity + ".");
        case 28: return finalize("Does " + field + " on the " + subject + " refer to " + entity + "?");
        case 29: return finalize("The " + field + " entry for the " + subject + " ends with " + entity + ".");
        case 30: return finalize(entity + " appears first; record it as " + field + " before " + action + ".");
        case 31: return finalize("When " + action + ", compare the current " + field + " with " + entity + ".");

        // Contrast templates: a described participant stands beside the named
        // one, and only the name is ever delimited.
        case 32: return finalize("Set " + field + " to " + entity + " rather than to " + person + ".");
        case 33: return finalize("The " + subject + " names " + entity + " while " + thing + " stays unnamed.");
        case 34: return finalize(open_question + "? For the " + subject + ", " + field + " is " + entity + ".");
        case 35: return finalize("Route the " + subject + " to " + entity + ", not to " + person + ".");
        case 36: return finalize("Although " + person + " raised the " + subject + ", " + field + " is " + entity + ".");
        case 37: return finalize(capitalize(person) + " asked that " + field + " be " + entity + " for the " + subject + ".");
        case 38: return finalize("Between " + person + " and " + entity + ", only the latter belongs in " + field + ".");
        case 39: return finalize("The " + subject + " lists " + field + "=" + entity + " and leaves the rest to " + thing + ".");
        case 40: return finalize("Who owns the " + subject + "? The " + field + " field points to " + entity + ".");
        case 41: return finalize("Replace " + person + " with " + entity + " in the " + field + " slot.");
        case 42: return finalize("The " + field + " on this " + subject + " is " + entity + ", not " + thing + ".");
        case 43: return finalize("Once " + action + " begins, " + field + " must read " + entity + " instead of " + person + ".");
        case 44: return finalize("A request from " + person + " assigns " + field + " to " + entity + ".");
        case 45: return finalize("The " + subject + " record shows " + field + ": " + entity + "; " + thing + " is only referenced.");
        case 46: return finalize(open_question + "? The " + subject + " answers with " + field + "=" + entity + ".");
        default: return finalize("Do not confuse " + person + " with " + entity + " when reading " + field + ".");
    }
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

    const bool has_entity =
        std::find(types.begin(), types.end(), ValueType::Entity) != types.end();
    const bool has_string =
        std::find(types.begin(), types.end(), ValueType::String) != types.end();
    const bool same_surface_contrast = has_entity && has_string && index % 4 == 0;
    const std::string contrast_surface = same_surface_contrast
        ? entityValue(700000 + index)
        : std::string{};

    std::vector<MixedField> fields;
    fields.reserve(types.size());
    for (std::size_t position = 0; position < types.size(); ++position) {
        const std::size_t value_index = 100000 + index * 7 + position * 100003;
        const std::size_t entity_surface_index = same_surface_contrast
            ? 700000 + index
            : value_index;
        std::string atom;
        if (same_surface_contrast &&
            (types[position] == ValueType::Entity ||
             types[position] == ValueType::String)) {
            const std::string tag = typeName(types[position]);
            atom = "<" + tag + ">" + contrast_surface + "</" + tag + ">";
        } else {
            atom = taggedValue(types[position], value_index);
        }
        fields.push_back(MixedField{
            types[position],
            types[position] == ValueType::Entity
                ? entityFieldFor(
                    entity_surface_index,
                    index + position * 5)
                : fieldFor(types[position], index + position * 5),
            std::move(atom),
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

    const std::size_t pattern = index % 24;
    const std::size_t context_index = index / 24;
    const std::string& subject = choose(kSubjects, context_index);
    const std::string& action = choose(kActions, context_index / kSubjects.size());
    const std::string& stage = choose(
        kStages,
        context_index / (kSubjects.size() * kActions.size()));
    const auto finalize = [&](std::string text) {
        return text + " This configuration applies during " + stage +
            " before " + action + ".";
    };
    switch (pattern) {
        case 0:
            return finalize("Configure the " + subject + " with " +
                join(equals, ", ", ", and ") + ".");
        case 1:
            return finalize("For the " + subject + ", set " +
                join(natural, "; ", "; and ") + ", then continue the workflow.");
        case 2:
            return finalize("The " + subject + " request carries " +
                join(natural, ", ", ", and ") + ".");
        case 3:
            return finalize("Use these " + subject + " values: " +
                join(equals, " | ") + ".");
        case 4:
            return finalize("Should the " + subject + " use " +
                join(equals, ", ", ", and ") + "?");
        case 5:
            return finalize("Update the " + subject + ": " +
                join(natural, ", ", ", while setting ") + ".");
        case 6:
            return finalize("Apply " + join(colon, "; ", "; and ") +
                " to the " + subject + ".");
        case 7:
            return finalize("Use " + join(reversed, ", ", ", and ") +
                " when handling the " + subject + ".");
        case 8:
            return finalize("The configuration for the " + subject + " is {" +
                join(equals, ", ") + "}.");
        case 9:
            return finalize("Assign the following " + subject + " values—" +
                join(equals, "; ") + "—then continue.");
        case 10:
            return finalize("On the " + subject + ", keep " +
                join(natural, ", ", ", and ") + ".");
        case 11:
            return finalize("Prepare the " + subject + " with " +
                join(equals, "; ", "; and ") + "; afterward, continue the workflow.");
        case 12:
            return finalize("The " + subject + " values are " +
                join(colon, ", ", ", and ") + ".");
        case 13:
            return finalize("Set " + join(natural, ", ", ", and ") +
                " for the " + subject + ".");
        case 14:
            return finalize("For the " + subject + ": " + join(equals, "; ") + ".");
        case 15:
            return finalize("Record " + join(reversed, ", ", ", and ") +
                " for the " + subject + ".");
        case 16:
            return finalize("The " + subject + " maps " +
                join(natural, "; ", "; and ") + ".");
        case 17:
            return finalize("Use (" + join(equals, ", ") + ") for the " + subject + ".");
        case 18:
            return finalize("Please verify " + join(equals, " / ") +
                " for the " + subject + ".");
        case 19:
            return finalize("For the " + subject + ", keep " +
                join(colon, "; ", "; and ") + ".");
        case 20:
            return finalize("Fields for the " + subject + "—" +
                join(equals, ", ") + ".");
        case 21:
            return finalize("Is the " + subject + " configured with " +
                join(equals, ", ", ", and ") + "?");
        case 22:
            return finalize("Apply " + join(natural, "; ", "; and ") +
                " to the " + subject + ".");
        default:
            return finalize("The " + subject + " uses " +
                join(reversed, ", ", ", and ") + ".");
    }
}

// ---------------------------------------------------------------------------
// v5 contrast families.
//
// Three shapes, all sharing one property: the noun phrase a reader would call
// "the thing being talked about" sits untagged next to the values that are
// genuinely atoms.
//
//   * same-type blocks  -- two atoms of one type plus untagged bait
//   * pair blocks       -- reuse the balanced kPairs table with a prose voice
//   * negative blocks   -- ordinary English with no atom and no numeral at all
//
// Per-type balance is preserved by construction: same-type blocks contribute
// exactly two occurrences to their own type, pair blocks walk the same
// combination table the mixed blocks do, and negative blocks contribute none.
// ---------------------------------------------------------------------------

std::string contrastAtom(ValueType type, std::size_t index) {
    switch (type) {
        case ValueType::Int: {
            // Plain readable magnitudes: the prose voice reads as a word
            // problem, so the sign and exponent variety of the config set would
            // only add noise. Still strict-validator clean.
            const std::size_t value = ((index * 37 + 11) % 4800) + 12;
            return "<INT>" + std::to_string(value) + "</INT>";
        }
        case ValueType::Float: {
            const std::size_t whole = ((index * 29 + 7) % 9000) + 3;
            const std::size_t fraction = (index * 61 + 13) % 100;
            std::ostringstream out;
            out << "<FLOAT>" << whole << '.' << std::setw(2)
                << std::setfill('0') << fraction << "</FLOAT>";
            return out.str();
        }
        case ValueType::String:
            return taggedValue(ValueType::String, 300000 + index);
        case ValueType::Bool:
            return index % 2 == 0 ? "<BOOL>true</BOOL>" : "<BOOL>false</BOOL>";
        case ValueType::Entity:
            return "<ENTITY>" + entitySingleSurface(400000 + index) + "</ENTITY>";
    }
    throw std::runtime_error("unknown ValueType");
}

// INT/FLOAT contrast: the shape the v4 model got wrong. Both quantities are
// tagged; the interrogative clause naming the object in question is not.
std::string makeContrastQuantitativeRaw(ValueType type, std::size_t index) {
    const std::size_t spread = (index * 7) % 3840;
    const std::size_t pattern = spread % 12;
    const std::string& noun = type == ValueType::Float
        ? kMeasureNouns[(spread / 12) % 20]
        : kCountNouns[(spread / 12) % 20];
    const std::string& container = kContainers[(spread / 240) % 16];

    const std::uint64_t mixed = mixIndex(index, 0x434f4e5451554eULL);
    const std::string& person = choose(kPersonRoles, mixed >> 8);
    const std::string& thing = choose(kThingRoles, mixed >> 16);
    const std::string& unnamed = choose(kUnnamedClauses, mixed >> 24);
    const std::string& tail_unnamed = otherUnnamedClause(unnamed, mixed >> 44);
    const std::string question = countQuestion(noun, mixed >> 32);

    // An explicit larger/smaller pair keeps "after using some, N remain" from
    // reading as an increase. The larger value is injective in index, so the
    // authored text stays unique without leaning on the template slots.
    std::string high;
    std::string low;
    if (type == ValueType::Int) {
        const std::size_t upper = ((index * 37 + 11) % 4800) + 80;
        const std::size_t delta = ((index * 13 + 5) % 60) + 3;
        high = "<INT>" + std::to_string(upper) + "</INT>";
        low = "<INT>" + std::to_string(upper - delta) + "</INT>";
    } else {
        const std::size_t whole = ((index * 29 + 7) % 9000) + 80;
        const std::size_t fraction = (index * 61 + 13) % 100;
        const std::size_t delta = ((index * 17 + 3) % 60) + 3;
        const auto wrap = [](std::size_t units, std::size_t hundredths) {
            std::ostringstream out;
            out << "<FLOAT>" << units << '.' << std::setw(2)
                << std::setfill('0') << hundredths << "</FLOAT>";
            return out.str();
        };
        high = wrap(whole, fraction);
        low = wrap(whole - delta, (fraction + 37) % 100);
    }

    const auto finalize = [&](std::string text) {
        switch ((mixed >> 48) % 4ULL) {
            case 0:
                return text + " The tally for the " + container +
                    " was confirmed by " + person + ", and " + tail_unnamed + ".";
            case 1:
                return text + " " + capitalize(person) + " logged the " +
                    container + " figures, though " + tail_unnamed + ".";
            case 2:
                return text + " This " + container + " tally was filed under " +
                    thing + " because " + tail_unnamed + ".";
            default:
                return text + " Ask " + person + " for the " + container +
                    " worksheet; " + tail_unnamed + ".";
        }
    };

    switch (pattern) {
        case 0:
            return finalize("A " + container + " holds " + high + " " + noun +
                ". After using some, " + low + " " + noun + " remain. " +
                question + "?");
        case 1:
            return finalize(question + "? The " + container + " started with " +
                high + " " + noun + " and ended with " + low + ".");
        case 2:
            return finalize("The first delivery brought " + low + " " + noun +
                " to the " + container + " and the second brought " + high +
                ". " + question + "?");
        case 3:
            return finalize("Counting the " + container + " twice gave " +
                high + " " + noun + " and then " + low + ". " + question + "?");
        case 4:
            return finalize("Of the " + high + " " + noun + " ordered, only " +
                low + " reached the " + container + ". " + question + "?");
        case 5:
            return finalize("The log records " + high + " " + noun +
                " in the " + container + " at intake and " + low +
                " at handoff. " + question + "?");
        case 6:
            return finalize("Someone moved " + noun + " out of the " + container +
                ": " + high + " went out and " + low + " came back. " +
                question + "?");
        case 7:
            return finalize("The " + container + " was rated for " + low + " " +
                noun + " but held " + high + ". " + question + "?");
        case 8:
            return finalize("Before the audit the " + container + " held " +
                high + " " + noun + "; afterwards, " + low + ". " +
                question + "?");
        case 9:
            return finalize("Two counts of the " + container + " disagree: " +
                high + " " + noun + " versus " + low + ". " + question + "?");
        case 10:
            return finalize("The " + container + " lost " + high + " " + noun +
                " overnight and gained " + low + " at dawn. " + question + "?");
        default:
            return finalize("A worker packed " + high + " " + noun + " into the " +
                container + " and unpacked " + low + ". " + question + "?");
    }
}

// STRING/BOOL/ENTITY contrast: two tagged values of one type surrounded by role
// descriptions and open questions that must stay bare. For ENTITY this is the
// direct lesson -- these two are names, that phrase beside them is not.
std::string makeContrastDescriptiveRaw(ValueType type, std::size_t index) {
    const std::size_t spread = (index * 7) % 3072;
    const std::size_t pattern = spread % 12;
    const std::string& subject = kSubjects[(spread / 12) % 16];
    const std::string& stage = kStages[(spread / 192) % 16];

    const std::uint64_t mixed = mixIndex(index, 0x434f4e5444455343ULL);
    const std::string& person = choose(kPersonRoles, mixed >> 8);
    const std::string& thing = choose(kThingRoles, mixed >> 16);
    const std::string& unnamed = choose(kUnnamedClauses, mixed >> 24);
    const std::string& tail_unnamed = otherUnnamedClause(unnamed, mixed >> 44);
    const std::string& open_question = choose(kOpenQuestions, mixed >> 32);

    const std::size_t first_index = index * 2;
    const std::size_t second_index = index * 2 + 1;
    const std::string first = contrastAtom(type, first_index);
    const std::string second = contrastAtom(type, second_index);
    const std::string& first_field = type == ValueType::Entity
        ? entitySingleField(400000 + first_index, index)
        : fieldFor(type, index);
    const std::string& second_field = type == ValueType::Entity
        ? entitySingleField(400000 + second_index, index + 5)
        : fieldFor(type, index + 5);

    const auto finalize = [&](std::string text) {
        switch ((mixed >> 48) % 4ULL) {
            case 0:
                return text + " The " + subject + " stays in " + stage +
                    " until " + tail_unnamed + ".";
            case 1:
                return text + " During " + stage + ", the " + subject +
                    " lists " + person + " only as a role.";
            case 2:
                return text + " The " + subject + " reached " + stage +
                    " while " + thing + " went unnamed.";
            default:
                return text + " No name is attached to the " + subject +
                    " at " + stage + ", since " + tail_unnamed + ".";
        }
    };

    switch (pattern) {
        case 0:
            return finalize("The " + subject + " lists " + first_field + "=" +
                first + " and " + second_field + "=" + second + ", but " +
                unnamed + ". " + open_question + "?");
        case 1:
            return finalize(open_question + "? The record sets " + first_field +
                " to " + first + " and " + second_field + " to " + second +
                ", while " + unnamed + ".");
        case 2:
            return finalize("Assign " + first + " to " + first_field + " and " +
                second + " to " + second_field + "; route the remainder to " +
                person + ".");
        case 3:
            return finalize("Both " + first_field + " (" + first + ") and " +
                second_field + " (" + second + ") are set, yet " + unnamed +
                ". " + open_question + "?");
        case 4:
            return finalize("For this request, " + first_field + " is " + first +
                " and " + second_field + " is " + second +
                ". Everything else stays with " + thing + ".");
        case 5:
            return finalize("Compare " + first + " in " + first_field +
                " against " + second + " in " + second_field +
                " before handing the work to " + person + ".");
        case 6:
            return finalize(capitalize(person) + " asked why " + first_field +
                " holds " + first + " while " + second_field + " holds " +
                second + ". " + open_question + "?");
        case 7:
            return finalize("One entry keeps " + first_field + "=" + first +
                "; " + thing + " keeps " + second_field + "=" + second + ".");
        case 8:
            return finalize("Update " + first_field + " to " + first + ", then " +
                second_field + " to " + second + ", and tell " + person +
                " that " + unnamed + ".");
        case 9:
            return finalize("Only " + first_field + " (" + first + ") and " +
                second_field + " (" + second + ") were filled in, so " +
                unnamed + ".");
        case 10:
            return finalize("The change set " + first_field + " to " + first +
                " and " + second_field + " to " + second + " even though " +
                unnamed + ".");
        default:
            return finalize(open_question + "? Neither " + person + " nor " +
                thing + " changed " + first_field + "=" + first + " or " +
                second_field + "=" + second + ".");
    }
}

std::string makeContrastSameTypeRaw(ValueType type, std::size_t index) {
    return type == ValueType::Int || type == ValueType::Float
        ? makeContrastQuantitativeRaw(type, index)
        : makeContrastDescriptiveRaw(type, index);
}

// Pair contrast. Walks the same kPairs table as the mixed blocks so the added
// occurrences stay balanced across types. Combinations containing ENTITY get a
// named-actor voice: the person is tagged, the question about the object is not.
std::string makeContrastPairRaw(std::vector<ValueType> types, std::size_t index) {
    if (index % 2 != 0) std::reverse(types.begin(), types.end());

    const std::size_t spread = (index * 7) % 3072;
    const std::size_t pattern = spread % 12;
    const std::string& subject = kSubjects[(spread / 12) % 16];
    const std::string& stage = kStages[(spread / 192) % 16];

    const std::uint64_t mixed = mixIndex(index, 0x434f4e54524150ULL);
    const std::string& person = choose(kPersonRoles, mixed >> 8);
    const std::string& thing = choose(kThingRoles, mixed >> 16);
    const std::string& unnamed = choose(kUnnamedClauses, mixed >> 24);
    const std::string& tail_unnamed = otherUnnamedClause(unnamed, mixed >> 48);
    const std::string& open_question = choose(kOpenQuestions, mixed >> 32);
    const std::string& noun = choose(kCountNouns, mixed >> 40);
    const std::string& container = choose(kContainers, mixed >> 44);

    std::vector<MixedField> parts;
    parts.reserve(types.size());
    for (std::size_t position = 0; position < types.size(); ++position) {
        const std::size_t value_index = 500000 + index * 3 + position;
        parts.push_back(MixedField{
            types[position],
            types[position] == ValueType::Entity
                ? entitySingleField(400000 + value_index, index + position)
                : fieldFor(types[position], index + position * 5),
            contrastAtom(types[position], value_index),
        });
    }

    const auto finalize = [&](std::string text) {
        switch ((mixed >> 52) % 4ULL) {
            case 0:
                return text + " The " + subject + " remains in " + stage +
                    " until " + tail_unnamed + ".";
            case 1:
                return text + " This " + subject + " entry was written during " +
                    stage + ".";
            case 2:
                return text + " " + capitalize(person) + " will close the " +
                    subject + " after " + stage + ".";
            default:
                return text + " The " + subject + " stays in " + stage +
                    " while " + thing + " is reviewed.";
        }
    };

    const auto entity_part = std::find_if(
        parts.begin(), parts.end(), [](const MixedField& part) {
            return part.type == ValueType::Entity;
        });

    if (entity_part == parts.end()) {
        const MixedField& one = parts.front();
        const MixedField& two = parts.back();
        switch (pattern) {
            case 0:
                return finalize("The " + subject + " report shows " + one.field +
                    "=" + one.atom + " and " + two.field + "=" + two.atom +
                    ", but " + unnamed + ". " + open_question + "?");
            case 1:
                return finalize(open_question + "? For the " + subject + ", " +
                    one.field + " is " + one.atom + " and " + two.field + " is " +
                    two.atom + ".");
            case 2:
                return finalize("Set " + one.field + " to " + one.atom + " and " +
                    two.field + " to " + two.atom + "; leave the " + subject +
                    " with " + person + ".");
            case 3:
                return finalize("The " + subject + " carries " + one.field + ": " +
                    one.atom + " and " + two.field + ": " + two.atom + ", while " +
                    thing + " is only described.");
            case 4:
                return finalize("Neither " + person + " nor " + thing +
                    " is a name, yet both appear beside " + one.field + "=" +
                    one.atom + " and " + two.field + "=" + two.atom + ".");
            case 5:
                return finalize("Check " + one.field + " (" + one.atom + ") and " +
                    two.field + " (" + two.atom + ") on the " + subject + "; " +
                    unnamed + ".");
            case 6:
                return finalize(capitalize(person) + " reviewed the " + subject +
                    " with " + one.field + "=" + one.atom + " and " + two.field +
                    "=" + two.atom + ". " + open_question + "?");
            case 7:
                return finalize("During " + stage + ", the " + subject +
                    " kept " + one.field + " at " + one.atom + " and " +
                    two.field + " at " + two.atom + ".");
            case 8:
                return finalize("A " + container + " was tagged with " + one.field +
                    "=" + one.atom + " and " + two.field + "=" + two.atom + ". " +
                    open_question + "?");
            case 9:
                return finalize("The " + subject + " belongs to " + thing +
                    ", and its " + one.field + " is " + one.atom + " with " +
                    two.field + " at " + two.atom + ".");
            case 10:
                return finalize(open_question + "? Nothing here is a name: " +
                    one.field + " is " + one.atom + " and " + two.field + " is " +
                    two.atom + ".");
            default:
                return finalize("Hand the " + subject + " to " + person +
                    " once " + one.field + "=" + one.atom + " and " + two.field +
                    "=" + two.atom + " are confirmed.");
        }
    }

    const std::string named = entity_part->atom;
    const MixedField& other = &parts.front() == &*entity_part
        ? parts.back() : parts.front();
    const bool numeric =
        other.type == ValueType::Int || other.type == ValueType::Float;

    if (numeric && pattern < 8) {
        const std::string question = countQuestion(noun, mixed >> 36);
        switch (pattern) {
            case 0:
                return finalize(named + " loaded " + other.atom + " " + noun +
                    " into the " + container + ". " + question + "?");
            case 1:
                return finalize("A " + container + " logged by " + named +
                    " held " + other.atom + " " + noun + ". " + question + "?");
            case 2:
                return finalize(question + "? " + named + " counted " +
                    other.atom + " " + noun + " in the " + container + ".");
            case 3:
                return finalize(named + " signed for " + other.atom + " " + noun +
                    " at the " + container + ", but " + unnamed + ". " +
                    question + "?");
            case 4:
                return finalize("The " + container + " was left with " +
                    other.atom + " " + noun + " after " + named +
                    " finished the round. " + question + "?");
            case 5:
                return finalize(named + " reported " + other.atom + " " + noun +
                    " while " + person + " reported none. " + question + "?");
            case 6:
                return finalize("Only " + named + " is a name in this note: the " +
                    container + " held " + other.atom + " " + noun + ", and " +
                    unnamed + ". " + question + "?");
            default:
                return finalize("After " + named + " emptied the " + container +
                    ", " + other.atom + " " + noun + " were still on the floor. " +
                    question + "?");
        }
    }

    switch (pattern) {
        case 0:
            return finalize(named + " updated " + other.field + " to " +
                other.atom + " for the " + subject + ". " + open_question + "?");
        case 1:
            return finalize("According to " + named + ", the " + subject +
                " now carries " + other.field + "=" + other.atom + ", though " +
                unnamed + ".");
        case 2:
            return finalize(open_question + "? " + named + " set " + other.field +
                " to " + other.atom + " while " + person + " watched.");
        case 3:
            return finalize("The " + subject + " passed from " + person + " to " +
                named + " with " + other.field + " at " + other.atom + ".");
        case 4:
            return finalize(named + " is named on the " + subject + "; " + person +
                " is not. The " + other.field + " remains " + other.atom + ".");
        case 5:
            return finalize("Neither " + person + " nor " + thing + " matches " +
                named + ", and " + other.field + " stays " + other.atom + ".");
        case 6:
            return finalize("Ask " + named + ", not " + person + ", why " +
                other.field + " is " + other.atom + " on the " + subject + ".");
        case 7:
            return finalize("The " + subject + " credits " + named + " with " +
                other.field + "=" + other.atom + " even though " + unnamed + ".");
        case 8:
            return finalize(open_question + "? Only " + named +
                " is a name here; " + thing + " is a description, and " +
                other.field + " is " + other.atom + ".");
        case 9:
            return finalize("During " + stage + ", " + named + " changed " +
                other.field + " to " + other.atom + " on behalf of " + person + ".");
        case 10:
            return finalize("Record " + named + " as the responsible party and " +
                other.atom + " as " + other.field + " for the " + subject + ".");
        default:
            return finalize(named + " handled the " + subject + " and left " +
                other.field + " at " + other.atom + ". " + open_question + "?");
    }
}

// Pure negatives: ordinary English, every participant described rather than
// named, and deliberately free of numerals so that nothing in the block is a
// candidate for any delimiter.
std::string makeNegativeOnlyRaw(std::size_t index) {
    const std::size_t spread = (index * 617) % 4096;
    const std::size_t pattern = spread % 16;
    const std::string& subject = kSubjects[(spread / 16) % 16];
    const std::string& person = kPersonRoles[(spread / 256) % 16];

    const std::uint64_t mixed = mixIndex(index, 0x4e45474154495645ULL);
    const std::string& thing = choose(kThingRoles, mixed >> 8);
    const std::string& unnamed = choose(kUnnamedClauses, mixed >> 16);
    const std::string& open_question = choose(kOpenQuestions, mixed >> 24);
    const std::string& noun = choose(kCountNouns, mixed >> 32);
    const std::string& container = choose(kContainers, mixed >> 40);

    switch (pattern) {
        case 0:
            return capitalize(unnamed) + ", so the " + subject + " lists " +
                person + " instead. " + open_question + "?";
        case 1:
            return "The " + subject + " mentions " + person + " and " + thing +
                ", but " + unnamed + ".";
        case 2:
            return open_question + "? The " + subject + " points at " + person +
                ", which describes a role rather than naming anyone.";
        case 3:
            return "Hand the " + subject + " to " + person + "; " + thing +
                " has no name on record.";
        case 4:
            return "Here the object in question is the " + noun +
                " count, not a named party, and the " + subject +
                " still waits on " + person + ".";
        case 5:
            return "No proper name appears on this " + subject + ": only " +
                person + " and " + thing + " are referenced.";
        case 6:
            return capitalize(person) + " opened the " + subject + " while " +
                unnamed + ". " + open_question + "?";
        case 7:
            return "The " + subject + " belongs to " + thing + " for now, since " +
                person + " has not been identified.";
        case 8:
            return "Ask " + person + " about the " + subject + " before the " +
                container + " is sealed; " + unnamed + ".";
        case 9:
            return open_question + "? Neither " + person + " nor " + thing +
                " is a name, and the " + subject + " has none either.";
        case 10:
            return "Every party on this " + subject +
                " is described rather than named, from " + person + " to " +
                thing + ".";
        case 11:
            return "The " + noun + " in the " + container + " were counted by " +
                person + ", and the " + subject + " records no name.";
        case 12:
            return capitalize(thing) + " raised the " + subject + ", though " +
                person + " signed for it and " + unnamed + ".";
        case 13:
            return "Leave the " + subject + " with " + person +
                " until a real name replaces " + thing + ".";
        case 14:
            return "This " + subject + " note refers to " + person +
                " throughout; " + unnamed + ".";
        default:
            return "Who handles the " + subject + "? For now, " + person +
                ", because " + unnamed + ".";
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
    std::array<int, kTypes.size()> occurrences{};
    std::array<int, kTypes.size()> singles{};
    int mixed_pairs = 0;
    int mixed_triples = 0;
    int mixed_all_types = 0;
    int contrast_same = 0;
    int contrast_pairs = 0;
    int negative_only = 0;
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
                type == ValueType::Entity
                    ? makeEntitySingleRaw(static_cast<std::size_t>(index))
                    : makeSingleRaw(type, static_cast<std::size_t>(index))));
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
    for (int index = 0; index < kMixedAllTypes; ++index) {
        ++mixed_id;
        generated.blocks.push_back(makeBlock(
            std::string(kBlockPrefix) + "mixed_" + zeroPadded(mixed_id),
            "Atom identification mixed all types " + zeroPadded(mixed_id),
            makeMixedRaw(all_types, mixed_index++)));
        addOccurrences(generated, all_types);
        ++generated.mixed_all_types;
    }

    int contrast_id = 0;
    for (const ValueType type : kTypes) {
        for (int index = 0; index < kContrastSamePerType; ++index) {
            ++contrast_id;
            generated.blocks.push_back(makeBlock(
                std::string(kBlockPrefix) + "contrast_" + zeroPadded(contrast_id),
                std::string("Atom identification contrast ") + typeName(type) +
                    " " + zeroPadded(contrast_id),
                makeContrastSameTypeRaw(type, static_cast<std::size_t>(index))));
            // Two atoms of the block's own type, so the family stays balanced.
            generated.occurrences[static_cast<std::size_t>(type)] += 2;
            ++generated.contrast_same;
        }
    }

    std::size_t contrast_pair_index = 0;
    for (const auto& combination : kPairs) {
        const std::vector<ValueType> types(combination.begin(), combination.end());
        for (int index = 0; index < kContrastPairPerCombination; ++index) {
            ++contrast_id;
            generated.blocks.push_back(makeBlock(
                std::string(kBlockPrefix) + "contrast_" + zeroPadded(contrast_id),
                "Atom identification contrast pair " + zeroPadded(contrast_id),
                makeContrastPairRaw(types, contrast_pair_index++)));
            addOccurrences(generated, types);
            ++generated.contrast_pairs;
        }
    }

    for (int index = 0; index < kNegativeOnlyBlocks; ++index) {
        generated.blocks.push_back(makeBlock(
            std::string(kBlockPrefix) + "negative_" + zeroPadded(index + 1),
            "Atom identification negative " + zeroPadded(index + 1),
            makeNegativeOnlyRaw(static_cast<std::size_t>(index))));
        ++generated.negative_only;
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

// The whole point of the v5 families is that described participants and
// interrogative clauses stay outside the delimiters. Enforce it rather than
// trusting the templates: any negative-inventory phrase that turns up inside an
// atom means a template was assembled wrong.
void validateNoDescribedPhrase(std::string_view content, const std::string& id) {
    const auto reject = [&](const std::string& phrase) {
        if (content.find(phrase) != std::string_view::npos) {
            throw std::runtime_error(
                id + ": atom content contains the described phrase \"" + phrase +
                "\", which must stay outside every delimiter");
        }
    };
    for (const auto& phrase : kPersonRoles) reject(phrase);
    for (const auto& phrase : kThingRoles) reject(phrase);
    for (const auto& phrase : kUnnamedClauses) reject(phrase);
    for (const auto& phrase : kOpenQuestions) reject(phrase);
}

std::array<int, kTypes.size()> validateAnnotatedRaw(const GRIM::ConceptBlock& block) {
    std::array<int, kTypes.size()> counts{};
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
            validateNoDescribedPhrase(content, block.id);
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
                case ValueType::Entity:
                    if (content.empty()) throw std::runtime_error(block.id + ": empty ENTITY");
                    if (std::any_of(content.begin(), content.end(), [](unsigned char byte) {
                            return byte > 0x7f;
                        })) {
                        throw std::runtime_error(
                            block.id + ": ENTITY surface must be ASCII in the English-focused curriculum");
                    }
                    if (content.front() == ' ' || content.front() == '\t' ||
                        content.front() == '\r' || content.front() == '\n' ||
                        content.back() == ' ' || content.back() == '\t' ||
                        content.back() == '\r' || content.back() == '\n') {
                        throw std::runtime_error(
                            block.id + ": ENTITY surface contains edge whitespace");
                    }
                    for (const std::string_view leaked_role :
                         std::array<std::string_view, 8>{
                             "Project ", "Repository ", "Document ", "Workspace ",
                             "Service ", "Customer ", "Team ", "Organization "}) {
                        if (content.starts_with(leaked_role)) {
                            throw std::runtime_error(
                                block.id + ": synthetic ENTITY surface leaks role prefix " +
                                std::string(leaked_role));
                        }
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
        throw std::runtime_error(
            "generated block count is not " + std::to_string(kExpectedBlockCount));
    }
    if (generated.mixed_pairs !=
            kMixedPairPerCombination * static_cast<int>(kPairs.size()) ||
        generated.mixed_triples !=
            kMixedTriplePerCombination * static_cast<int>(kTriples.size()) ||
        generated.mixed_all_types != kMixedAllTypes) {
        throw std::runtime_error("mixed distribution is incorrect");
    }
    if (generated.contrast_same !=
            kContrastSamePerType * static_cast<int>(kTypes.size()) ||
        generated.contrast_pairs !=
            kContrastPairPerCombination * static_cast<int>(kPairs.size()) ||
        generated.negative_only != kNegativeOnlyBlocks) {
        throw std::runtime_error("contrast distribution is incorrect");
    }

    std::unordered_set<std::string> ids;
    std::unordered_set<std::string> raws;
    ids.reserve(generated.blocks.size() * 2);
    raws.reserve(generated.blocks.size() * 2);
    std::array<int, kTypes.size()> audited_occurrences{};
    int entity_string_same_surface_contrasts = 0;
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

        if (block.id.find("_negative_") != std::string::npos) {
            const int tagged = std::accumulate(counts.begin(), counts.end(), 0);
            if (tagged != 0) {
                throw std::runtime_error(
                    block.id + ": negative-only block carries " +
                    std::to_string(tagged) + " atoms");
            }
            // A bare numeral in a negative block would teach the opposite of
            // what the family is for, so keep them numeral-free.
            if (std::any_of(block.raw.begin(), block.raw.end(), [](unsigned char byte) {
                    return byte >= '0' && byte <= '9';
                })) {
                throw std::runtime_error(
                    block.id + ": negative-only block contains an untagged numeral");
            }
        }

        const std::size_t entity_open = block.raw.find("<ENTITY>");
        const std::size_t string_open = block.raw.find("<STRING>");
        if (entity_open != std::string::npos && string_open != std::string::npos) {
            const std::size_t entity_begin = entity_open + std::string_view("<ENTITY>").size();
            const std::size_t entity_end = block.raw.find("</ENTITY>", entity_begin);
            const std::size_t string_begin = string_open + std::string_view("<STRING>").size();
            const std::size_t string_end = block.raw.find("</STRING>", string_begin);
            if (entity_end == std::string::npos || string_end == std::string::npos) {
                throw std::runtime_error(block.id + ": missing contrast close delimiter");
            }
            if (block.raw.substr(entity_begin, entity_end - entity_begin) ==
                block.raw.substr(string_begin, string_end - string_begin)) {
                ++entity_string_same_surface_contrasts;
            }
        }
    }

    if (entity_string_same_surface_contrasts !=
        kExpectedEntityStringSameSurfaceContrasts) {
        throw std::runtime_error(
            "ENTITY/STRING same-surface contrast count is " +
            std::to_string(entity_string_same_surface_contrasts) +
            ", expected " +
            std::to_string(kExpectedEntityStringSameSurfaceContrasts));
    }

    for (std::size_t type = 0; type < kTypes.size(); ++type) {
        const int expected = expectedOccurrencesPerType(kTypes[type]);
        if (generated.singles[type] != kSinglePerType ||
            generated.occurrences[type] != expected ||
            audited_occurrences[type] != expected) {
            throw std::runtime_error(
                std::string("type balance failed for ") + typeName(kTypes[type]) +
                ": bookkeeping " + std::to_string(generated.occurrences[type]) +
                ", audited " + std::to_string(audited_occurrences[type]) +
                ", expected " + std::to_string(expected));
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

struct JsonlMergeStats {
    std::size_t kept = 0;
    std::size_t replaced = 0;
};

bool isAtomCurriculumJsonlEntry(const std::string& line) {
    if (line.find(kBlockPrefix) == std::string::npos &&
        line.find(kSourceId) == std::string::npos) {
        return false;
    }
    const json entry = json::parse(line);
    if (!entry.is_object()) return false;
    const std::string id = entry.value("id", std::string{});
    const std::string source_id = entry.value("source_sequence_id", std::string{});
    return startsWith(id, kBlockPrefix) || source_id == kSourceId;
}

JsonlMergeStats mergeCanonicalJsonl(
    const fs::path& path,
    const std::vector<GRIM::ConceptBlock>& blocks) {
    fs::path temporary = path;
    temporary += ".atom_curriculum.tmp";

    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) throw std::runtime_error("cannot open " + path.string());
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output.is_open()) throw std::runtime_error("cannot write " + temporary.string());

    JsonlMergeStats stats;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        try {
            if (isAtomCurriculumJsonlEntry(line)) {
                ++stats.replaced;
                continue;
            }
        } catch (const std::exception& error) {
            throw std::runtime_error(
                path.string() + ":" + std::to_string(line_number) +
                ": cannot inspect JSONL entry: " + error.what());
        }
        output << line << '\n';
        ++stats.kept;
    }
    if (!input.eof()) throw std::runtime_error("failed while reading " + path.string());

    for (const auto& block : blocks) output << blockJson(block).dump() << '\n';
    if (!output.good()) throw std::runtime_error("failed while writing " + temporary.string());
    output.close();
    input.close();
    replaceFile(temporary, path);
    return stats;
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
        {"generation_version", 5},
        {"deterministic", true},
        {"total_blocks", generated.blocks.size()},
        {"single_blocks", kSinglePerType * static_cast<int>(kTypes.size())},
        {"single_blocks_by_type", std::move(single_counts)},
        {"mixed_blocks", generated.mixed_pairs + generated.mixed_triples +
                         generated.mixed_all_types},
        {"mixed_pair_blocks", generated.mixed_pairs},
        {"mixed_triple_blocks", generated.mixed_triples},
        {"mixed_all_types_blocks", generated.mixed_all_types},
        {"contrast_blocks", generated.contrast_same + generated.contrast_pairs},
        {"contrast_same_type_blocks", generated.contrast_same},
        {"contrast_pair_blocks", generated.contrast_pairs},
        {"negative_only_blocks", generated.negative_only},
        {"atom_occurrences_by_type", std::move(occurrence_counts)},
        {"format_type", "raw"},
        {"training_stage", "pt"},
        {"model_input_encoding", "exact_utf8_bytes"},
        {"entity_role_outside_tag", true},
        {"entity_surface_language", "English"},
        {"entity_surface_encoding", "ASCII"},
        {"entity_single_template_count", 48},
        {"mixed_template_count", 24},
        {"contrast_quantitative_template_count", 12},
        {"contrast_descriptive_template_count", 12},
        {"contrast_pair_template_count", 12},
        {"negative_only_template_count", 16},
        {"negative_phrase_pool", {
            {"person_roles", static_cast<int>(kPersonRoles.size())},
            {"thing_roles", static_cast<int>(kThingRoles.size())},
            {"unnamed_clauses", static_cast<int>(kUnnamedClauses.size())},
            {"open_questions", static_cast<int>(kOpenQuestions.size())},
            {"count_nouns", static_cast<int>(kCountNouns.size())},
            {"containers", static_cast<int>(kContainers.size())},
        }},
        {"described_phrases_outside_tag", true},
        {"entity_string_same_surface_contrasts",
         kExpectedEntityStringSameSurfaceContrasts},
    };
}

fs::path parseDataDirectory(int argc, char** argv) {
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) == "--data-dir" && index + 1 < argc) {
            return fs::absolute(argv[index + 1]).lexically_normal();
        }
    }
    throw std::runtime_error(
        "usage: atom_curriculum_builder --data-dir <training-data-directory> "
        "[--preview-out <directory>]");
}

// Preview writes only the standalone curriculum JSONL and its manifest, leaving
// concept_blocks.{jsonl,fb} and the registry untouched. Useful for eyeballing a
// generation change before committing it to the canonical dataset.
fs::path parsePreviewOut(int argc, char** argv) {
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) == "--preview-out" && index + 1 < argc) {
            return fs::absolute(argv[index + 1]).lexically_normal();
        }
    }
    return {};
}

} // namespace

int main(int argc, char** argv) {
    try {
        const fs::path preview_directory = parsePreviewOut(argc, argv);
        if (!preview_directory.empty()) {
            fs::create_directories(preview_directory);
            GeneratedCurriculum preview = generateCurriculum();
            auditGenerated(preview);
            writeJsonl(preview_directory / "atom_identification_v1.jsonl",
                       preview.blocks);
            writeJsonAtomic(
                preview_directory / "atom_identification_v1_manifest.json",
                makeManifest(preview));
            std::cout << "Preview: generated and audited " << preview.blocks.size()
                      << " blocks into " << preview_directory.string()
                      << " (canonical dataset untouched)\n";
            return 0;
        }

        const fs::path data_directory = parseDataDirectory(argc, argv);
        const fs::path flatbuffer_path = data_directory / "concept_blocks.fb";
        const fs::path jsonl_path = data_directory / "concept_blocks.jsonl";
        const fs::path registry_path = data_directory / "curriculum_registry.json";
        if (!fs::is_regular_file(flatbuffer_path) || !fs::is_regular_file(jsonl_path) ||
            !fs::is_regular_file(registry_path)) {
            throw std::runtime_error(
                "data directory lacks concept_blocks.jsonl, concept_blocks.fb, or curriculum_registry.json");
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

        const JsonlMergeStats jsonl_stats = mergeCanonicalJsonl(jsonl_path, generated.blocks);
        if (!GRIM::ConceptBlockIO::saveFlatBuffer(flatbuffer_path, existing, &io_error)) {
            throw std::runtime_error("failed to save concept blocks: " + io_error);
        }
        updateRegistry(registry_path, generated.blocks);
        writeJsonl(data_directory / "atom_identification_v1.jsonl", generated.blocks);
        writeJsonAtomic(
            data_directory / "atom_identification_v1_manifest.json",
            makeManifest(generated));

        std::cout << "Replaced prior curriculum blocks: " << replaced << '\n'
                  << "Canonical JSONL kept/replaced/appended: "
                  << jsonl_stats.kept << '/' << jsonl_stats.replaced << '/'
                  << generated.blocks.size() << '\n'
                  << "Concept-block total: " << existing.size() << '\n'
                  << "Curriculum: " << kCurriculumName << " (" << kCurriculumId << ")\n"
                  << "Contrast blocks (same-type/pair): "
                  << generated.contrast_same << '/'
                  << generated.contrast_pairs << '\n'
                  << "Negative-only blocks: " << generated.negative_only
                  << '\n'
                  << "Per-type occurrences: "
                  << expectedOccurrencesPerType(ValueType::Int) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "atom_curriculum_builder: " << error.what() << '\n';
        return 1;
    }
}

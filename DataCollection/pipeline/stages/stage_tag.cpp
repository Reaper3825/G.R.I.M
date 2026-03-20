#include "stage_tag.hpp"
#include "../pipeline_context.hpp"
#include "../chunk_spool.hpp"

#include <nlohmann/json.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace GRIM {
namespace Pipeline {

namespace fs = std::filesystem;
using json = nlohmann::json;

// Deterministic ID from content hash (FNV-1a 128-bit → hex)
static std::string generateEntryId(const std::string& content) {
    uint64_t h1 = 14695981039346656037ULL;
    uint64_t h2 = 14695981039346656037ULL;
    for (size_t i = 0; i < content.size(); ++i) {
        uint8_t c = static_cast<uint8_t>(content[i]);
        h1 ^= c;
        h1 *= 1099511628211ULL;
        if (i + 1 < content.size()) {
            h2 ^= static_cast<uint8_t>(content[i + 1]);
            h2 *= 1099511628211ULL;
        }
    }
    std::ostringstream oss;
    oss << std::hex << std::setfill('0')
        << std::setw(16) << h1
        << std::setw(16) << h2;
    return oss.str();
}

static std::string classifyQualityTier(float reliabilityScore) {
    if (reliabilityScore >= 0.8f) return "high";
    if (reliabilityScore >= 0.5f) return "medium";
    return "low";
}

static std::string classifySubject(const std::string& content) {
    // Keyword-based heuristic classification; LLM-powered version later.
    struct SubjectKeywords {
        const char* subject;
        std::vector<std::string> keywords;
    };

    static const SubjectKeywords subjects[] = {
        {"code",    {"function", "class ", "def ", "import ", "return ", "#include",
                     "public:", "private:", "void ", "int ", "std::"}},
        {"math",    {"equation", "theorem", "proof", "integral", "derivative",
                     "matrix", "vector", "polynomial", "calculus"}},
        {"science", {"experiment", "hypothesis", "molecule", "atom", "quantum",
                     "physics", "chemistry", "biology", "genome", "evolution"}},
        {"history", {"century", "civilization", "empire", "dynasty", "revolution",
                     "ancient", "medieval", "colonial", "war "}},
        {"medical", {"patient", "diagnosis", "symptom", "treatment", "clinical",
                     "disease", "therapy", "surgical", "medication"}},
        {"legal",   {"statute", "jurisdiction", "plaintiff", "defendant",
                     "court", "amendment", "regulation", "contract"}},
    };

    std::string lower = content;
    for (auto& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    int bestScore = 0;
    const char* bestSubject = "general";

    for (const auto& s : subjects) {
        int score = 0;
        for (const auto& kw : s.keywords) {
            if (lower.find(kw) != std::string::npos) score++;
        }
        if (score > bestScore) {
            bestScore = score;
            bestSubject = s.subject;
        }
    }

    return (bestScore >= 2) ? bestSubject : "general";
}

static std::vector<std::string> autoTags(const std::string& sourceType,
                                          const std::string& qualityTier,
                                          const std::string& subject) {
    std::vector<std::string> tags;
    if (!sourceType.empty()) tags.push_back("source:" + sourceType);
    if (!qualityTier.empty()) tags.push_back("quality:" + qualityTier);
    if (!subject.empty() && subject != "general") tags.push_back("subject:" + subject);
    return tags;
}

StageResult StageTag::execute(PipelineContext& ctx) {
    auto startTime = std::chrono::steady_clock::now();
    StageResult result;

    std::cout << "=== TAGGING ===\n\n";

    ChunkSpool spool(ctx.run.spoolRoot);
    const std::string outputSpool = "tag";
    size_t outputChunkIdx = 0;

    auto reportProgress = [&](float p) {
        if (ctx.onProgress) ctx.onProgress(PipelineState::Tag, p, "tagging");
    };

    auto now = std::chrono::system_clock::now();
    int64_t timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();

    size_t totalTagged = 0;
    size_t totalChunks = ctx.preprocessCursor.chunkFiles.size();
    size_t chunksProcessed = 0;

    for (const auto& chunkFile : ctx.preprocessCursor.chunkFiles) {
        if (ctx.stopRequested.load()) {
            result.success = false;
            result.errorMessage = "Stopped by user";
            return result;
        }

        std::vector<TaggedEntry> tagged;
        {
            std::ifstream file(chunkFile);
            std::string line;
            while (std::getline(file, line)) {
                if (line.empty()) continue;
                try {
                    json j = json::parse(line);
                    std::string content = j.value("content", std::string());
                    if (content.empty()) continue;

                    TaggedEntry entry;
                    entry.content          = std::move(content);
                    entry.id               = generateEntryId(entry.content);
                    entry.sourceUrl        = j.value("source_url", std::string());
                    entry.sourceType       = j.value("source_type", std::string("unknown"));
                    entry.reliabilityScore = j.value("reliability_score", 0.7f);
                    entry.qualityTier      = classifyQualityTier(entry.reliabilityScore);
                    entry.subject          = classifySubject(entry.content);
                    entry.tags             = autoTags(entry.sourceType, entry.qualityTier, entry.subject);
                    entry.timestamp        = timestamp;

                    tagged.push_back(std::move(entry));
                } catch (...) { continue; }
            }
        }

        if (!tagged.empty()) {
            fs::path outFile = spool.createChunkFile(outputSpool, outputChunkIdx++);
            std::ofstream out(outFile, std::ios::trunc);
            for (const auto& entry : tagged) {
                json j;
                j["id"]                = entry.id;
                j["content"]           = entry.content;
                j["source_url"]        = entry.sourceUrl;
                j["source_type"]       = entry.sourceType;
                j["quality_tier"]      = entry.qualityTier;
                j["subject"]           = entry.subject;
                j["reliability_score"] = entry.reliabilityScore;
                j["timestamp"]         = entry.timestamp;
                j["tags"]              = entry.tags;
                out << j.dump() << "\n";
            }
            totalTagged += tagged.size();
        }

        chunksProcessed++;
        if (totalChunks > 0) {
            reportProgress(static_cast<float>(chunksProcessed) / static_cast<float>(totalChunks) * 100.0f);
        }
    }

    ctx.stats.entriesTagged = totalTagged;

    ctx.tagCursor.stageName = outputSpool;
    ctx.tagCursor.chunkFiles = spool.enumerateChunks(outputSpool);
    ctx.tagCursor.nextChunk = 0;

    std::cout << "  Tagged: " << totalTagged << " entries\n\n";

    auto elapsed = std::chrono::steady_clock::now() - startTime;
    result.durationSeconds = std::chrono::duration<float>(elapsed).count();
    std::cout << "[Tag] Complete (" << result.durationSeconds << "s)\n\n";
    return result;
}

} // namespace Pipeline
} // namespace GRIM

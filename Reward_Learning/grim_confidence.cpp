#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include "grim_confidence.hpp"
#include "logger.hpp"

float refresh_rate = 1000.0f; // in milliseconds

namespace GRIM {

    namespace Confidence{
namespace {
    constexpr float kConfidenceMin = 0.0f;
    constexpr float kConfidenceMax = 1.0f;

    struct ChannelInfo {
        float value;
        bool active;
        const char* label;
    };

    std::array<ChannelInfo, 8> buildChannelInfo(const GC& con)
    {
        return {{
            {con.txt_conf, con.active_contexts.a_text, "Text"},
            {con.vis_conf, con.active_contexts.a_visual, "Visual"},
            {con.pvs_conf, con.active_contexts.a_pvisual, "PhysicalVisual"},
            {con.nlp_conf, con.active_contexts.a_nlp, "NLP"},
            {con.aud_conf, con.active_contexts.a_audio, "Audio"},
            {con.emt_conf, con.active_contexts.a_emotion, "Emotion"},
            {con.mem_conf, con.active_contexts.a_memory, "Memory"},
            {con.ctx_conf, con.active_contexts.a_context, "Context"},
        }};
    }

    struct AggregateStats {
        float sum = 0.0f;
        std::size_t count = 0;
    };

    AggregateStats computeActiveStats(const GC& con)
    {
        AggregateStats stats;
        for (const auto& channel : buildChannelInfo(con))
        {
            if (channel.active)
            {
                stats.sum += channel.value;
                ++stats.count;
            }
        }
        return stats;
    }

    std::string joinTags(const std::vector<std::string>& tags)
    {
        if (tags.empty())
            return "[]";

        std::ostringstream oss;
        oss << "[";
        for (std::size_t i = 0; i < tags.size(); ++i)
        {
            if (i > 0)
                oss << ", ";
            oss << tags[i];
        }
        oss << "]";
        return oss.str();
    }
} // namespace

    GC CGC; // Global Confidence struct Define
    void zeroConfidence(GC& con) {
        con.txt_conf = 0.0f;
        con.vis_conf = 0.0f;
        con.pvs_conf = 0.0f;
        con.nlp_conf = 0.0f;
        con.aud_conf = 0.0f;
        con.emt_conf = 0.0f;
        con.mem_conf = 0.0f;
        con.ctx_conf = 0.0f;
    }


void logGrimState(const GC& con)
{
    std::ostringstream oss;
    oss << "Confidence snapshot => ";
    for (const auto& channel : buildChannelInfo(con))
    {
        oss << channel.label << "=" << channel.value
            << (channel.active ? "(active)" : "(inactive)") << " ";
    }

    oss << "tags=" << joinTags(con.context_tags)
        << " environment=" << joinTags(con.environment_tags);

    LOG_DEBUG("GrimConfidence", oss.str());
}

void offsetConfidence(float& base, float offset)
{
    base = std::clamp(base + offset, kConfidenceMin, kConfidenceMax);
}

float computeAverageConfidence(const GC& con)
{
    const auto channels = buildChannelInfo(con);
    const float sum = std::accumulate(
        channels.begin(), channels.end(), 0.0f,
        [](float acc, const ChannelInfo& ch) { return acc + ch.value; });

    return channels.empty() ? 0.0f : sum / static_cast<float>(channels.size());
}

float computeMeanConfidence(const GC& con)
{
    const AggregateStats stats = computeActiveStats(con);
    if (stats.count == 0)
        return 0.0f;

    return stats.sum / static_cast<float>(stats.count);
}

float computeMaxConfidence(const GC& con)
{
    const auto channels = buildChannelInfo(con);
    const auto it = std::max_element(
        channels.begin(), channels.end(),
        [](const ChannelInfo& a, const ChannelInfo& b) { return a.value < b.value; });

    return (it != channels.end()) ? it->value : 0.0f;
}

float computeMinConfidence(const GC& con)
{
    const auto channels = buildChannelInfo(con);
    const auto it = std::min_element(
        channels.begin(), channels.end(),
        [](const ChannelInfo& a, const ChannelInfo& b) { return a.value < b.value; });

    return (it != channels.end()) ? it->value : 0.0f;
}

float computeGrimConfidence(const GC& con)
{
    const AggregateStats stats = computeActiveStats(con);
    float confidence = (stats.count > 0)
        ? stats.sum / static_cast<float>(stats.count)
        : computeAverageConfidence(con);

    // Provide a small boost for each context/environment tag to reflect richer awareness.
    const float tagBonus = 0.01f * static_cast<float>(
        con.context_tags.size() + con.environment_tags.size());

    confidence = std::clamp(confidence + tagBonus, kConfidenceMin, kConfidenceMax);
    return confidence;
}

float getGrimConfidence(const GC& con)
{
    return computeGrimConfidence(con);
}

void shutdownGrimConfidence() {
    gconfidence_running = false;
    LOG_DEBUG("GrimConfidence", "GRIM Confidence system stopped.");

}


//Startup function
void startupGrimConfidence() {
    gconfidence_running = true;
    LOG_DEBUG("GrimConfidence", "GRIM Confidence system started.");
    zeroConfidence(CGC);



    //=========================================================
    //                   Confidence Loop 
    //=========================================================
    while (gconfidence_running) {
        computeGrimConfidence(CGC);
        




        std::this_thread::sleep_for(
            std::chrono::milliseconds(static_cast<int64_t>(refresh_rate)));
    }

    shutdownGrimConfidence();
    
}



} // namespace Confidence

} // namespace GRIM

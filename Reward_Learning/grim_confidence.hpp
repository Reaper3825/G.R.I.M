#pragma once
#include <string>
#include <vector>

// GRIM's internal representation of its state, this payload is what drives every response from GRIM, 
namespace GRIM {

    namespace Confidence{

    bool gconfidence_running = false;
        // What modalities should
    enum class Modality {
        Text,
        Visual,
        PhysicalVisual,
        NLP,
        Audio,
        Emotion,
        Memory,
        Context
    };

    // context activation flags (should use confidence or should not)
    struct AC {
        bool a_audio = true;
        bool a_visual = true;
        bool a_text = true;
        bool a_context = true;
        bool a_emotion = true;
        bool a_memory = true;
        bool a_pvisual = true;
        bool a_nlp = true;
    };
    
    struct GC {
        float txt_conf = 0.0f;
        float vis_conf = 0.0f;
        float pvs_conf = 0.0f;
        float nlp_conf = 0.0f;
        float aud_conf = 0.0f;
        float emt_conf = 0.0f;
        float mem_conf = 0.0f;
        float ctx_conf = 0.0f;
        AC active_contexts{}; 
        std::vector<std::string> context_tags; // e.g., "coding", "gaming", etc.
        std::vector<std::string> environment_tags; // e.g., "online", "app", "crowded", "loud", "low light" etc.
    };

    void startupGrimConfidence();
    void shutdownGrimConfidence();
    void logGrimState(const GC& con);
    void offsetConfidence(float& base, float offset);
    void zeroConfidence(GC& con);
    float computeGrimConfidence(const GC& con);
    float getGrimConfidence(const GC& con);
    float computeAverageConfidence(const GC& con);
    float computeMeanConfidence(const GC& con);
    float computeMaxConfidence(const GC& con);
    float computeMinConfidence(const GC& con);
    

} // namespace Confidence

} // namespace GRIM

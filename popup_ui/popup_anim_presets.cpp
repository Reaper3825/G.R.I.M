#include "popup_anim_presets.hpp"
#include "objects/popup_mesh_cache_loader.hpp"
#include "logger.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <cmath>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <algorithm>

// ===========================================================
// Popup animation preset / clip engine implementation
// ===========================================================

namespace
{
float clamp01(float t) { return t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t); }

float applyEasing(PopupEasing e, float t)
{
    t = clamp01(t);
    switch (e)
    {
        case PopupEasing::Linear:
            return t;
        case PopupEasing::EaseInCubic:
            return t * t * t;
        case PopupEasing::EaseOutCubic:
        {
            float f = t - 1.0f;
            return f * f * f + 1.0f;
        }
        case PopupEasing::EaseInOutCubic:
            if (t < 0.5f) return 4.0f * t * t * t;
            else { float f = 2.0f * t - 2.0f; return 0.5f * f * f * f + 1.0f; }
        case PopupEasing::EaseOutBack:
        {
            const float c1 = 1.70158f, c3 = c1 + 1.0f;
            float f = t - 1.0f;
            return 1.0f + c3 * f * f * f + c1 * f * f;
        }
        case PopupEasing::EaseInBack:
        {
            const float c1 = 1.70158f, c3 = c1 + 1.0f;
            return c3 * t * t * t - c1 * t * t;
        }
    }
    return t;
}

float evalTween(const PopupTween& tw, float nt, float fallback)
{
    if (!tw.enabled) return fallback;
    return tw.from + (tw.to - tw.from) * applyEasing(tw.easing, nt);
}

PopupEasing easingFromString(const std::string& s)
{
    if (s == "linear")          return PopupEasing::Linear;
    if (s == "easeInCubic")     return PopupEasing::EaseInCubic;
    if (s == "easeOutCubic")    return PopupEasing::EaseOutCubic;
    if (s == "easeInOutCubic")  return PopupEasing::EaseInOutCubic;
    if (s == "easeOutBack")     return PopupEasing::EaseOutBack;
    if (s == "easeInBack")      return PopupEasing::EaseInBack;
    return PopupEasing::EaseOutCubic;
}
} // namespace

struct PopupClipEngine
{
    std::vector<PopupAnimPreset> presets;
    std::unordered_map<std::string, int> nameToIndex;

    // Mesh caches keyed by .gmc filename (preset.meshCache).
    std::unordered_map<std::string, PopupMeshCache> caches;

    // Cross-thread playback request (UI thread -> render thread).
    std::atomic<uint32_t> requestToken{ 0 };
    std::atomic<int>      requestIdx{ -1 };

    // Render-thread-only playback state.
    uint32_t lastToken = 0;
    int      curIdx    = -1;
    double   startTime = 0.0;

    void rebuildIndex()
    {
        nameToIndex.clear();
        for (int i = 0; i < static_cast<int>(presets.size()); ++i)
            nameToIndex[presets[i].name] = i;
    }
};

PopupClipEngine* popupClipEngineCreate()
{
    return new PopupClipEngine();
}

void popupClipEngineDestroy(PopupClipEngine* engine)
{
    delete engine;
}

void popupClipEngineLoadDefaults(PopupClipEngine* engine)
{
    if (!engine) return;
    engine->presets.clear();
    engine->caches.clear();

    // ---- load_in: scale-pop + fade-in + brief emissive flash ----
    {
        PopupAnimPreset p;
        p.name        = "load_in";
        p.durationSec = 0.45f;
        p.loop        = false;
        p.holdLastValue = true;
        p.alpha    = { true, 0.0f, 1.0f, PopupEasing::EaseOutCubic };
        p.scale    = { true, 0.8f, 1.0f, PopupEasing::EaseOutBack };
        p.emissive = { true, 0.5f, 0.0f, PopupEasing::EaseOutCubic };
        engine->presets.push_back(p);
    }
    // ---- fade_out: fade + slight shrink ----
    {
        PopupAnimPreset p;
        p.name        = "fade_out";
        p.durationSec = 0.30f;
        p.loop        = false;
        p.holdLastValue = true;
        p.alpha = { true, 1.0f, 0.0f, PopupEasing::EaseInOutCubic };
        p.scale = { true, 1.0f, 0.92f, PopupEasing::EaseInCubic };
        engine->presets.push_back(p);
    }
    // ---- idle_spin: gentle continuous spin (opt-in) ----
    {
        PopupAnimPreset p;
        p.name        = "idle_spin";
        p.durationSec = 6.0f;
        p.loop        = true;
        p.spinTurns   = 1.0f;
        engine->presets.push_back(p);
    }

    engine->rebuildIndex();
}

bool popupClipEngineLoadJson(PopupClipEngine* engine,
                             const std::string& jsonPath,
                             const std::string& popup3dDir,
                             uint32_t defaultColorABGR)
{
    if (!engine) return false;

    std::ifstream f(jsonPath);
    if (!f.is_open())
    {
        LOG_DEBUG("Popup3D", "Anim presets JSON not found, using defaults: " + jsonPath);
        return false;
    }

    nlohmann::json root;
    try { f >> root; }
    catch (const std::exception& e)
    {
        LOG_ERROR("Popup3D", std::string("Anim presets JSON parse error: ") + e.what());
        return false;
    }

    if (!root.contains("presets") || !root["presets"].is_array())
    {
        LOG_ERROR("Popup3D", "Anim presets JSON missing 'presets' array");
        return false;
    }

    std::vector<PopupAnimPreset> parsed;

    auto parseTween = [](const nlohmann::json& j, PopupTween& tw)
    {
        tw.enabled = true;
        tw.from   = j.value("from", 0.0f);
        tw.to     = j.value("to", 1.0f);
        tw.easing = easingFromString(j.value("easing", std::string("easeOutCubic")));
    };

    for (const auto& jp : root["presets"])
    {
        PopupAnimPreset p;
        p.name          = jp.value("name", std::string());
        if (p.name.empty()) continue;
        p.durationSec   = jp.value("duration", 0.45f);
        p.loop          = jp.value("loop", false);
        p.holdLastValue = jp.value("holdLastValue", true);
        p.spinTurns     = jp.value("spinTurns", 0.0f);
        p.meshCache     = jp.value("meshCache", std::string());

        if (jp.contains("alpha")    && jp["alpha"].is_object())    parseTween(jp["alpha"],    p.alpha);
        if (jp.contains("scale")    && jp["scale"].is_object())    parseTween(jp["scale"],    p.scale);
        if (jp.contains("emissive") && jp["emissive"].is_object()) parseTween(jp["emissive"], p.emissive);

        parsed.push_back(std::move(p));
    }

    if (parsed.empty())
    {
        LOG_ERROR("Popup3D", "Anim presets JSON contained no valid presets");
        return false;
    }

    engine->presets = std::move(parsed);
    engine->caches.clear();

    // Preload referenced mesh caches.
    for (auto& p : engine->presets)
    {
        if (p.meshCache.empty()) continue;
        if (engine->caches.count(p.meshCache)) continue;
        std::string full = popup3dDir + "/" + p.meshCache;
        try
        {
            engine->caches[p.meshCache] = loadPopupMeshCache(full, defaultColorABGR);
            LOG_DEBUG("Popup3D", "Loaded mesh cache: " + full + " (" +
                      std::to_string(engine->caches[p.meshCache].frames.size()) + " frames)");
        }
        catch (const std::exception& e)
        {
            LOG_ERROR("Popup3D", std::string("Failed to load mesh cache '") + full + "': " + e.what());
            // Keep the preset but drop the geometry reference so it still tweens.
            p.meshCache.clear();
        }
    }

    engine->rebuildIndex();
    return true;
}

void popupClipEngineTrigger(PopupClipEngine* engine, const char* presetName)
{
    if (!engine || !presetName) return;
    auto it = engine->nameToIndex.find(presetName);
    if (it == engine->nameToIndex.end())
    {
        LOG_DEBUG("Popup3D", std::string("Trigger: unknown preset '") + presetName + "'");
        return;
    }
    engine->requestIdx.store(it->second, std::memory_order_relaxed);
    engine->requestToken.fetch_add(1, std::memory_order_release);
}

bool popupClipEngineEvaluate(PopupClipEngine* engine, double nowSeconds, PopupClipEval& out)
{
    if (!engine) return false;

    // Pick up a new playback request.
    uint32_t tok = engine->requestToken.load(std::memory_order_acquire);
    if (tok != engine->lastToken)
    {
        engine->lastToken = tok;
        engine->curIdx    = engine->requestIdx.load(std::memory_order_relaxed);
        engine->startTime = nowSeconds;

        if (engine->curIdx >= 0 && engine->curIdx < static_cast<int>(engine->presets.size()))
        {
            const PopupAnimPreset& sp = engine->presets[engine->curIdx];
            if (!sp.meshCache.empty())
                LOG_DEBUG("Popup3D", "Playing preset '" + sp.name +
                          "' using mesh cache: " + sp.meshCache);
            else
                LOG_DEBUG("Popup3D", "Playing preset '" + sp.name +
                          "' (tween only, no mesh cache)");
        }
    }

    if (engine->curIdx < 0 || engine->curIdx >= static_cast<int>(engine->presets.size()))
        return false;

    const PopupAnimPreset& p = engine->presets[engine->curIdx];

    double elapsed = nowSeconds - engine->startTime;
    if (elapsed < 0.0) elapsed = 0.0;

    const float tweenDur = (p.durationSec > 0.0f) ? p.durationSec : 0.0001f;

    float nt;          // normalized tween time [0,1]
    float spinPhase;   // continuous phase for spin
    bool  finished = false;
    if (p.loop)
    {
        spinPhase = static_cast<float>(elapsed / tweenDur);
        nt = spinPhase - std::floor(spinPhase);
    }
    else
    {
        if (elapsed >= tweenDur) { nt = 1.0f; finished = true; }
        else                      nt = static_cast<float>(elapsed / tweenDur);
        spinPhase = nt;
    }

    out.active      = true;
    out.alphaMul    = evalTween(p.alpha,    nt, 1.0f);
    out.scaleMul    = evalTween(p.scale,    nt, 1.0f);
    out.emissiveAdd = evalTween(p.emissive, nt, 0.0f);
    out.spinY       = p.spinTurns * 6.28318530718f * spinPhase;
    out.finished    = finished;
    out.frame       = nullptr;

    // Geometry frame selection.
    //
    // The whole baked clip is spread evenly across the preset's 'duration', so
    // 'duration' is the single source of truth for playback speed and the mesh
    // stays in lockstep with the alpha/scale/emissive tweens. The baked fps only
    // determines how many frames exist (smoothness), not how fast they play.
    if (!p.meshCache.empty())
    {
        auto it = engine->caches.find(p.meshCache);
        if (it != engine->caches.end() && !it->second.empty())
        {
            const PopupMeshCache& cache = it->second;
            const int frameCount = static_cast<int>(cache.frames.size());

            int fi;
            if (p.loop)
            {
                // Map the looping phase [0,1) across every frame so one full clip
                // plays per 'duration' seconds.
                float phase = spinPhase - std::floor(spinPhase);
                long long m = static_cast<long long>(std::floor(phase * frameCount)) % frameCount;
                if (m < 0) m += frameCount;
                fi = static_cast<int>(m);
            }
            else
            {
                // nt sweeps 0->1 over 'duration'; spread frames across it and
                // hold the last frame once finished.
                fi = static_cast<int>(std::floor(nt * frameCount));
                if (fi >= frameCount) fi = frameCount - 1;
                if (fi < 0) fi = 0;
            }
            out.frame = &cache.frames[fi];
        }
    }

    return true;
}

void popupClipEngineMaxBuffer(const PopupClipEngine* engine,
                              uint32_t* maxVerts, uint32_t* maxIndices)
{
    uint32_t mv = 0, mi = 0;
    if (engine)
    {
        for (const auto& kv : engine->caches)
        {
            mv = std::max(mv, kv.second.maxVertices);
            mi = std::max(mi, kv.second.maxIndices);
        }
    }
    if (maxVerts)   *maxVerts   = mv;
    if (maxIndices) *maxIndices = mi;
}

bool popupClipEngineHasGeometry(const PopupClipEngine* engine)
{
    if (!engine) return false;
    return !engine->caches.empty();
}

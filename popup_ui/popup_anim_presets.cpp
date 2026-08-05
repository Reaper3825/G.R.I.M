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

PopupAnimType animTypeFromString(const std::string& s)
{
    if (s == "blendPose" || s == "pose") return PopupAnimType::BlendPose;
    return PopupAnimType::OneShot;
}

// Map elapsed seconds to a normalized tween time and a continuous spin phase.
// Looping presets can ping-pong (0->1->0 triangle) so a pose bob/pulse eases at
// both extremes instead of snapping back to the start.
void clipPhase(const PopupAnimPreset& p, double elapsed,
               float& nt, float& spinPhase, bool& finished)
{
    if (elapsed < 0.0) elapsed = 0.0;
    const float dur = (p.durationSec > 0.0f) ? p.durationSec : 0.0001f;
    finished = false;
    if (p.loop)
    {
        spinPhase = static_cast<float>(elapsed / dur);
        float f = spinPhase - std::floor(spinPhase);
        nt = p.pingPong ? (f < 0.5f ? f * 2.0f : 2.0f - f * 2.0f) : f;
    }
    else
    {
        if (elapsed >= dur) { nt = 1.0f; finished = true; }
        else                 nt = static_cast<float>(elapsed / dur);
        spinPhase = nt;
    }
}

// Evaluate a preset's scalar tween channels at the given normalized time.
void evalPresetTweens(const PopupAnimPreset& p, float nt, float spinPhase,
                      float& alpha, float& scale, float& emis, float& spin,
                      float& px, float& py, float& pz)
{
    alpha = evalTween(p.alpha,    nt, 1.0f);
    scale = evalTween(p.scale,    nt, 1.0f);
    emis  = evalTween(p.emissive, nt, 0.0f);
    spin  = p.spinTurns * 6.28318530718f * spinPhase;
    px    = evalTween(p.posX, nt, 0.0f);
    py    = evalTween(p.posY, nt, 0.0f);
    pz    = evalTween(p.posZ, nt, 0.0f);
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

    // Cross-thread blend-pose request (UI thread -> render thread).
    std::atomic<uint32_t> poseStartToken{ 0 };
    std::atomic<uint32_t> poseStopToken{ 0 };
    std::atomic<int>      poseRequestIdx{ -1 };

    // Render-thread-only playback state (transient one-shot track).
    uint32_t lastToken = 0;
    int      curIdx    = -1;
    double   startTime = 0.0;

    // Render-thread-only playback state (persistent blend-pose track).
    uint32_t lastPoseStartToken = 0;
    uint32_t lastPoseStopToken  = 0;
    int      poseIdx      = -1;
    double   poseStartTime = 0.0;
    float    poseBlend    = 0.0f;   // current blend alpha [0,1]
    bool     poseStopping = false;  // ramping out toward idle

    double   lastEvalTime = 0.0;    // for blend-ramp dt

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

    // ---- presence_idle: persistent presence pose (gentle bob + breathe) ----
    {
        PopupAnimPreset p;
        p.name        = "presence_idle";
        p.type        = PopupAnimType::BlendPose;
        p.durationSec = 4.0f;
        p.loop        = true;
        p.pingPong    = true;
        p.blendInSec  = 0.40f;
        p.blendOutSec = 0.50f;
        p.posY  = { true, -0.02f, 0.02f, PopupEasing::EaseInOutCubic };
        p.scale = { true,  0.99f, 1.01f, PopupEasing::EaseInOutCubic };
        engine->presets.push_back(p);
    }

    // ---- speech_pulse: persistent speaking pose (scale + emissive pulse) ----
    {
        PopupAnimPreset p;
        p.name        = "speech_pulse";
        p.type        = PopupAnimType::BlendPose;
        p.durationSec = 0.50f;
        p.loop        = true;
        p.pingPong    = true;
        p.blendInSec  = 0.15f;
        p.blendOutSec = 0.30f;
        p.scale    = { true, 1.00f, 1.06f, PopupEasing::EaseInOutCubic };
        p.emissive = { true, 0.00f, 0.40f, PopupEasing::EaseInOutCubic };
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

        p.type          = animTypeFromString(jp.value("type", std::string("oneShot")));
        p.blendInSec    = jp.value("blendIn", 0.25f);
        p.blendOutSec   = jp.value("blendOut", 0.35f);
        p.pingPong      = jp.value("pingPong", false);

        if (jp.contains("alpha")    && jp["alpha"].is_object())    parseTween(jp["alpha"],    p.alpha);
        if (jp.contains("scale")    && jp["scale"].is_object())    parseTween(jp["scale"],    p.scale);
        if (jp.contains("emissive") && jp["emissive"].is_object()) parseTween(jp["emissive"], p.emissive);
        if (jp.contains("posX")     && jp["posX"].is_object())     parseTween(jp["posX"],     p.posX);
        if (jp.contains("posY")     && jp["posY"].is_object())     parseTween(jp["posY"],     p.posY);
        if (jp.contains("posZ")     && jp["posZ"].is_object())     parseTween(jp["posZ"],     p.posZ);

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
    if (engine->presets[it->second].type != PopupAnimType::OneShot)
    {
        LOG_ERROR("Popup3D", std::string("Trigger: preset '") + presetName +
                  "' is a blendPose — use StartPose instead");
        return;
    }
    engine->requestIdx.store(it->second, std::memory_order_relaxed);
    engine->requestToken.fetch_add(1, std::memory_order_release);
}

void popupClipEngineStartPose(PopupClipEngine* engine, const char* presetName)
{
    if (!engine || !presetName) return;
    auto it = engine->nameToIndex.find(presetName);
    if (it == engine->nameToIndex.end())
    {
        LOG_DEBUG("Popup3D", std::string("StartPose: unknown preset '") + presetName + "'");
        return;
    }
    if (engine->presets[it->second].type != PopupAnimType::BlendPose)
    {
        LOG_ERROR("Popup3D", std::string("StartPose: preset '") + presetName +
                  "' is a oneShot — use Trigger instead");
        return;
    }
    engine->poseRequestIdx.store(it->second, std::memory_order_relaxed);
    engine->poseStartToken.fetch_add(1, std::memory_order_release);
}

void popupClipEngineStopPose(PopupClipEngine* engine)
{
    if (!engine) return;
    engine->poseStopToken.fetch_add(1, std::memory_order_release);
}

bool popupClipEngineEvaluate(PopupClipEngine* engine, double nowSeconds, PopupClipEval& out)
{
    if (!engine) return false;

    // Frame delta used to advance the pose blend ramps.
    double dt = (engine->lastEvalTime > 0.0) ? (nowSeconds - engine->lastEvalTime) : 0.0;
    if (dt < 0.0) dt = 0.0;
    engine->lastEvalTime = nowSeconds;

    // =====================================================================
    // Track 1 — transient one-shot clip (load_in / fade_out / idle_spin).
    // Owns baked geometry playback and returns to idle when finished.
    // =====================================================================
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

    bool  transientActive   = false;
    bool  transientFinished = false;
    float tAlpha = 1.0f, tScale = 1.0f, tEmis = 0.0f, tSpin = 0.0f;
    const PopupMeshFrame* transientFrame = nullptr;
    const PopupMaterialProgram* transientMaterialProgram = nullptr;

    if (engine->curIdx >= 0 && engine->curIdx < static_cast<int>(engine->presets.size()))
    {
        const PopupAnimPreset& p = engine->presets[engine->curIdx];
        const double elapsed = nowSeconds - engine->startTime;

        float nt, spinPhase;
        clipPhase(p, elapsed, nt, spinPhase, transientFinished);

        float dummyX, dummyY, dummyZ;
        evalPresetTweens(p, nt, spinPhase, tAlpha, tScale, tEmis, tSpin, dummyX, dummyY, dummyZ);
        transientActive = true;

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
                    float phase = spinPhase - std::floor(spinPhase);
                    long long m = static_cast<long long>(std::floor(phase * frameCount)) % frameCount;
                    if (m < 0) m += frameCount;
                    fi = static_cast<int>(m);
                }
                else
                {
                    fi = static_cast<int>(std::floor(nt * frameCount));
                    if (fi >= frameCount) fi = frameCount - 1;
                    if (fi < 0) fi = 0;
                }
                transientFrame = &cache.frames[fi];
                transientMaterialProgram = &cache.materialProgram;
            }
        }
    }

    // =====================================================================
    // Track 2 — persistent blend-pose (presence / speech / movement).
    // Transform-only: blends scale / alpha / emissive / spin / position from
    // the idle neutral toward the pose target by a blendAlpha that ramps in on
    // start and ramps out on stop (or when a non-looping pose finishes).
    // =====================================================================
    uint32_t startTok = engine->poseStartToken.load(std::memory_order_acquire);
    if (startTok != engine->lastPoseStartToken)
    {
        engine->lastPoseStartToken = startTok;
        int idx = engine->poseRequestIdx.load(std::memory_order_relaxed);
        if (idx >= 0 && idx < static_cast<int>(engine->presets.size()) &&
            engine->presets[idx].type == PopupAnimType::BlendPose)
        {
            engine->poseIdx       = idx;
            engine->poseStartTime = nowSeconds;
            engine->poseStopping  = false;   // keep current poseBlend for a smooth crossfade
            LOG_DEBUG("Popup3D", "Pose start: '" + engine->presets[idx].name + "'");
        }
    }
    uint32_t stopTok = engine->poseStopToken.load(std::memory_order_acquire);
    if (stopTok != engine->lastPoseStopToken)
    {
        engine->lastPoseStopToken = stopTok;
        engine->poseStopping = true;
    }

    bool  poseActive = false;
    float pAlpha = 1.0f, pScale = 1.0f, pEmis = 0.0f, pSpin = 0.0f;
    float pX = 0.0f, pY = 0.0f, pZ = 0.0f;

    if (engine->poseIdx >= 0 && engine->poseIdx < static_cast<int>(engine->presets.size()))
    {
        const PopupAnimPreset& pp = engine->presets[engine->poseIdx];

        const float target  = engine->poseStopping ? 0.0f : 1.0f;
        const float rampSec  = engine->poseStopping ? pp.blendOutSec : pp.blendInSec;
        if (rampSec <= 0.0f)
        {
            engine->poseBlend = target;
        }
        else
        {
            const float step = static_cast<float>(dt) / rampSec;
            if (engine->poseBlend < target)
                engine->poseBlend = std::min(target, engine->poseBlend + step);
            else if (engine->poseBlend > target)
                engine->poseBlend = std::max(target, engine->poseBlend - step);
        }

        if (engine->poseStopping && engine->poseBlend <= 0.0f)
        {
            // Fully blended out — pose track returns to idle.
            engine->poseIdx      = -1;
            engine->poseStopping = false;
            engine->poseBlend    = 0.0f;
        }
        else
        {
            const double elapsed = nowSeconds - engine->poseStartTime;
            float nt, spinPhase;
            bool  poseFinished = false;
            clipPhase(pp, elapsed, nt, spinPhase, poseFinished);

            // A non-looping pose that reaches its end auto-blends back to idle.
            if (!pp.loop && poseFinished && !engine->poseStopping)
                engine->poseStopping = true;

            float a, s, e, sp, x, y, z;
            evalPresetTweens(pp, nt, spinPhase, a, s, e, sp, x, y, z);

            const float b = engine->poseBlend;
            pAlpha = 1.0f + (a - 1.0f) * b;   // multiplicative → neutral 1
            pScale = 1.0f + (s - 1.0f) * b;   // multiplicative → neutral 1
            pEmis  = e  * b;                  // additive       → neutral 0
            pSpin  = sp * b;
            pX = x * b; pY = y * b; pZ = z * b;
            poseActive = true;
        }
    }

    // =====================================================================
    // Combine tracks. Transient supplies geometry; pose is transform-only.
    // =====================================================================
    out.active = transientActive || poseActive;
    if (!out.active)
    {
        out.frame           = nullptr;
        out.materialProgram = nullptr;
        out.poseBlend       = engine->poseBlend;
        return false;
    }

    out.alphaMul     = tAlpha * pAlpha;
    out.scaleMul     = tScale * pScale;
    out.emissiveAdd  = tEmis + pEmis;
    out.spinY        = tSpin + pSpin;
    out.posOffset[0] = pX;
    out.posOffset[1] = pY;
    out.posOffset[2] = pZ;
    out.frame           = transientFrame;
    out.materialProgram = transientMaterialProgram;
    out.finished        = transientFinished;
    out.poseBlend       = engine->poseBlend;
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

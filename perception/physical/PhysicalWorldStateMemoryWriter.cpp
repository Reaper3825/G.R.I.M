#include "PhysicalWorldStateMemoryWriter.hpp"

#include "PhysicalWorldStateBus.hpp"
#include "PhysicalWorldStateLogTag.hpp"
#include "PhysicalWorldStateResult.hpp"
#include "memory/MemoryFacade.hpp"
#include "memory/unified_memory.hpp"
#include "bootstrap/bootstrap.hpp"   // g_memoryFacade
#include "logger.hpp"

#include <algorithm>
#include <cstdint>
#include <ctime>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

// Minimum dwell (ns) before a Confirmed entity is summarised on disappearance.
constexpr int64_t  kSummaryMinDwellNs    = 60LL * 1000LL * 1000LL * 1000LL; // 60s
// Bound the per-object state map. Trackers can spawn many short-lived ids.
constexpr size_t   kMaxTrackedObjects    = 4096;
// Bound how many transitions we emit per single tick (prevents storms).
constexpr size_t   kMaxRecordsPerTick    = 32;

struct EntityMemoryState {
    uint64_t                       appeared_memory_id   = 0;
    PhysicalEntityTrackState       last_track_state     = PhysicalEntityTrackState::Tentative;
    bool                           ever_confirmed       = false;
    bool                           had_text             = false;
    PhysicalSupportSurfaceClass    last_surface         = PhysicalSupportSurfaceClass::Unknown;
    bool                           was_path_blocked     = false;
    std::string                    class_label;
    int32_t                        class_id             = -1;
    int64_t                        first_seen_steady_ns = 0;
    int64_t                        last_seen_steady_ns  = 0;
    float                          last_range_meters    = 0.0f;
    bool                           had_metric_depth     = false;
    std::vector<std::string>       last_text_on_object;
    uint64_t                       last_seen_frame_ctr  = 0;
};

struct WriterState {
    std::mutex                                       mutex;
    bool                                             shutting_down = false;
    uint64_t                                         last_seen_frame_counter = 0;
    std::unordered_map<uint64_t, EntityMemoryState>  tracked;
};

WriterState& State() {
    static WriterState s;
    return s;
}

GRIM::MMO::MemoryFacade& RequireFacade() {
    if (!g_memoryFacade) {
        throw std::runtime_error(
            "PhysicalWorldStateMemoryWriter: g_memoryFacade is NULL — "
            "memory subsystem must be initialised before perception writer "
            "ticks (init order violation)");
    }
    return *g_memoryFacade;
}

std::string SurfaceWord(PhysicalSupportSurfaceClass s) {
    switch (s) {
        case PhysicalSupportSurfaceClass::Floor:   return "floor";
        case PhysicalSupportSurfaceClass::Table:   return "table";
        case PhysicalSupportSurfaceClass::Wall:    return "wall";
        case PhysicalSupportSurfaceClass::Unknown: return "";
    }
    return "";
}

std::string MetricDepthOrEmpty(const PhysicalWorldEntity& e) {
    if (!e.has_depth || e.depth_units != DepthUnits::Meters) return {};
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%.2fm", e.range_value_meters);
    return buf;
}

GRIM::UnifiedMemoryObject MakeBaseRecord(
    GRIM::TypeTag type,
    const std::string& normalized,
    float confidence)
{
    GRIM::UnifiedMemoryObject obj(
        GRIM::MemoryDomain::FIELD,
        type,
        GRIM::ContextType::CONVERSATION,
        normalized,
        confidence);
    obj.modality   = GRIM::Modality::VISION;
    obj.comm_type  = GRIM::CommType::UNKNOWN;
    obj.importance = confidence;
    obj.tags.push_back("physical");
    obj.tags.push_back("perception");
    return obj;
}

uint64_t EmitAppeared(GRIM::MMO::MemoryFacade& facade,
                      const PhysicalWorldEntity& e) {
    std::ostringstream txt;
    txt << e.class_label << '#' << e.object_id << " appeared";
    auto surface = SurfaceWord(e.support_surface);
    if (!surface.empty()) txt << " on " << surface;
    auto depth = MetricDepthOrEmpty(e);
    if (!depth.empty()) txt << " at " << depth;

    auto rec = MakeBaseRecord(GRIM::TypeTag::STRING,
                              txt.str(), e.confidence);
    rec.tags.push_back("entity_appeared");
    rec.tags.push_back("class:" + e.class_label);
    rec.tags.push_back("object:" + std::to_string(e.object_id));
    facade.recordInteraction(rec);
    return rec.id;
}

void EmitOcrFact(GRIM::MMO::MemoryFacade& facade,
                 const PhysicalWorldEntity& e,
                 uint64_t parent_id) {
    std::ostringstream txt;
    txt << e.class_label << '#' << e.object_id << " labelled [";
    for (size_t i = 0; i < e.text_on_object.size(); ++i) {
        if (i) txt << ", ";
        txt << '"' << e.text_on_object[i] << '"';
    }
    txt << ']';
    auto rec = MakeBaseRecord(GRIM::TypeTag::STRING,
                              txt.str(), e.confidence);
    rec.parent_id = parent_id;
    rec.tags.push_back("ocr");
    rec.tags.push_back("class:" + e.class_label);
    rec.tags.push_back("object:" + std::to_string(e.object_id));
    facade.recordInteraction(rec);
}

void EmitSurfaceChange(GRIM::MMO::MemoryFacade& facade,
                       const PhysicalWorldEntity& e,
                       PhysicalSupportSurfaceClass from,
                       uint64_t parent_id) {
    std::ostringstream txt;
    txt << e.class_label << '#' << e.object_id << " moved from "
        << (SurfaceWord(from).empty() ? std::string("unknown") : SurfaceWord(from))
        << " to "
        << (SurfaceWord(e.support_surface).empty()
                ? std::string("unknown") : SurfaceWord(e.support_surface));
    auto rec = MakeBaseRecord(GRIM::TypeTag::STRING,
                              txt.str(), e.support_surface_score);
    rec.parent_id = parent_id;
    rec.tags.push_back("surface_change");
    rec.tags.push_back("class:" + e.class_label);
    rec.tags.push_back("object:" + std::to_string(e.object_id));
    facade.recordInteraction(rec);
}

void EmitPathBlocked(GRIM::MMO::MemoryFacade& facade,
                     const PhysicalWorldEntity& e,
                     uint64_t parent_id) {
    std::ostringstream txt;
    txt << "path blocked by " << e.class_label << '#' << e.object_id;
    auto depth = MetricDepthOrEmpty(e);
    if (!depth.empty()) txt << " at " << depth;
    auto rec = MakeBaseRecord(GRIM::TypeTag::STRING,
                              txt.str(), e.path_block_score);
    rec.parent_id = parent_id;
    rec.tags.push_back("path_blocked");
    rec.tags.push_back("class:" + e.class_label);
    rec.tags.push_back("object:" + std::to_string(e.object_id));
    facade.recordInteraction(rec);
}

void EmitLost(GRIM::MMO::MemoryFacade& facade,
              const EntityMemoryState& s,
              uint64_t object_id) {
    std::ostringstream txt;
    txt << s.class_label << '#' << object_id << " lost";
    auto rec = MakeBaseRecord(GRIM::TypeTag::STRING,
                              txt.str(), 1.0f);
    rec.parent_id = s.appeared_memory_id;
    rec.tags.push_back("entity_lost");
    rec.tags.push_back("class:" + s.class_label);
    rec.tags.push_back("object:" + std::to_string(object_id));
    facade.recordInteraction(rec);
}

void EmitSummary(GRIM::MMO::MemoryFacade& facade,
                 const EntityMemoryState& s,
                 uint64_t object_id) {
    int64_t dwell_s = (s.last_seen_steady_ns - s.first_seen_steady_ns) / 1000000000LL;
    std::ostringstream txt;
    txt << s.class_label << '#' << object_id << " seen for " << dwell_s << "s";
    auto surface = SurfaceWord(s.last_surface);
    if (!surface.empty()) txt << " on " << surface;
    if (s.had_metric_depth) {
        char buf[24];
        std::snprintf(buf, sizeof(buf), " at %.2fm", s.last_range_meters);
        txt << buf;
    }
    if (!s.last_text_on_object.empty()) {
        txt << ", labelled [";
        for (size_t i = 0; i < s.last_text_on_object.size(); ++i) {
            if (i) txt << ", ";
            txt << '"' << s.last_text_on_object[i] << '"';
        }
        txt << ']';
    }
    auto rec = MakeBaseRecord(GRIM::TypeTag::STRING,
                              txt.str(), 1.0f);
    rec.parent_id = s.appeared_memory_id;
    rec.importance = std::min(2.0f, 1.0f + static_cast<float>(dwell_s) / 300.0f);
    rec.tags.push_back("entity_summary");
    rec.tags.push_back("class:" + s.class_label);
    rec.tags.push_back("object:" + std::to_string(object_id));
    facade.recordInteraction(rec);
}

bool TextChanged(const std::vector<std::string>& a,
                 const std::vector<std::string>& b) {
    if (a.size() != b.size()) return true;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return true;
    return false;
}

} // anonymous namespace

void TickPhysicalWorldStateMemoryWriter() {
    auto& s = State();
    std::lock_guard<std::mutex> lk(s.mutex);
    if (s.shutting_down) return;

    PhysicalWorldStateBus::SnapshotView view;
    const bool advanced =
        PhysicalWorldStateBus::Instance()
            .PullLatestPhysicalWorldStateSnapshotView(view, s.last_seen_frame_counter);
    if (!advanced) return;

    if (view.snapshot.source_frame_counter == 0) {
        throw std::runtime_error(
            "PhysicalWorldStateMemoryWriter: pulled snapshot with "
            "source_frame_counter==0; PhysicalWorldStateBus producer "
            "violated its publish invariant");
    }

    auto& facade = RequireFacade();

    size_t records_emitted = 0;
    auto emit_budget_exhausted = [&]() {
        if (records_emitted >= kMaxRecordsPerTick) {
            LOG_ERROR(PHYSICAL_WORLD_STATE_LOG_TAG,
                std::string("MemoryWriter: tick budget exhausted (")
                + std::to_string(kMaxRecordsPerTick)
                + " records); remaining transitions deferred to next tick");
            return true;
        }
        return false;
    };

    std::unordered_set<uint64_t> seen_this_tick;
    seen_this_tick.reserve(view.snapshot.entities.size());

    // Forward pass: appearances + per-entity transitions.
    for (const auto& e : view.snapshot.entities) {
        if (e.object_id == 0) continue;
        seen_this_tick.insert(e.object_id);

        auto it = s.tracked.find(e.object_id);
        if (it == s.tracked.end()) {
            // First time we see this object_id. Only record an "appeared"
            // event once the tracker promotes it to Confirmed — Tentative
            // tracks are noisy and frequently die before stabilising.
            if (e.track_state != PhysicalEntityTrackState::Confirmed) continue;
            if (s.tracked.size() >= kMaxTrackedObjects) {
                LOG_ERROR(PHYSICAL_WORLD_STATE_LOG_TAG,
                    std::string("MemoryWriter: tracked-object cap reached (")
                    + std::to_string(kMaxTrackedObjects)
                    + "); skipping new entity #" + std::to_string(e.object_id));
                continue;
            }
            if (emit_budget_exhausted()) break;

            EntityMemoryState st;
            st.last_track_state     = e.track_state;
            st.ever_confirmed       = true;
            st.had_text             = !e.text_on_object.empty();
            st.last_text_on_object  = e.text_on_object;
            st.last_surface         = e.support_surface;
            st.was_path_blocked     = e.path_blocked;
            st.class_label          = e.class_label;
            st.class_id             = e.class_id;
            st.first_seen_steady_ns = e.first_seen_steady_ns;
            st.last_seen_steady_ns  = e.last_seen_steady_ns;
            st.last_range_meters    = e.range_value_meters;
            st.had_metric_depth     = (e.has_depth && e.depth_units == DepthUnits::Meters);
            st.last_seen_frame_ctr  = view.snapshot.source_frame_counter;
            st.appeared_memory_id   = EmitAppeared(facade, e);
            ++records_emitted;

            if (st.had_text) {
                if (!emit_budget_exhausted()) {
                    EmitOcrFact(facade, e, st.appeared_memory_id);
                    ++records_emitted;
                }
            }
            if (e.path_blocked) {
                if (!emit_budget_exhausted()) {
                    EmitPathBlocked(facade, e, st.appeared_memory_id);
                    ++records_emitted;
                }
            }
            s.tracked.emplace(e.object_id, std::move(st));
            continue;
        }

        // Existing entity — diff against stored state.
        auto& st = it->second;
        st.last_seen_steady_ns = e.last_seen_steady_ns;
        st.last_seen_frame_ctr = view.snapshot.source_frame_counter;
        if (e.has_depth && e.depth_units == DepthUnits::Meters) {
            st.last_range_meters = e.range_value_meters;
            st.had_metric_depth  = true;
        }
        if (e.track_state == PhysicalEntityTrackState::Confirmed) st.ever_confirmed = true;
        st.last_track_state = e.track_state;

        // First-non-empty OCR text, or text content change.
        if (!e.text_on_object.empty() &&
            (!st.had_text || TextChanged(st.last_text_on_object, e.text_on_object))) {
            if (!emit_budget_exhausted()) {
                EmitOcrFact(facade, e, st.appeared_memory_id);
                ++records_emitted;
                st.had_text = true;
                st.last_text_on_object = e.text_on_object;
            }
        }

        // Support-surface change (only when both sides are known).
        if (e.support_surface != st.last_surface &&
            e.support_surface != PhysicalSupportSurfaceClass::Unknown &&
            st.last_surface   != PhysicalSupportSurfaceClass::Unknown) {
            if (!emit_budget_exhausted()) {
                EmitSurfaceChange(facade, e, st.last_surface, st.appeared_memory_id);
                ++records_emitted;
            }
        }
        // Always update last_surface when we see a known one (lets
        // future Unknown→Known transitions still fire correctly).
        if (e.support_surface != PhysicalSupportSurfaceClass::Unknown) {
            st.last_surface = e.support_surface;
        }

        // path_blocked rising edge.
        if (e.path_blocked && !st.was_path_blocked) {
            if (!emit_budget_exhausted()) {
                EmitPathBlocked(facade, e, st.appeared_memory_id);
                ++records_emitted;
            }
        }
        st.was_path_blocked = e.path_blocked;
    }

    // Reverse pass: detect tracker-dropped entities (absent from snapshot
    // entirely; Coasting still appears as an entity so this triggers ONLY
    // on true tracker death).
    for (auto it = s.tracked.begin(); it != s.tracked.end(); ) {
        if (seen_this_tick.find(it->first) == seen_this_tick.end()) {
            if (it->second.ever_confirmed) {
                if (records_emitted < kMaxRecordsPerTick) {
                    EmitLost(facade, it->second, it->first);
                    ++records_emitted;

                    int64_t dwell_ns = it->second.last_seen_steady_ns
                                     - it->second.first_seen_steady_ns;
                    if (dwell_ns >= kSummaryMinDwellNs &&
                        records_emitted < kMaxRecordsPerTick) {
                        EmitSummary(facade, it->second, it->first);
                        ++records_emitted;
                    }
                }
            }
            it = s.tracked.erase(it);
        } else {
            ++it;
        }
    }

    LOG_DEBUG(PHYSICAL_WORLD_STATE_LOG_TAG,
              std::string("MemoryWriter: frame=")
              + std::to_string(view.snapshot.source_frame_counter)
              + " tracked=" + std::to_string(s.tracked.size())
              + " emitted=" + std::to_string(records_emitted));
}

void ShutdownPhysicalWorldStateMemoryWriter() {
    auto& s = State();
    std::lock_guard<std::mutex> lk(s.mutex);
    s.shutting_down = true;

    // On shutdown, summarise any long-dwell confirmed entities still
    // tracked so their dwell time isn't lost. Best-effort: skip if the
    // facade is gone (memory subsystem may have shut down first).
    if (g_memoryFacade) {
        for (const auto& [object_id, st] : s.tracked) {
            if (!st.ever_confirmed) continue;
            int64_t dwell_ns = st.last_seen_steady_ns - st.first_seen_steady_ns;
            if (dwell_ns < kSummaryMinDwellNs) continue;
            try {
                EmitSummary(*g_memoryFacade, st, object_id);
            } catch (const std::exception& e) {
                LOG_ERROR(PHYSICAL_WORLD_STATE_LOG_TAG,
                          std::string("MemoryWriter: shutdown summary failed: ")
                          + e.what());
            }
        }
    }

    s.tracked.clear();
    s.last_seen_frame_counter = 0;
}

}}} // namespace GRIM::Perception::Physical

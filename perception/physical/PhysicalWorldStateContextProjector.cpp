#include "PhysicalWorldStateContextProjector.hpp"

#include "PhysicalWorldStateBus.hpp"
#include "PhysicalWorldStateLogTag.hpp"
#include "PhysicalWorldStateResult.hpp"
#include "MMO/Core/SessionContextManager.hpp"
#include "logger.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

// Must match MemoryFacade::kDefaultSession (file-static there). Keep in sync.
constexpr const char* kDefaultSession = "default";

// Caps to bound router prompt size. Tune via observation, not config.
constexpr size_t kMaxEntitiesInFocus     = 8;
constexpr size_t kMaxRelationsInProjection = 8;
constexpr size_t kMaxAlertsInProjection  = 4;
constexpr size_t kMaxLabelsInOcrSummary  = 2;

struct ProjectorState {
    std::mutex mutex;
    bool       shutting_down = false;
    uint64_t   last_seen_frame_counter = 0;
};

ProjectorState& State() {
    static ProjectorState s;
    return s;
}

std::string FormatRangeMeters(const PhysicalWorldEntity& e) {
    if (!e.has_depth) return {};
    if (e.depth_units == DepthUnits::Meters) {
        char buf[24];
        std::snprintf(buf, sizeof(buf), "%.2fm", e.range_value_meters);
        return buf;
    }
    // Relative depth — emit a unitless ordinal so the LM doesn't see fake metres.
    char buf[32];
    std::snprintf(buf, sizeof(buf), "depth=%.2f(rel)", e.range_value);
    return buf;
}

std::string DescribeSurface(PhysicalSupportSurfaceClass s) {
    switch (s) {
        case PhysicalSupportSurfaceClass::Floor:   return "floor";
        case PhysicalSupportSurfaceClass::Table:   return "table";
        case PhysicalSupportSurfaceClass::Wall:    return "wall";
        case PhysicalSupportSurfaceClass::Unknown: return "";
    }
    return "";
}

std::string ComposeEntityLine(const PhysicalWorldEntity& e) {
    std::ostringstream oss;
    oss << e.class_label << '#' << e.object_id;
    auto surface = DescribeSurface(e.support_surface);
    if (!surface.empty()) oss << " on " << surface;
    auto depth = FormatRangeMeters(e);
    if (!depth.empty()) oss << ", " << depth;
    if (!e.text_on_object.empty()) {
        oss << ", holds [";
        size_t shown = std::min(e.text_on_object.size(), kMaxLabelsInOcrSummary);
        for (size_t i = 0; i < shown; ++i) {
            if (i) oss << ", ";
            oss << '"' << e.text_on_object[i] << '"';
        }
        if (e.text_on_object.size() > shown) {
            oss << ", +" << (e.text_on_object.size() - shown);
        }
        oss << ']';
    }
    return oss.str();
}

std::string ComposeRelationLine(const PhysicalWorldEntity& e,
                                const PhysicalEntitySpatialRelation& r) {
    std::ostringstream oss;
    oss << e.class_label << '#' << e.object_id
        << ' ' << DescribePhysicalEntityRelationKind(r.kind)
        << " #" << r.other_object_id;
    char buf[16];
    std::snprintf(buf, sizeof(buf), " (%.2f)", r.strength);
    oss << buf;
    return oss.str();
}

void BuildProjection(const PhysicalWorldStateSnapshot& snap,
                     ::GRIM::MMO::VisualContext::PhysicalVisual& out) {
    out = {}; // overwrite; we are the sole owner of this struct each frame

    // 1-line headline — counts only.
    {
        std::ostringstream hdr;
        hdr << snap.entities.size() << " tracked, "
            << snap.num_visible_entities << " visible, "
            << snap.num_occluded_entities << " occluded, "
            << snap.num_coasting_entities << " coasting";
        out.scene_summary = hdr.str();
    }

    // Class label histogram — "cup x2, person x1".
    {
        std::map<std::string, int> counts;
        for (const auto& e : snap.entities) ++counts[e.class_label];
        // Stable order by count desc then label asc.
        std::vector<std::pair<std::string,int>> sorted(counts.begin(), counts.end());
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) {
                      if (a.second != b.second) return a.second > b.second;
                      return a.first < b.first;
                  });
        for (const auto& [label, n] : sorted) {
            std::ostringstream item;
            item << label;
            if (n > 1) item << " x" << n;
            out.detected_objects.push_back(item.str());
        }
    }

    // Entities in focus — top by confidence, visible only.
    {
        std::vector<const PhysicalWorldEntity*> ranked;
        ranked.reserve(snap.entities.size());
        for (const auto& e : snap.entities) {
            if (e.visibility == PhysicalEntityVisibility::Visible) ranked.push_back(&e);
        }
        std::sort(ranked.begin(), ranked.end(),
                  [](const PhysicalWorldEntity* a, const PhysicalWorldEntity* b) {
                      return a->confidence > b->confidence;
                  });
        size_t take = std::min(ranked.size(), kMaxEntitiesInFocus);
        for (size_t i = 0; i < take; ++i) {
            out.entities_in_focus.push_back(ComposeEntityLine(*ranked[i]));
        }
    }

    // Top-K relations across ALL entities, ranked by strength.
    {
        struct Ranked { const PhysicalWorldEntity* e; const PhysicalEntitySpatialRelation* r; };
        std::vector<Ranked> all;
        for (const auto& e : snap.entities) {
            for (const auto& r : e.relations) {
                if (r.kind == PhysicalEntityRelationKind::None) continue;
                all.push_back({&e, &r});
            }
        }
        std::sort(all.begin(), all.end(),
                  [](const Ranked& a, const Ranked& b) {
                      return a.r->strength > b.r->strength;
                  });
        size_t take = std::min(all.size(), kMaxRelationsInProjection);
        for (size_t i = 0; i < take; ++i) {
            out.spatial_relations_top_k.push_back(
                ComposeRelationLine(*all[i].e, *all[i].r));
        }
    }

    // Active alerts: path_blocked is the only one Stage-4 currently flags.
    {
        std::vector<const PhysicalWorldEntity*> blockers;
        for (const auto& e : snap.entities) {
            if (e.path_blocked) blockers.push_back(&e);
        }
        std::sort(blockers.begin(), blockers.end(),
                  [](const PhysicalWorldEntity* a, const PhysicalWorldEntity* b) {
                      return a->path_block_score > b->path_block_score;
                  });
        size_t take = std::min(blockers.size(), kMaxAlertsInProjection);
        for (size_t i = 0; i < take; ++i) {
            std::ostringstream a;
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%.2f", blockers[i]->path_block_score);
            a << "path_blocked: " << blockers[i]->class_label
              << '#' << blockers[i]->object_id << " (" << buf << ')';
            out.active_alerts.push_back(a.str());
        }
    }

    out.provenance_frame_counter = snap.source_frame_counter;
}

} // anonymous namespace

void TickPhysicalWorldStateContextProjector() {
    auto& s = State();
    std::lock_guard<std::mutex> lk(s.mutex);
    if (s.shutting_down) return;

    PhysicalWorldStateBus::SnapshotView view;
    const bool advanced =
        PhysicalWorldStateBus::Instance()
            .PullLatestPhysicalWorldStateSnapshotView(view, s.last_seen_frame_counter);
    if (!advanced) return;

    if (view.snapshot.source_frame_counter == 0) {
        // Producer must never publish a zero-counter snapshot. If we see one
        // it is an upstream invariant violation — fail loud (Rule 20).
        throw std::runtime_error(
            "PhysicalWorldStateContextProjector: pulled snapshot with "
            "source_frame_counter==0; PhysicalWorldStateBus producer "
            "violated its publish invariant");
    }

    ::GRIM::MMO::VisualContext::PhysicalVisual projection;
    BuildProjection(view.snapshot, projection);

    ::GRIM::MMO::SessionContextManager::instance().updatePhysicalVisual(
        kDefaultSession, projection);

    LOG_DEBUG(PHYSICAL_WORLD_STATE_LOG_TAG,
              std::string("ContextProjector: published projection frame=")
              + std::to_string(view.snapshot.source_frame_counter)
              + " entities=" + std::to_string(view.snapshot.entities.size())
              + " focus=" + std::to_string(projection.entities_in_focus.size())
              + " relations=" + std::to_string(projection.spatial_relations_top_k.size())
              + " alerts=" + std::to_string(projection.active_alerts.size()));
}

void ShutdownPhysicalWorldStateContextProjector() {
    auto& s = State();
    std::lock_guard<std::mutex> lk(s.mutex);
    s.shutting_down = true;
    s.last_seen_frame_counter = 0;
}

}}} // namespace GRIM::Perception::Physical

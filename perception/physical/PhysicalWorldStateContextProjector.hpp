#pragma once

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalWorldStateContextProjector — perception → SessionContextManager.
//
//  Single-tick consumer of PhysicalWorldStateBus that PROJECTS the latest
//  identity-keyed PhysicalWorldStateSnapshot down to the LM-readable
//  fields of VisualContext::PhysicalVisual on SessionContextManager.
//
//  Architectural rules (MMO):
//    * The model never writes to memory. This projector ONLY mutates live
//      session context (in-RAM working memory) — it never calls
//      MemoryFacade. Long-term writes belong to PhysicalWorldStateMemoryWriter.
//    * No pixel coordinates, boxes, velocities, or per-frame floats are
//      projected — only natural-language tokens the router can attend to.
//    * Rule 20: throws on missing dependencies; never silently no-ops.
//
//  Mainloop call site: invoke ONCE per frame, AFTER TickPhysicalWorldState().
// ─────────────────────────────────────────────────────────────────────────────

namespace GRIM { namespace Perception { namespace Physical {

// Mainloop entry point. Cheap; safe to call every frame. Lazy-inits.
// No-ops cleanly (no throw) until the first world-state snapshot is published.
void TickPhysicalWorldStateContextProjector();

// Stop processing. Safe to call multiple times.
void ShutdownPhysicalWorldStateContextProjector();

}}} // namespace GRIM::Perception::Physical

#pragma once

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalWorldStateMemoryWriter — perception → MemoryFacade.
//
//  Single-tick consumer of PhysicalWorldStateBus that DIFFs successive
//  identity-keyed PhysicalWorldStateSnapshots and emits durable
//  UnifiedMemoryObject records to MemoryFacade for state transitions only.
//
//  Why diff-based, not per-frame:
//    The personality LoRA is trained from MemoryFacade content. Per-frame
//    snapshots would flood the FlatBuffer store with noise and pollute the
//    LoRA training corpus with coordinate tokens. State changes (entity
//    appeared / lost / surface change / ocr label / path block) are the
//    only signal worth a long-term record.
//
//  Architectural rules (MMO):
//    * The MODEL never writes to memory. This writer is a perception-side
//      consumer of the bus; the model is not in its call chain.
//    * Records carry NO pixel coordinates, boxes, velocities, or per-frame
//      floats — only durable identity, class, surface, ocr text, depth in
//      metres, and human-readable normalized text. This keeps the LoRA
//      training corpus in language space.
//    * Rule 20: throws on missing g_memoryFacade; never silently no-ops.
//
//  Mainloop call site: invoke ONCE per frame, AFTER TickPhysicalWorldState()
//  AND AFTER TickPhysicalWorldStateContextProjector() (order doesn't matter
//  between the two consumers — they each track their own bus cursor —
//  but doing the projector first keeps the live router context fresher).
// ─────────────────────────────────────────────────────────────────────────────

namespace GRIM { namespace Perception { namespace Physical {

// Mainloop entry point. Cheap; safe to call every frame. Lazy-inits.
// Throws std::runtime_error if g_memoryFacade is null when a snapshot
// transition would otherwise be written.
void TickPhysicalWorldStateMemoryWriter();

// Stop processing and clear per-object state. Safe to call multiple times.
void ShutdownPhysicalWorldStateMemoryWriter();

}}} // namespace GRIM::Perception::Physical

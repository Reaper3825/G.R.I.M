// MMO ModelRouter — thin parser for grim-text route responses
//
// The router (grim-text with LoRA) returns a JSON response
// that the body must parse into a RouteDecision.  This class
// is that parser.  It owns NO model logic — all intelligence
// is in grim-text; the body merely validates and extracts.
//
// Expected router response JSON:
//   {
//     "sub_model_id": "science_brick",
//     "composed_generation": "TASK: Explain photosynthesis\n...",
//     "confidence": 0.92,
//     "diagnostics": { ... }          // optional
//   }
//
// If the router refuses:
//   {
//     "sub_model_id": "",
//     "refusal": "cannot route: ambiguous domain",
//     "confidence": 0.0
//   }
//======================================================//
#pragma once

#include "../Core/Contracts.hpp"        // RouteDecision
#include "../Shared/MMD.hpp"            // ResponseEnvelope, ResponseStatus

#include <string>

namespace GRIM::MMO {

// =========================================================
// RouteConfidence — parsed confidence scores from router
// =========================================================
struct RouteConfidence {
    float overall           = 0.0f;  // router's routing confidence
    float user_intent       = 0.0f;  // how well router understood intent
    float domain_match      = 0.0f;  // domain relevance to chosen sub-model
};

// =========================================================
// ParsedRouteResult — full parse output from router response
// =========================================================
struct ParsedRouteResult {
    bool            success = false;
    RouteDecision   decision;        // sub_model_id + composed_generation + diagnostics
    RouteConfidence confidence;
    std::string     refusal;         // non-empty when router refused
    std::string     error;           // non-empty on parse failure
};

// =========================================================
// ModelRouter — stateless parser
//
// Usage:
//   ModelRouter router;
//   auto parsed = router.parseRouteResponse(response_envelope);
//   if (parsed.success) {
//       use(parsed.decision.sub_model_id);
//       use(parsed.decision.composed_generation);
//   }
// =========================================================
class ModelRouter {
public:
    // Parse a ResponseEnvelope from the router backend into
    // a structured RouteDecision with confidence scores.
    //
    // Returns ParsedRouteResult with success=false and error
    // set if the JSON is malformed or missing required fields.
    ParsedRouteResult parseRouteResponse(const ResponseEnvelope& response) const;

    // Parse raw JSON text directly (for testing or when the
    // response has already been extracted from the envelope).
    ParsedRouteResult parseRouteJson(const std::string& json_text) const;
};

} // namespace GRIM::MMO

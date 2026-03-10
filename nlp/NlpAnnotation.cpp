// NlpAnnotation.cpp — builds the canonical NlpAnnotation payload
//
// Phase 0.5: This is the bridge layer that wraps existing NLP infrastructure
// (NLP::parse, IntentGate::decide, FastClassifier::evaluate) into the new
// structured annotation format consumed by the router metadata builder,
// memory tagger, and action policy evaluator.
//
// No new code should add direct NLP -> command execution coupling.
// NLP output is metadata; execution goes through ToolRegistry + ActionPolicy.
//======================================================//

#include "NlpAnnotation.hpp"
#include "nlp.hpp"
#include "../ai/intent_gate.hpp"
#include "../ai/fast_classifier.hpp"
#include "../MMO/Core/SessionContextManager.hpp"
#include "../memory/context_snapshot.hpp"
#include <algorithm>
#include <cctype>
#include <regex>

extern NLP g_nlp;

namespace GRIM {

// ─── Text normalization ── lowercase + whitespace collapse ────

static std::string normalizeText(const std::string& raw) {
    std::string out;
    out.reserve(raw.size());
    bool prev_space = false;
    for (char c : raw) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!prev_space) { out += ' '; prev_space = true; }
        } else {
            out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            prev_space = false;
        }
    }
    // trim trailing space
    if (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

// ─── Language detection (lightweight heuristic) ───────────────────────

static std::string detectLanguage(const std::string& text) {
    // Count characters in non-Latin Unicode ranges
    int cjk = 0, cyrillic = 0, arabic = 0, latin = 0;
    for (size_t i = 0; i < text.size(); ++i) {
        auto c = static_cast<unsigned char>(text[i]);
        if (c < 0x80) {
            if (std::isalpha(c)) ++latin;
        } else if (c >= 0xD0 && c <= 0xD4) {
            // Cyrillic block (UTF-8 two-byte: 0xD0xx-0xD4xx)
            ++cyrillic; ++i;
        } else if (c >= 0xD8 && c <= 0xDB) {
            // Arabic block (UTF-8 two-byte: 0xD8xx-0xDBxx)
            ++arabic; ++i;
        } else if (c >= 0xE4 && c <= 0xE9) {
            // CJK Unified Ideographs range (UTF-8 three-byte: 0xE4xxxx-0xE9xxxx)
            ++cjk; i += 2;
        }
    }
    int total = latin + cjk + cyrillic + arabic;
    if (total == 0) return "en";

    if (cjk * 3 > total) return "zh";       // Chinese/Japanese/Korean predominant
    if (cyrillic * 3 > total) return "ru";   // Cyrillic predominant
    if (arabic * 3 > total) return "ar";     // Arabic predominant
    return "en";
}

// ─── Entity extraction (regex-based, aligns with GRIM-text atoms) ─────

static void extractEntities(const std::string& normalized,
                             std::vector<Entity>& out) {
    // URL
    {
        static const std::regex url_re(
            R"(https?://[^\s]+)",
            std::regex::icase | std::regex::optimize);
        std::sregex_iterator it(normalized.begin(), normalized.end(), url_re);
        std::sregex_iterator end;
        for (; it != end; ++it) {
            Entity e;
            e.text  = it->str();
            e.type  = "url";
            e.start_offset = static_cast<int>(it->position());
            e.end_offset   = e.start_offset + static_cast<int>(it->length());
            out.push_back(std::move(e));
        }
    }
    // Email
    {
        static const std::regex email_re(
            R"([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})",
            std::regex::optimize);
        std::sregex_iterator it(normalized.begin(), normalized.end(), email_re);
        std::sregex_iterator end;
        for (; it != end; ++it) {
            Entity e;
            e.text  = it->str();
            e.type  = "email";
            e.start_offset = static_cast<int>(it->position());
            e.end_offset   = e.start_offset + static_cast<int>(it->length());
            out.push_back(std::move(e));
        }
    }
    // File path (Windows or Unix)
    {
        static const std::regex path_re(
            R"((?:[a-zA-Z]:\\|/)(?:[^\s/\\]+[/\\])*[^\s/\\]+)",
            std::regex::optimize);
        std::sregex_iterator it(normalized.begin(), normalized.end(), path_re);
        std::sregex_iterator end;
        for (; it != end; ++it) {
            Entity e;
            e.text  = it->str();
            e.type  = "path";
            e.start_offset = static_cast<int>(it->position());
            e.end_offset   = e.start_offset + static_cast<int>(it->length());
            out.push_back(std::move(e));
        }
    }
    // Numbers (integers and decimals)
    {
        static const std::regex num_re(
            R"(\b\d+(?:\.\d+)?\b)",
            std::regex::optimize);
        std::sregex_iterator it(normalized.begin(), normalized.end(), num_re);
        std::sregex_iterator end;
        for (; it != end; ++it) {
            Entity e;
            e.text  = it->str();
            e.type  = "number";
            e.start_offset = static_cast<int>(it->position());
            e.end_offset   = e.start_offset + static_cast<int>(it->length());
            out.push_back(std::move(e));
        }
    }
}

// ─── Reference detection (pronouns / deictics) ────────────────

static void extractReferences(const std::string& normalized,
                               std::vector<std::string>& out) {
    static const std::regex ref_re(
        R"(\b(?:it|that|this|those|these|them|same|the same|that file|that folder|same folder|that one)\b)",
        std::regex::icase | std::regex::optimize);
    std::sregex_iterator it(normalized.begin(), normalized.end(), ref_re);
    std::sregex_iterator end;
    for (; it != end; ++it) {
        out.push_back(it->str());
    }
}

// ─── Risk tag detection ───────────────────────────────────────

static void extractRiskTags(const std::string& normalized,
                             const std::vector<std::string>& affordances,
                             std::vector<std::string>& out) {
    // Check for destructive keywords
    static const std::regex destroy_re(
        R"(\b(?:delete|remove|kill|drop|format|wipe|erase|destroy|uninstall|rm|shutdown|reboot|restart)\b)",
        std::regex::icase | std::regex::optimize);
    if (std::regex_search(normalized, destroy_re)) {
        out.push_back("destructive");
    }

    // Check for system-level
    static const std::regex sys_re(
        R"(\b(?:registry|regedit|service|driver|firewall|admin|sudo|root|system32)\b)",
        std::regex::icase | std::regex::optimize);
    if (std::regex_search(normalized, sys_re)) {
        out.push_back("system");
    }

    // Check for network
    static const std::regex net_re(
        R"(\b(?:download|upload|fetch|curl|wget|http|ssh|ftp|connect|socket|port)\b)",
        std::regex::icase | std::regex::optimize);
    if (std::regex_search(normalized, net_re)) {
        out.push_back("network");
    }

    // Check for credential-sensitive
    static const std::regex cred_re(
        R"(\b(?:password|token|secret|key|credential|api.?key|auth)\b)",
        std::regex::icase | std::regex::optimize);
    if (std::regex_search(normalized, cred_re)) {
        out.push_back("credential_sensitive");
    }

    // Affordance cross-check
    for (const auto& a : affordances) {
        if (a == "dangerous_delete") {
            if (std::find(out.begin(), out.end(), "destructive") == out.end())
                out.push_back("destructive");
        }
    }
}

// ─── Action affordance detection ──────────────────────────────

static void extractAffordances(const std::string& normalized,
                                std::vector<std::string>& out) {
    struct { const char* pattern; const char* tag; } checks[] = {
        { R"(\b(?:open|launch|start|run)\b)",      "open"       },
        { R"(\b(?:search|find|look\s?up|query)\b)", "search"     },
        { R"(\b(?:go\s?to|navigate|browse)\b)",     "navigate"   },
        { R"(\b(?:edit|modify|change|update)\b)",   "edit"       },
        { R"(\b(?:inspect|check|examine|view)\b)",  "inspect"    },
        { R"(\b(?:delete|remove|erase|drop)\b)",    "dangerous_delete" },
        { R"(\b(?:create|make|new|add)\b)",         "create"     },
        { R"(\b(?:copy|clone|duplicate)\b)",        "copy"       },
        { R"(\b(?:move|rename|transfer)\b)",        "move"       },
        { R"(\b(?:close|stop|quit|exit)\b)",        "close"      },
    };
    for (auto& [pat, tag] : checks) {
        std::regex re(pat, std::regex::icase | std::regex::optimize);
        if (std::regex_search(normalized, re)) {
            out.push_back(tag);
        }
    }
}

// ─── Context tag detection ────────────────────────────────────

static void extractContextTags(const std::string& normalized,
                                const std::vector<Entity>& entities,
                                std::vector<std::string>& out) {
    // Domain markers
    static const std::regex code_re(R"(\b(?:code|compile|build|debug|git|commit|branch|merge|function|class|variable)\b)", std::regex::icase | std::regex::optimize);
    static const std::regex fs_re(R"(\b(?:file|folder|directory|path|drive|disk|copy|move|rename)\b)", std::regex::icase | std::regex::optimize);
    static const std::regex web_re(R"(\b(?:web|browser|url|http|site|page|download|upload)\b)", std::regex::icase | std::regex::optimize);
    static const std::regex sys_re(R"(\b(?:system|process|service|registry|cpu|memory|task|manager)\b)", std::regex::icase | std::regex::optimize);
    static const std::regex ui_re(R"(\b(?:window|click|button|ui|menu|dialog|popup|screen)\b)", std::regex::icase | std::regex::optimize);
    static const std::regex voice_re(R"(\b(?:say|speak|tell|voice|listen|hear|read\s?aloud)\b)", std::regex::icase | std::regex::optimize);

    if (std::regex_search(normalized, code_re))  out.push_back("coding");
    if (std::regex_search(normalized, fs_re))    out.push_back("filesystem");
    if (std::regex_search(normalized, web_re))   out.push_back("web");
    if (std::regex_search(normalized, sys_re))   out.push_back("system");
    if (std::regex_search(normalized, ui_re))    out.push_back("ui");
    if (std::regex_search(normalized, voice_re)) out.push_back("voice");

    // Entity-driven tags
    for (const auto& e : entities) {
        if (e.type == "path" || e.type == "file") {
            if (std::find(out.begin(), out.end(), "filesystem") == out.end())
                out.push_back("filesystem");
        }
        if (e.type == "url") {
            if (std::find(out.begin(), out.end(), "web") == out.end())
                out.push_back("web");
        }
    }
}

// =========================================================
// annotate() — the main entry point
//
// Produces a full NlpAnnotation from raw user input.
// Wraps existing NLP, IntentGate, and FastClassifier.
// =========================================================
NlpAnnotation annotate(const std::string& raw_input,
                        const ContextSnapshot& context) {
    NlpAnnotation ann;
    ann.raw_text        = raw_input;
    ann.normalized_text = normalizeText(raw_input);
    ann.language        = detectLanguage(raw_input);

    // ─── Utterance priors from IntentGate + FastClassifier ──
    IntentResult gate = IntentGate::decide(raw_input, context);
    FastResult   fast = FastClassifier::evaluate(raw_input, context);

    // Blend: gate is the primary signal, fast provides spread
    ann.utterance_priors.command  = (gate.type == IntentType::Command)  ? gate.confidence : 0.0f;
    ann.utterance_priors.question = (gate.type == IntentType::Question) ? gate.confidence : 0.0f;
    ann.utterance_priors.banter   = (gate.type == IntentType::Banter)   ? gate.confidence : 0.0f;
    ann.utterance_priors.unknown  = (gate.type == IntentType::Unknown)  ? gate.confidence : 0.0f;

    // Spread fast classifier signal into non-dominant categories
    float fast_spread = (1.0f - gate.confidence) * fast.confidence;
    if (fast.guess == IntentType::Command  && gate.type != IntentType::Command)
        ann.utterance_priors.command  += fast_spread;
    if (fast.guess == IntentType::Question && gate.type != IntentType::Question)
        ann.utterance_priors.question += fast_spread;
    if (fast.guess == IntentType::Banter   && gate.type != IntentType::Banter)
        ann.utterance_priors.banter   += fast_spread;

    // ─── Legacy NLP parse for slot extraction ───────────────
    Intent intent = g_nlp.parse(raw_input);
    ann.legacy_intent_name = intent.name;
    ann.legacy_category    = intent.category;
    ann.legacy_matched     = intent.matched;

    // ─── Entity extraction ──────────────────────────────────
    extractEntities(ann.normalized_text, ann.entities);

    // ─── Affordances ────────────────────────────────────────
    extractAffordances(ann.normalized_text, ann.action_affordances);

    // ─── References ─────────────────────────────────────────
    extractReferences(ann.normalized_text, ann.references);

    // ─── Context tags ───────────────────────────────────────
    extractContextTags(ann.normalized_text, ann.entities, ann.context_tags);

    // ─── Risk tags ──────────────────────────────────────────
    extractRiskTags(ann.normalized_text, ann.action_affordances, ann.risk_tags);

    // ─── Router tags (from NLP category + entities) ─────────
    if (!intent.category.empty() && intent.category != "general") {
        ann.router_tags.push_back(intent.category);
    }
    // Add entity types as router hints
    for (const auto& e : ann.entities) {
        std::string tag = "has:" + e.type;
        if (std::find(ann.router_tags.begin(), ann.router_tags.end(), tag) == ann.router_tags.end())
            ann.router_tags.push_back(tag);
    }

    // ─── Memory tags (context + entities + affordances) ─────
    ann.memory_tags = ann.context_tags; // start with context
    for (const auto& a : ann.action_affordances) {
        ann.memory_tags.push_back("action:" + a);
    }

    // ─── Confidence summary ─────────────────────────────────
    float entity_conf = ann.entities.empty() ? 0.0f : 1.0f;
    for (const auto& e : ann.entities) {
        entity_conf = std::min(entity_conf, e.confidence);
    }
    ann.confidence_summary.entity_confidence = entity_conf;
    ann.confidence_summary.intent_confidence = static_cast<float>(intent.confidence);
    ann.confidence_summary.prior_confidence  = gate.confidence;
    ann.confidence_summary.overall = (entity_conf + static_cast<float>(intent.confidence) + gate.confidence) / 3.0f;

    return ann;
}

// =========================================================
// annotate() V2 overload — accepts ContextSnapshotV2
//
// Projects V2 → V1 for IntentGate/FastClassifier (which still
// consume ContextSnapshot), then delegates to the V1 overload.
// =========================================================
NlpAnnotation annotate(const std::string& raw_input,
                        const GRIM::MMO::ContextSnapshotV2& v2) {
    // Project V2 → V1 for legacy consumers
    ContextSnapshot v1;
    v1.currentMood         = v2.current_mood;
    v1.lastNlpCategory     = v2.lastNlpCategory;
    v1.consecutiveCommands = v2.consecutiveCommands;
    v1.conversationDepth   = static_cast<int>(v2.recent_turn_summaries.size());

    // Populate recentIntents from utterance_priors strings
    v1.recentIntents = v2.utterance_priors;

    return annotate(raw_input, v1);
}

} // namespace GRIM

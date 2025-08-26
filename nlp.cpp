#include "NLP.hpp"
#include <regex>
#include <fstream>
#include <sstream>
#include <cctype>

// Minimal JSON loader without deps.
// If you prefer, swap with nlohmann/json later.
// For now, we expect a very simple, fixed structure and parse crudely.
namespace mini_json {
    // Not a general parser; assumes your file is formatted like in the example below.
    // If you want robust JSON, use nlohmann/json single-header later.
    static std::string read_all(const std::string& path) {
        std::ifstream f(path);
        std::ostringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }

    static std::string strip_ws(std::string s) {
        s.erase(remove_if(s.begin(), s.end(), [](unsigned char c){return std::isspace(c);}), s.end());
        return s;
    }
}

// --- NLP impl ---
bool NLP::load_rules(const std::string& json_path, std::string* error_out) {
    rules_.clear();

    auto txt = mini_json::read_all(json_path);
    if (txt.empty()) {
        if (error_out) *error_out = "Failed to read rules file or it was empty: " + json_path;
        return false;
    }

    // SUPER SIMPLE parsing expecting an array of objects like:
    // [
    //   {"intent":"open_app","pattern":"^open\\s+(\\w+)$","slot_names":["app"],"score_boost":0.2},
    //   ...
    // ]
    // To keep this tiny, we’ll search for objects and pull fields with naive slicing.
    // Swap to a real JSON library when you want robustness.

    size_t i = 0;
    auto next = [&](char c){
        while (i < txt.size() && std::isspace((unsigned char)txt[i])) ++i;
        return (i < txt.size() && txt[i] == c);
    };

    // Find start of array
    while (i < txt.size() && txt[i] != '[') ++i;
    if (i == txt.size()) {
        if (error_out) *error_out = "Rules JSON missing opening '['";
        return false;
    }
    ++i; // skip [

    auto read_quoted = [&]() -> std::string {
        while (i < txt.size() && txt[i] != '"') ++i;
        if (i == txt.size()) return {};
        ++i; // skip "
        std::string out;
        while (i < txt.size() && txt[i] != '"') {
            if (txt[i] == '\\' && i + 1 < txt.size()) {
                out.push_back(txt[i+1]);
                i += 2;
            } else {
                out.push_back(txt[i++]);
            }
        }
        if (i < txt.size() && txt[i] == '"') ++i; // closing "
        return out;
    };

    auto skip_ws = [&](){
        while (i < txt.size() && std::isspace((unsigned char)txt[i])) ++i;
    };

    auto read_number = [&]() -> float {
        skip_ws();
        size_t start = i;
        while (i < txt.size() && (std::isdigit((unsigned char)txt[i]) || txt[i]=='.' || txt[i]=='-' || txt[i]=='+')) ++i;
        try {
            return std::stof(txt.substr(start, i-start));
        } catch (...) { return 0.0f; }
    };

    auto read_bool = [&]() -> bool {
        skip_ws();
        if (txt.compare(i, 4, "true") == 0) { i += 4; return true; }
        if (txt.compare(i, 5, "false") == 0) { i += 5; return false; }
        return true;
    };

    auto read_slot_array = [&]() -> std::vector<std::string> {
        std::vector<std::string> out;
        skip_ws();
        if (i >= txt.size() || txt[i] != '[') return out;
        ++i; // [
        skip_ws();
        while (i < txt.size() && txt[i] != ']') {
            skip_ws();
            if (txt[i] == '"') out.push_back(read_quoted());
            skip_ws();
            if (i < txt.size() && txt[i] == ',') { ++i; }
            skip_ws();
        }
        if (i < txt.size() && txt[i] == ']') ++i;
        return out;
    };

    // Read objects
    skip_ws();
    while (i < txt.size() && txt[i] != ']') {
        skip_ws();
        if (txt[i] != '{') { ++i; continue; }
        ++i; // {

        Rule r;
        r.case_insensitive = true;

        // read fields until '}'
        while (i < txt.size() && txt[i] != '}') {
            skip_ws();
            if (txt[i] == '"') {
                auto key = read_quoted();
                skip_ws();
                if (i < txt.size() && txt[i] == ':') ++i;
                skip_ws();

                if (key == "intent") r.intent = read_quoted();
                else if (key == "description") r.description = read_quoted();
                else if (key == "pattern") r.pattern = read_quoted();
                else if (key == "slot_names") r.slot_names = read_slot_array();
                else if (key == "score_boost") r.score_boost = read_number();
                else if (key == "case_insensitive") r.case_insensitive = read_bool();
                else {
                    // skip value naively
                    if (i < txt.size() && txt[i] == '"') read_quoted();
                    else if (i < txt.size() && txt[i] == '[') (void)read_slot_array();
                    else if (i < txt.size() && std::isdigit((unsigned char)txt[i])) (void)read_number();
                    else ++i;
                }

                skip_ws();
                if (i < txt.size() && txt[i] == ',') ++i;
            } else {
                ++i;
            }
        }

        if (i < txt.size() && txt[i] == '}') ++i; // }

        // store rule if valid
        if (!r.intent.empty() && !r.pattern.empty())
            rules_.push_back(std::move(r));

        skip_ws();
        if (i < txt.size() && txt[i] == ',') ++i;
        skip_ws();
    }

    if (rules_.empty()) {
        if (error_out) *error_out = "No valid rules found in " + json_path;
        return false;
    }
    return true;
}

Intent NLP::parse(const std::string& text) const {
    Intent best;
    for (const auto& r : rules_) {
        std::regex_constants::syntax_option_type flags = std::regex_constants::ECMAScript;
        if (r.case_insensitive) flags = (std::regex_constants::syntax_option_type)(flags | std::regex_constants::icase);

        std::regex rx(r.pattern, flags);
        std::smatch m;
        if (std::regex_search(text, m, rx)) {
            Intent cur;
            cur.name = r.intent;
            cur.matched = true;
            cur.score = 0.5f + r.score_boost; // base score; you can tune this

            // Capture groups mapped to slot_names
            for (size_t gi = 1; gi < m.size() && gi-1 < r.slot_names.size(); ++gi) {
                cur.slots[r.slot_names[gi-1]] = m[gi].str();
            }

            // Prefer higher score or first match if equal
            if (!best.matched || cur.score > best.score) best = std::move(cur);
        }
    }
    return best;
}

std::vector<std::string> NLP::list_intents() const {
    std::vector<std::string> out;
    out.reserve(rules_.size());
    for (auto& r : rules_) out.push_back(r.intent);
    return out;
}

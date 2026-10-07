#include "../resources/models/GRIM-text/Shared/Forward/InferenceInput.hpp"
#include "../resources/models/GRIM-text/Shared/Forward/InferenceResultDecode.hpp"
#include "../resources/models/GRIM-text/Shared/UnigramByte/Detectors/DetectorRegistry.hpp"
#include "../resources/models/GRIM-text/Shared/UnigramByte/TextUtils.hpp"
#include "../resources/models/GRIM-text/Shared/DataLoader/ConceptBlockGrmtCompiler.hpp"
#include <cassert>
#include <iostream>

template<class F> void rejects(F&& f) {
    bool rejected = false;
    try { f(); } catch (const std::exception&) { rejected = true; }
    assert(rejected);
}

int main() {
    using namespace GRIM;
    using nlohmann::json;
    auto registry = Tokenizer::Detector::makeDefaultRawTextDetectorRegistry();
    const Tokenizer::Detector::RawTextDetectorOptions options(true, true);
    for (const auto& d : registry.scan("120 liters and 84 remain", options)) assert(!d.emitsAtom());
    const auto annotated = registry.scan("<INT>120</INT> liters", options);
    assert(annotated.front().emitsAtom());
    assert(annotated.front().atom_type == Tokenizer::AtomType::ATOM_INT);

    NamedConceptSpanDefinition prompt;
    prompt.name = "question";
    prompt.source_path = "/prompt";
    prompt.supervision = ConceptSpanSupervision::Context;
    prompt.open_delimiter = "<q>";
    prompt.close_delimiter = "</q>";
    NamedConceptSpanDefinition goal = prompt;
    goal.name = "target_state";
    goal.source_path = "/goal/target_state";
    goal.open_delimiter = "<target>";
    goal.close_delimiter = "</target>";
    NamedConceptSpanDefinitions definitions{prompt, goal};
    int next = 1000;
    resolveConceptSpanDelimiters(definitions, [&](const auto&) { return next++; });
    const auto source = Forward::inferenceInputSource("120 liters", {{"goal", {{"target_state", "Quantity"}}}});
    const auto rendered = SpanText::renderInputWithSpans(source, definitions);
    assert(rendered.text == "<q>\n120 liters\n</q>\n\n<target>\nQuantity\n</target>\n\n");
    for (const auto* output : {"determine", "define", "execute", "update", "answer", "raw"})
        rejects([&] { Forward::inferenceInputSource("Q", {{output, "future output"}}); });
    rejects([&] { Forward::inferenceInputSource("Q", {{"goal", {{"answer", "future output"}}}}); });
    rejects([&] { Forward::inferenceInputSource("Q", {{"goal", {{"success_criteria", {{{"criterion", "C"}, {"answer", "leak"}}}}}}}); });
    rejects([&] { Forward::inferenceInputSource("", json::object()); });
    rejects([&] { Forward::inferenceInputSource("Q", {{"knowns", "invalid"}}); });

    // Exercise the actual spacing normalization that caused the byte-prefix
    // mismatch: the dummy leading marker is behind the reinserted opening tag.
    auto tokenize = [](const std::string& content, const auto& boundaries, auto* counts) {
        Tokenizer::UniByteResult t;
        const auto normalized = Tokenizer::normalizeSpaces(content);
        for (unsigned char c : normalized) t.token_ids.push_back(Tokenizer::BYTE_TOKEN_OFFSET + c);
        for (auto boundary : boundaries)
            counts->push_back(Tokenizer::normalizeSpaces(content.substr(0, boundary)).size());
        // At offset zero the dummy prefix belongs to the first content segment.
        counts->front() = 0;
        const auto n = t.token_ids.size();
        t.is_byte_fallback.assign(n, true);
        t.token_numeric_values.assign(n, 0);
        t.token_atom_flags.assign(n, 0);
        t.token_atom_mask.assign(n, 0);
        t.atom_entry_ids.assign(n, Tokenizer::kAtomEntryNone);
        t.token_local_atom_indices.assign(n, Tokenizer::kLocalAtomIndexNone);
        return t;
    };
    const auto input = SpanText::encodeSpanText(rendered, tokenize);
    const auto training = encodeConceptBlockRender(rendered, nullptr, tokenize).value();
    assert(input.tokens.token_ids == training.token_ids);
    validateNamedConceptSpans(*input.spans, input.tokens.token_ids.size());
    assert(input.spans->entries[0].name == "question");
    assert(input.spans->entries[0].span.begin == 0);
    assert(input.spans->entries[0].span.end < input.spans->entries[1].span.end);

    std::string decoded_prefix;
    for (int id : input.tokens.token_ids) {
        if (id == definitions[0].open_delimiter_id) decoded_prefix += "<q>";
        else if (id == definitions[0].close_delimiter_id) decoded_prefix += "</q>";
        else if (id == definitions[1].open_delimiter_id) decoded_prefix += "<target>";
        else if (id == definitions[1].close_delimiter_id) decoded_prefix += "</target>";
        else decoded_prefix += static_cast<char>(id - Tokenizer::BYTE_TOKEN_OFFSET);
    }
    decoded_prefix = Tokenizer::denormalizeSpaces(decoded_prefix);
    assert(decoded_prefix.find("<q> \n") == 0);
    assert(decoded_prefix != rendered.text); // The old string-prefix guard fails here.

    // Span-delimiter interleaving must move an already-identified atom's
    // numeric/registry/local-address channels together, without rediscovery.
    const auto typed_render = SpanText::renderInputWithSpans(
        Forward::inferenceInputSource("<INT>120</INT>", json::object()), definitions);
    auto typed = SpanText::encodeSpanText(typed_render,
        [&](const std::string& content, const auto& boundaries, auto* counts) {
            auto t = tokenize(content, boundaries, counts);
            const auto detections = registry.scan(content, options);
            for (const auto& d : detections) {
                if (!d.emitsAtom()) continue;
                const auto offset = Tokenizer::normalizeSpaces(content.substr(0, d.start)).size();
                t.token_ids.at(offset) = Tokenizer::atomTypeToOpenTokenId(d.atom_type);
                t.token_numeric_values.at(offset) = 120;
                t.token_atom_mask.at(offset) = 1;
                t.token_atom_flags.at(offset) = 123;
                t.atom_entry_ids.at(offset) = 9;
                t.token_local_atom_indices.at(offset) = 2;
            }
            return t;
        });
    const auto open = std::find(typed.tokens.token_ids.begin(), typed.tokens.token_ids.end(),
                              Tokenizer::atomTypeToOpenTokenId(Tokenizer::AtomType::ATOM_INT));
    assert(open != typed.tokens.token_ids.end());
    const auto anchor = static_cast<size_t>(open - typed.tokens.token_ids.begin());
    assert(typed.tokens.token_numeric_values.at(anchor) == 120);
    assert(typed.tokens.token_atom_mask.at(anchor) == 1);
    assert(typed.tokens.token_atom_flags.at(anchor) == 123);
    assert(typed.tokens.atom_entry_ids.at(anchor) == 9);
    assert(typed.tokens.token_local_atom_indices.at(anchor) == 2);

    GeneratedSequence sequence;
    sequence.token_ids = input.tokens.token_ids;
    sequence.token_ids.push_back(Tokenizer::BYTE_TOKEN_OFFSET + 'A');
    const auto n = sequence.token_ids.size();
    sequence.atom_entry_ids.assign(n, Tokenizer::kAtomEntryNone);
    sequence.token_numeric_values.assign(n, 0);
    sequence.token_atom_mask.assign(n, 0);
    sequence.atom_entry_ids.back() = 7;
    sequence.token_numeric_values.back() = 36;
    sequence.token_atom_mask.back() = 1;
    auto continuation = Forward::continuationDecodeRequest(sequence, input.tokens.token_ids);
    assert(continuation.token_count == 1);
    assert(continuation.token_ids[0] == Tokenizer::BYTE_TOKEN_OFFSET + 'A');
    assert(continuation.atom_entry_ids[0] == 7 && continuation.token_numeric_values[0] == 36);
    assert(continuation.token_atom_mask[0] == 1);
    sequence.token_ids[0] = 99;
    rejects([&] { Forward::continuationDecodeRequest(sequence, input.tokens.token_ids); });
    sequence.token_ids = input.tokens.token_ids;
    sequence.atom_entry_ids.resize(sequence.token_ids.size());
    sequence.token_numeric_values.resize(sequence.token_ids.size());
    sequence.token_atom_mask.resize(sequence.token_ids.size());
    assert(Forward::continuationDecodeRequest(sequence, input.tokens.token_ids).token_count == 0);
    sequence.token_atom_mask.pop_back();
    rejects([&] { Forward::continuationDecodeRequest(sequence, input.tokens.token_ids); });
    sequence.token_ids.clear();
    rejects([&] { Forward::continuationDecodeRequest(sequence, input.tokens.token_ids); });
    std::cout << "Inference payload boundaries, continuation isolation, and authored atom handling passed\n";
}

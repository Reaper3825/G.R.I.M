// Detector and numeric exclusion prepass for the TokenizerHP entrypoint.

#include "../Unigram.hpp"
#include "UnigramTrainingValidation.hpp"
#include "../AtomTable.hpp"
#include "../Detectors/DetectorRegistry.hpp"
#include "../NumericTokens.hpp"
#include "../ManualVocabSource.hpp"
#include "HyperParameters/HyperparameterGroupings.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace GRIM {
namespace Tokenizer {

bool UnigramLM::trainFromCorpus(const std::vector<std::string>& texts,
                                 const ::GRIM::HyperParameters::TokenizerHP& tokenizer_hp) {
    validateTrainFromCorpusParameters(texts, tokenizer_hp);
    Detector::DetectorRegistry detector_registry;
    if (tokenizer_hp.enable_atom_reasoning) {
        detector_registry = Detector::makeDefaultRawTextDetectorRegistry();
        if (detector_registry.empty()) {
            throw std::runtime_error("UnigramLM::trainFromCorpus: detector registry initialized empty while atom reasoning is enabled");
        }
    }

    std::vector<std::vector<AtomSpan>> all_atom_spans;
    all_atom_spans.reserve(texts.size());

    size_t total_atoms = 0;
    size_t total_numeric_spans = 0;
    const bool detect = tokenizer_hp.enable_atom_reasoning;
    for (const auto& text : texts) {
        std::vector<AtomSpan> spans;
        if (detect) {
            const Detector::RawTextDetectorOptions detector_options(
                true,
                true);
            const auto detections = detector_registry.scan(text, detector_options);
            const AtomTableFromDetectionsResult atom_table_build = createAtomTableFromRawTextDetections(
                std::string_view(text.data(), text.size()),
                detections,
                "UnigramLM::trainFromCorpus");
            spans.reserve(atom_table_build.atom_tokens.size());
            for (const AtomTokenizationPayload& atom_payload : atom_table_build.atom_tokens) {
                spans.push_back({atom_payload.span.start, atom_payload.span.end});
            }
            total_atoms += spans.size();
        }

        std::vector<NumericTokenSpan> atom_exclusions;
        atom_exclusions.reserve(spans.size());
        for (const AtomSpan& span : spans) {
            atom_exclusions.push_back(NumericTokenSpan{span.start, span.end});
        }
        const std::vector<NumericTokenSpan> numeric_spans = findNumericTokenSpans(
            std::string_view(text.data(), text.size()), atom_exclusions);
        spans.reserve(spans.size() + numeric_spans.size());
        for (const NumericTokenSpan& span : numeric_spans) {
            spans.push_back(AtomSpan{span.start, span.end});
        }
        std::sort(spans.begin(), spans.end(), [](const AtomSpan& lhs, const AtomSpan& rhs) {
            return lhs.start < rhs.start;
        });
        total_numeric_spans += numeric_spans.size();
        all_atom_spans.push_back(std::move(spans));
    }

    std::cout << "[UnigramLM] Detected " << total_atoms << " atoms across "
              << texts.size() << " texts (will skip during vocab training); "
              << "atom_reasoning=" << (detect ? "on" : "off") << std::endl;
    std::cout << "[UnigramLM] Identified " << total_numeric_spans
              << " non-atom numeric spans (fixed numeric vocabulary; will skip during learned-vocab training)"
              << std::endl;

    const auto manual_pieces = loadManualVocabPieces(tokenizer_hp.manual_vocab_path);
    const bool trained = trainFromCorpus(texts, all_atom_spans, tokenizer_hp, manual_pieces);
    if (trained) {
        requireRuntimeReadyForLastTraining("UnigramLM::trainFromCorpus(TokenizerHP)");
    }
    return trained;
}

} // namespace Tokenizer
} // namespace GRIM

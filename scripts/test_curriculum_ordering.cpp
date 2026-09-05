// Standalone CPU unit test. No trainer, CUDA runtime, or model dependencies.
#include "../resources/models/GRIM-text/Shared/Batching/CurriculumOrdering.hpp"
#include "../resources/models/GRIM-text/Shared/Batching/EpochBatching.hpp"
#include "../resources/models/GRIM-text/Shared/Batching/PackerPolicy.hpp"
#include <cassert>
#include <iostream>
#include <set>

using namespace GRIM;
using namespace GRIM::Batching;

template<class F> void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
}

int main() {
    CurriculumMetadata c;
    c.courses = {{"a", {"a1", "a2", "a3"}}, {"b", {"b1", "b2", "b3"}}};
    const std::vector<std::string> authored{"a1", "a2", "a3", "b1", "b2", "b3"};
    assert(parseCurriculumOrdering("CURRICULUM") == CurriculumOrdering::CURRICULUM);
    assert(parseCurriculumOrdering("PRESERVE") == CurriculumOrdering::PRESERVE);
    assert(parseCurriculumOrdering("RANDOM") == CurriculumOrdering::RANDOM);
    rejects([] { parseCurriculumOrdering("invalid"); });
    for (int flags = 0; flags < 4; ++flags) {
        c.randomize_course_order = (flags & 1) != 0;
        c.randomize_concept_block_order = (flags & 2) != 0;
        assert(orderedConceptBlocks(c, CurriculumOrdering::PRESERVE, 42) == authored);
        const auto random = orderedConceptBlocks(c, CurriculumOrdering::RANDOM, 42);
        auto baseline = c;
        baseline.randomize_course_order = baseline.randomize_concept_block_order = false;
        assert(random == orderedConceptBlocks(baseline, CurriculumOrdering::RANDOM, 42));
        const auto order = orderedConceptBlocks(c, CurriculumOrdering::CURRICULUM, 42);
        assert(order == orderedConceptBlocks(c, CurriculumOrdering::CURRICULUM, 42));
        assert(std::set<std::string>(order.begin(), order.end()) == std::set<std::string>(authored.begin(), authored.end()));
        // Course boundaries are retained for all four flag combinations.
        for (int start : {0, 3})
            for (int i = start; i < start + 3; ++i) assert(order[i][0] == order[start][0]);
        if (!(flags & 1)) assert(order.front()[0] == 'a');
        if (!(flags & 2))
            for (int start : {0, 3}) assert(order[start][1] == '1' && order[start+1][1] == '2' && order[start+2][1] == '3');
    }
    bool global_mix_seen = false;
    bool course_shuffle_seen = false;
    bool within_shuffle_seen = false;
    for (uint64_t seed = 0; seed < 32; ++seed) {
        const auto random = orderedConceptBlocks(c, CurriculumOrdering::RANDOM, seed);
        global_mix_seen |= !(random[0][0] == random[1][0] && random[1][0] == random[2][0]);
        c.randomize_course_order = true; c.randomize_concept_block_order = false;
        course_shuffle_seen |= orderedConceptBlocks(c, CurriculumOrdering::CURRICULUM, seed).front()[0] == 'b';
        c.randomize_course_order = false; c.randomize_concept_block_order = true;
        within_shuffle_seen |= orderedConceptBlocks(c, CurriculumOrdering::CURRICULUM, seed) != authored;
    }
    assert(global_mix_seen && course_shuffle_seen && within_shuffle_seen);
    // Duplicate membership across courses is exposed once, in first authored course.
    c.courses[1].concept_block_ids.push_back("a1");
    assert(orderedConceptBlocks(c, CurriculumOrdering::PRESERVE, 1) == authored);
    const std::vector<uint32_t> lengths{4, 4, 4, 4};
    const std::vector<std::string> ids{"b1", "a1", "b1", "a2"};
    const std::vector<uint32_t> expected{1, 3, 0, 2};
    assert(orderedCourseSequences(lengths, ids, c, CurriculumOrdering::PRESERVE, 1) == expected);
    auto schedule = buildEpochBatches(lengths, 4, 2, 0, 0, 42, c, ids, CurriculumOrdering::PRESERVE, {});
    assert(schedule.batches.size() == 2);
    assert((schedule.batches[0].seq_ids == std::vector<uint32_t>{1, 3}));
    assert((schedule.batches[1].seq_ids == std::vector<uint32_t>{0, 2}));
    // Explicit order is authoritative even if the low-level caller requests shuffles.
    PackerPolicy policy;
    policy.sequence_order = expected;
    policy.batch_ordering = BatchOrdering::RANDOM;
    policy.rng_seed = 42;
    assert(buildBatches(lengths, 4, 2, policy).batches[0].seq_ids == schedule.batches[0].seq_ids);
    policy.sequence_order = std::vector<uint32_t>{0, 0, 2, 3};
    rejects([&] { buildBatches(lengths, 4, 2, policy); });
    policy.sequence_order = std::vector<uint32_t>{0};
    rejects([&] { buildBatches(lengths, 4, 2, policy); });
    rejects([&] { orderedCourseSequences(lengths, {"", "a1", "b1", "a2"}, c, CurriculumOrdering::RANDOM, 1); });
    rejects([&] { orderedCourseSequences(lengths, {"missing", "a1", "b1", "a2"}, c, CurriculumOrdering::PRESERVE, 1); });
    rejects([&] { orderedCourseSequences(lengths, {}, c, CurriculumOrdering::CURRICULUM, 1); });
    rejects([&] { orderedConceptBlocks(CurriculumMetadata{}, CurriculumOrdering::PRESERVE, 1); });
    assert((orderedCourseSequences({0, 4}, {"", "a1"}, c, CurriculumOrdering::PRESERVE, 1) == std::vector<uint32_t>{1}));
    // In RANDOM, all windows of a block remain together and ordered.
    const auto rows = orderedCourseSequences(lengths, ids, c, CurriculumOrdering::RANDOM, 7);
    auto first = std::find(rows.begin(), rows.end(), uint32_t{0});
    assert(first != rows.end() && first + 1 != rows.end() && *(first + 1) == 2);
    std::cout << "Course ordering and fixed batch packing tests passed\n";
}

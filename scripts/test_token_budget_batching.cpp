// Host-only tests: exercise the production planner and accumulation formulas.
#include "../resources/models/GRIM-text/Shared/Batching/Batching_GPU.hpp"
#include "../resources/models/GRIM-text/Shared/Batching/PackerPolicy.hpp"
#include "../resources/models/GRIM-text/Shared/Batching/TokenBudget.hpp"
#include "../resources/models/GRIM-text/Shared/Batching/AccumulationWeights.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>

using namespace GRIM::Batching;

template<class F> void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
}

int main() {
    PackerPolicy policy;
    const std::vector<uint32_t> lengths{1024, 16, 256, 800, 1024, 1, 512, 64, 0};
    const auto plan = buildBatches(lengths, 1024, 4096, policy);
    assert(plan.batch_size == 4 && plan.token_budget == 4096);
    assert(plan.batches.size() == 2 && plan.discarded_tail_sequences == 0);
    assert((plan.batches[0].seq_ids == std::vector<uint32_t>{0, 1, 2, 3}));
    assert((plan.batches[1].seq_ids == std::vector<uint32_t>{4, 5, 6, 7}));
    assert(plan.projected_total_tokens == 8192);
    assert(plan.actual_tokens == 3697);
    assert(plan.projected_padding_tokens == 8192 - 3697);

    // An unfinished rectangle is emitted, and context width is still 1024.
    const auto tail = buildBatches({4, 5, 6, 7, 8}, 1024, 4500, policy);
    assert(tail.batches.size() == 2);
    assert(tail.batches.back().seq_ids == std::vector<uint32_t>{4});
    assert(tail.projected_total_tokens == 5 * 1024);
    assert(tail.batch_size == 4); // remaining 404 slots cannot fit another row
    assert(buildBatches({10}, 1024, 1024, policy).batches.size() == 1);
    assert(buildBatches({}, 1024, 4096, policy).batches.empty());
    assert(buildBatches({0, 0}, 1024, 4096, policy).batches.empty());
    rejects([&] { buildBatches({1025}, 1024, 4096, policy); });
    rejects([&] { buildBatches({1}, 1024, 1023, policy); });
    rejects([&] { buildBatches({1}, 0, 4096, policy); });
    rejects([&] { buildBatches({1}, 1024, UINT32_MAX, policy); });
    rejects([] { tokenBudgetForRows(std::numeric_limits<int>::max(), 1024); });
    assert(tokenBudgetForRows(10, 1024) == 10240);

    policy.sequence_order = std::vector<uint32_t>{4, 3, 2, 1, 0};
    const auto ordered = buildBatches({1, 2, 3, 4, 5}, 1024, 2048, policy);
    std::vector<uint32_t> observed;
    for (const auto& batch : ordered.batches)
        observed.insert(observed.end(), batch.seq_ids.begin(), batch.seq_ids.end());
    assert(observed == *policy.sequence_order);
    policy.sequence_order.reset();
    policy.batch_ordering = BatchOrdering::RANDOM;
    policy.rng_seed = 17;
    const auto random_a = buildBatches(lengths, 1024, 2048, policy);
    const auto random_b = buildBatches(lengths, 1024, 2048, policy);
    for (size_t i = 0; i < random_a.batches.size(); ++i)
        assert(random_a.batches[i].seq_ids == random_b.batches[i].seq_ids);

    // Across varied budgets, preserve every non-empty row exactly once without
    // crossing the padded-slot budget, irrespective of actual content length.
    policy = PackerPolicy{};
    std::mt19937 rng(42);
    for (int trial = 0; trial < 200; ++trial) {
        std::vector<uint32_t> rows(1 + rng() % 150);
        for (auto& length : rows) length = rng() % 1025;
        const uint32_t budget = 1024 + rng() % 16000;
        const auto schedule = buildBatches(rows, 1024, budget, policy);
        std::vector<uint32_t> actual, expected;
        uint64_t compute = 0;
        for (uint32_t i = 0; i < rows.size(); ++i)
            if (rows[i]) expected.push_back(i);
        for (const auto& batch : schedule.batches) {
            assert(!batch.seq_ids.empty());
            assert(batch.seq_ids.size() * 1024 <= budget);
            compute += batch.seq_ids.size() * 1024;
            actual.insert(actual.end(), batch.seq_ids.begin(), batch.seq_ids.end());
        }
        assert(actual == expected && schedule.projected_total_tokens == compute);
    }

    // A microbatch's mean gradient is weighted by supervised rows, with the
    // clipping boundary's existing 1/accum_steps applied once afterwards.
    const float weighted = (objectiveAccumulationScale(100, 400, 2) * 2.0f +
                            objectiveAccumulationScale(300, 400, 2) * 6.0f) / 2;
    assert(std::abs(weighted - 5.0f) < 1e-6f);
    assert(objectiveAccumulationScale(0, 100, 2) == 0);
    assert(objectiveAccumulationScale(0, 0, 2) == 0);
    assert(objectiveAccumulationScale(100, 100, 4) / 4 == 1); // short final window
    rejects([] { objectiveAccumulationScale(10, 5, 2); });
    rejects([] { objectiveAccumulationScale(-1, 5, 2); });
    assert(optimizerWindowsPerEpoch(5, 2) == 3);
    assert(optimizerWindowsPerEpoch(1, 4) == 1);
    assert(optimizerWindowsPerEpoch(8, 2) == 4);
    std::cout << "Token-budget batching and accumulation tests passed\n";
}

"""CPU regression for production epoch-boundary saving and .opt resume state.

Execute the real runEpoch boundary and optimizer metadata assignments with
host stubs. No trainer or CUDA target is built or run.
"""
from pathlib import Path

from test_add_gradfn_host import compile_and_run
from test_unary_gradfn_host import method


def main():
    training = Path(__file__).resolve().parent.parent / 'training'
    loop = (training / 'Phases/Phase2_TrainingLoop.cu').read_text(encoding='utf-8')
    optimizer = (training / 'OptimizerCheckpoint.cu').read_text(encoding='utf-8')
    epoch = method(loop, 'EpochResult runEpoch(')
    boundary = epoch[epoch.index('result.validation = runValidation(ctx);'):]
    boundary = boundary[:boundary.rfind('}')]
    save = optimizer[optimizer.index('OptFileHeader header{};', optimizer.index('bool saveOptimizerState(')):]
    save = save[:save.index('uint64_t expected_file_bytes')]
    load = optimizer[optimizer.index('// Restore metadata into TrainingContext'):]
    load = load[:load.index('EmitModuleInfo(')]
    source = r'''
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
constexpr uint32_t OPT_MAGIC = 0x47524F50, OPT_VERSION = 1;
constexpr size_t RESERVED_BYTES = 32;
''' + method(optimizer, 'struct OptFileHeader') + r''';
struct TrainingContext {
    struct Optimizer {
        struct { int step = 0; } optimizer_step;
        int slot = 0;
        int accumulationSlot() const { return slot; }
        void setAccumulationSlot(int value) { slot = value; }
    } optimizer;
    int epochs_completed = 0, global_step = 0;
    float best_val_loss = 1.5f;
};
OptFileHeader saveHeader(const TrainingContext& ctx) {
    struct { size_t size() const { return 175; } } groups;
''' + save + r'''
    return header;
}
void loadHeader(TrainingContext& ctx, const OptFileHeader& header) {
''' + load + r'''
}
struct EpochResult { bool validation = false; };
bool validation_fails = false;
bool runValidation(TrainingContext&) {
    if (validation_fails) throw std::runtime_error("validation failed");
    return true;
}
OptFileHeader saved;
void finalizeEpochOutcome(TrainingContext& ctx, int, EpochResult& result,
                          int epoch_idx, float, int) {
    assert(result.validation);
    saved = saveHeader(ctx);
    assert(saved.current_epoch == epoch_idx + 1);
}
EpochResult completeEpoch(TrainingContext& ctx, int epoch_idx) {
    EpochResult result;
    int state = 0, epoch_start = 0;
    float epoch_loss = 0;
''' + boundary + r'''
}
int main() {
    // Fresh run's epoch-1 save must resume at zero-based epoch index 1.
    TrainingContext first;
    first.global_step = 9278;
    first.optimizer.optimizer_step.step = 4639;
    completeEpoch(first, 0);
    assert(saved.current_epoch == 1);
    TrainingContext second;
    loadHeader(second, saved);
    assert(second.epochs_completed == 1);
    assert(second.global_step == 9278);
    assert(second.optimizer.optimizer_step.step == 4639);
    assert(second.best_val_loss == 1.5f);
    second.global_step += 9278;
    second.optimizer.optimizer_step.step += 4639;
    completeEpoch(second, second.epochs_completed);
    TrainingContext third;
    loadHeader(third, saved);
    assert(third.epochs_completed == 2);
    assert(third.global_step == 18556);
    assert(third.optimizer.optimizer_step.step == 9278);
    // The final configured epoch also writes a fully completed resume cursor.
    completeEpoch(third, 5);
    assert(saved.current_epoch == 6);
    // Failed validation must not publish a completed epoch or save a boundary.
    validation_fails = true;
    const auto old_saved = saved;
    try { completeEpoch(third, 6); assert(false); }
    catch (const std::runtime_error&) {}
    assert(third.epochs_completed == 6);
    assert(std::memcmp(&old_saved, &saved, sizeof(saved)) == 0);
    std::cout << "Checkpoint epoch resume host tests passed\n";
}
'''
    compile_and_run(source)


if __name__ == '__main__':
    main()

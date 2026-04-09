#pragma once
/**
 * @file TelemetryCsvLogger.hpp
 * @brief Per-step CSV export of ALL measured telemetry state (global + local)
 * 
 * Dumps the full TelemetryState for every stream at every lattice level
 * after each update. Only measured/computed values — no constants.
 * 
 * CSV columns:
 *   global_step, stream_name, level,
 *   <raw input observation>,
 *   <all 20 measured TelemetryState fields>,
 *   <10 TelemetryVector output fields>
 */

#include "TelemetryLattice_GPU.hpp"
#include "TelemetryState_GPU.hpp"
#include <string>
#include <fstream>
#include <cstdint>

namespace GRIM::Telemetry {

class TelemetryCsvLogger {
public:
    /**
     * @brief Open CSV file and write header row
     * @param csv_path  Absolute path to output CSV file (created/truncated)
     * @param lattice   Reference to the lattice (used for config dimensions)
     * 
     * Throws std::runtime_error if file cannot be opened (Rule 20).
     */
    TelemetryCsvLogger(const std::string& csv_path, const TelemetryLattice& lattice);
    ~TelemetryCsvLogger();

    TelemetryCsvLogger(const TelemetryCsvLogger&) = delete;
    TelemetryCsvLogger& operator=(const TelemetryCsvLogger&) = delete;
    TelemetryCsvLogger(TelemetryCsvLogger&&) noexcept;
    TelemetryCsvLogger& operator=(TelemetryCsvLogger&&) noexcept;

    /**
     * @brief Snapshot all lattice state to one CSV row per (stream, level) pair
     * 
     * Reads every TelemetryState from GPU → host, writes measured fields.
     * Only emits rows for levels that actually updated at this global_step.
     * 
     * @param lattice       GPU-resident lattice
     * @param raw_obs       Raw input observations [num_streams] (host ptr)
     * @param global_step   Current training step
     */
    void log(const TelemetryLattice& lattice,
             const float* raw_obs,
             uint32_t global_step);

    /**
     * @brief Flush buffered writes to disk (call periodically or at shutdown)
     */
    void flush();

private:
    std::ofstream file_;
    int num_levels_ = 0;
    int num_streams_ = 0;

    void writeHeader();
};

} // namespace GRIM::Telemetry

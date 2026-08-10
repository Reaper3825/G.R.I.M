//======================================================//
//  DataStatsDump.cu
//  Implementation of dumpDataStats.
//======================================================//

#include "DataStatsDump.hpp"

#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#  include <windows.h>
#  include <psapi.h>
#elif defined(__APPLE__)
#  include <mach/mach.h>
#else
#  include <sys/resource.h>
#  include <unistd.h>
#endif

namespace GRIMText { namespace Training {

namespace {

// Format byte counts as "<bytes> B (<MiB> MiB / <GiB> GiB)" for readability.
std::string fmtBytes(std::size_t bytes) {
    const double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
    const double gib = mib / 1024.0;
    std::ostringstream oss;
    oss << bytes << " B (" << std::fixed << std::setprecision(2)
        << mib << " MiB, " << std::setprecision(3) << gib << " GiB)";
    return oss.str();
}

// Resident set size of the current process (host RAM actually held).
// Returns 0 if the platform query fails.
std::size_t queryHostResidentBytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<std::size_t>(pmc.WorkingSetSize);
    }
    return 0;
#elif defined(__APPLE__)
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        return static_cast<std::size_t>(info.resident_size);
    }
    return 0;
#else
    struct rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        // ru_maxrss is KB on Linux, bytes on macOS (handled above).
        return static_cast<std::size_t>(ru.ru_maxrss) * 1024;
    }
    return 0;
#endif
}

void appendMemoryStats(
    std::vector<std::pair<std::string, std::string>>& rows)
{
    const std::size_t host_rss = queryHostResidentBytes();
    rows.emplace_back("memory.host.rss",
        host_rss > 0 ? fmtBytes(host_rss) : std::string("(unavailable)"));
}

} // namespace

void dumpDataStats(
    const DataStatsSnapshot& snap,
    const DataStatsDumpOptions& opts,
    const DataStatsLogFn& log_fn)
{
    if (!log_fn) return;

    // Single (name, value) table — same shape as ConfigDump rows so the
    // visual block matches when they're emitted back-to-back.
    std::vector<std::pair<std::string, std::string>> rows;
    rows.reserve(20);
    rows.emplace_back("data_path",            snap.data_path);
    rows.emplace_back("vocab_path",           snap.vocab_path);
    rows.emplace_back("actual_vocab_size",    std::to_string(snap.actual_vocab_size));
    rows.emplace_back("train_sequence_count", std::to_string(snap.train_sequence_count));
    rows.emplace_back("val_sequence_count",   std::to_string(snap.val_sequence_count));
    appendMemoryStats(rows);

    std::size_t longest_name = 0;
    for (const auto& kv : rows) {
        if (kv.first.size() > longest_name) longest_name = kv.first.size();
    }
    const std::size_t name_w =
        (opts.name_col_width > 0) ? opts.name_col_width : longest_name;

    auto rule = [](char ch, std::size_t n) { return std::string(n, ch); };

    // Section divider — same style as ConfigDump section markers.
    {
        std::ostringstream s;
        s << "--- " << opts.banner_label << " ";
        std::string line = s.str();
        std::size_t pad = (opts.banner_width > line.size())
                              ? (opts.banner_width - line.size()) : 0;
        log_fn(opts.prefix + line + rule('-', pad));
    }

    for (const auto& kv : rows) {
        std::ostringstream line;
        line << "  " << std::left << std::setw(static_cast<int>(name_w))
             << kv.first << " = " << kv.second;
        log_fn(opts.prefix + line.str());
    }

    if (opts.show_footer) {
        log_fn(opts.prefix + rule('=', opts.banner_width));
    }
}

} } // namespace GRIMText::Training

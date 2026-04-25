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

namespace GRIMText { namespace Training {

void dumpDataStats(
    const DataStatsSnapshot& snap,
    const DataStatsDumpOptions& opts,
    const DataStatsLogFn& log_fn)
{
    if (!log_fn) return;

    // Single (name, value) table — same shape as ConfigDump rows so the
    // visual block matches when they're emitted back-to-back.
    std::vector<std::pair<std::string, std::string>> rows;
    rows.reserve(8);
    rows.emplace_back("data_path",            snap.data_path);
    rows.emplace_back("vocab_path",           snap.vocab_path);
    rows.emplace_back("tokenizer_vocab_size", std::to_string(snap.tokenizer_vocab_size));
    rows.emplace_back("actual_vocab_size",    std::to_string(snap.actual_vocab_size));
    rows.emplace_back("vocab_match",
        (snap.tokenizer_vocab_size == snap.actual_vocab_size) ? "OK" : "MISMATCH");
    rows.emplace_back("train_sequence_count", std::to_string(snap.train_sequence_count));
    rows.emplace_back("val_sequence_count",   std::to_string(snap.val_sequence_count));

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

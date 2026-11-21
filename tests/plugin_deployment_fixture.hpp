#pragma once
#include <filesystem>
#include <fstream>
#include <stdexcept>

inline void writePluginFixtureRevision(const std::filesystem::path& bundle) {
    std::ofstream marker(bundle / ".deployment");
    marker << "fixture-revision\n";
    marker.close();
    if (!marker) throw std::runtime_error("Cannot write fixture deployment marker");
}

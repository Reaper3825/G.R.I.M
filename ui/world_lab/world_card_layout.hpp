#pragma once
#include "world_document.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace GRIM::WorldLab::CardLayout {
inline constexpr float width = 230, header = 44, rowHeight = 25, footer = 12;
inline float height(size_t rows) { return header + std::max<size_t>(1, rows) * rowHeight + footer; }

// Live-world layout only. Use the same card geometry as rendering/hit testing.
// The hub stays fixed; provider cards separate along the least-overlapping axis.
inline void separate(Document& document, float gap = 36) {
    const auto count = document.layout.size();
    if (count < 2) return;
    std::unordered_map<std::string, std::string> parents;
    for (const auto& entity : document.entities) parents.emplace(entity.id, entity.parent);
    std::unordered_map<std::string, size_t> rows;
    for (const auto& entity : document.entities) {
        auto parent = entity.parent;
        // Bounded traversal also tolerates a malformed document before validation.
        for (size_t depth = 0; !parent.empty() && depth < document.entities.size(); ++depth) {
            ++rows[parent];
            const auto it = parents.find(parent);
            if (it == parents.end()) break;
            parent = it->second;
        }
    }
    std::vector<float> heights;
    heights.reserve(count);
    for (const auto& p : document.layout) heights.push_back(height(rows[p.id]));
    const auto overlap = [&](size_t i, size_t j) {
        const auto& a = document.layout[i]; const auto& b = document.layout[j];
        return std::pair{
            width + gap - std::abs(a.x - b.x),
            (heights[i] + heights[j]) / 2 + gap -
                std::abs((a.y + heights[i] / 2) - (b.y + heights[j] / 2))};
    };
    for (unsigned pass = 0; pass < 96; ++pass) {
        bool moved = false;
        for (size_t i = 0; i < count; ++i) for (size_t j = i + 1; j < count; ++j) {
            const auto [x, y] = overlap(i, j);
            if (x <= 0 || y <= 0) continue;
            auto& a = document.layout[i]; auto& b = document.layout[j];
            const float share = i == 0 ? 0.f : .5f;
            if (x < y) {
                const float delta = (x + .25f) * (b.x >= a.x ? 1.f : -1.f);
                a.x -= delta * share; b.x += delta * (1 - share);
            } else {
                const float delta = (y + .25f) *
                    (b.y + heights[j] / 2 >= a.y + heights[i] / 2 ? 1.f : -1.f);
                a.y -= delta * share; b.y += delta * (1 - share);
            }
            moved = true;
        }
        if (!moved) return;
    }
    // Dense layouts may not converge within the relaxation budget. Finish with
    // deterministic downward placement so no overlapping cards remain.
    for (size_t j = 1; j < count; ++j) {
        for (size_t i = 0; i < j;) {
            const auto [x, y] = overlap(i, j);
            if (x > 0 && y > 0) {
                document.layout[j].y = document.layout[i].y + heights[i] + gap + .25f;
                i = 0;
            } else ++i;
        }
    }
}
}

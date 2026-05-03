#include "overlay_renderer.hpp"
#include <algorithm>
#include <cmath>

void OverlayRenderer::blurRegion(int x, int y, int w, int h, int radius)
{
    std::lock_guard<std::mutex> lock(m_renderMutex);

    if (!m_pixels || w <= 0 || h <= 0 || radius <= 0)
        return;

    int x1 = std::max(0, x);
    int y1 = std::max(0, y);
    int x2 = std::min(m_width, x + w);
    int y2 = std::min(m_height, y + h);
    int rw = x2 - x1;
    int rh = y2 - y1;
    if (rw <= 0 || rh <= 0) return;

    size_t needed = static_cast<size_t>(rw) * rh;
    if (m_blurTemp.size() < needed)
        m_blurTemp.resize(needed);

    int diameter = radius * 2 + 1;
    if (m_gaussianKernel.size() != static_cast<size_t>(diameter)) {
        m_gaussianKernel.resize(diameter);
        float sigma = radius / 2.0f;
        if (sigma < 0.5f) sigma = 0.5f;
        float sum = 0;
        for (int k = -radius; k <= radius; ++k) {
            float g = std::exp(-(k * k) / (2.0f * sigma * sigma));
            m_gaussianKernel[k + radius] = g;
            sum += g;
        }
        for (int i = 0; i < diameter; ++i)
            m_gaussianKernel[i] /= sum;
    }

    uint32_t* pixels = static_cast<uint32_t*>(m_pixels);

    for (int row = 0; row < rh; ++row) {
        int py = y1 + row;
        for (int col = 0; col < rw; ++col) {
            int px = x1 + col;
            float sumA = 0, sumR = 0, sumG = 0, sumB = 0;
            for (int k = -radius; k <= radius; ++k) {
                int sx = std::clamp(px + k, x1, x2 - 1);
                float weight = m_gaussianKernel[k + radius];
                uint32_t c = pixels[py * m_width + sx];
                sumA += weight * ((c >> 24) & 0xFF);
                sumR += weight * ((c >> 16) & 0xFF);
                sumG += weight * ((c >> 8) & 0xFF);
                sumB += weight * (c & 0xFF);
            }
            auto clamp8 = [](float v) { return static_cast<uint32_t>(std::min(255, std::max(0, static_cast<int>(v + 0.5f)))); };
            m_blurTemp[row * rw + col] =
                (clamp8(sumA) << 24) | (clamp8(sumR) << 16) | (clamp8(sumG) << 8) | clamp8(sumB);
        }
    }

    auto clamp8 = [](float v) { return static_cast<uint32_t>(std::min(255, std::max(0, static_cast<int>(v + 0.5f)))); };
    for (int col = 0; col < rw; ++col) {
        for (int row = 0; row < rh; ++row) {
            float sumA = 0, sumR = 0, sumG = 0, sumB = 0;
            for (int k = -radius; k <= radius; ++k) {
                int sy = std::clamp(row + k, 0, rh - 1);
                float weight = m_gaussianKernel[k + radius];
                uint32_t c = m_blurTemp[sy * rw + col];
                sumA += weight * ((c >> 24) & 0xFF);
                sumR += weight * ((c >> 16) & 0xFF);
                sumG += weight * ((c >> 8) & 0xFF);
                sumB += weight * (c & 0xFF);
            }
            int py = y1 + row;
            int px = x1 + col;
            pixels[py * m_width + px] =
                (clamp8(sumA) << 24) | (clamp8(sumR) << 16) | (clamp8(sumG) << 8) | clamp8(sumB);
        }
    }
}

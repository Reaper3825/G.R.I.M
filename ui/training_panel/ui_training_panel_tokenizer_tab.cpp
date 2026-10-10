// UITrainingPanel: Tokenizer tab
#include "ui_training_panel_internal.hpp"
#include "ai/grim_text_server_manager.hpp"

using namespace GRIMText;
using namespace UITheme;
using namespace UITrainingPanelDetail;

namespace {
void drawCenteredTokenizerText(OverlayRenderer& renderer, float x, float y, float width,
                               const std::string& text, uint32_t color, float fontSize) {
    const float textWidth = renderer.measureTextWidth(text, fontSize);
    renderer.drawText({x + (width - textWidth) * 0.5f, y}, text, color, fontSize);
}
} // namespace

// ============================================================
// Tokenizer Runner
// ============================================================

void UITrainingPanel::handleRunTokenizer() {
    if (tokenizerFuture_.valid()) return;
    const auto url = GRIM::GRIMTextServerManager::getInstance().getServerURL();
    tokenizerRunning_ = true;
    tokenizerComplete_ = false;
    tokenizerSuccess_ = false;
    tokenizerStatusMessage_ = "Validating the loaded model tokenizer...";
    tokenizerFuture_ = std::async(std::launch::async, [url]() {
        return TokenizerClient(url).runTokenizer();
    });
}

// Background tasks return values; only the UI thread publishes widget state.
void UITrainingPanel::applyTokenizerResults() {
    const auto ready = [](auto& future) {
        return future.valid() && future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    };
    if (ready(tokenizerHealthFuture_)) serverConnected = tokenizerHealthFuture_.get();
    if (ready(tokenizerFuture_)) {
        lastTokenizerResult_ = tokenizerFuture_.get();
        tokenizerSuccess_ = lastTokenizerResult_.success;
        tokenizerComplete_ = true;
        tokenizerRunning_ = false;
        const auto& r = lastTokenizerResult_;
        tokenizerStatusMessage_ = r.success
            ? "Tokenizer OK: " + std::to_string(r.total_vocab_size) + " tokens (" +
              std::to_string(r.validation_tests_passed) + "/" + std::to_string(r.validation_tests_total) + " checks passed)"
            : "Tokenizer FAILED: " + r.error;
    }
    if (ready(encodeFuture_)) {
        lastEncodeResult_ = encodeFuture_.get();
        encodeSuccess_ = lastEncodeResult_.success;
        encodeErrorMessage_ = lastEncodeResult_.error;
        encodeComplete_ = true;
        encodeRunning_ = false;
    }
}

float UITrainingPanel::drawTokenizerStatus(OverlayRenderer& renderer, float x, float y, float width) {
    if (!tokenizerComplete_ && !tokenizerRunning_) return 0.0f;
    const uint32_t color = tokenizerRunning_ ? Colors::Warning :
        (tokenizerSuccess_ ? Colors::Success : Colors::Danger);
    const auto lines = renderer.wrapText(tokenizerStatusMessage_, width - 24.0f, 18.0f);
    const float h = 20.0f + 26.0f * lines.size();
    renderer.drawRoundedRect({x, y}, {width, h}, Colors::CardSurface, Sizes::WidgetRadius);
    renderer.drawRect({x, y + 10.0f}, {3.0f, h - 20.0f}, color);
    float textY = y + 10.0f;
    for (const auto& line : lines) {
        drawCenteredTokenizerText(renderer, x, textY, width, line, color, 18.0f);
        textY += 26.0f;
    }
    float used = h;
    if (tokenizerComplete_ && tokenizerSuccess_) {
        const auto& r = lastTokenizerResult_;
        const std::string details = std::to_string(r.byte_vocab_size) + " byte  /  " +
            std::to_string(r.numeric_vocab_size) + " numeric  /  " +
            std::to_string(r.atom_vocab_size) + " atom  /  " +
            std::to_string(r.newline_vocab_size) + " newline  /  " +
            std::to_string(r.special_token_count) + " special  /  " +
            std::to_string(static_cast<int>(r.validation_time_ms)) + " ms validation";
        used += 10.0f;
        for (const auto& line : renderer.wrapText(details, width, 16.0f)) {
            drawCenteredTokenizerText(renderer, x, y + used, width, line, Colors::TextSecondary, 16.0f);
            used += 24.0f;
        }
    }
    return used;
}

void UITrainingPanel::drawStatCard(OverlayRenderer& renderer, const Vec2& pos, const Vec2& size,
                                   const std::string& label, const std::string& value,
                                   uint32_t accentColor) {
    renderer.drawRoundedRect(pos, size, Colors::CardSurface, Sizes::WidgetRadius);
    renderer.drawRoundedBorder(pos, size, Colors::BorderSubtle, Sizes::WidgetRadius);
    renderer.drawRect({pos.x, pos.y + 6.0f}, {3.0f, size.y - 12.0f}, accentColor);
    drawCenteredTokenizerText(renderer, pos.x, pos.y + 8.0f, size.x, label, Colors::TextSecondary, 17.0f);
    drawCenteredTokenizerText(renderer, pos.x, pos.y + 36.0f, size.x, value, Colors::TextPrimary, 24.0f);
}

// ============================================================
// Tokenizer Tab
// ============================================================

void UITrainingPanel::drawTokenizerTab(OverlayRenderer& renderer, const PanelRect& content) {
    const float outerPad = 16.0f;
    const float panelW = std::min(1040.0f, content.size.x - 2.0f * outerPad);
    const float panelH = content.size.y - 2.0f * outerPad;
    if (panelW <= 64.0f || panelH <= 64.0f) return;
    const float panelX = content.origin.x + (content.size.x - panelW) * 0.5f;
    const float panelY = content.origin.y + outerPad;
    const float padding = panelW < 600.0f ? 16.0f : 24.0f;
    const float x = panelX + padding;
    const float w = panelW - 2.0f * padding;
    const float bottom = panelY + panelH - padding;
    float y = panelY + padding;

    renderer.drawRoundedRect({panelX, panelY}, {panelW, panelH}, Colors::ContentAreaBg, 16.0f);
    renderer.drawRoundedBorder({panelX, panelY}, {panelW, panelH}, Colors::BorderPrimary, 16.0f);
    renderer.pushClipRect({panelX, panelY}, {panelW, panelH});

    drawCenteredTokenizerText(renderer, x, y, w, "Tokenizer", Colors::TextHeader, 24.0f);
    y += 32.0f;
    for (const auto& line : renderer.wrapText(
            "Explore the loaded vocabulary and see how your text becomes tokens.", w, 17.0f)) {
        drawCenteredTokenizerText(renderer, x, y, w, line, Colors::TextSecondary, 17.0f);
        y += 24.0f;
    }
    y += 6.0f;
    renderer.drawRect({x, y}, {w, 1.0f}, Colors::DividerLine);
    y += 12.0f;

    if (tokenizerComplete_ || tokenizerRunning_) {
        y += drawTokenizerStatus(renderer, x, y, w) + 16.0f;
    } else {
        for (const auto& line : renderer.wrapText(
                "Load a model, then run validation to inspect its vocabulary.", w, 17.0f)) {
            drawCenteredTokenizerText(renderer, x, y, w, line, Colors::TextSecondary, 17.0f);
            y += 24.0f;
        }
        y += 12.0f;
    }

    if (tokenizerComplete_ && !tokenizerSuccess_ && !lastTokenizerResult_.failures.empty()) {
        // Keep the encode controls accessible even when validation reports many failures.
        const auto lines = renderer.wrapText(lastTokenizerResult_.failures.front(), w, 16.0f);
        for (size_t i = 0; i < std::min<size_t>(lines.size(), 2); ++i) {
            drawCenteredTokenizerText(renderer, x, y, w, lines[i], Colors::Danger, 16.0f);
            y += 24.0f;
        }
        if (lastTokenizerResult_.failures.size() > 1) {
            drawCenteredTokenizerText(renderer, x, y, w, std::to_string(lastTokenizerResult_.failures.size() - 1) +
                              " more validation failures", Colors::Danger, 16.0f);
            y += 24.0f;
        }
    }

    if (tokenizerComplete_ && tokenizerSuccess_) {
        const auto& r = lastTokenizerResult_;
        const float cardW = (w - 24.0f) / 3.0f;
        const float cardH = 72.0f;
        drawStatCard(renderer, {x, y}, {cardW, cardH},
                     "Total vocab", std::to_string(r.total_vocab_size), Colors::Primary);
        drawStatCard(renderer, {x + cardW + 12.0f, y}, {cardW, cardH},
                     "Unigram", std::to_string(r.unigram_vocab_size), Colors::AccentBlue);
        drawStatCard(renderer, {x + 2.0f * (cardW + 12.0f), y}, {cardW, cardH},
                     "Fixed tokens", std::to_string(r.total_vocab_size - r.unigram_vocab_size), Colors::Warning);
        y += cardH + 12.0f;
        const std::string special = "PAD " + std::to_string(r.pad_id) + "   UNK " +
            std::to_string(r.unk_id) + "   BOS " + std::to_string(r.bos_id) +
            "   EOS " + std::to_string(r.eos_id);
        drawCenteredTokenizerText(renderer, x, y, w, special, Colors::TextSecondary, 16.0f);
        y += 28.0f;
    }

    renderer.drawRect({x, y}, {w, 1.0f}, Colors::DividerLine);
    y += 14.0f;
    drawCenteredTokenizerText(renderer, x, y, w, "Encode text", Colors::TextHeader, 20.0f);
    y += 30.0f;

    const float buttonW = 90.0f;
    const float clearW = 70.0f;
    const float gap = 10.0f;
    const bool stacked = w < 480.0f;
    const float inputW = stacked ? w : w - buttonW - clearW - 2.0f * gap;
    encodeInputBox_->setPosition(x, y);
    encodeInputBox_->setSize(inputW, 40.0f);
    encodeInputBox_->drawOverlay(renderer, position);
    const float buttonY = stacked ? y + 50.0f : y;
    const float buttonX = stacked ? x + (w - buttonW - clearW - gap) * 0.5f : x + inputW + gap;
    encodeButton_->setPosition(buttonX, buttonY);
    encodeButton_->setSize(buttonW, 40.0f);
    encodeButton_->drawOverlay(renderer, position);
    clearEncodeButton_->setPosition(buttonX + buttonW + gap, buttonY);
    clearEncodeButton_->setSize(clearW, 40.0f);
    clearEncodeButton_->drawOverlay(renderer, position);
    y = buttonY + 40.0f + 16.0f;

    if (encodeRunning_) {
        drawCenteredTokenizerText(renderer, x, y, w, "Encoding...", Colors::Warning, 18.0f);
    } else if (!encodeComplete_) {
        drawCenteredTokenizerText(renderer, x, y, w, "Token pieces and their IDs will appear here.", Colors::TextMuted, 17.0f);
    }
    if (bottom - y > 40.0f) drawEncodeResults(renderer, x, y, w, bottom - y);
    renderer.popClipRect();
}

void UITrainingPanel::drawEncodeResults(OverlayRenderer& renderer, float x, float y, float width, float maxHeight) {
    if (!encodeComplete_) return;
    const float bottom = y + maxHeight;
    if (!encodeSuccess_) {
        for (const auto& line : renderer.wrapText("Encode failed: " + encodeErrorMessage_, width, 18.0f)) {
            if (y + 26.0f > bottom) break;
            drawCenteredTokenizerText(renderer, x, y, width, line, Colors::Danger, 18.0f);
            y += 26.0f;
        }
        return;
    }
    const auto& r = lastEncodeResult_;
    std::string summary = std::to_string(r.token_count) + " tokens  /  " +
        std::to_string(static_cast<int>(r.encode_time_ms * 1000.0)) + " us encode";
    if (r.total_vocab_size > 0) summary += "  /  vocab " + std::to_string(r.total_vocab_size);
    const auto summaryLines = renderer.wrapText(summary, width - 24.0f, 18.0f);
    const float summaryH = 20.0f + 26.0f * summaryLines.size();
    renderer.drawRoundedRect({x, y}, {width, summaryH}, Colors::CardSurface, Sizes::WidgetRadius);
    for (const auto& line : summaryLines) {
        drawCenteredTokenizerText(renderer, x, y + 10.0f, width, line, Colors::TextPrimary, 18.0f);
        y += 26.0f;
    }
    y += 20.0f + 16.0f;
    if (r.decoded_text != r.input_text) {
        drawCenteredTokenizerText(renderer, x, y, width, "Round-trip mismatch", Colors::Danger, 17.0f);
        y += 30.0f;
    }

    constexpr float chipH = 76.0f;
    constexpr float chipGap = 8.0f;
    constexpr float chipPad = 12.0f;
    constexpr float pieceSize = 20.0f;
    constexpr float idSize = 16.0f;
    const float flowBottom = bottom - 30.0f;
    float chipY = y;
    size_t shown = 0;
    static const uint32_t colors[] = {
        0xF0303B50, 0xF0344438, 0xF04A3E30,
        0xF03F3450, 0xF0304548, 0xF04A3442,
    };
    struct Chip {
        std::string text, id;
        float width, textWidth, idWidth;
    };
    std::vector<Chip> chips;
    chips.reserve(r.tokens.size());
    for (size_t i = 0; i < r.tokens.size(); ++i) {
        const auto& tok = r.tokens[i];
        std::string text = tok.piece;
        if (text.empty()) text = tok.type == "unigram" ? " " : "<" + std::to_string(tok.id) + ">";
        // Use valid UTF-8 for visible control characters; spaces remain spaces.
        std::string visible;
        for (const unsigned char c : text) {
            if (c == '\n') visible += "\xC2\xAC";
            else if (c == '\t') visible += "\xC2\xBB";
            else if (c < 0x20) visible += "\xC2\xB7";
            else visible.push_back(static_cast<char>(c));
        }
        const std::string id = std::to_string(tok.id);
        const float textW = renderer.measureTextWidth(visible, pieceSize);
        const float idW = renderer.measureTextWidth(id, idSize);
        const float chipW = std::min(width, std::max(52.0f, std::max(textW, idW) + 2.0f * chipPad));
        chips.push_back({std::move(visible), id, chipW, textW, idW});
    }
    // Measure each wrapped row before drawing so even the last row is centered.
    for (size_t rowStart = 0; rowStart < chips.size();) {
        size_t rowEnd = rowStart;
        float rowWidth = 0.0f;
        while (rowEnd < chips.size()) {
            const float nextWidth = rowWidth + (rowEnd > rowStart ? chipGap : 0.0f) + chips[rowEnd].width;
            if (rowEnd > rowStart && nextWidth > width) break;
            rowWidth = nextWidth;
            ++rowEnd;
        }
        if (chipY + chipH > flowBottom) break;
        float chipX = x + (width - rowWidth) * 0.5f;
        for (size_t i = rowStart; i < rowEnd; ++i) {
            const auto& chip = chips[i];
            const auto& visible = chip.text;
            const auto& id = chip.id;
            const float chipW = chip.width;
            const float textW = chip.textWidth;
            const float idW = chip.idWidth;
            renderer.drawRoundedRect({chipX, chipY}, {chipW, chipH}, colors[i % 6], 9.0f);
            renderer.drawRoundedBorder({chipX, chipY}, {chipW, chipH}, Colors::BorderSubtle, 9.0f);
            renderer.pushClipRect({chipX + 4.0f, chipY}, {chipW - 8.0f, chipH});
            renderer.drawText({chipX + std::max(4.0f, (chipW - textW) * 0.5f), chipY + 10.0f},
                              visible, Colors::TextPrimary, pieceSize);
            renderer.drawRect({chipX + 8.0f, chipY + 43.0f}, {chipW - 16.0f, 1.0f}, Colors::DividerLine);
            renderer.drawText({chipX + (chipW - idW) * 0.5f, chipY + 52.0f}, id, Colors::TextSecondary, idSize);
            renderer.popClipRect();
            chipX += chipW + chipGap;
            ++shown;
        }
        rowStart = rowEnd;
        chipY += chipH + chipGap;
    }
    const std::string footer = shown < r.tokens.size()
        ? std::to_string(r.tokens.size() - shown) + " more tokens beyond this view"
        : "Token text above / token ID below";
    if (bottom - y >= 24.0f)
        drawCenteredTokenizerText(renderer, x, bottom - 24.0f, width, footer, Colors::TextMuted, 16.0f);
}

void UITrainingPanel::drawTokenizerBottomBar(OverlayRenderer& renderer, float barY, float barWidth, float barX) {
    float btnW = 120.0f;
    float btnH = Sizes::ButtonHeight;
    float gap = Spacing::Small;
    float totalW = btnW + 90.0f + gap;
    float startX = barX + (barWidth - totalW) / 2.0f;

    tokenizerRunValidationBtn_->setPosition(startX, barY);
    tokenizerRunValidationBtn_->setSize(btnW, btnH);
    tokenizerRunValidationBtn_->drawOverlay(renderer, position);

    tokenizerCloseBtn_->setPosition(startX + btnW + gap, barY);
    tokenizerCloseBtn_->setSize(90.0f, btnH);
    tokenizerCloseBtn_->drawOverlay(renderer, position);
}

void UITrainingPanel::handleEncodeText() {
    if (encodeFuture_.valid() || encodeInputBuffer_.empty()) return;
    const auto url = GRIM::GRIMTextServerManager::getInstance().getServerURL();
    const auto text = encodeInputBuffer_;
    encodeRunning_ = true;
    encodeComplete_ = false;
    encodeSuccess_ = false;
    encodeErrorMessage_.clear();
    encodeFuture_ = std::async(std::launch::async, [url, text]() {
        return TokenizerClient(url).encodeText(text);
    });
}

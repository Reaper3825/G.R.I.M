// UITrainingPanel: Tokenizer tab
#include "ui_training_panel_internal.hpp"

using namespace GRIMText;
using namespace UITheme;
using namespace UITrainingPanelDetail;

// ============================================================
// Tokenizer Runner
// ============================================================

void UITrainingPanel::handleRunTokenizer() {
    if (tokenizerRunning_) {
        LOG_DEBUG("UITrainingPanel", "Tokenizer already running");
        return;
    }
    if (!trainingController) {
        LOG_ERROR("UITrainingPanel", "Cannot run tokenizer: controller not initialized");
        return;
    }
    if (!serverConnected) {
        pollServer();
        if (!serverConnected) {
            LOG_ERROR("UITrainingPanel", "Cannot run tokenizer: server not connected");
            return;
        }
    }
    
    tokenizerRunning_ = true;
    tokenizerComplete_ = false;
    tokenizerSuccess_ = false;
    tokenizerStatusMessage_ = "Running tokenizer validation...";
    LOG_DEBUG("UITrainingPanel", "Starting tokenizer validation...");
    
    // Run async to avoid blocking UI
    std::thread([this]() {
        auto result = trainingController->runTokenizer();
        
        lastTokenizerResult_ = result;
        tokenizerSuccess_ = result.success;
        tokenizerComplete_ = true;
        tokenizerRunning_ = false;
        
        if (result.success) {
            tokenizerStatusMessage_ = "Tokenizer OK: " + 
                std::to_string(result.total_vocab_size) + " tokens (" +
                std::to_string(result.validation_tests_passed) + "/" +
                std::to_string(result.validation_tests_total) + " tests passed)";
            LOG_DEBUG("UITrainingPanel", tokenizerStatusMessage_);
        } else {
            tokenizerStatusMessage_ = "Tokenizer FAILED: " + result.error;
            LOG_ERROR("UITrainingPanel", tokenizerStatusMessage_);
            for (const auto& f : result.failures) {
                LOG_ERROR("UITrainingPanel", "  - " + f);
            }
        }
    }).detach();
}

void UITrainingPanel::drawTokenizerStatus(OverlayRenderer& renderer, float x, float y, float width) {
    if (!tokenizerComplete_ && !tokenizerRunning_) return;
    
    uint32_t bgColor = tokenizerRunning_ ? Colors::ContentAreaBg :
                        (tokenizerSuccess_ ? 0xFF1A3A1A : 0xFF3A1A1A);
    uint32_t textColor = tokenizerRunning_ ? Colors::TextSecondary :
                          (tokenizerSuccess_ ? Colors::Success : Colors::Danger);
    
    float h = 24.0f;
    renderer.drawRoundedRect({x, y}, {width, h}, bgColor, Sizes::WidgetRadius);
    renderer.drawText({x + Spacing::Small, y + 4.0f}, tokenizerStatusMessage_, textColor);
    
    if (tokenizerComplete_ && tokenizerSuccess_) {
        float detailY = y + h + 2.0f;
        auto& r = lastTokenizerResult_;
        std::string details = "Vocab: " + std::to_string(r.unigram_vocab_size) + " unigram + " +
            std::to_string(r.byte_vocab_size) + " byte + " +
            std::to_string(r.atom_vocab_size) + " atom | " +
            "Load: " + std::to_string(static_cast<int>(r.load_time_ms)) + "ms | " +
            "Val: " + std::to_string(static_cast<int>(r.validation_time_ms)) + "ms";
        renderer.drawText({x + Spacing::Small, detailY}, details, Colors::TextSecondary);
    }
}

void UITrainingPanel::drawStatCard(OverlayRenderer& renderer, const Vec2& pos, const Vec2& size,
                                   const std::string& label, const std::string& value,
                                   uint32_t accentColor) {
    renderer.drawRoundedRect(pos, size, Colors::CardSurface, Sizes::WidgetRadius);
    renderer.drawRoundedBorder(pos, size, Colors::BorderSubtle, Sizes::WidgetRadius);
    renderer.drawRect({pos.x, pos.y + 6.0f}, {3.0f, size.y - 12.0f}, accentColor);
    renderer.drawText({pos.x + 12.0f, pos.y + 8.0f}, label, Colors::TextSecondary);
    renderer.drawText({pos.x + 12.0f, pos.y + 30.0f}, value, Colors::TextPrimary);
}

// ============================================================
// Tokenizer Tab
// ============================================================

void UITrainingPanel::drawTokenizerTab(OverlayRenderer& renderer, const PanelRect& content) {
    float x = content.origin.x + Spacing::PaddingX;
    float y = content.origin.y + Spacing::Small;
    float w = content.size.x - 2.0f * Spacing::PaddingX;

    // ── Section 1: Validation ──
    UIDrawHelpers::drawSectionHeader(renderer, {x - Spacing::PaddingX, y}, content.size.x,
                                     "Tokenizer Validation", Colors::SectionAI);
    y += Sizes::HeaderHeight + Spacing::Small;

    // Run Validation button
    tokenizerRunValidationBtn_->setPosition(x, y);
    tokenizerRunValidationBtn_->drawOverlay(renderer, position);

    // Running indicator
    if (tokenizerRunning_) {
        renderer.drawText({x + 130.0f, y + 8.0f}, "Running...", Colors::Warning);
    }

    y += Sizes::ButtonHeight + Spacing::Medium;

    // Validation status
    drawTokenizerStatus(renderer, x, y, w);
    if (tokenizerComplete_ || tokenizerRunning_) {
        y += 24.0f + Spacing::Small;
        if (tokenizerComplete_ && tokenizerSuccess_) {
            y += 18.0f + Spacing::Small; // detail line
        }
    }

    // Validation detail: test failures
    if (tokenizerComplete_ && !tokenizerSuccess_ && !lastTokenizerResult_.failures.empty()) {
        float failY = y;
        renderer.drawText({x, failY}, "Failed tests:", Colors::Danger);
        failY += 18.0f;
        for (const auto& f : lastTokenizerResult_.failures) {
            if (failY > content.origin.y + content.size.y - 40.0f) break;
            renderer.drawText({x + Spacing::Medium, failY}, "- " + f, Colors::TextSecondary);
            failY += 16.0f;
        }
        y = failY + Spacing::Small;
    }

    // Vocab info cards (show after successful validation)
    if (tokenizerComplete_ && tokenizerSuccess_) {
        auto& r = lastTokenizerResult_;
        float cardW = (w - 2.0f * kStatCardGap) / 3.0f;

        drawStatCard(renderer, {x, y}, {cardW, kStatCardH},
                     "Total Vocab", std::to_string(r.total_vocab_size), Colors::Primary);
        drawStatCard(renderer, {x + cardW + kStatCardGap, y}, {cardW, kStatCardH},
                     "Unigram", std::to_string(r.unigram_vocab_size), Colors::AccentBlue);
        drawStatCard(renderer, {x + 2.0f * (cardW + kStatCardGap), y}, {cardW, kStatCardH},
                     "Byte + Atom", std::to_string(r.byte_vocab_size + r.atom_vocab_size), Colors::Warning);
        y += kStatCardH + Spacing::Large;

        // Special token IDs
        std::string specialStr = "PAD=" + std::to_string(r.pad_id) +
            "  UNK=" + std::to_string(r.unk_id) +
            "  BOS=" + std::to_string(r.bos_id) +
            "  EOS=" + std::to_string(r.eos_id);
        renderer.drawText({x, y}, specialStr, Colors::TextSecondary);
        y += 18.0f + Spacing::Small;
    }

    y += Spacing::Medium;

    // ── Section 2: Encode Text ──
    UIDrawHelpers::drawSectionHeader(renderer, {x - Spacing::PaddingX, y}, content.size.x,
                                     "Encode Text", Colors::SectionNeutral);
    y += Sizes::HeaderHeight + Spacing::Small;

    // Input box + buttons row
    float inputW = w - 90.0f - 70.0f - 2.0f * Spacing::Small;
    encodeInputBox_->setPosition(x, y);
    encodeInputBox_->setSize(inputW, 28.0f);
    encodeInputBox_->drawOverlay(renderer, position);

    encodeButton_->setPosition(x + inputW + Spacing::Small, y - 2.0f);
    encodeButton_->drawOverlay(renderer, position);

    clearEncodeButton_->setPosition(x + inputW + Spacing::Small + 90.0f + Spacing::Small, y - 2.0f);
    clearEncodeButton_->drawOverlay(renderer, position);

    if (encodeRunning_.load()) {
        renderer.drawText({x, y + 32.0f}, "Encoding...", Colors::Warning);
    }

    y += 28.0f + Spacing::Medium;

    // ── Encode results ──
    float remainingH = (content.origin.y + content.size.y) - y - Spacing::Small;
    if (remainingH > 40.0f) {
        drawEncodeResults(renderer, x, y, w, remainingH);
    }
}

void UITrainingPanel::drawEncodeResults(OverlayRenderer& renderer, float x, float y, float width, float maxHeight) {
    if (!encodeComplete_ && !encodeRunning_.load()) return;

    if (!encodeSuccess_ && encodeComplete_) {
        // Error display
        renderer.drawRoundedRect({x, y}, {width, 24.0f}, 0xFF3A1A1A, Sizes::WidgetRadius);
        renderer.drawText({x + Spacing::Small, y + 4.0f},
                          "Encode failed: " + encodeErrorMessage_, Colors::Danger);
        return;
    }

    if (!encodeComplete_) return;

    std::lock_guard<std::mutex> lock(encodeMutex_);
    auto& r = lastEncodeResult_;

    // Summary bar
    std::string summary = std::to_string(r.token_count) + " tokens | " +
        std::to_string(static_cast<int>(r.encode_time_ms * 1000.0)) + "us encode";
    if (r.total_vocab_size > 0) {
        summary += " | vocab " + std::to_string(r.total_vocab_size);
    }
    renderer.drawRoundedRect({x, y}, {width, 22.0f}, Colors::CardSurface, Sizes::SmallRadius);
    renderer.drawText({x + Spacing::Small, y + 3.0f}, summary, Colors::TextPrimary);
    y += 22.0f + Spacing::Small;

    // Round-trip check
    if (r.decoded_text != r.input_text) {
        renderer.drawText({x, y}, "Round-trip mismatch!", Colors::Danger);
        y += 16.0f;
    }

    // Token flow — colored chips showing the tokenization
    float chipX = x;
    float chipY = y;
    float chipH = 26.0f;
    float chipGap = 3.0f;
    float chipPadX = 6.0f;

    // Alternating colors for visual token boundary separation
    static const uint32_t kTokenColors[] = {
        0xD0283050,  // blue-ish
        0xD0304028,  // green-ish
        0xD0403028,  // amber-ish
        0xD0382850,  // purple-ish
        0xD0284040,  // teal-ish
        0xD0402840,  // magenta-ish
    };
    static constexpr int kNumTokenColors = 6;

    for (size_t i = 0; i < r.tokens.size(); ++i) {
        const auto& tok = r.tokens[i];

        // Determine display text
        std::string displayText = tok.piece;
        if (displayText.empty()) {
            displayText = "<" + std::to_string(tok.id) + ">";
        }
        // Replace control chars for display
        for (char& c : displayText) {
            if (c == '\n') c = '\xAC';  // ¬ for newline
            else if (c == '\t') c = '\xBB'; // » for tab
            else if (c < 0x20 && c >= 0) c = '\xB7'; // · for other control
        }

        float textW = UIDrawHelpers::getTextWidth(displayText);
        float chipW = textW + 2.0f * chipPadX;

        // Wrap to next line
        if (chipX + chipW > x + width && chipX > x) {
            chipX = x;
            chipY += chipH + chipGap;
            if (chipY + chipH > y + maxHeight - 40.0f) break; // out of space
        }

        // Chip background
        uint32_t bgColor = kTokenColors[i % kNumTokenColors];
        renderer.drawRoundedRect({chipX, chipY}, {chipW, chipH}, bgColor, Sizes::SmallRadius);

        // Token text
        renderer.drawText({chipX + chipPadX, chipY + 5.0f}, displayText, Colors::TextPrimary);

        // Token ID subscript
        std::string idStr = std::to_string(tok.id);
        renderer.drawText({chipX + chipPadX, chipY + chipH - 10.0f}, idStr, Colors::TextSecondary);

        chipX += chipW + chipGap;
    }

    // Token type legend at bottom
    chipY += chipH + Spacing::Medium;
    if (chipY < y + maxHeight - 20.0f) {
        renderer.drawText({x, chipY},
                          "Types: special | byte | atom | unigram", Colors::TextSecondary);
    }
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
    if (encodeRunning_.load()) return;
    if (encodeInputBuffer_.empty()) return;
    if (!trainingController) {
        encodeErrorMessage_ = "Controller not initialized";
        encodeComplete_ = true;
        encodeSuccess_ = false;
        return;
    }
    if (!serverConnected) {
        pollServer();
        if (!serverConnected) {
            encodeErrorMessage_ = "Server not connected";
            encodeComplete_ = true;
            encodeSuccess_ = false;
            return;
        }
    }

    encodeRunning_.store(true);
    encodeComplete_ = false;
    encodeSuccess_ = false;
    encodeErrorMessage_.clear();

    std::string textCopy = encodeInputBuffer_;
    std::thread([this, textCopy]() {
        auto result = trainingController->encodeText(textCopy);

        {
            std::lock_guard<std::mutex> lock(encodeMutex_);
            lastEncodeResult_ = result;
        }
        encodeSuccess_ = result.success;
        encodeComplete_ = true;
        encodeRunning_.store(false);

        if (!result.success) {
            encodeErrorMessage_ = result.error;
            LOG_ERROR("UITrainingPanel", "Encode failed: " + result.error);
        } else {
            LOG_DEBUG("UITrainingPanel", "Encoded " + std::to_string(result.token_count) + " tokens");
        }
    }).detach();
}

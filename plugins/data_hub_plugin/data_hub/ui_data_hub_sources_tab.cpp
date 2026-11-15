#include "ui_data_hub_internal.hpp"

using namespace UIDataHubDetail;

// Sources tab layout
// =========================================================

void UIDataHubPanel::drawSourcesTab(OverlayRenderer& renderer, const PanelRect& content) {
    float x = content.origin.x + 15.0f;
    float fullW = content.size.x - 30.0f;
    float scrollAreaTop = content.origin.y + 5.0f;
    float scrollAreaH   = content.size.y - 60.0f;

    // Count enabled sources for header
    size_t enabledCount = 0;
    for (const auto& c : sourceCards_) { if (c.enabled) enabledCount++; }
    std::string header = "Data Sources (" + std::to_string(enabledCount)
                       + "/" + std::to_string(sourceCards_.size()) + ")";
    renderer.drawText({x + 2.0f, scrollAreaTop - 1.0f}, header,
                      UITheme::Colors::TextSecondary);

    // Scroll area background
    float cardAreaTop = scrollAreaTop + 16.0f;
    float cardAreaH   = scrollAreaH - 16.0f;
    renderer.drawRoundedRect({x, cardAreaTop}, {fullW, cardAreaH},
                             UITheme::Colors::ContentAreaBg, UITheme::Sizes::WidgetRadius);
    renderer.drawRoundedBorder({x, cardAreaTop}, {fullW, cardAreaH},
                               UITheme::Colors::BorderSubtle, UITheme::Sizes::WidgetRadius);

    // Clip to card area
    renderer.pushClipRect({x, cardAreaTop}, {fullW, cardAreaH});

    float cardY = cardAreaTop + 8.0f - sourcesScrollOffset_;

    for (size_t i = 0; i < sourceCards_.size(); i++) {
        auto& card = sourceCards_[i];
        float cardX = x + 8.0f;
        float cardW = fullW - 16.0f;

        if (cardY + kCardHeight < cardAreaTop || cardY > cardAreaTop + cardAreaH) {
            cardY += kCardHeight + kCardGap;
            continue;
        }

        // ── Card background ─────────────────────────
        renderer.drawRoundedRect({cardX, cardY}, {cardW, kCardHeight},
                                 UITheme::Colors::CardSurface, UITheme::Sizes::WidgetRadius);
        renderer.drawRoundedBorder({cardX, cardY}, {cardW, kCardHeight},
                                   UITheme::Colors::BorderSubtle, UITheme::Sizes::WidgetRadius);

        float innerX = cardX + kCardInnerPad;
        float innerW = cardW - 2.0f * kCardInnerPad;
        float rowY   = cardY;

        // ── Top bar: number + enabled toggle + delete ──
        renderer.drawText({innerX, rowY + 7.0f},
                          std::to_string(i + 1),
                          UITheme::Colors::TextMuted);

        float toggleW = 50.0f;
        float delW = 28.0f;
        float topRightX = cardX + cardW - kCardInnerPad;

        card.deleteBtn->setPosition(topRightX - delW, rowY + 2.0f);
        card.deleteBtn->setSize(delW, 24.0f);
        card.deleteBtn->drawOverlay(renderer, position);

        card.enabledToggle->setPosition(topRightX - delW - toggleW - 8.0f, rowY + 2.0f);
        card.enabledToggle->setSize(toggleW, 24.0f);
        card.enabledToggle->drawOverlay(renderer, position);

        // Divider
        rowY += kCardTopBarH;
        renderer.drawRect({innerX, rowY - 1.0f}, {innerW, 1.0f}, UITheme::Colors::DividerLine);

        // ── Row 1: Name | URL ───────────────────────
        float halfW = (innerW - 10.0f) / 2.0f;

        card.nameInput->setPosition(innerX, rowY + 2.0f);
        card.nameInput->setSize(halfW, kCardFieldH);
        card.nameInput->drawOverlay(renderer, position);

        card.urlInput->setPosition(innerX + halfW + 10.0f, rowY + 2.0f);
        card.urlInput->setSize(halfW, kCardFieldH);
        card.urlInput->drawOverlay(renderer, position);

        float labelY = rowY + kCardFieldH + kCardLabelGap;
        renderer.drawText({innerX + 2.0f, labelY}, "NAME",
                          UITheme::Colors::TextSecondary);
        renderer.drawText({innerX + halfW + 12.0f, labelY}, "URL",
                          UITheme::Colors::TextSecondary);

        rowY += kCardRowH;

        // ── Row 2: Priority | Depth | Limit ──
        float col3W = (innerW - 20.0f) / 3.0f;
        float qx = innerX;

        card.prioritySlider->setPosition(qx, rowY + 2.0f);
        card.prioritySlider->setSize(col3W, kCardFieldH);
        card.prioritySlider->drawOverlay(renderer, position);

        qx += col3W + 10.0f;
        card.depthSlider->setPosition(qx, rowY + 2.0f);
        card.depthSlider->setSize(col3W, kCardFieldH);
        card.depthSlider->drawOverlay(renderer, position);

        qx += col3W + 10.0f;
        card.limitSlider->setPosition(qx, rowY + 2.0f);
        card.limitSlider->setSize(col3W, kCardFieldH);
        card.limitSlider->drawOverlay(renderer, position);

        labelY = rowY + kCardFieldH + kCardLabelGap;
        qx = innerX;
        renderer.drawText({qx + 2.0f, labelY}, "PRIORITY",
                          UITheme::Colors::TextSecondary);
        qx += col3W + 10.0f;
        renderer.drawText({qx + 2.0f, labelY}, "CRAWL DEPTH",
                          UITheme::Colors::TextSecondary);
        qx += col3W + 10.0f;
        renderer.drawText({qx + 2.0f, labelY}, "FETCH LIMIT",
                          UITheme::Colors::TextSecondary);

        cardY += kCardHeight + kCardGap;
    }

    if (sourceCards_.empty()) {
        float cx = x + fullW / 2.0f - 120.0f;
        float cy = cardAreaTop + cardAreaH / 2.0f - 10.0f;
        renderer.drawText({cx, cy},
                          "No sources configured. Click '+ Add Source' below.",
                          UITheme::Colors::TextDisabled);
    }

    renderer.popClipRect();

    // Scroll indicator (thin bar on right side)
    float totalH = sourceCards_.size() * (kCardHeight + kCardGap) + 10.0f;
    if (totalH > cardAreaH) {
        float scrollRatio  = sourcesScrollOffset_ / (totalH - cardAreaH);
        float thumbRatio   = cardAreaH / totalH;
        float thumbH       = std::max(20.0f, cardAreaH * thumbRatio);
        float thumbY       = cardAreaTop + scrollRatio * (cardAreaH - thumbH);
        float barX         = x + fullW - 6.0f;
        renderer.drawRoundedRect({barX, thumbY}, {4.0f, thumbH},
                                 UITheme::Colors::ScrollThumb, 2.0f);
    }

    // ── Add Source button (centered below card area) ──
    float addBtnW = 220.0f;
    float addBtnH = 36.0f;
    float addBtnX = x + (fullW - addBtnW) / 2.0f;
    float addBtnY = cardAreaTop + cardAreaH + 10.0f;
    btnAddCard_->setPosition(addBtnX, addBtnY);
    btnAddCard_->setSize(addBtnW, addBtnH);
    btnAddCard_->drawOverlay(renderer, position);
    btnReloadSources_->setPosition(addBtnX + addBtnW + 10.0f, addBtnY);
    btnReloadSources_->setSize(150.0f, addBtnH);
    btnReloadSources_->drawOverlay(renderer, position);
}

// =========================================================
// HuggingFace tab layout
// =========================================================

// =========================================================
// Sources tab actions
// =========================================================

UIDataHubPanel::SourceCard UIDataHubPanel::buildSourceCard(
    const std::string& name, const std::string& url,
    int priority, int depth, int limit, bool enabled)
{
    SourceCard card;
    card.cardId     = nextSourceCardId_++;
    card.name       = name;
    card.url        = url;
    card.priority   = priority;
    card.enabled    = enabled;
    card.crawlDepth = depth;
    card.fetchLimit = limit;

    card.nameInput = std::make_shared<UIInputBox>();
    card.nameInput->setPlaceholder("Source name...");
    if (!name.empty() && name != "New Source") card.nameInput->setText(name);

    card.urlInput = std::make_shared<UIInputBox>();
    card.urlInput->setPlaceholder("https://...");
    if (!url.empty()) card.urlInput->setText(url);

    card.prioritySlider = std::make_shared<UISlider>(
        "Priority", 1.0f, 10.0f, static_cast<float>(priority),
        [this](float) { sourcesDirty_ = true; }, 1.0f);

    card.depthSlider = std::make_shared<UISlider>(
        "Crawl Depth", 1.0f, 5.0f, static_cast<float>(depth),
        [this](float) { sourcesDirty_ = true; }, 1.0f);

    card.limitSlider = std::make_shared<UISlider>(
        "Fetch Limit", 0.0f, 5000.0f, static_cast<float>(limit),
        [this](float) { sourcesDirty_ = true; }, 50.0f);

    card.enabledToggle = std::make_shared<UIToggle>(
        "", enabled, [this](bool) { sourcesDirty_ = true; });

    size_t id = card.cardId;
    card.deleteBtn = std::make_shared<UIButton>("x", [this, id]() {
        cardToDelete_ = static_cast<int>(id);
    });

    return card;
}

void UIDataHubPanel::removeSourceCard(size_t cardId) {
    sourceCards_.erase(
        std::remove_if(sourceCards_.begin(), sourceCards_.end(),
            [cardId](const SourceCard& c) { return c.cardId == cardId; }),
        sourceCards_.end());
    sourcesDirty_ = true;
}


void UIDataHubPanel::loadSourceCards() {
    const auto rows = DataHub::sources();
    std::vector<SourceCard> cards;
    for (const auto& row : rows) {
        auto card = buildSourceCard(row.at("name").get<std::string>(), row.at("url").get<std::string>(),
            row.at("priority").get<int>(), row.at("crawl_depth").get<int>(),
            row.at("fetch_limit").get<int>(), row.at("enabled").get<bool>());
        card.requiresAuth = row.value("requires_auth", false);
        cards.push_back(std::move(card));
    }
    sourceCards_ = std::move(cards);
    sourceBaseline_ = rows;
    sourcesDirty_ = false;
    totalSources_ = static_cast<int>(std::count_if(sourceCards_.begin(), sourceCards_.end(),
        [](const auto& card) { return card.enabled; }));
}
void UIDataHubPanel::saveSourceCards() {
    nlohmann::json rows = nlohmann::json::array();
    for (const auto& card : sourceCards_)
        rows.push_back({{"name",card.name},{"url",card.url},{"source_type",detectFetcherFromUrl(card.url)},
            {"priority",card.priority},{"enabled",card.enabled},{"crawl_depth",card.crawlDepth},
            {"fetch_limit",card.fetchLimit},{"requires_auth",card.requiresAuth}});
    try {
        const auto saved = DataHub::saveSources(rows, sourceBaseline_);
        sourceBaseline_ = saved;
        if (saved.size() != rows.size()) loadSourceCards();
        addLog("Sources saved", 0);
    } catch (const std::exception& error) { addLog(error.what(), 2); }
    sourcesDirty_ = false;
}

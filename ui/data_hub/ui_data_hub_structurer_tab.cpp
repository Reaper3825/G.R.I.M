#include "ui_data_hub_internal.hpp"

using namespace UIDataHubDetail;

// Structurer tab layout
// =========================================================

void UIDataHubPanel::drawStructurerTab(OverlayRenderer& renderer, const PanelRect& content) {
    float x = content.origin.x + 15.0f;
    float y = content.origin.y + 10.0f;
    float fullW = content.size.x - 30.0f;

    // ── Toolbar row 1 ───────────────────────────────────
    float ddW = 260.0f;
    float sliderW = 280.0f;
    float gap = 14.0f;
    float rowH = 36.0f;
    float genW = 90.0f;

    float cx = x;
    modelDropdown_->setPosition(cx, y);
    modelDropdown_->setSize(ddW, rowH);
    modelDropdown_->drawOverlay(renderer, position);
    cx += ddW + gap;

    formatDropdown_->setPosition(cx, y);
    formatDropdown_->setSize(ddW, rowH);
    formatDropdown_->drawOverlay(renderer, position);
    cx += ddW + 8.0f;

    btnGenerate_->setPosition(cx, y);
    btnGenerate_->setSize(genW, rowH);
    btnGenerate_->drawOverlay(renderer, position);
    cx += genW + gap;

    viewModeDropdown_->setPosition(cx, y);
    viewModeDropdown_->setSize(ddW, rowH);
    viewModeDropdown_->drawOverlay(renderer, position);

    float rightX = x + fullW;
    float menuW = 110.0f;
    float btnGap = 8.0f;

    structureActionMenu_->setPosition(rightX - menuW, y);
    structureActionMenu_->setSize(menuW, 28.0f);
    structureActionMenu_->drawOverlay(renderer, position);
    y += rowH + kStructRowGap;

    // ── Toolbar row 2 ───────────────────────────────────

    sliderMaxEntries_->setPosition(x, y);
    sliderMaxEntries_->setSize(sliderW, rowH);
    sliderMaxEntries_->drawOverlay(renderer, position);

    sliderParallel_->setPosition(x + sliderW + gap, y);
    sliderParallel_->setSize(sliderW, rowH);
    sliderParallel_->drawOverlay(renderer, position);

    datasetActionMenu_->setPosition(rightX - menuW, y);
    datasetActionMenu_->setSize(menuW, 28.0f);
    datasetActionMenu_->drawOverlay(renderer, position);
    y += rowH + kStructRowGap;

    // ── Search bar (Dataset View) or header (Sequence View) ──

    float searchH = 28.0f;

    if (structViewMode_ == 0) {
        structSearchInput_->setPosition(x, y);
        structSearchInput_->setSize(fullW * 0.4f, searchH);
        structSearchInput_->drawOverlay(renderer, position);

        float navX = x + fullW * 0.4f + 20.0f;
        btnPrevSeq_->setPosition(navX, y);
        btnPrevSeq_->setSize(35.0f, searchH);
        btnPrevSeq_->drawOverlay(renderer, position);

        std::string seqLabel = std::to_string(currentSequenceIndex_ + 1)
                             + " / " + std::to_string(totalSequences_);
        renderer.drawText({navX + 42.0f, y + 6.0f}, seqLabel, UITheme::Colors::TextPrimary);

        btnNextSeq_->setPosition(navX + 130.0f, y);
        btnNextSeq_->setSize(35.0f, searchH);
        btnNextSeq_->drawOverlay(renderer, position);
        y += searchH + kStructSectionGap;
    } else if (structViewMode_ == 1) {
        size_t enabledCards = sequenceCards_.size();
        std::string header = "Sequences (" + std::to_string(enabledCards) + ")";
        renderer.drawText({x + 2.0f, y + 6.0f}, header, UITheme::Colors::TextSecondary);
        y += searchH + kStructSectionGap;
    }

    UIDrawHelpers::drawDivider(renderer, {x, y}, fullW);
    y += kStructLabelSpace;

    // ── Bottom-anchored layout ──────────────────────────
    // Status bar + custom prompt section are anchored to bottom

    float statusY     = content.origin.y + content.size.y - 30.0f;
    float promptTextH = 50.0f;
    float promptLabelH = 16.0f;
    float promptGap    = 6.0f;
    float promptTopY   = statusY - promptTextH - promptGap;
    float promptLabelY = promptTopY - promptLabelH - 2.0f;
    float contentEndY  = promptLabelY - 8.0f;

    // ── Content split: Dataset View / Sequence View / Curriculum View ──

    if (structViewMode_ == 2) {
        drawCurriculumView(renderer, content, x, y, fullW, contentEndY);
        return;
    }

    if (structViewMode_ == 0) {
        // ── Dual text areas ─────────────────────────────
        float areaH = contentEndY - y;
        if (areaH < 100.0f) areaH = 100.0f;
        float halfW = (fullW - 10.0f) / 2.0f;

        rawTextArea_->setPosition(x, y);
        rawTextArea_->setSize(halfW, areaH);
        rawTextArea_->drawOverlay(renderer, position);

        structuredTextArea_->setPosition(x + halfW + 10.0f, y);
        structuredTextArea_->setSize(halfW, areaH);
        structuredTextArea_->drawOverlay(renderer, position);
        y += areaH + 5.0f;
    } else if (structViewMode_ == 1) {
        // ── Sequence cards (scrollable) ─────────────────
        float cardAreaH = contentEndY - y - 46.0f;
        if (cardAreaH < 120.0f) cardAreaH = 120.0f;

        seqCardAreaTop_ = y;
        seqCardAreaH_   = cardAreaH;

        renderer.drawRoundedRect({x, y}, {fullW, cardAreaH},
                                 UITheme::Colors::ContentAreaBg, UITheme::Sizes::WidgetRadius);
        renderer.drawRoundedBorder({x, y}, {fullW, cardAreaH},
                                   UITheme::Colors::BorderSubtle, UITheme::Sizes::WidgetRadius);

        renderer.pushClipRect({x, y}, {fullW, cardAreaH});

        float cardY = y + 8.0f - seqScrollOffset_;

        for (size_t ci = 0; ci < sequenceCards_.size(); ++ci) {
            auto& card = sequenceCards_[ci];
            float cardH = sequenceCardHeight(card);
            float cardX = x + 8.0f;
            float cardW = fullW - 16.0f;

            if (cardY + cardH < y || cardY > y + cardAreaH) {
                cardY += cardH + kCardGap;
                continue;
            }

            // Card background
            renderer.drawRoundedRect({cardX, cardY}, {cardW, cardH},
                                     UITheme::Colors::CardSurface, UITheme::Sizes::WidgetRadius);
            renderer.drawRoundedBorder({cardX, cardY}, {cardW, cardH},
                                       UITheme::Colors::BorderSubtle, UITheme::Sizes::WidgetRadius);

            float innerX = cardX + kSeqCardInnerPad;
            float innerW = cardW - 2.0f * kSeqCardInnerPad;
            float rowY = cardY;

            // Top bar: number + format dropdown + generate + delete
            renderer.drawText({innerX, rowY + 9.0f},
                              "#" + std::to_string(ci + 1),
                              UITheme::Colors::TextMuted);

            float fmtDdW = 180.0f;
            float genBtnW = 80.0f;
            float delBtnW = 28.0f;

            card.formatDropdown->setPosition(innerX + 30.0f, rowY + 2.0f);
            card.formatDropdown->setSize(fmtDdW, 28.0f);
            card.formatDropdown->drawOverlay(renderer, position);

            card.generateBtn->setPosition(innerX + 30.0f + fmtDdW + 8.0f, rowY + 2.0f);
            card.generateBtn->setSize(genBtnW, 28.0f);
            card.generateBtn->drawOverlay(renderer, position);

            float topRightX = cardX + cardW - kSeqCardInnerPad;
            card.deleteBtn->setPosition(topRightX - delBtnW, rowY + 2.0f);
            card.deleteBtn->setSize(delBtnW, 28.0f);
            card.deleteBtn->drawOverlay(renderer, position);

            rowY += kSeqCardTopBarH;
            renderer.drawRect({innerX, rowY - 1.0f}, {innerW, 1.0f}, UITheme::Colors::DividerLine);

            // Entry fields
            for (auto& entry : card.entries) {
                renderer.drawText({innerX + 2.0f, rowY + 2.0f}, entry.prefix,
                                  UITheme::Colors::TextSecondary);
                rowY += kSeqCardEntryLabelH;

                entry.textArea->setPosition(innerX, rowY);
                entry.textArea->setSize(innerW, kSeqCardEntryTextH);
                entry.textArea->drawOverlay(renderer, position);
                rowY += kSeqCardEntryTextH + kSeqCardEntryGapH;
            }

            // Add Entry button (Thought format only)
            if (card.formatIndex == 1) {
                float addW = 90.0f;
                card.addEntryBtn->setPosition(innerX, rowY);
                card.addEntryBtn->setSize(addW, 24.0f);
                card.addEntryBtn->drawOverlay(renderer, position);
                rowY += kSeqCardBtnRowH;
            }

            // Save button row
            float saveBtnW = 80.0f;
            card.saveBtn->setPosition(innerX + innerW - saveBtnW, rowY);
            card.saveBtn->setSize(saveBtnW, 24.0f);
            card.saveBtn->drawOverlay(renderer, position);

            cardY += cardH + kCardGap;
        }

        if (sequenceCards_.empty()) {
            float emptyX = x + fullW / 2.0f - 160.0f;
            float emptyY = y + cardAreaH / 2.0f - 10.0f;
            renderer.drawText({emptyX, emptyY},
                              "No sequences. Click '+ Add Sequence' below.",
                              UITheme::Colors::TextDisabled);
        }

        renderer.popClipRect();

        // Scroll indicator
        float totalCardH = 10.0f;
        for (const auto& c : sequenceCards_)
            totalCardH += sequenceCardHeight(c) + kCardGap;
        if (totalCardH > cardAreaH) {
            float scrollRatio = seqScrollOffset_ / (totalCardH - cardAreaH);
            float thumbRatio  = cardAreaH / totalCardH;
            float thumbH      = std::max(20.0f, cardAreaH * thumbRatio);
            float thumbY      = y + scrollRatio * (cardAreaH - thumbH);
            float barX        = x + fullW - 6.0f;
            renderer.drawRoundedRect({barX, thumbY}, {4.0f, thumbH},
                                     UITheme::Colors::ScrollThumb, 2.0f);
        }

        y += cardAreaH + 10.0f;

        // Add Sequence button (centered below card area)
        float addBtnW = 220.0f;
        float addBtnH = 36.0f;
        float addBtnX = x + (fullW - addBtnW) / 2.0f;
        btnAddSequence_->setPosition(addBtnX, y);
        btnAddSequence_->setSize(addBtnW, addBtnH);
        btnAddSequence_->drawOverlay(renderer, position);

        y += addBtnH + 5.0f;
    }

    // ── Custom prompt + type ────────────────────────────

    renderer.drawText({x + 2.0f, promptLabelY}, "Custom Prompt",
                      UITheme::Colors::TextSecondary);

    std::string currentFmt = formatDropdown_ ? formatDropdown_->getSelectedItem() : "Q/A";
    float fmtLabelW = UIDrawHelpers::getTextWidth("Type: " + currentFmt);
    renderer.drawText({x + fullW - fmtLabelW - 2.0f, promptLabelY},
                      "Type: " + currentFmt, UITheme::Colors::TextMuted);

    float appendBtnW = 80.0f;
    float promptAreaW = fullW - appendBtnW - 8.0f;

    customPromptArea_->setPosition(x, promptTopY);
    customPromptArea_->setSize(promptAreaW, promptTextH);
    customPromptArea_->drawOverlay(renderer, position);

    btnAppendEntry_->setPosition(x + promptAreaW + 8.0f, promptTopY);
    btnAppendEntry_->setSize(appendBtnW, promptTextH);
    btnAppendEntry_->drawOverlay(renderer, position);

    // ── Status bar ──────────────────────────────────────

    renderer.drawRoundedRect({x, statusY}, {fullW, 25.0f},
                             UITheme::Colors::Background, UITheme::Sizes::SmallRadius);

    std::string status = "Total: " + std::to_string(totalSequences_)
                       + "  |  Assigned: " + std::to_string(assignedSequences_)
                       + "  |  Structured: " + std::to_string(structuredCount_)
                       + "  |  Failed: " + std::to_string(failedCount_);
    renderer.drawText({x + 10.0f, statusY + 5.0f}, status, UITheme::Colors::TextSecondary);
}

// =========================================================
// Curriculum view
// =========================================================

static uint32_t subjectBadgeColor(const std::string& subj) {
    if (subj == "code")    return 0xFF5090FF;
    if (subj == "math")    return 0xFFF0B040;
    if (subj == "science") return 0xFF50E080;
    if (subj == "history") return 0xFFA090FF;
    if (subj == "medical") return 0xFFE84060;
    if (subj == "legal")   return 0xFF6B90F0;
    return 0xFF8888A0;
}

static uint32_t qualityBadgeColor(const std::string& q) {
    if (q == "high")   return 0xFF50E080;
    if (q == "medium") return 0xFFF0B040;
    if (q == "low")    return 0xFFE84060;
    return 0xFF8888A0;
}

void UIDataHubPanel::rebuildFilteredPool() {
    filteredPoolIndices_.clear();
    if (!datasetTarget_) return;

    std::string subj;
    if (subjectFilterDropdown_ && filterSubjectIdx_ > 0)
        subj = subjectFilterDropdown_->getSelectedItem();

    std::string qual;
    if (qualityFilterDropdown_ && filterQualityIdx_ > 0)
        qual = qualityFilterDropdown_->getSelectedItem();

    std::string query;
    if (poolSearchInput_)
        query = poolSearchInput_->getText();

    filteredPoolIndices_ = datasetTarget_->filterSequences(subj, qual, query);
    poolFilterDirty_ = false;
}

void UIDataHubPanel::selectPoolRow(int row) {
    selectedPoolRow_ = row;
    selectedCurrRow_ = -1;
    if (row >= 0 && static_cast<size_t>(row) < filteredPoolIndices_.size()) {
        loadDetailForSequence(filteredPoolIndices_[row]);
    }
}

void UIDataHubPanel::selectCurriculumRow(int row) {
    selectedCurrRow_ = row;
    selectedPoolRow_ = -1;
    if (!datasetTarget_ || row < 0) return;
    const auto& order = datasetTarget_->curriculumOrder();
    if (static_cast<size_t>(row) >= order.size()) return;
    const std::string& seqId = order[row];
    for (size_t i = 0; i < datasetTarget_->massDatasetSize(); ++i) {
        auto seq = datasetTarget_->getSequence(i);
        if (seq.id == seqId) {
            loadDetailForSequence(i);
            return;
        }
    }
}

void UIDataHubPanel::loadDetailForSequence(size_t seqIndex) {
    detailSeqIndex_ = seqIndex;
    if (!datasetTarget_ || seqIndex >= datasetTarget_->massDatasetSize()) return;
    auto seq = datasetTarget_->getSequence(seqIndex);
    if (detailContentArea_)    detailContentArea_->setText(seq.content);
    if (detailStructuredArea_) detailStructuredArea_->setText(seq.structured);
}

void UIDataHubPanel::drawPoolTable(OverlayRenderer& renderer,
                                    float x, float y, float w, float h) {
    renderer.drawRoundedRect({x, y}, {w, h},
                             UITheme::Colors::ContentAreaBg, UITheme::Sizes::SmallRadius);
    renderer.drawRoundedBorder({x, y}, {w, h},
                               UITheme::Colors::BorderSubtle, UITheme::Sizes::SmallRadius);

    // Header row
    renderer.drawRect({x, y}, {w, kPoolHeaderH}, UITheme::Colors::TableHeaderBg);
    float colX = x + 4.0f;
    float hdrY = y + 6.0f;
    renderer.drawText({colX, hdrY}, "#", UITheme::Colors::TextWhite);
    colX += w * kColNum;
    renderer.drawText({colX, hdrY}, "Subject", UITheme::Colors::TextWhite);
    colX += w * kColSubject;
    renderer.drawText({colX, hdrY}, "Quality", UITheme::Colors::TextWhite);
    colX += w * kColQuality;
    renderer.drawText({colX, hdrY}, "Source", UITheme::Colors::TextWhite);
    colX += w * kColSource;
    renderer.drawText({colX, hdrY}, "Str?", UITheme::Colors::TextWhite);
    colX += w * kColStructured;
    renderer.drawText({colX, hdrY}, "Preview", UITheme::Colors::TextWhite);

    float bodyY = y + kPoolHeaderH;
    float bodyH = h - kPoolHeaderH;
    if (bodyH <= 0) return;

    renderer.pushClipRect({x, bodyY}, {w, bodyH});

    int visibleRows = static_cast<int>(bodyH / kPoolRowH) + 1;
    int startRow = static_cast<int>(poolScrollOffset_ / kPoolRowH);
    startRow = std::max(0, std::min(startRow, static_cast<int>(filteredPoolIndices_.size()) - 1));

    int totalRows = static_cast<int>(filteredPoolIndices_.size());

    for (int i = startRow; i < std::min(startRow + visibleRows, totalRows); ++i) {
        size_t seqIdx = filteredPoolIndices_[i];
        auto seq = datasetTarget_->getSequence(seqIdx);

        float rowY = bodyY + (i - startRow) * kPoolRowH
                   - (poolScrollOffset_ - startRow * kPoolRowH);

        bool isAssigned = datasetTarget_->isAssigned(seqIdx);
        uint32_t rowColor = (i == selectedPoolRow_) ? UITheme::Colors::RowSelected :
                            (i == hoveredPoolRow_)  ? UITheme::Colors::RowHover :
                            (i % 2 == 0)            ? UITheme::Colors::RowEven
                                                    : UITheme::Colors::RowOdd;
        renderer.drawRect({x, rowY}, {w, kPoolRowH}, rowColor);

        if (isAssigned) {
            renderer.drawRect({x, rowY}, {3.0f, kPoolRowH}, UITheme::Colors::Success);
        }

        float textY = rowY + 4.0f;
        colX = x + 4.0f;
        renderer.drawText({colX, textY}, std::to_string(seqIdx + 1), UITheme::Colors::TextMuted);
        colX += w * kColNum;

        renderer.drawText({colX, textY}, seq.subject, subjectBadgeColor(seq.subject));
        colX += w * kColSubject;

        renderer.drawText({colX, textY}, seq.quality_tier, qualityBadgeColor(seq.quality_tier));
        colX += w * kColQuality;

        std::string srcShort = seq.source_type;
        if (srcShort.size() > 10) srcShort = srcShort.substr(0, 10);
        renderer.drawText({colX, textY}, srcShort, UITheme::Colors::TextLight);
        colX += w * kColSource;

        renderer.drawText({colX, textY}, seq.is_structured ? "Y" : "-",
                          seq.is_structured ? UITheme::Colors::Success : UITheme::Colors::TextDisabled);
        colX += w * kColStructured;

        float previewW = w - (colX - x) - 6.0f;
        int maxChars = static_cast<int>(previewW / 7.5f);
        std::string preview = seq.content.substr(0, std::min(seq.content.size(), static_cast<size_t>(maxChars)));
        std::replace(preview.begin(), preview.end(), '\n', ' ');
        if (static_cast<int>(seq.content.size()) > maxChars) preview += "...";
        renderer.drawText({colX, textY}, preview, UITheme::Colors::TextPrimary);
    }

    renderer.popClipRect();

    // Scrollbar
    float totalH = totalRows * kPoolRowH;
    if (totalH > bodyH) {
        float ratio = poolScrollOffset_ / (totalH - bodyH);
        float thumbRatio = bodyH / totalH;
        float thumbH = std::max(20.0f, bodyH * thumbRatio);
        float thumbY = bodyY + ratio * (bodyH - thumbH);
        float barX = x + w - 6.0f;
        renderer.drawRoundedRect({barX, thumbY}, {4.0f, thumbH},
                                 UITheme::Colors::ScrollThumb, 2.0f);
    }
}

void UIDataHubPanel::drawCurriculumList(OverlayRenderer& renderer,
                                         float x, float y, float w, float h) {
    renderer.drawRoundedRect({x, y}, {w, h},
                             UITheme::Colors::ContentAreaBg, UITheme::Sizes::SmallRadius);
    renderer.drawRoundedBorder({x, y}, {w, h},
                               UITheme::Colors::BorderSubtle, UITheme::Sizes::SmallRadius);

    if (!datasetTarget_) return;
    const auto& order = datasetTarget_->curriculumOrder();
    const auto& phases = datasetTarget_->phaseMarkers();

    // Header
    std::string hdr = "Curriculum (" + std::to_string(order.size()) + ")";
    renderer.drawRect({x, y}, {w, kPoolHeaderH}, UITheme::Colors::TableHeaderBg);
    renderer.drawText({x + 8.0f, y + 6.0f}, hdr, UITheme::Colors::TextWhite);

    // Curriculum list action menu in header
    float menuW = 100.0f;
    float btnH = 22.0f;
    float btnY = y + 3.0f;
    if (currListActionMenu_) {
        currListActionMenu_->setPosition(x + w - menuW - 8.0f, btnY);
        currListActionMenu_->setSize(menuW, btnH);
        currListActionMenu_->drawOverlay(renderer, position);
    }

    float bodyY = y + kPoolHeaderH;
    float bodyH = h - kPoolHeaderH;
    if (bodyH <= 0) return;

    renderer.pushClipRect({x, bodyY}, {w, bodyH});

    // Build a flat list of items: phases interleaved with sequences
    struct CurrItem { bool isPhase; size_t index; size_t phaseIdx; };
    std::vector<CurrItem> items;
    items.reserve(order.size() + phases.size());

    size_t phaseI = 0;
    for (size_t i = 0; i < order.size(); ++i) {
        while (phaseI < phases.size() && phases[phaseI].position <= i) {
            items.push_back({true, i, phaseI});
            phaseI++;
        }
        items.push_back({false, i, 0});
    }
    while (phaseI < phases.size()) {
        items.push_back({true, order.size(), phaseI});
        phaseI++;
    }

    float itemY = bodyY - currScrollOffset_;
    for (size_t ii = 0; ii < items.size(); ++ii) {
        const auto& item = items[ii];
        float rowH = item.isPhase ? kPhaseRowH : kCurrRowH;

        if (itemY + rowH >= bodyY && itemY < bodyY + bodyH) {
            if (item.isPhase) {
                renderer.drawRect({x, itemY}, {w, rowH}, UITheme::Colors::SectionNeutral);
                renderer.drawRect({x, itemY + rowH - 1.0f}, {w, 1.0f}, UITheme::Colors::DividerLine);
                const std::string& label = phases[item.phaseIdx].label;
                renderer.drawText({x + 10.0f, itemY + 3.0f}, label, UITheme::Colors::TextSecondary);
            } else {
                size_t ci = item.index;
                int row = static_cast<int>(ci);
                uint32_t rowColor = (row == selectedCurrRow_) ? UITheme::Colors::RowSelected :
                                    (row == hoveredCurrRow_)  ? UITheme::Colors::RowHover :
                                    (ci % 2 == 0)             ? UITheme::Colors::RowEven
                                                              : UITheme::Colors::RowOdd;
                renderer.drawRect({x, itemY}, {w, rowH}, rowColor);

                const std::string& seqId = order[ci];
                SequenceHandle seq;
                for (size_t si = 0; si < datasetTarget_->massDatasetSize(); ++si) {
                    auto s = datasetTarget_->getSequence(si);
                    if (s.id == seqId) { seq = s; break; }
                }

                float textY = itemY + 5.0f;
                float cx = x + 6.0f;

                renderer.drawText({cx, textY}, std::to_string(ci + 1), UITheme::Colors::TextMuted);
                cx += 30.0f;

                renderer.drawText({cx, textY}, seq.subject, subjectBadgeColor(seq.subject));
                cx += 65.0f;

                renderer.drawText({cx, textY}, seq.quality_tier, qualityBadgeColor(seq.quality_tier));
                cx += 55.0f;

                float previewW = w - (cx - x) - 60.0f;
                int maxChars = static_cast<int>(previewW / 7.5f);
                if (maxChars < 0) maxChars = 0;
                std::string preview = seq.content.substr(
                    0, std::min(seq.content.size(), static_cast<size_t>(maxChars)));
                std::replace(preview.begin(), preview.end(), '\n', ' ');
                if (static_cast<int>(seq.content.size()) > maxChars) preview += "...";
                renderer.drawText({cx, textY}, preview, UITheme::Colors::TextPrimary);

                // Move up/down arrows on right side
                float arrowX = x + w - 50.0f;
                if (ci > 0) {
                    renderer.drawText({arrowX, textY}, "^", UITheme::Colors::TextLink);
                }
                if (ci + 1 < order.size()) {
                    renderer.drawText({arrowX + 18.0f, textY}, "v", UITheme::Colors::TextLink);
                }
            }
        }

        itemY += rowH;
    }

    renderer.popClipRect();

    // Scrollbar
    float totalH = 0;
    for (const auto& it : items)
        totalH += it.isPhase ? kPhaseRowH : kCurrRowH;
    if (totalH > bodyH) {
        float ratio = currScrollOffset_ / (totalH - bodyH);
        float thumbRatio = bodyH / totalH;
        float thumbH = std::max(20.0f, bodyH * thumbRatio);
        float thumbY = bodyY + ratio * (bodyH - thumbH);
        float barX = x + w - 6.0f;
        renderer.drawRoundedRect({barX, thumbY}, {4.0f, thumbH},
                                 UITheme::Colors::ScrollThumb, 2.0f);
    }
}

void UIDataHubPanel::drawDetailEditor(OverlayRenderer& renderer,
                                       float x, float y, float w, float h) {
    // Divider bar (clickable to toggle)
    renderer.drawRect({x, y}, {w, kDetailDividerH}, UITheme::Colors::SectionNeutral);
    renderer.drawRect({x, y + kDetailDividerH - 1.0f}, {w, 1.0f}, UITheme::Colors::DividerLine);
    std::string divLabel = detailPanelOpen_ ? "Detail  [click to collapse]" : "Detail  [click to expand]";
    renderer.drawText({x + 10.0f, y + 2.0f}, divLabel, UITheme::Colors::TextSecondary);

    if (!detailPanelOpen_) return;

    float panelY = y + kDetailDividerH;
    float panelH = h - kDetailDividerH;
    if (panelH < 40.0f) return;

    renderer.drawRoundedRect({x, panelY}, {w, panelH},
                             UITheme::Colors::ContentAreaBg, UITheme::Sizes::SmallRadius);

    if (detailSeqIndex_ == SIZE_MAX) {
        renderer.drawText({x + 20.0f, panelY + panelH / 2.0f - 8.0f},
                          "Select a sequence to view details",
                          UITheme::Colors::TextDisabled);
        return;
    }

    float halfW = (w - 18.0f) / 2.0f;
    float areaH = panelH - 32.0f;
    if (areaH < 30.0f) areaH = 30.0f;

    detailContentArea_->setPosition(x + 4.0f, panelY + 4.0f);
    detailContentArea_->setSize(halfW, areaH);
    detailContentArea_->drawOverlay(renderer, position);

    detailStructuredArea_->setPosition(x + halfW + 14.0f, panelY + 4.0f);
    detailStructuredArea_->setSize(halfW, areaH);
    detailStructuredArea_->drawOverlay(renderer, position);

    float saveBtnW = 80.0f;
    float saveBtnH = 24.0f;
    btnDetailSave_->setPosition(x + w - saveBtnW - 8.0f, panelY + panelH - saveBtnH - 4.0f);
    btnDetailSave_->setSize(saveBtnW, saveBtnH);
    btnDetailSave_->drawOverlay(renderer, position);
}

void UIDataHubPanel::drawCurriculumView(OverlayRenderer& renderer,
                                         const PanelRect& /*content*/,
                                         float x, float y, float fullW,
                                         float contentEndY) {
    if (poolFilterDirty_) rebuildFilteredPool();

    float availH = contentEndY - y;
    if (availH < 100.0f) return;

    // Filter bar
    float filterX = x;
    float ddW = 140.0f;
    float gap = 8.0f;
    float searchW = 200.0f;

    subjectFilterDropdown_->setPosition(filterX, y);
    subjectFilterDropdown_->setSize(ddW, kFilterBarH);
    subjectFilterDropdown_->drawOverlay(renderer, position);
    filterX += ddW + gap;

    qualityFilterDropdown_->setPosition(filterX, y);
    qualityFilterDropdown_->setSize(ddW, kFilterBarH);
    qualityFilterDropdown_->drawOverlay(renderer, position);
    filterX += ddW + gap;

    poolSearchInput_->setPosition(filterX, y + 2.0f);
    poolSearchInput_->setSize(searchW, kFilterBarH - 4.0f);
    poolSearchInput_->drawOverlay(renderer, position);
    filterX += searchW + gap;

    btnAssignSelected_->setPosition(filterX, y + 2.0f);
    btnAssignSelected_->setSize(100.0f, kFilterBarH - 4.0f);
    btnAssignSelected_->drawOverlay(renderer, position);

    y += kFilterBarH + 6.0f;

    // Rebuild filter if search text changed
    std::string curSearch = poolSearchInput_ ? poolSearchInput_->getText() : "";
    if (curSearch != filterSearchQuery_) {
        filterSearchQuery_ = curSearch;
        rebuildFilteredPool();
    }

    // Layout: left pool (60%), right curriculum (40%)
    float detailH = detailPanelOpen_ ? detailPanelHeight_ : kDetailDividerH;
    float splitH = (contentEndY - y) - detailH - 6.0f;
    if (splitH < 80.0f) splitH = 80.0f;

    float leftW  = fullW * 0.58f;
    float rightW = fullW - leftW - 10.0f;

    drawPoolTable(renderer, x, y, leftW, splitH);
    drawCurriculumList(renderer, x + leftW + 10.0f, y, rightW, splitH);

    float detailY = y + splitH + 6.0f;
    drawDetailEditor(renderer, x, detailY, fullW, detailH);

    // Status bar
    float statusY = contentEndY + 2.0f;
    renderer.drawRoundedRect({x, statusY}, {fullW, 25.0f},
                             UITheme::Colors::Background, UITheme::Sizes::SmallRadius);
    std::string status = "Pool: " + std::to_string(filteredPoolIndices_.size())
                       + " / " + std::to_string(totalSequences_)
                       + "  |  Curriculum: " + std::to_string(assignedSequences_)
                       + " sequences";
    if (datasetTarget_) {
        status += "  |  " + std::to_string(datasetTarget_->phaseMarkers().size()) + " phases";
    }
    renderer.drawText({x + 10.0f, statusY + 5.0f}, status, UITheme::Colors::TextSecondary);
}

// =========================================================
// Structurer helpers
// =========================================================

void UIDataHubPanel::populateModelDropdown() {
    auto models = GRIM::MMO::ModelRegistry::instance().getAllModels();
    std::vector<std::string> names;
    for (const auto* m : models) names.push_back(m->name);
    if (names.empty()) names.push_back("(no models)");
    if (modelDropdown_) modelDropdown_->setItems(names);
}

void UIDataHubPanel::refreshStructurerState() {
    if (!datasetTarget_) return;
    datasetTarget_->loadMassDataset();
    totalSequences_ = datasetTarget_->massDatasetSize();
    assignedSequences_ = datasetTarget_->assignedCount();
    structuredCount_ = 0;
    for (size_t i = 0; i < totalSequences_; ++i)
        if (datasetTarget_->getSequence(i).is_structured) structuredCount_++;
    if (currentSequenceIndex_ >= totalSequences_ && totalSequences_ > 0)
        currentSequenceIndex_ = totalSequences_ - 1;
    loadCurrentSequence();
}

void UIDataHubPanel::loadCurrentSequence() {
    if (!datasetTarget_ || datasetTarget_->massDatasetSize() == 0) {
        if (rawTextArea_) rawTextArea_->setText("");
        if (structuredTextArea_) structuredTextArea_->setText("");
        return;
    }
    auto seq = datasetTarget_->getSequence(currentSequenceIndex_);
    if (rawTextArea_) rawTextArea_->setText(seq.content);
    if (structuredTextArea_) structuredTextArea_->setText(seq.structured);
}

// =========================================================
// Sequence card helpers
// =========================================================

float UIDataHubPanel::sequenceCardHeight(const SequenceCard& card) const {
    float h = kSeqCardTopBarH;
    h += card.entries.size() * kSeqCardEntryH;
    if (card.formatIndex == 1) h += kSeqCardBtnRowH;
    h += kSeqCardBtnRowH;
    h += kSeqCardPadBot;
    return h;
}

void UIDataHubPanel::applyFormatTemplate(SequenceCard& card, int formatIndex) {
    card.formatIndex = formatIndex;
    card.entries.clear();

    auto makeField = [](const std::string& prefix) {
        UIDataHubPanel::SequenceCard::EntryField f;
        f.prefix = prefix;
        f.textArea = std::make_shared<UITextArea>(prefix, "", [](const std::string&) {});
        return f;
    };

    switch (formatIndex) {
        case 0: // Q/A
            card.entries.push_back(makeField("Q:"));
            card.entries.push_back(makeField("A:"));
            break;
        case 1: // Thought
            card.entries.push_back(makeField("T:"));
            break;
        case 2: // Conversation
            card.entries.push_back(makeField("Human:"));
            card.entries.push_back(makeField("Assistant:"));
            break;
        case 3: // Instruct
            card.entries.push_back(makeField("Instruction:"));
            card.entries.push_back(makeField("Response:"));
            break;
        case 4: // Raw
        default:
            card.entries.push_back(makeField("Content:"));
            break;
    }
}

UIDataHubPanel::SequenceCard UIDataHubPanel::buildSequenceCard(int formatIndex) {
    SequenceCard card;
    card.cardId      = nextSeqCardId_++;
    card.formatIndex = formatIndex;

    size_t id = card.cardId;

    std::vector<std::string> formats = {"Q/A", "Thought", "Conversation", "Instruct", "Raw"};
    card.formatDropdown = std::make_shared<UIDropdown>(
        "Format", formats, formatIndex,
        [this, id](int idx, const std::string&) {
            for (auto& c : sequenceCards_) {
                if (c.cardId == id) { applyFormatTemplate(c, idx); break; }
            }
        });
    card.formatDropdown->setMaxVisibleItems(5);

    card.generateBtn = std::make_shared<UIButton>("Generate", [this, id]() {
        generateForCard(id);
    });

    card.addEntryBtn = std::make_shared<UIButton>("+ Entry", [this, id]() {
        for (auto& c : sequenceCards_) {
            if (c.cardId == id) {
                SequenceCard::EntryField f;
                f.prefix = "T:";
                f.textArea = std::make_shared<UITextArea>("T:", "", [](const std::string&) {});
                c.entries.push_back(std::move(f));
                break;
            }
        }
    });

    card.deleteBtn = std::make_shared<UIButton>("x", [this, id]() {
        seqCardToDelete_ = static_cast<int>(id);
    });

    card.saveBtn = std::make_shared<UIButton>("Save", [this, id]() {
        saveSequenceCard(id);
    });

    applyFormatTemplate(card, formatIndex);
    return card;
}

void UIDataHubPanel::removeSequenceCard(size_t cardId) {
    sequenceCards_.erase(
        std::remove_if(sequenceCards_.begin(), sequenceCards_.end(),
            [cardId](const SequenceCard& c) { return c.cardId == cardId; }),
        sequenceCards_.end());
}

void UIDataHubPanel::generateForCard(size_t cardId) {
    SequenceCard* card = nullptr;
    for (auto& c : sequenceCards_) {
        if (c.cardId == cardId) { card = &c; break; }
    }
    if (!card || !structurer_) return;

    std::string raw = rawTextArea_ ? rawTextArea_->getText() : "";
    if (raw.empty()) { addLog("Put raw text in 'Raw Source' first", 1); return; }

    static const char* modes[] = {"qa", "thought", "conversation", "instruct", "raw"};
    std::string mode = (card->formatIndex >= 0 && card->formatIndex < 5)
                           ? modes[card->formatIndex] : "qa";

    std::string prompt = customPromptArea_ ? customPromptArea_->getText() : "";
    addLog("Generating " + mode + " format...", 0);

    auto results = structurer_->structureEntry(raw, mode, prompt);
    if (results.empty()) {
        std::string err = structurer_->lastError();
        addLog("Generation failed: " + (err.empty() ? "LLM returned no output" : err), 2);
        return;
    }

    auto makeField = [](const std::string& prefix, const std::string& text) {
        UIDataHubPanel::SequenceCard::EntryField f;
        f.prefix = prefix;
        f.textArea = std::make_shared<UITextArea>(prefix, text, [](const std::string&) {});
        return f;
    };

    if (mode == "qa" || mode == "instruct") {
        while (card->entries.size() < 2) {
            std::string pfx = card->entries.empty() ? "Q:" : "A:";
            card->entries.push_back(makeField(pfx, ""));
        }
        std::string result = results[0];
        size_t aPos = result.find("\n\nA: ");
        if (aPos != std::string::npos) {
            std::string q = result.substr(3, aPos - 3);
            std::string a = result.substr(aPos + 5);
            card->entries[0].textArea->setText(q);
            card->entries[1].textArea->setText(a);
        } else {
            card->entries[0].textArea->setText(result);
        }
    } else if (mode == "thought") {
        card->entries.clear();
        for (const auto& r : results) {
            std::string text = r;
            if (text.size() > 3 && text.substr(0, 3) == "T: ") text = text.substr(3);
            card->entries.push_back(makeField("T:", text));
        }
        if (card->entries.empty())
            card->entries.push_back(makeField("T:", ""));
    } else if (mode == "conversation") {
        card->entries.clear();
        if (!results.empty()) {
            std::istringstream ss(results[0]);
            std::string line;
            std::string accum;
            std::string curPrefix;
            while (std::getline(ss, line)) {
                if (line.substr(0, 7) == "Human: " || line.substr(0, 11) == "Assistant: ") {
                    if (!curPrefix.empty())
                        card->entries.push_back(makeField(curPrefix, accum));
                    if (line.substr(0, 7) == "Human: ") {
                        curPrefix = "Human:";
                        accum = line.substr(7);
                    } else {
                        curPrefix = "Assistant:";
                        accum = line.substr(11);
                    }
                } else if (!curPrefix.empty() && !line.empty()) {
                    accum += "\n" + line;
                }
            }
            if (!curPrefix.empty())
                card->entries.push_back(makeField(curPrefix, accum));
        }
        if (card->entries.empty()) {
            card->entries.push_back(makeField("Human:", ""));
            card->entries.push_back(makeField("Assistant:", ""));
        }
    } else {
        if (!card->entries.empty() && !results.empty()) {
            std::string combined;
            for (size_t i = 0; i < results.size(); ++i) {
                if (i > 0) combined += "\n\n";
                combined += results[i];
            }
            card->entries[0].textArea->setText(combined);
        }
    }

    addLog("Generated " + std::to_string(results.size()) + " entries in " + mode + " format", 0);
}

void UIDataHubPanel::saveSequenceCard(size_t cardId) {
    SequenceCard* card = nullptr;
    for (auto& c : sequenceCards_) {
        if (c.cardId == cardId) { card = &c; break; }
    }
    if (!card || !datasetTarget_) return;

    std::string structured;
    for (size_t i = 0; i < card->entries.size(); ++i) {
        if (i > 0) structured += "\n\n";
        structured += card->entries[i].prefix + " " + card->entries[i].textArea->getText();
    }

    if (structured.empty()) {
        addLog("Nothing to save — fill in the sequence fields", 1);
        return;
    }

    std::string rawContent;
    for (size_t i = 0; i < card->entries.size(); ++i) {
        if (i > 0) rawContent += "\n\n";
        rawContent += card->entries[i].textArea->getText();
    }

    if (datasetTarget_->appendStructuredEntry(rawContent, structured)) {
        totalSequences_ = datasetTarget_->massDatasetSize();
        structuredCount_++;
        addLog("Saved sequence card #" + std::to_string(card->cardId) + " to dataset", 0);
    } else {
        addLog("Failed to save sequence card", 2);
    }
}

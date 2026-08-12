#include "ui_data_hub_internal.hpp"

using namespace UIDataHubDetail;

// =========================================================
// Curriculum tab layout
// =========================================================

void UIDataHubPanel::drawCurriculumTab(OverlayRenderer& renderer,
                                        const PanelRect& content) {
    float x     = content.origin.x + 15.0f;
    float y     = content.origin.y + 10.0f;
    float fullW = content.size.x - 30.0f;

    // ── Toolbar row 1: Model + Curriculum + Manage ────────
    float gap   = 8.0f;
    float rowH  = 36.0f;
    float btnW  = 80.0f;
    float menuW = 100.0f;

    // Model and curriculum share the flexible space; stage and format metadata
    // keep fixed, readable slots.
    float stageW = 240.0f;
    float formatBadgeSlotW = 84.0f;
    float ddW = (fullW - stageW - menuW - formatBadgeSlotW - 4.0f * gap) * 0.5f;
    if (ddW < 220.0f) ddW = 220.0f;

    float cx = x;
    cbModelDropdown_->setPosition(cx, y);
    cbModelDropdown_->setSize(ddW, rowH);
    cbModelDropdown_->drawOverlay(renderer, position);
    cx += ddW + gap;

    if (renamingCurriculum_ && cbCurriculumRenameInput_) {
        cbCurriculumRenameInput_->setPosition(cx, y);
        cbCurriculumRenameInput_->setSize(ddW, rowH);
        cbCurriculumRenameInput_->drawOverlay(renderer, position);
    } else {
        cbCurriculumDropdown_->setPosition(cx, y);
        cbCurriculumDropdown_->setSize(ddW, rowH);
        cbCurriculumDropdown_->drawOverlay(renderer, position);
    }
    cx += ddW + gap;

    cbTrainingStageDropdown_->setPosition(cx, y);
    cbTrainingStageDropdown_->setSize(stageW, rowH);
    cbTrainingStageDropdown_->drawOverlay(renderer, position);
    cx += stageW + gap;

    curriculumActionMenu_->setPosition(cx, y + 4.0f);
    curriculumActionMenu_->setSize(menuW, rowH - 8.0f);
    curriculumActionMenu_->drawOverlay(renderer, position);
    cx += menuW + gap;

    // Format-mode badge is rendering metadata, separate from training stage.
    if (!activeCurriculumId_.empty() && datasetTarget_) {
        auto activeCurr = datasetTarget_->getCurriculumById(activeCurriculumId_);
        if (!activeCurr.id.empty()) {
            const char* modeLabel = activeCurr.format_as_concept ? "Concept" : "Plain Text";
            uint32_t badgeBg = activeCurr.format_as_concept
                ? UITheme::Colors::Success : 0xFF88AADD;
            float badgeW = activeCurr.format_as_concept ? 68.0f : 84.0f;
            float badgeH = rowH - 12.0f;
            float badgeY = y + 6.0f;
            renderer.drawRoundedRect({cx, badgeY}, {badgeW, badgeH}, badgeBg, 4.0f);
            renderer.drawText({cx + 6.0f, badgeY + 3.0f}, modeLabel, 0xFFFFFFFF);
        }
    }

    y += rowH + 6.0f;

    // ── Toolbar row 2: Curriculum filter + Format filter + Generate + Search ──
    float toggleW = 130.0f;
    cx = x;
    cbCurriculumFilterToggle_->setPosition(cx, y + 4.0f);
    cbCurriculumFilterToggle_->setSize(toggleW, rowH - 8.0f);
    cbCurriculumFilterToggle_->drawOverlay(renderer, position);
    cx += toggleW + gap;

    float filterDdW = 275.0f;
    cbTypeFilterDropdown_->setPosition(cx, y);
    cbTypeFilterDropdown_->setSize(filterDdW, rowH);
    cbTypeFilterDropdown_->drawOverlay(renderer, position);
    cx += filterDdW + gap;

    btnCBGenerate_->setPosition(cx, y + 4.0f);
    btnCBGenerate_->setSize(btnW, rowH - 8.0f);
    btnCBGenerate_->drawOverlay(renderer, position);
    cx += btnW + gap;

    float searchW = x + fullW - cx - 8.0f;
    if (searchW < 100.0f) searchW = 100.0f;
    cbSearchInput_->setPosition(cx, y + 4.0f);
    cbSearchInput_->setSize(searchW, rowH - 8.0f);
    cbSearchInput_->drawOverlay(renderer, position);

    y += rowH + 12.0f;

    // ── Split: list left (38%) | editor right (62%) ─────
    const CurriculumTabLayout layout = computeCurriculumTabLayout(content);
    y = layout.listY;
    float availH = layout.listH;
    float listW  = layout.listW;
    float editorX = layout.editorX;
    float editorW = layout.editorW;
    float bottomBarH = layout.bottomBarH;
    float statusBarH = layout.statusBarH;

    // Determine if active curriculum uses concept block formatting
    bool conceptMode = true; // default when no curriculum selected
    if (!activeCurriculumId_.empty() && datasetTarget_) {
        auto activeCurr = datasetTarget_->getCurriculumById(activeCurriculumId_);
        if (!activeCurr.id.empty())
            conceptMode = activeCurr.format_as_concept;
    }

    // ── ConceptBlock list ────────────────────────────────
    renderer.drawRoundedRect({x, y}, {listW, availH},
                             UITheme::Colors::Background, UITheme::Sizes::SmallRadius);

    // Header
    renderer.drawRoundedRect({x, y}, {listW, kPoolHeaderH},
                             UITheme::Colors::WidgetBg, UITheme::Sizes::SmallRadius);
    const float nameColW = listW * 0.34f;
    const float qColX    = x + 8.0f + nameColW;
    renderer.drawText({x + 8.0f, y + 6.0f}, "Name", UITheme::Colors::TextSecondary);
    renderer.drawText({qColX, y + 6.0f},
                      conceptMode ? "Prompt (preview)" : "Text (preview)",
                      UITheme::Colors::TextSecondary);
    renderer.drawText({x + listW - 118.0f, y + 6.0f}, "Type", UITheme::Colors::TextSecondary);

    float bodyY = y + kPoolHeaderH;
    float bodyH = availH - kPoolHeaderH;
    renderer.pushClipRect({x, bodyY}, {listW, bodyH});

    const int cbRows = cbCurriculumListRowCount();
    int startRow = static_cast<int>(cbListScrollOffset_ / kCBListRowH);
    float offsetY = bodyY - (cbListScrollOffset_ - startRow * kCBListRowH);
    for (int i = startRow; i < cbRows; ++i) {
        if (offsetY > bodyY + bodyH) break;
        if (offsetY + kCBListRowH < bodyY) {
            offsetY += kCBListRowH;
            continue;
        }

        std::string nameStr;
        std::string qRaw;
        std::string formatKey = "chain_of_thought";
        std::string blockId;
        bool isDraft = cbCurriculumRowIsDraft(i);

        if (isDraft) {
            nameStr = cbNameInput_ ? cbNameInput_->getText() : "";
            qRaw    = cbPromptArea_ ? cbPromptArea_->getText() : "";
            int pti = cbListTypeDropdown_ ? cbListTypeDropdown_->getSelectedIndex() : 1;
            if (pti >= 0 && pti < GRIM::kConceptPresetCount)
                formatKey = GRIM::kConceptPresets[pti].key;
            if (nameStr.empty())
                nameStr = "(new block — unsaved)";
        } else {
            size_t realIdx = 0;
            if (!cbCurriculumRowToBlockIndex(i, realIdx)) {
                offsetY += kCBListRowH;
                continue;
            }
            auto cb = datasetTarget_ ? datasetTarget_->getConceptBlock(realIdx) : GRIM::ConceptBlock{};
            nameStr   = cb.name;
            qRaw      = cb.format_type == "raw" ? cb.raw : cb.prompt;
            formatKey = cb.format_type;
            blockId   = cb.id;
        }

        bool selected = (i == selectedCBRow_);
        bool hovered  = (i == hoveredCBRow_);
        uint32_t rowBg = selected ? UITheme::Colors::Primary
                       : hovered  ? UITheme::Colors::WidgetBgHover
                       : (i % 2 == 0) ? 0x00000000 : 0x08FFFFFF;

        if (rowBg != 0x00000000)
            renderer.drawRect({x, offsetY}, {listW, kCBListRowH}, rowBg);

        uint32_t textCol = selected ? 0xFFFFFFFF : UITheme::Colors::TextPrimary;
        std::string nameLine = cbSingleLinePreview(nameStr, 36);
        std::string qLine    = cbSingleLinePreview(qRaw, 48);
        if (qLine.empty())
            qLine = "—";

        renderer.drawText({x + 8.0f, offsetY + 4.0f}, nameLine, textCol);
        renderer.drawText({x + 8.0f, offsetY + 22.0f}, qLine,
                          selected ? 0xCCFFFFFF : UITheme::Colors::TextSecondary);

        if (!(selected && cbListTypeDropdown_)) {
            int pi = GRIM::presetIndexForKey(formatKey);
            std::string fmtLabel = (pi >= 0 && pi < GRIM::kConceptPresetCount)
                ? GRIM::kConceptPresets[pi].label : formatKey;
            if (fmtLabel.size() > 14)
                fmtLabel = fmtLabel.substr(0, 12) + "..";
            renderer.drawText({x + listW - 118.0f, offsetY + 14.0f}, fmtLabel,
                              selected ? 0xCCFFFFFF : UITheme::Colors::TextSecondary);
        }

        if (!isDraft && datasetTarget_ && !blockId.empty()
            && !activeCurriculumId_.empty()
            && datasetTarget_->isConceptBlockInCurriculum(blockId, activeCurriculumId_))
            renderer.drawRoundedRect({x + listW - 14.0f, offsetY + 10.0f}, {8.0f, 8.0f},
                                     UITheme::Colors::Success, 4.0f);

        offsetY += kCBListRowH;
    }
    renderer.popClipRect();

    layoutCBListTypeDropdownInList(x, y, listW);
    if (cbListTypeDropdown_)
        cbListTypeDropdown_->drawOverlay(renderer, position);

    // ── Editor panel ─────────────────────────────────────
    renderer.drawRoundedRect({editorX, y}, {editorW, availH},
                             UITheme::Colors::Background, 6.0f);

    float ePad = 16.0f;
    float eInnerW = editorW - 2.0f * ePad;
    float fieldH = 30.0f;
    float areaH  = 56.0f;
    float sectionGap = 6.0f;   // gap between section divider and next label
    float sectionPad = 6.0f;   // internal padding inside section boxes
    float sectionRad = 5.0f;   // corner radius for section backgrounds

    int presetIdx = cbListTypeDropdown_ ? cbListTypeDropdown_->getSelectedIndex() : 1;
    if (presetIdx < 0 || presetIdx >= GRIM::kConceptPresetCount) presetIdx = 1;
    const auto& preset = GRIM::kConceptPresets[presetIdx];
    const bool rawMode = std::string(preset.key) == "raw";
    const bool structuredEditorMode = conceptMode && !rawMode;

    // ── Scrollable editor content ───────────────────────
    renderer.pushClipRect({editorX, y}, {editorW, availH});

    float ey = y + 12.0f - cbEditorScrollOffset_;

    // ─── Name ───────────────────────────────────────────
    renderer.drawText({editorX + ePad, ey}, "Name", UITheme::Colors::TextSecondary);
    ey += 20.0f;
    cbNameInput_->setPosition(editorX + ePad, ey);
    cbNameInput_->setSize(eInnerW, fieldH);
    cbNameInput_->drawOverlay(renderer, position);
    ey += fieldH + 14.0f;

    // ─── Q: Question / Raw Text ───────────────────────────
    // Divider line
    renderer.drawRect({editorX + ePad, ey}, {eInnerW, 1.0f}, 0x18FFFFFF);
    ey += sectionGap;
    if (rawMode) {
        renderer.drawText({editorX + ePad, ey}, "Raw", UITheme::Colors::TextSecondary);
    } else if (structuredEditorMode) {
        std::string qLabel = std::string("Q: ") + preset.questionLabel;
        renderer.drawText({editorX + ePad, ey}, qLabel, UITheme::Colors::TextSecondary);
    } else {
        renderer.drawText({editorX + ePad, ey}, "Text", UITheme::Colors::TextSecondary);
    }
    ey += 20.0f;
    cbPromptArea_->setPosition(editorX + ePad, ey);
    cbPromptArea_->setSize(eInnerW, structuredEditorMode ? areaH : areaH * 2.0f);
    cbPromptArea_->drawOverlay(renderer, position);
    ey += (structuredEditorMode ? areaH : areaH * 2.0f) + 16.0f;

    if (structuredEditorMode) {
    // ─── Goal identifier ────────────────────────────────
    renderer.drawRect({editorX + ePad, ey}, {eInnerW, 1.0f}, 0x18FFFFFF);
    ey += sectionGap;
    {
        const float goalStartY = ey;
        const float innerLeft = editorX + ePad + sectionPad;
        const float innerW = eInnerW - 2.0f * sectionPad;
        const float criterionAreaH = 44.0f;
        const float criterionBlockH = 148.0f;
        const float goalSectionH = sectionPad + 22.0f + 20.0f + areaH + 14.0f
            + criterionBlockH * static_cast<float>(cbSuccessCriterionRows_.size())
            + 32.0f + sectionPad;

        renderer.drawRoundedRect({editorX + ePad, goalStartY},
                                 {eInnerW, goalSectionH}, 0x0CFFFFFF, sectionRad);

        ey += sectionPad;
        renderer.drawText({innerLeft, ey}, "GOAL IDENTIFIER",
                          UITheme::Colors::TextSecondary);
        ey += 22.0f;

        renderer.drawText({innerLeft, ey}, "Target State (optional)",
                          UITheme::Colors::TextMuted);
        ey += 20.0f;
        cbTargetStateArea_->setPosition(innerLeft, ey);
        cbTargetStateArea_->setSize(innerW, areaH);
        cbTargetStateArea_->drawOverlay(renderer, position);
        ey += areaH + 14.0f;

        for (size_t ci = 0; ci < cbSuccessCriterionRows_.size(); ++ci) {
            auto& row = cbSuccessCriterionRows_[ci];
            renderer.drawText({innerLeft, ey},
                              "Success Criterion " + std::to_string(ci + 1),
                              UITheme::Colors::TextMuted);
            ey += 20.0f;
            row.criterionArea->setPosition(innerLeft, ey);
            row.criterionArea->setSize(innerW, criterionAreaH);
            row.criterionArea->drawOverlay(renderer, position);
            ey += criterionAreaH + 8.0f;

            renderer.drawText({innerLeft, ey}, "Evidence (required)",
                              UITheme::Colors::TextMuted);
            ey += 20.0f;
            row.evidenceArea->setPosition(innerLeft, ey);
            row.evidenceArea->setSize(innerW, criterionAreaH);
            row.evidenceArea->drawOverlay(renderer, position);
            ey += criterionAreaH + 12.0f;
        }

        successCriteriaActionMenu_->setPosition(innerLeft, ey);
        successCriteriaActionMenu_->setSize(108.0f, 26.0f);
        successCriteriaActionMenu_->drawOverlay(renderer, position);
        ey += 32.0f + sectionPad;
    }
    ey += 16.0f;
    }

    if (structuredEditorMode) {
    // ─── EXP: Explanation / Intermediates ────────────────
    if (preset.intermediatesLabel) {
        renderer.drawRect({editorX + ePad, ey}, {eInnerW, 1.0f}, 0x18FFFFFF);
        ey += sectionGap;
        {
            float expStartY = ey;
            ey += sectionPad;
            float innerLeft = editorX + ePad + sectionPad;

            std::string expLabel = std::string("EXP: ") + preset.intermediatesLabel;
            renderer.drawText({innerLeft, ey}, expLabel,
                              UITheme::Colors::TextSecondary);
            ey += 22.0f;

            float stepAreaH = 44.0f;
            float expInnerW = eInnerW - 2.0f * sectionPad;

            for (size_t si = 0; si < cbIntermediateAreas_.size(); ++si) {
                cbIntermediateAreas_[si]->setPosition(innerLeft, ey);
                cbIntermediateAreas_[si]->setSize(expInnerW, stepAreaH);
                cbIntermediateAreas_[si]->drawOverlay(renderer, position);
                ey += stepAreaH + 10.0f;
            }

            stepActionMenu_->setPosition(innerLeft, ey);
            stepActionMenu_->setSize(90.0f, 26.0f);
            stepActionMenu_->drawOverlay(renderer, position);
            ey += 32.0f + sectionPad;

            // Section background
            renderer.drawRoundedRect({editorX + ePad, expStartY},
                                     {eInnerW, ey - expStartY}, 0x0CFFFFFF, sectionRad);
        }
        ey += 16.0f;
    }

    // ─── EXEC: Execution Steps ──────────────────────────
    renderer.drawRect({editorX + ePad, ey}, {eInnerW, 1.0f}, 0x18FFFFFF);
    ey += sectionGap;
    {
        float execStartY = ey;
        ey += sectionPad;
        float innerLeft = editorX + ePad + sectionPad;
        float innerW = eInnerW - 2.0f * sectionPad;

        renderer.drawText({innerLeft, ey}, "EXEC Steps",
                          UITheme::Colors::TextSecondary);
        ey += 20.0f;

        float opW    = 80.0f;
        float slotsW = 90.0f;
        float colGap = 6.0f;
        float controlW = 74.0f;
        float argsW  = (innerW - opW - slotsW - controlW - colGap * 4.0f) * 0.5f;
        float resW   = argsW;

        // Column sub-headers
        if (!cbExecStepRows_.empty()) {
            float hx = innerLeft;
            renderer.drawText({hx, ey}, "op", UITheme::Colors::TextMuted);
            hx += opW + colGap;
            renderer.drawText({hx, ey}, "arg_slots", UITheme::Colors::TextMuted);
            hx += slotsW + colGap;
            renderer.drawText({hx, ey}, "args", UITheme::Colors::TextMuted);
            hx += argsW + colGap;
            renderer.drawText({hx, ey}, "=> result", UITheme::Colors::TextMuted);
            hx += resW + colGap;
            renderer.drawText({hx, ey}, "control", UITheme::Colors::TextMuted);
            ey += 16.0f;
        }

        for (size_t ei = 0; ei < cbExecStepRows_.size(); ++ei) {
            auto& row = cbExecStepRows_[ei];

            float cx = innerLeft;
            row.opDropdown->setPosition(cx, ey);
            row.opDropdown->setSize(opW, fieldH);
            row.opDropdown->drawOverlay(renderer, position);
            cx += opW + colGap;

            row.argSlotsInput->setPosition(cx, ey);
            row.argSlotsInput->setSize(slotsW, fieldH);
            row.argSlotsInput->drawOverlay(renderer, position);
            cx += slotsW + colGap;

            row.argsInput->setPosition(cx, ey);
            row.argsInput->setSize(argsW, fieldH);
            row.argsInput->drawOverlay(renderer, position);
            cx += argsW + colGap;

            row.resultInput->setPosition(cx, ey);
            row.resultInput->setSize(resW, fieldH);
            row.resultInput->drawOverlay(renderer, position);
            cx += resW + colGap;

            const bool final_step = ei + 1 == cbExecStepRows_.size();
            renderer.drawText({cx, ey + 7.0f}, final_step ? "STOP" : "CONTINUE",
                              final_step ? UITheme::Colors::Danger
                                         : UITheme::Colors::Success);

            ey += fieldH + 8.0f;
        }

        // Exec step add/remove menu
        execStepActionMenu_->setPosition(innerLeft, ey);
        execStepActionMenu_->setSize(120.0f, 26.0f);
        execStepActionMenu_->drawOverlay(renderer, position);
        ey += 32.0f;

        ey += sectionPad;

        // Section background
        renderer.drawRoundedRect({editorX + ePad, execStartY},
                                 {eInnerW, ey - execStartY}, 0x0CFFFFFF, sectionRad);
    }
    ey += 16.0f;

    } // end if (conceptMode) — EXP / EXEC hidden in PT mode

    // ─── A: Answer (concept mode only) ────────────────────
    if (structuredEditorMode) {
    renderer.drawRect({editorX + ePad, ey}, {eInnerW, 1.0f}, 0x18FFFFFF);
    ey += sectionGap;
    std::string aLabel = std::string("A: ") + preset.answerLabel;
    renderer.drawText({editorX + ePad, ey}, aLabel,
                      UITheme::Colors::TextSecondary);
    ey += 20.0f;
    cbAnswerArea_->setPosition(editorX + ePad, ey);
    cbAnswerArea_->setSize(eInnerW, areaH);
    cbAnswerArea_->drawOverlay(renderer, position);
    ey += areaH + 18.0f;
    } // end if (conceptMode)

    // ─── Training Preview (read-only) ───────────────────
    renderer.drawRect({editorX + ePad, ey}, {eInnerW, 1.0f}, 0x18FFFFFF);
    ey += sectionGap;
    {
        renderer.drawText({editorX + ePad, ey},
                          structuredEditorMode ? "Training Preview (logical delimiters)"
                                               : "Training Preview (raw)",
                          UITheme::Colors::TextSecondary);
        ey += 22.0f;

        // Build a ConceptBlock from current editor state for preview
        GRIM::ConceptBlock previewCB;
        std::string previewValidationError;
        const bool previewValid =
            buildConceptBlockFromEditor(previewCB, previewValidationError);
        if (structuredEditorMode && !previewValid) {
            renderer.drawText({editorX + ePad, ey},
                              "Validation: " + previewValidationError,
                              UITheme::Colors::Danger);
            ey += 20.0f;
        }
        std::string preview = buildTrainingPreview(previewCB, structuredEditorMode);

        // Pre-count wrapped lines so we can draw the background first.
        std::vector<std::pair<std::string, uint32_t>> previewLines;
        const float previewTextW = eInnerW - 20.0f;
        {
            std::istringstream pss(preview);
            std::string pline;
            while (std::getline(pss, pline)) {
                uint32_t lineCol = UITheme::Colors::TextMuted;
                if (structuredEditorMode) {
                    if (pline.size() >= 2 && pline[0] == 'Q' && pline[1] == ':')
                        lineCol = UITheme::Colors::TextPrimary;
                    else if (pline.size() >= 2 && pline[0] == 'A' && pline[1] == ':')
                        lineCol = UITheme::Colors::TextPrimary;
                    else if (pline.size() >= 5 && pline.substr(0, 5) == "STATE")
                        lineCol = UITheme::Colors::Success;
                    else if (pline.size() >= 4 && pline.substr(0, 4) == "EXEC")
                        lineCol = 0xFF88CCFF;
                    else if (pline.size() >= 4 && pline.substr(0, 4) == "EXP:")
                        lineCol = UITheme::Colors::TextSecondary;
                }
                for (const std::string& wrapped : renderer.wrapText(pline, previewTextW))
                    previewLines.push_back({wrapped, lineCol});
            }
        }

        // Draw background FIRST (correct z-order)
        float lineH = 18.0f;
        float previewH = static_cast<float>(previewLines.size()) * lineH + 16.0f;
        renderer.drawRoundedRect({editorX + ePad, ey - 4.0f},
                                 {eInnerW, previewH}, 0x14FFFFFF, 6.0f);

        // Then draw text on top
        float textY = ey + 4.0f;
        for (const auto& [text, col] : previewLines) {
            renderer.drawText({editorX + ePad + 10.0f, textY}, text, col);
            textY += lineH;
        }
        ey += previewH + 12.0f;
    }

    float totalEditorContentH = (ey + cbEditorScrollOffset_) - (y + 12.0f);
    renderer.popClipRect();

    // Handle editor scroll
    float maxEditorScroll = std::max(0.0f, totalEditorContentH - availH);
    cbEditorScrollOffset_ = std::clamp(cbEditorScrollOffset_, 0.0f, maxEditorScroll);

    // ── Bottom action bar ────────────────────────────────
    float barY = y + availH + 6.0f;
    float bx = x;
    float bGap = 8.0f;
    float bBtnW = 90.0f;
    float bBtnH = 28.0f;

    float blockMenuW = 90.0f;
    blockActionMenu_->setPosition(bx, barY + 4.0f);
    blockActionMenu_->setSize(blockMenuW, bBtnH);
    blockActionMenu_->drawOverlay(renderer, position);
    bx += blockMenuW + bGap;

    float menuW2 = 110.0f;
    blockCurriculumMenu_->setPosition(bx, barY + 4.0f);
    blockCurriculumMenu_->setSize(menuW2, bBtnH);
    blockCurriculumMenu_->drawOverlay(renderer, position);
    bx += menuW2;

    // Custom prompt (right side of action bar)
    float promptW = fullW - (bx + bGap - x) - 10.0f;
    if (promptW > 200.0f) {
        float promptX = bx + bBtnW + bGap + 10.0f;
        cbCustomPromptArea_->setPosition(promptX, barY);
        cbCustomPromptArea_->setSize(promptW, bottomBarH);
        cbCustomPromptArea_->drawOverlay(renderer, position);
    }

    // ── Status bar ───────────────────────────────────────
    float statusY = barY + bottomBarH + 4.0f;
    renderer.drawRoundedRect({x, statusY}, {fullW, statusBarH},
                             UITheme::Colors::Background, UITheme::Sizes::SmallRadius);
    std::string status = "Blocks: " + std::to_string(cbTotalCount_)
                       + "  |  Showing: " + std::to_string(filteredCBIndices_.size())
                       + "  |  In Curriculum: " + std::to_string(cbInCurrCount_);
    if (cbFormatFilterIdx_ > 0 && cbFormatFilterIdx_ <= GRIM::kConceptPresetCount)
        status += std::string("  |  Filter: ") + GRIM::kConceptPresets[cbFormatFilterIdx_ - 1].label;
    if (presetIdx >= 0 && presetIdx < GRIM::kConceptPresetCount)
        status += std::string("  |  Format: ") + GRIM::kConceptPresets[presetIdx].label;
    renderer.drawText({x + 10.0f, statusY + 5.0f}, status, UITheme::Colors::TextSecondary);
}

// =========================================================
// Curriculum tab helpers
// =========================================================

void UIDataHubPanel::refreshCurriculumTabState() {
    if (!datasetTarget_) return;
    if (!datasetTarget_->loadCurriculumRegistry()) {
        addLog("Failed to reload curriculum registry", 2);
    }
    datasetTarget_->loadConceptBlocks();
    cbTotalCount_ = datasetTarget_->conceptBlockCount();
    populateCBCurriculumDropdown();
    syncCurriculumTrainingStageDropdown();
    // Count blocks in active curriculum
    if (!activeCurriculumId_.empty()) {
        auto curr = datasetTarget_->getCurriculumById(activeCurriculumId_);
        cbInCurrCount_ = curr.id.empty() ? 0 : static_cast<int>(curr.concept_block_ids.size());
    } else {
        cbInCurrCount_ = 0;
    }
    cbFilterDirty_ = true;
    rebuildFilteredCBList();
    populateCBModelDropdown();
}

void UIDataHubPanel::rebuildFilteredCBList() {
    cbFilterDirty_ = false;
    filteredCBIndices_.clear();
    if (!datasetTarget_) return;

    std::string formatFilter;
    if (cbFormatFilterIdx_ > 0 && cbFormatFilterIdx_ <= GRIM::kConceptPresetCount) {
        // Index 0 = "All Types"; indices 1..N map to kConceptPresets[0..N-1]
        formatFilter = GRIM::kConceptPresets[cbFormatFilterIdx_ - 1].key;
    }
    filteredCBIndices_ = datasetTarget_->filterConceptBlocks(formatFilter, cbFilterSearch_);

    // Optional: restrict to blocks in the active curriculum
    if (cbCurriculumFilterActive_ && !activeCurriculumId_.empty()) {
        auto curr = datasetTarget_->getCurriculumById(activeCurriculumId_);
        if (!curr.id.empty()) {
            std::unordered_set<std::string> inCurr(
                curr.concept_block_ids.begin(), curr.concept_block_ids.end());
            std::vector<size_t> filtered;
            filtered.reserve(filteredCBIndices_.size());
            for (size_t idx : filteredCBIndices_) {
                auto cb = datasetTarget_->getConceptBlock(idx);
                if (inCurr.count(cb.id))
                    filtered.push_back(idx);
            }
            filteredCBIndices_ = std::move(filtered);
        }
    }

    cbTotalCount_ = datasetTarget_->conceptBlockCount();
}

void UIDataHubPanel::loadConceptBlockIntoEditor(size_t cbIndex) {
    if (!datasetTarget_) return;
    auto cb = datasetTarget_->getConceptBlock(cbIndex);
    if (cb.id.empty()) return;

    if (cbNameInput_)    cbNameInput_->setText(cb.name);
    if (cbPromptArea_) {
        cbPromptArea_->setText(cb.format_type == "raw" ? cb.raw : cb.prompt);
    }
    if (cbTargetStateArea_) {
        cbTargetStateArea_->setText(
            cb.goal.has_value() ? cb.goal->target_state : std::string());
    }
    syncSuccessCriterionRows(
        cb.goal.has_value() ? static_cast<int>(cb.goal->success_criteria.size()) : 0);
    if (cb.goal.has_value()) {
        for (size_t i = 0; i < cb.goal->success_criteria.size(); ++i) {
            cbSuccessCriterionRows_[i].criterionArea->setText(
                cb.goal->success_criteria[i].criterion);
            cbSuccessCriterionRows_[i].evidenceArea->setText(
                cb.goal->success_criteria[i].evidence);
        }
    }
    if (cbAnswerArea_)   cbAnswerArea_->setText(cb.answer);

    int pi = GRIM::presetIndexForKey(cb.format_type);
    if (cbListTypeDropdown_ && pi >= 0) cbListTypeDropdown_->setSelectedIndex(pi);

    syncIntermediateAreas(static_cast<int>(cb.intermediates.size()));
    for (size_t i = 0; i < cb.intermediates.size() && i < cbIntermediateAreas_.size(); ++i) {
        cbIntermediateAreas_[i]->setText(cb.intermediates[i]);
    }

    // Execution steps
    syncExecStepRows(static_cast<int>(cb.execution.size()));
    for (size_t i = 0; i < cb.execution.size() && i < cbExecStepRows_.size(); ++i) {
        const auto& step = cb.execution[i];
        auto& row = cbExecStepRows_[i];

        // Map op string to dropdown index
        int opIdx = 0;
        if (step.op == "add") opIdx = 0;
        else if (step.op == "sub") opIdx = 1;
        else if (step.op == "mul") opIdx = 2;
        else if (step.op == "div") opIdx = 3;
        if (row.opDropdown) row.opDropdown->setSelectedIndex(opIdx);

        if (row.argSlotsInput) {
            std::ostringstream oss;
            for (size_t j = 0; j < step.arg_slots.size(); ++j) {
                if (j > 0) oss << ", ";
                oss << step.arg_slots[j];
            }
            row.argSlotsInput->setText(oss.str());
        }
        if (row.argsInput) {
            std::ostringstream oss;
            for (size_t j = 0; j < step.args.size(); ++j) {
                if (j > 0) oss << ", ";
                oss << step.args[j];
            }
            row.argsInput->setText(oss.str());
        }
        if (row.resultInput) row.resultInput->setText(std::to_string(step.result));
    }

    cbEditorScrollOffset_ = 0.0f;
}

void UIDataHubPanel::clearCBEditor() {
    if (cbNameInput_)    cbNameInput_->setText("");
    if (cbPromptArea_) cbPromptArea_->setText("");
    if (cbTargetStateArea_) cbTargetStateArea_->setText("");
    if (cbAnswerArea_)   cbAnswerArea_->setText("");
    cbSuccessCriterionRows_.clear();
    cbIntermediateAreas_.clear();
    cbExecStepRows_.clear();
    cbEditorScrollOffset_ = 0.0f;
}

void UIDataHubPanel::syncSuccessCriterionRows(int count) {
    if (count < 0) count = 0;
    while (static_cast<int>(cbSuccessCriterionRows_.size()) < count) {
        CBSuccessCriterionRow row;
        row.criterionArea = std::make_shared<UITextArea>(
            "", "",
            [](const std::string&) {});
        row.evidenceArea = std::make_shared<UITextArea>(
            "", "",
            [](const std::string&) {});
        cbSuccessCriterionRows_.push_back(std::move(row));
    }
    while (static_cast<int>(cbSuccessCriterionRows_.size()) > count) {
        cbSuccessCriterionRows_.pop_back();
    }
}

void UIDataHubPanel::syncIntermediateAreas(int count) {
    if (count < 0) count = 0;
    while (static_cast<int>(cbIntermediateAreas_.size()) < count) {
        auto area = std::make_shared<UITextArea>(
            "EXP " + std::to_string(cbIntermediateAreas_.size() + 1), "",
            [](const std::string&) {});
        cbIntermediateAreas_.push_back(area);
    }
    while (static_cast<int>(cbIntermediateAreas_.size()) > count) {
        cbIntermediateAreas_.pop_back();
    }
}

void UIDataHubPanel::syncExecStepRows(int count) {
    if (count < 0) count = 0;
    while (static_cast<int>(cbExecStepRows_.size()) < count) {
        CBExecStepRow row;
        row.opDropdown = std::make_shared<UIDropdown>(
            "", std::vector<std::string>{"add", "sub", "mul", "div"}, 0,
            [](int, const std::string&) {});
        row.opDropdown->setMaxVisibleItems(4);
        row.argSlotsInput = std::make_shared<UIInputBox>();
        row.argSlotsInput->setPlaceholder("slots: 0, 1");
        row.argsInput = std::make_shared<UIInputBox>();
        row.argsInput->setPlaceholder("args: 2.0, 3.0");
        row.resultInput = std::make_shared<UIInputBox>();
        row.resultInput->setPlaceholder("result");
        cbExecStepRows_.push_back(std::move(row));
    }
    while (static_cast<int>(cbExecStepRows_.size()) > count) {
        cbExecStepRows_.pop_back();
    }
}

bool UIDataHubPanel::buildConceptBlockFromEditor(
    GRIM::ConceptBlock& out,
    std::string& validation_error) const
{
    validation_error.clear();
    out = GRIM::ConceptBlock{};
    out.name = cbNameInput_ ? cbNameInput_->getText() : "";
    out.prompt = cbPromptArea_ ? cbPromptArea_->getText() : "";
    out.answer = cbAnswerArea_ ? cbAnswerArea_->getText() : "";
    auto trim = [](std::string value) {
        const auto first = value.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) return std::string{};
        const auto last = value.find_last_not_of(" \t\r\n");
        return value.substr(first, last - first + 1);
    };
    const std::string target_state = cbTargetStateArea_
        ? trim(cbTargetStateArea_->getText()) : std::string{};
    if (!target_state.empty() || !cbSuccessCriterionRows_.empty()) {
        GRIM::ConceptBlockGoal goal;
        goal.target_state = target_state;
        for (size_t i = 0; i < cbSuccessCriterionRows_.size(); ++i) {
            const auto& row = cbSuccessCriterionRows_[i];
            const std::string criterion = row.criterionArea
                ? trim(row.criterionArea->getText()) : std::string{};
            const std::string evidence = row.evidenceArea
                ? trim(row.evidenceArea->getText()) : std::string{};
            if (criterion.empty()) {
                validation_error = "Success criterion " + std::to_string(i + 1)
                    + " cannot be empty";
                return false;
            }
            goal.success_criteria.push_back(
                GRIM::ConceptBlockSuccessCriterion{criterion, evidence});
        }
        out.goal = std::move(goal);
    }

    const int preset_index = cbListTypeDropdown_
        ? cbListTypeDropdown_->getSelectedIndex() : 1;
    out.format_type = (preset_index >= 0 && preset_index < GRIM::kConceptPresetCount)
        ? GRIM::kConceptPresets[preset_index].key : "chain_of_thought";

    if (out.format_type == "raw") {
        out.raw = std::move(out.prompt);
        out.prompt.clear();
        out.answer.clear();
        out.goal.reset();
        if (trim(out.raw).empty()) {
            validation_error = "Raw text cannot be empty";
            return false;
        }
        out.recomputeDerived();
        return true;
    }

    for (const auto& area : cbIntermediateAreas_) {
        out.intermediates.push_back(area ? area->getText() : "");
    }

    auto parseDoubles = [&](const std::string& text, const std::string& field,
                            std::vector<double>& values) {
        if (trim(text).empty()) return true;
        std::istringstream input(text);
        std::string token;
        while (std::getline(input, token, ',')) {
            token = trim(token);
            try {
                size_t used = 0;
                const double value = std::stod(token, &used);
                if (token.empty() || used != token.size() || !std::isfinite(value)) {
                    validation_error = field + " contains invalid number '" + token + "'";
                    return false;
                }
                values.push_back(value);
            } catch (...) {
                validation_error = field + " contains invalid number '" + token + "'";
                return false;
            }
        }
        return true;
    };
    auto parseInts = [&](const std::string& text, const std::string& field,
                         std::vector<int>& values) {
        if (trim(text).empty()) return true;
        std::istringstream input(text);
        std::string token;
        while (std::getline(input, token, ',')) {
            token = trim(token);
            try {
                size_t used = 0;
                const int value = std::stoi(token, &used);
                if (token.empty() || used != token.size()) {
                    validation_error = field + " contains invalid integer '" + token + "'";
                    return false;
                }
                values.push_back(value);
            } catch (...) {
                validation_error = field + " contains invalid integer '" + token + "'";
                return false;
            }
        }
        return true;
    };

    static const char* op_names[] = {"add", "sub", "mul", "div"};
    for (size_t i = 0; i < cbExecStepRows_.size(); ++i) {
        const auto& row = cbExecStepRows_[i];
        GRIM::ConceptExecutionStep step;
        const int op_index = row.opDropdown ? row.opDropdown->getSelectedIndex() : 0;
        step.op = (op_index >= 0 && op_index < 4) ? op_names[op_index] : "add";
        const std::string field_prefix = "EXEC step " + std::to_string(i + 1);
        if (row.argSlotsInput
            && !parseInts(row.argSlotsInput->getText(), field_prefix + " arg_slots",
                          step.arg_slots)) {
            return false;
        }
        if (row.argsInput
            && !parseDoubles(row.argsInput->getText(), field_prefix + " args", step.args)) {
            return false;
        }
        const std::string result_text = row.resultInput
            ? trim(row.resultInput->getText()) : std::string{};
        try {
            size_t used = 0;
            step.result = std::stod(result_text, &used);
            if (result_text.empty() || used != result_text.size()
                || !std::isfinite(step.result)) {
                validation_error = field_prefix + " has an invalid result";
                return false;
            }
        } catch (...) {
            validation_error = field_prefix + " has an invalid result";
            return false;
        }
        out.execution.push_back(std::move(step));
    }

    out.recomputeDerived();
    validation_error = GRIM::validateConceptBlockExecutionControl(out);
    return validation_error.empty();
}

std::string UIDataHubPanel::buildTrainingPreview(const GRIM::ConceptBlock& cb, bool conceptMode) const {
    return conceptMode
        ? GRIM::ConceptCanonical::renderLogicalTrainingPreview(cb)
        : GRIM::ConceptCanonical::renderPlainText(cb);
}

void UIDataHubPanel::generateConceptBlock() {
    if (!structurer_ || !cbPromptArea_) return;
    std::string input = cbPromptArea_->getText();
    if (input.empty()) {
        addLog("Enter a prompt to generate from", 1);
        return;
    }

    std::string customPrompt = cbCustomPromptArea_ ? cbCustomPromptArea_->getText() : "";
    auto results = structurer_->structureEntry(input, "concept_block", customPrompt);
    if (results.empty()) {
        std::string err = structurer_->lastError();
        addLog("Generation failed: " + (err.empty() ? "LLM returned no output" : err), 2);
        return;
    }

    std::vector<std::string> intermediates;
    std::string prompt;
    std::string answer;

    for (const auto& line : results) {
        if (line.size() > 3 && line.substr(0, 3) == "Q: ") {
            prompt = line.substr(3);
        } else if (line.size() > 3 && line.substr(0, 3) == "T: ") {
            intermediates.push_back(line.substr(3));
        } else if (line.size() > 3 && line.substr(0, 3) == "A: ") {
            answer = line.substr(3);
        }
    }

    if (!prompt.empty() && cbPromptArea_)
        cbPromptArea_->setText(prompt);

    syncIntermediateAreas(static_cast<int>(intermediates.size()));
    for (size_t i = 0; i < intermediates.size() && i < cbIntermediateAreas_.size(); ++i) {
        cbIntermediateAreas_[i]->setText(intermediates[i]);
    }

    if (!answer.empty() && cbAnswerArea_)
        cbAnswerArea_->setText(answer);

    addLog("Generated ConceptBlock with " + std::to_string(intermediates.size()) + " steps", 0);
}

int UIDataHubPanel::cbCurriculumListRowCount() const {
    return static_cast<int>(filteredCBIndices_.size()) + (cbDraftPreviewActive_ ? 1 : 0);
}

bool UIDataHubPanel::cbCurriculumRowIsDraft(int listRow) const {
    return cbDraftPreviewActive_ && listRow == 0;
}

bool UIDataHubPanel::cbCurriculumRowToBlockIndex(int listRow, size_t& outBlockIndex) const {
    if (listRow < 0) return false;
    if (cbCurriculumRowIsDraft(listRow)) return false;
    const int adj = cbDraftPreviewActive_ ? (listRow - 1) : listRow;
    if (adj < 0 || adj >= static_cast<int>(filteredCBIndices_.size())) return false;
    outBlockIndex = filteredCBIndices_[static_cast<size_t>(adj)];
    return true;
}

void UIDataHubPanel::syncCBListTypeDropdownFromToolbar() {
    // No-op: toolbar format dropdown removed; cbListTypeDropdown_ is now the single source of truth.
}

void UIDataHubPanel::layoutCBListTypeDropdownInList(float listX, float listY, float listW) {
    if (!cbListTypeDropdown_) return;
    if (selectedCBRow_ < 0 || selectedCBRow_ >= cbCurriculumListRowCount()) {
        cbListTypeDropdown_->setPosition(-2000.0f, -2000.0f);
        return;
    }
    const float bodyY  = listY + kPoolHeaderH;
    const float rowTop = bodyY + selectedCBRow_ * kCBListRowH - cbListScrollOffset_;
    const float typeColX = listX + listW - 128.0f;
    cbListTypeDropdown_->setPosition(typeColX - 150.0f, rowTop + 4.0f);
    cbListTypeDropdown_->setSize(290.0f, 36.0f);
}

void UIDataHubPanel::populateCBModelDropdown() {
    if (!cbModelDropdown_) return;
    namespace fs = std::filesystem;
    fs::path modelStoreRoot = resolvePathFromGrimRoot(
        aiConfig.at("paths").at("grim_text").at("model_store").get<std::string>());

    std::vector<std::string> names;
    try {
        if (fs::exists(modelStoreRoot)) {
            for (const auto& entry : fs::directory_iterator(modelStoreRoot)) {
                if (entry.is_directory())
                    names.push_back(entry.path().filename().string());
            }
        }
    } catch (...) {}

    if (names.empty()) names.push_back("(none)");
    cbModelDropdown_->setItems(names);

    if (datasetTarget_) {
        // Update curriculum count for active curriculum
        if (!activeCurriculumId_.empty()) {
            auto curr = datasetTarget_->getCurriculumById(activeCurriculumId_);
            cbInCurrCount_ = curr.id.empty() ? 0 : static_cast<int>(curr.concept_block_ids.size());
        } else {
            cbInCurrCount_ = 0;
        }
    }
}

// ── Curriculum dropdown helpers ─────────────────────────

void UIDataHubPanel::populateCBCurriculumDropdown() {
    if (!cbCurriculumDropdown_ || !datasetTarget_) return;
    const auto& curricula = datasetTarget_->getCurriculums();
    std::vector<std::string> names;
    names.push_back("(none)");
    int selectedIdx = 0;
    for (size_t i = 0; i < curricula.size(); ++i) {
        names.push_back(curricula[i].name);
        if (curricula[i].id == activeCurriculumId_)
            selectedIdx = static_cast<int>(i) + 1; // +1 for "(none)" entry
    }
    cbCurriculumDropdown_->setItems(names);
    cbCurriculumDropdown_->setSelectedIndex(selectedIdx);
}

void UIDataHubPanel::syncCurriculumTrainingStageDropdown() {
    if (!cbTrainingStageDropdown_) return;

    int selectedIdx = 0;
    if (datasetTarget_ && !activeCurriculumId_.empty()) {
        const auto curr = datasetTarget_->getCurriculumById(activeCurriculumId_);
        if (curr.training_stage == "pt") selectedIdx = 1;
        else if (curr.training_stage == "sft") selectedIdx = 2;
        else if (curr.training_stage == "dpo") selectedIdx = 3;
        else if (curr.training_stage == "rlhf") selectedIdx = 4;
    }
    cbTrainingStageDropdown_->setSelectedIndex(selectedIdx);
}

void UIDataHubPanel::selectActiveCurriculum(int dropdownIndex) {
    if (!datasetTarget_) return;
    const auto& curricula = datasetTarget_->getCurriculums();
    if (dropdownIndex <= 0 || dropdownIndex > static_cast<int>(curricula.size())) {
        activeCurriculumId_.clear();
        cbInCurrCount_ = 0;
    } else {
        const auto& curr = curricula[dropdownIndex - 1]; // -1 for "(none)" entry
        activeCurriculumId_ = curr.id;
        cbInCurrCount_ = static_cast<int>(curr.concept_block_ids.size());
    }
    syncCurriculumTrainingStageDropdown();
    cbFilterDirty_ = true;
}

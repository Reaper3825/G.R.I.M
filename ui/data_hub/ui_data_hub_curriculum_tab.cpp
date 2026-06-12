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

    // Proportional: two dropdowns share remaining space after menu
    float ddW = (fullW - menuW - 3.0f * gap) * 0.5f;
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

    curriculumActionMenu_->setPosition(cx, y + 4.0f);
    curriculumActionMenu_->setSize(menuW, rowH - 8.0f);
    curriculumActionMenu_->drawOverlay(renderer, position);
    cx += menuW + gap;

    // Format-mode badge: show whether active curriculum is Concept or PT
    if (!activeCurriculumId_.empty() && datasetTarget_) {
        auto activeCurr = datasetTarget_->getCurriculumById(activeCurriculumId_);
        if (!activeCurr.id.empty()) {
            const char* modeLabel = activeCurr.format_as_concept ? "Concept" : "PT";
            uint32_t badgeBg = activeCurr.format_as_concept
                ? UITheme::Colors::Success : 0xFF88AADD;
            float badgeW = activeCurr.format_as_concept ? 68.0f : 38.0f;
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
                      conceptMode ? "Question (preview)" : "Text (preview)",
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
            qRaw    = cbQuestionArea_ ? cbQuestionArea_->getText() : "";
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
            qRaw      = cb.question;
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
    if (conceptMode) {
        std::string qLabel = std::string("Q: ") + preset.questionLabel;
        renderer.drawText({editorX + ePad, ey}, qLabel, UITheme::Colors::TextSecondary);
    } else {
        renderer.drawText({editorX + ePad, ey}, "Text", UITheme::Colors::TextSecondary);
    }
    ey += 20.0f;
    cbQuestionArea_->setPosition(editorX + ePad, ey);
    cbQuestionArea_->setSize(eInnerW, conceptMode ? areaH : areaH * 2.0f);
    cbQuestionArea_->drawOverlay(renderer, position);
    ey += (conceptMode ? areaH : areaH * 2.0f) + 16.0f;

    // ─── STATE0: Bootstrap Atoms (concept mode only) ────
    if (conceptMode) {
    renderer.drawRect({editorX + ePad, ey}, {eInnerW, 1.0f}, 0x18FFFFFF);
    ey += sectionGap;
    {
        float s0StartY = ey;
        ey += sectionPad;

        renderer.drawText({editorX + ePad + sectionPad, ey}, "STATE0  (bootstrap atoms)",
                          UITheme::Colors::TextSecondary);
        ey += 20.0f;
        float halfW = (eInnerW - 2.0f * sectionPad - 12.0f) * 0.5f;
        float innerLeft = editorX + ePad + sectionPad;
        renderer.drawText({innerLeft, ey}, "Type", UITheme::Colors::TextMuted);
        renderer.drawText({innerLeft + halfW + 12.0f, ey}, "Atoms (comma-separated)",
                          UITheme::Colors::TextMuted);
        ey += 16.0f;
        cbState0TypeInput_->setPosition(innerLeft, ey);
        cbState0TypeInput_->setSize(halfW, fieldH);
        cbState0TypeInput_->drawOverlay(renderer, position);
        cbState0AtomsInput_->setPosition(innerLeft + halfW + 12.0f, ey);
        cbState0AtomsInput_->setSize(halfW, fieldH);
        cbState0AtomsInput_->drawOverlay(renderer, position);
        ey += fieldH + sectionPad;

        // Section background (drawn behind)
        renderer.drawRoundedRect({editorX + ePad, s0StartY},
                                 {eInnerW, ey - s0StartY}, 0x0CFFFFFF, sectionRad);
    }
    ey += 16.0f;

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
        float argsW  = (innerW - opW - slotsW - colGap * 3.0f) * 0.5f;
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

            ey += fieldH + 8.0f;
        }

        // Exec step add/remove menu
        execStepActionMenu_->setPosition(innerLeft, ey);
        execStepActionMenu_->setSize(120.0f, 26.0f);
        execStepActionMenu_->drawOverlay(renderer, position);
        ey += 32.0f;

        // STATE1 derived result (inside the exec box)
        if (!cbExecStepRows_.empty()) {
            renderer.drawRect({innerLeft, ey}, {innerW, 1.0f}, 0x10FFFFFF);
            ey += 8.0f;
            std::string s1Label = "STATE1  result = ";
            auto& lastRow = cbExecStepRows_.back();
            std::string resText = lastRow.resultInput ? lastRow.resultInput->getText() : "?";
            s1Label += resText.empty() ? "?" : resText;
            renderer.drawText({innerLeft, ey}, s1Label, UITheme::Colors::Success);
            ey += 20.0f;
        }

        ey += sectionPad;

        // Section background
        renderer.drawRoundedRect({editorX + ePad, execStartY},
                                 {eInnerW, ey - execStartY}, 0x0CFFFFFF, sectionRad);
    }
    ey += 16.0f;

    } // end if (conceptMode) — STATE0 / EXP / EXEC hidden in PT mode

    // ─── A: Answer (concept mode only) ────────────────────
    if (conceptMode) {
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
                          conceptMode ? "Training Preview (canonical format)"
                                      : "Training Preview (raw)",
                          UITheme::Colors::TextSecondary);
        ey += 22.0f;

        // Build a ConceptBlock from current editor state for preview
        GRIM::ConceptBlock previewCB;
        previewCB.question = cbQuestionArea_ ? cbQuestionArea_->getText() : "";
        previewCB.answer   = (conceptMode && cbAnswerArea_) ? cbAnswerArea_->getText() : "";
        previewCB.state_0.type = cbState0TypeInput_ ? cbState0TypeInput_->getText() : "";
        if (conceptMode && cbState0AtomsInput_) {
            std::string atomsStr = cbState0AtomsInput_->getText();
            if (!atomsStr.empty()) {
                std::istringstream iss(atomsStr);
                std::string tok;
                while (std::getline(iss, tok, ',')) {
                    try { previewCB.state_0.atoms.push_back(std::stod(tok)); }
                    catch (...) {}
                }
            }
        }
        if (conceptMode) {
        for (const auto& row : cbExecStepRows_) {
            GRIM::ConceptExecutionStep step;
            static const char* opNames[] = {"add", "sub", "mul", "div"};
            int opIdx = row.opDropdown ? row.opDropdown->getSelectedIndex() : 0;
            step.op = (opIdx >= 0 && opIdx < 4) ? opNames[opIdx] : "add";
            if (row.argsInput) {
                std::istringstream iss(row.argsInput->getText());
                std::string tok;
                while (std::getline(iss, tok, ',')) {
                    try { step.args.push_back(std::stod(tok)); }
                    catch (...) {}
                }
            }
            if (row.resultInput) {
                try { step.result = std::stod(row.resultInput->getText()); }
                catch (...) { step.result = 0.0; }
            }
            previewCB.execution.push_back(std::move(step));
        }
        if (!previewCB.execution.empty()) {
            previewCB.state_1.result = previewCB.execution.back().result;
            previewCB.state_1.has_result = true;
        }
        for (const auto& area : cbIntermediateAreas_) {
            previewCB.explanation.push_back(area ? area->getText() : "");
        }
        } // end if (conceptMode)

        std::string preview = buildTrainingPreview(previewCB, conceptMode);

        // Pre-count lines so we can draw the background FIRST
        std::vector<std::pair<std::string, uint32_t>> previewLines;
        {
            std::istringstream pss(preview);
            std::string pline;
            while (std::getline(pss, pline)) {
                uint32_t lineCol = UITheme::Colors::TextMuted;
                if (conceptMode) {
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
                previewLines.push_back({pline, lineCol});
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
    datasetTarget_->loadConceptBlocks();
    cbTotalCount_ = datasetTarget_->conceptBlockCount();
    populateCBCurriculumDropdown();
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
    if (cbQuestionArea_) cbQuestionArea_->setText(cb.question);
    if (cbAnswerArea_)   cbAnswerArea_->setText(cb.answer);

    int pi = GRIM::presetIndexForKey(cb.format_type);
    if (cbListTypeDropdown_ && pi >= 0) cbListTypeDropdown_->setSelectedIndex(pi);

    syncIntermediateAreas(static_cast<int>(cb.intermediates.size()));
    for (size_t i = 0; i < cb.intermediates.size() && i < cbIntermediateAreas_.size(); ++i) {
        cbIntermediateAreas_[i]->setText(cb.intermediates[i]);
    }

    // State 0
    if (cbState0TypeInput_)  cbState0TypeInput_->setText(cb.state_0.type);
    if (cbState0AtomsInput_) {
        std::ostringstream oss;
        for (size_t i = 0; i < cb.state_0.atoms.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << cb.state_0.atoms[i];
        }
        cbState0AtomsInput_->setText(oss.str());
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
    if (cbQuestionArea_) cbQuestionArea_->setText("");
    if (cbAnswerArea_)   cbAnswerArea_->setText("");
    cbIntermediateAreas_.clear();
    if (cbState0TypeInput_)   cbState0TypeInput_->setText("");
    if (cbState0AtomsInput_)  cbState0AtomsInput_->setText("");
    cbExecStepRows_.clear();
    cbEditorScrollOffset_ = 0.0f;
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

std::string UIDataHubPanel::buildTrainingPreview(const GRIM::ConceptBlock& cb, bool conceptMode) const {
    std::ostringstream oss;

    if (conceptMode) {
        oss << "Q: " << cb.question << "\n";

        if (!cb.state_0.atoms.empty()) {
            oss << "STATE0 type=" << cb.state_0.type;
            for (double a : cb.state_0.atoms)
                oss << " " << a;
            oss << "\n";
        }

        for (const auto& exp : cb.explanation)
            oss << "EXP: " << exp << "\n";

        for (const auto& step : cb.execution) {
            oss << "EXEC " << step.op;
            for (double a : step.args)
                oss << " " << a;
            oss << " => " << step.result << "\n";
        }

        if (cb.state_1.has_result)
            oss << "STATE1 result=" << cb.state_1.result << "\n";

        oss << "A: " << cb.answer;
    } else {
        // Plain text / raw mode — no structural prefixes
        oss << cb.question;
        if (!cb.answer.empty())
            oss << "\n" << cb.answer;
    }

    return oss.str();
}

void UIDataHubPanel::generateConceptBlock() {
    if (!structurer_ || !cbQuestionArea_) return;
    std::string input = cbQuestionArea_->getText();
    if (input.empty()) {
        addLog("Enter a question/prompt to generate from", 1);
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
    std::string question;
    std::string answer;

    for (const auto& line : results) {
        if (line.size() > 3 && line.substr(0, 3) == "Q: ") {
            question = line.substr(3);
        } else if (line.size() > 3 && line.substr(0, 3) == "T: ") {
            intermediates.push_back(line.substr(3));
        } else if (line.size() > 3 && line.substr(0, 3) == "A: ") {
            answer = line.substr(3);
        }
    }

    if (!question.empty() && cbQuestionArea_)
        cbQuestionArea_->setText(question);

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
    cbFilterDirty_ = true;
}

#pragma once
#include "schedule_provider.hpp"
#include "core/calendar_time.hpp"
#include "core/input_parser.hpp"
#include "ui/primitives/ui_panel.hpp"
#include "ui/primitives/ui_button.hpp"
#include "ui/primitives/ui_inputbox.hpp"
#include "ui/primitives/ui_scrollbox.hpp"
#include "ui/overlay_renderer.hpp"
#include "ui/ui_theme.hpp"
#include <array>
#include <random>
#include <sstream>

namespace grim::scheduling {
class CalendarPanel : public UIPanel {
    using Json = nlohmann::json;
    grim::PluginHost api_;
    Provider& provider_;
    std::chrono::year_month month_;
    std::chrono::sys_days day_;
    std::chrono::steady_clock::time_point nextRefresh_{};
    std::vector<Json> entries_;
    std::array<int, 42> counts_{};
    std::string selected_, status_;
    std::shared_ptr<UIScrollBox> scroll_;
    std::shared_ptr<UIInputBox> titleInput_, startInput_, endInput_;
    std::shared_ptr<UIButton> previousMonth_, today_, nextMonth_, add_, previous_, next_, delete_;
    std::array<std::shared_ptr<UIButton>, 42> days_;
    struct Label { std::string text; Vec2 position, size; uint32_t color; };
    std::vector<Label> labels_;

    static int64_t milliseconds(std::chrono::sys_days day) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(day.time_since_epoch()).count();
    }
    void chooseDay(std::chrono::sys_days day) {
        const int year = int(std::chrono::year_month_day{day}.year());
        if (year < 1 || year > 9999) throw std::invalid_argument("Calendar date must be in years 1..9999");
        day_ = day;
        selected_.clear();
        startInput_->setText(formatCalendarTime(milliseconds(day) + 9 * 3600000));
        endInput_->setText(formatCalendarTime(milliseconds(day) + 10 * 3600000));
    }
    Json request(const std::string& op, const Json& args) {
        auto result = provider_.execute(op, args);
        if (!result.at("success").get<bool>()) throw std::runtime_error(result.at("message").get<std::string>());
        if (result.contains("resource_error"))
            api_->log(api_.instance(), GRIM_LOG_ERROR, result.at("resource_error").get_ref<const std::string&>().c_str());
        return result;
    }
    void refresh() {
        std::vector<Json> entries;
        std::array<int, 42> counts{};
        const auto grid = milliseconds(calendarGridStart(month_));
        const auto day = milliseconds(day_);
        size_t offset = 0;
        for (;;) {
            const auto page = request("list", {{"kind", "event"}, {"from_utc_ms", grid},
                {"until_utc_ms", grid + 42LL * 86400000}, {"limit", 200}, {"offset", offset}});
            const auto& rows = page.at("items");
            for (const auto& event : rows) {
                const auto start = event.at("start_utc_ms").get<int64_t>(), end = event.at("end_utc_ms").get<int64_t>();
                for (size_t i = 0; i < counts.size(); ++i) {
                    const auto cell = grid + int64_t(i) * 86400000;
                    if (start < cell + 86400000 && end > cell) ++counts[i];
                }
                if (start < day + 86400000 && end > day) entries.push_back(event);
            }
            offset += rows.size();
            if (rows.empty() || offset >= page.at("total").get<size_t>()) break;
        }
        const bool hadSelection = !selected_.empty();
        const bool retained = std::any_of(entries.begin(), entries.end(), [&](const auto& e) { return e.at("id") == selected_; });
        if (!retained) {
            selected_.clear();
            if (hadSelection) status_ = "Selected event is no longer available.";
            else if (!entries.empty()) selected_ = entries.front().at("id").get<std::string>();
        }
        entries_ = std::move(entries);
        counts_ = counts;
    }
    template<class Action> void act(Action action) {
        try { action(); refresh(); }
        catch (const std::exception& error) { status_ = error.what(); api_->log(api_.instance(), GRIM_LOG_ERROR, error.what()); }
        nextRefresh_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    }
    void changeMonth(int offset) {
        const auto next = month_ + std::chrono::months{offset};
        if (int(next.year()) < 1 || int(next.year()) > 9999) throw std::invalid_argument("Calendar month is out of range");
        month_ = next;
        chooseDay(std::chrono::sys_days{month_ / 1});
    }
    void chooseToday() {
        const auto today = std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now());
        const std::chrono::year_month_day date{today};
        month_ = date.year() / date.month();
        chooseDay(today);
    }
    auto selectedEntry() {
        return std::find_if(entries_.begin(), entries_.end(), [&](const auto& e) { return e.at("id") == selected_; });
    }
    void selectRelative(int offset) {
        if (entries_.empty()) return;
        const auto selected = selectedEntry();
        const auto index = selected == entries_.end() ? 0 : int(selected - entries_.begin());
        selected_ = entries_[std::clamp(index + offset, 0, int(entries_.size()) - 1)].at("id").get<std::string>();
    }
    void deleteSelected() {
        const auto selected = selectedEntry();
        if (selected == entries_.end()) return;
        request("cancel", {{"id", selected->at("id")}, {"revision", selected->at("revision")}});
        status_ = "Event deleted.";
    }
    void addEvent() {
        int64_t start, end;
        const auto title = titleInput_->getText();
        if (title.find_first_not_of(" \t\r\n") == std::string::npos || title.size() > 120)
            throw std::invalid_argument("Enter a title (1-120 bytes).");
        if (!parseCalendarTime(startInput_->getText(), start) || !parseCalendarTime(endInput_->getText(), end) || end <= start)
            throw std::invalid_argument("Use valid UTC dates; end must be after start.");
        std::random_device random;
        std::ostringstream key; key << "calendar-" << std::hex;
        for (int i = 0; i < 4; ++i) key << random() << '-';
        auto saved = request("create", {{"id", key.str()}, {"kind", "event"}, {"title", title},
            {"start_utc_ms", start}, {"end_utc_ms", end}});
        titleInput_->clear();
        day_ = std::chrono::floor<std::chrono::days>(std::chrono::sys_time<std::chrono::milliseconds>{std::chrono::milliseconds{start}});
        const std::chrono::year_month_day date{day_}; month_ = date.year() / date.month();
        selected_ = saved.at("item").at("id").get<std::string>();
        status_ = "Event saved.";
    }
    std::shared_ptr<UIButton> button(const std::string& text, std::function<void()> action) {
        auto result = std::make_shared<UIButton>(text, [this, action = std::move(action)] { act(action); });
        addChild(result);
        return result;
    }
    void layout() {
        namespace Colors = UITheme::Colors;
        const auto content = getContentRect();
        scroll_->setPosition(content.origin); scroll_->setSize(content.size);
        const float width = (std::max)(1.0f, content.size.x - 40);
        const bool wide = width >= 900;
        const float gridWidth = wide ? width * 0.46f : width;
        const float gridHeight = wide ? (std::max)(360.0f, content.size.y - 130) : 360.0f;
        const float rowHeight = (gridHeight - 100) / 6;
        scroll_->setContentHeight(wide ? (std::max)(gridHeight + 70, 490.0f) : gridHeight + 560);
        const float x = content.origin.x + 12, y = content.origin.y + 8 - scroll_->getScrollOffset();
        const float editorX = wide ? x + gridWidth + 24 : x;
        const float editorY = wide ? y : y + gridHeight + 70;
        const float editorWidth = wide ? width - gridWidth - 24 : width;
        auto place = [](Widget& widget, float px, float py, float w, float h) {
            widget.setPosition(px, py); widget.setSize(w, h);
        };
        labels_.clear();
        auto label = [&](std::string text, float px, float py, float w, uint32_t color = Colors::TextPrimary) {
            labels_.push_back({std::move(text), {px, py}, {w, 24}, color});
        };
        const char* months[] = {"January","February","March","April","May","June","July","August","September","October","November","December"};
        label(std::string(months[unsigned(month_.month()) - 1]) + " " + std::to_string(int(month_.year())), x, y, gridWidth);
        const float nav = (std::min)(100.0f, (gridWidth - 24) / 3);
        place(*previousMonth_, x, y + 32, nav, 28);
        place(*today_, x + (gridWidth - nav) / 2, y + 32, nav, 28);
        place(*nextMonth_, x + gridWidth - nav, y + 32, nav, 28);
        const char* weekdays[] = {"Mon","Tue","Wed","Thu","Fri","Sat","Sun"};
        for (int i = 0; i < 7; ++i) label(weekdays[i], x + float(i) * gridWidth / 7, y + 76, gridWidth / 7, Colors::TextSecondary);
        for (size_t i = 0; i < days_.size(); ++i) {
            const auto date = calendarGridStart(month_) + std::chrono::days{int(i)};
            auto text = std::to_string(unsigned(std::chrono::year_month_day{date}.day()));
            if (counts_[i] && gridWidth / 7 >= 64) text += " (" + std::to_string(counts_[i]) + ")";
            days_[i]->setText(text);
            days_[i]->setColors(date == day_ ? Colors::Primary : Colors::WidgetBg, Colors::WidgetBgHover, Colors::WidgetBgActive);
            place(*days_[i], x + float(i % 7) * (gridWidth + 10) / 7, y + 100 + float(i / 7) * rowHeight,
                  (gridWidth - 60) / 7, rowHeight - 10);
        }
        label("Selected: " + formatCalendarTime(milliseconds(day_)).substr(0,10) + " UTC", x, y + 110 + 6 * rowHeight, gridWidth);
        auto editorLabel = [&](const std::string& text, float offset) { label(text, editorX, editorY + offset, editorWidth); };
        editorLabel("Saved local calendar | UTC", 0);
        editorLabel("Title", 30); editorLabel("Start (UTC)", 86); editorLabel("End (UTC)", 142);
        place(*titleInput_, editorX, editorY + 52, editorWidth, 28);
        place(*startInput_, editorX, editorY + 108, editorWidth, 28);
        place(*endInput_, editorX, editorY + 164, editorWidth, 28);
        place(*add_, editorX, editorY + 210, 120, 28);
        editorLabel(status_, 254);
        const auto selected = selectedEntry();
        delete_->setVisible(selected != entries_.end());
        previous_->setVisible(!entries_.empty()); next_->setVisible(!entries_.empty());
        if (selected == entries_.end()) editorLabel(entries_.empty() ? "No events on this day." : "Select an event.", 296);
        else {
            editorLabel("Event " + std::to_string(selected - entries_.begin() + 1) + " of " + std::to_string(entries_.size()), 290);
            editorLabel(selected->at("title").get<std::string>(), 316);
            editorLabel("Start: " + formatCalendarTime(selected->at("start_utc_ms").get<int64_t>()), 348);
            editorLabel("End: " + formatCalendarTime(selected->at("end_utc_ms").get<int64_t>()), 374);
        }
        place(*previous_, editorX, editorY + 426, (editorWidth - 24) / 3, 28);
        place(*next_, editorX + (editorWidth + 12) / 3, editorY + 426, (editorWidth - 24) / 3, 28);
        place(*delete_, editorX + 2 * (editorWidth + 12) / 3, editorY + 426, (editorWidth - 24) / 3, 28);
    }
public:
    CalendarPanel(grim::PluginHost api, Provider& provider)
        : UIPanel("Calendar", true, "calendar", false), api_(api), provider_(provider) {
        if (!api_ || !api_->log) throw std::invalid_argument("Calendar requires provider logging");
        setPosition(240, 180); setSize(1160, 600); setVisible(false);
        scroll_ = std::make_shared<UIScrollBox>(); addChild(scroll_);
        titleInput_ = std::make_shared<UIInputBox>(); addChild(titleInput_);
        startInput_ = std::make_shared<UIInputBox>(); addChild(startInput_);
        endInput_ = std::make_shared<UIInputBox>(); addChild(endInput_);
        previousMonth_ = button("< Month", [this] { changeMonth(-1); });
        today_ = button("Today", [this] { chooseToday(); });
        nextMonth_ = button("Month >", [this] { changeMonth(1); });
        for (size_t i = 0; i < days_.size(); ++i)
            days_[i] = button("", [this, i] { chooseDay(calendarGridStart(month_) + std::chrono::days{int(i)}); });
        add_ = button("Add event", [this] { addEvent(); });
        previous_ = button("Previous", [this] { selectRelative(-1); });
        next_ = button("Next", [this] { selectRelative(1); });
        delete_ = button("Delete event", [this] { deleteSelected(); });
        chooseToday();
        layout();
    }
    void setVisible(bool value) override {
        if (value && !isVisible()) nextRefresh_ = {};
        UIPanel::setVisible(value);
    }
    void update(const InputState& input, float dt) override {
        if (!isVisible()) return;
        if (!isMinimized() && std::chrono::steady_clock::now() >= nextRefresh_) act([] {});
        layout();
        UIPanel::update(input, dt);
        layout();
    }
    bool drawOverlay(OverlayRenderer& renderer) override {
        if (!UIPanel::drawOverlay(renderer)) return false;
        layout();
        for (const auto& child : children)
            if (child->isVisible()) child->drawOverlay(renderer, getPosition());
        for (const auto& label : labels_) {
            renderer.pushClipRect(label.position, label.size);
            renderer.drawText(label.position, label.text, label.color);
            renderer.popClipRect();
        }
        renderer.popClipRect();
        return true;
    }
};
}

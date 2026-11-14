#pragma once
#include "schedule_provider.hpp"
#include "core/calendar_time.hpp"
#include "ui/ui_theme.hpp"
#include <array>
#include <map>
#include <random>
#include <sstream>

namespace grim::scheduling {
class CalendarPanel {
    using Json = nlohmann::json;
    const GrimPluginAPI* api_;
    Provider& provider_;
    GrimTimerHandle timer_ = nullptr;
    uint64_t runtime_ = 0, revision_ = 1;
    bool visible_ = false;
    float width_ = 1148, height_ = 558;
    std::chrono::year_month month_;
    std::chrono::sys_days day_;
    std::chrono::steady_clock::time_point nextRefresh_{};
    struct Draft { std::string value; uint64_t revision = 1; };
    std::map<std::string, Draft> drafts_{{"title", {}}, {"start", {}}, {"end", {}}};
    std::vector<Json> entries_;
    std::array<int, 42> counts_{};
    std::string selected_, status_;

    static int64_t milliseconds(std::chrono::sys_days day) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(day.time_since_epoch()).count();
    }
    void chooseDay(std::chrono::sys_days day) {
        const int year = int(std::chrono::year_month_day{day}.year());
        if (year < 1 || year > 9999) throw std::invalid_argument("Calendar date must be in years 1..9999");
        day_ = day;
        selected_.clear();
        drafts_.at("start") = {formatCalendarTime(milliseconds(day) + 9 * 3600000), drafts_.at("start").revision + 1};
        drafts_.at("end") = {formatCalendarTime(milliseconds(day) + 10 * 3600000), drafts_.at("end").revision + 1};
    }
    Json request(const std::string& op, const Json& args) {
        auto result = provider_.execute(op, args);
        if (!result.at("success").get<bool>()) throw std::runtime_error(result.at("message").get<std::string>());
        if (result.contains("resource_error")) throw std::runtime_error(result.at("resource_error").get<std::string>());
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
    void edit(const std::string& id, const Json& value) {
        auto& draft = drafts_.at(id);
        if (value.at("value_revision").get<uint64_t>() == draft.revision) {
            const auto text = value.at("value").get<std::string>();
            if (text.size() > 16384) throw std::invalid_argument("Calendar input is too long");
            draft.value = text;
        }
    }
    void action(const Json& event) {
        for (auto it = event.at("values").begin(); it != event.at("values").end(); ++it) edit(it.key(), it.value());
        const auto& action = event.at("action");
        const auto op = action.at("op").get<std::string>();
        if (op == "day") chooseDay(std::chrono::sys_days{std::chrono::days{action.at("day").get<int64_t>()}});
        else if (op == "month") {
            const auto next = month_ + std::chrono::months{action.at("offset").get<int>()};
            if (int(next.year()) < 1 || int(next.year()) > 9999) throw std::invalid_argument("Calendar month is out of range");
            month_ = next;
            chooseDay(std::chrono::sys_days{month_ / 1});
        } else if (op == "today") {
            const auto today = std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now());
            const std::chrono::year_month_day date{today}; month_ = date.year() / date.month(); chooseDay(today);
        } else if (op == "select") selected_ = action.at("id").get<std::string>();
        else if (op == "delete") {
            request("cancel", {{"id", action.at("id")}, {"revision", action.at("revision")}});
            status_ = "Event deleted.";
        } else if (op == "add") {
            int64_t start, end;
            const auto& title = drafts_.at("title").value;
            if (title.find_first_not_of(" \t\r\n") == std::string::npos || title.size() > 120)
                throw std::invalid_argument("Enter a title (1-120 bytes).");
            if (!parseCalendarTime(drafts_.at("start").value, start) || !parseCalendarTime(drafts_.at("end").value, end) || end <= start)
                throw std::invalid_argument("Use valid UTC dates; end must be after start.");
            std::random_device random;
            std::ostringstream key; key << "calendar-" << std::hex;
            for (int i = 0; i < 4; ++i) key << random() << '-';
            auto saved = request("create", {{"id", key.str()}, {"kind", "event"}, {"title", title},
                {"start_utc_ms", start}, {"end_utc_ms", end}});
            drafts_.at("title") = {"", drafts_.at("title").revision + 1};
            day_ = std::chrono::floor<std::chrono::days>(std::chrono::sys_time<std::chrono::milliseconds>{std::chrono::milliseconds{start}});
            const std::chrono::year_month_day date{day_}; month_ = date.year() / date.month();
            selected_ = saved.at("item").at("id").get<std::string>();
            status_ = "Event saved.";
        } else throw std::invalid_argument("Unknown Calendar action");
        refresh();
    }
    void publish() {
        if (!runtime_) throw std::logic_error("Calendar has not activated");
        ++revision_;
        const auto text = definition().dump();
        if (api_->update_panel_definition(api_, text.data(), text.size()) != GRIM_OK)
            throw std::runtime_error("Calendar definition update rejected");
    }
    static void tick(uint64_t, void* context) noexcept {
        auto& self = *static_cast<CalendarPanel*>(context);
        try { self.pump(); }
        catch (const std::exception& error) { self.api_->log(self.api_, GRIM_LOG_ERROR, error.what()); }
    }
public:
    CalendarPanel(const GrimPluginAPI* api, Provider& provider) : api_(api), provider_(provider) {
        if (!api_ || !api_->register_panel_definition || !api_->update_panel_definition || !api_->poll_panel_events ||
            !api_->string_free || !api_->create_timer || !api_->cancel_timer || !api_->log)
            throw std::invalid_argument("Calendar requires copied panel, timer and logging services");
        const auto today = std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now());
        const std::chrono::year_month_day date{today}; month_ = date.year() / date.month(); chooseDay(today);
        status_ = "Calendar loading.";
    }
    ~CalendarPanel() {
        if (timer_ && api_->cancel_timer(api_, timer_) != GRIM_OK)
            api_->log(api_, GRIM_LOG_ERROR, "Calendar timer cancellation failed");
    }
    void prepare() {
        const auto text = definition().dump();
        if (api_->register_panel_definition(api_, text.data(), text.size()) != GRIM_OK)
            throw std::runtime_error("Calendar panel registration rejected");
        timer_ = api_->create_timer(api_, 50, true, tick, this);
        if (!timer_) throw std::runtime_error("Calendar event timer creation failed");
    }
    void pump() {
        const char* raw = api_->poll_panel_events(api_);
        if (!raw) throw std::runtime_error("Calendar event polling failed");
        struct Release { const GrimPluginAPI* api; const char* text; ~Release() { api->string_free(api, text); } } release{api_, raw};
        const auto response = Json::parse(raw);
        if (response.at("version") != 1) throw std::runtime_error("Unsupported Calendar event version");
        bool dirty = false;
        for (const auto& event : response.at("events")) {
            if (event.at("panel_id") != "calendar") throw std::runtime_error("Foreign panel event delivered to Calendar");
            const auto type = event.at("type").get<std::string>();
            if (type == "activated") {
                if (!event.at("success").get<bool>()) throw std::runtime_error(event.at("error").get<std::string>());
                runtime_ = event.at("runtime_id").get<uint64_t>();
                dirty = true;
                continue;
            }
            if (!runtime_ || event.at("runtime_id").get<uint64_t>() != runtime_) continue;
            try {
                if (type == "visibility") visible_ = event.at("visible").get<bool>();
                else if (type == "geometry") { width_ = event.at("width").get<float>(); height_ = event.at("height").get<float>(); }
                else if (type == "change") { edit(event.at("widget_id").get<std::string>(), event); continue; }
                else if (type == "click") action(event);
                else if (type == "error") throw std::runtime_error(event.at("error").get<std::string>());
                else throw std::runtime_error("Unknown Calendar event");
            } catch (const std::exception& error) {
                status_ = error.what();
                api_->log(api_, GRIM_LOG_ERROR, error.what());
            }
            dirty = true;
        }
        if (runtime_ && visible_ && std::chrono::steady_clock::now() >= nextRefresh_) {
            try { refresh(); if (status_ == "Calendar loading.") status_.clear(); }
            catch (const std::exception& error) { status_ = error.what(); api_->log(api_, GRIM_LOG_ERROR, error.what()); }
            nextRefresh_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
            dirty = true;
        }
        if (runtime_ && dirty) publish();
    }
    Json definition() const {
        using namespace UITheme;
        const float width = (std::max)(1.0f, width_ - 40), gridWidth = width >= 900 ? width * 0.46f : width;
        const bool wide = width >= 900;
        const float gridHeight = wide ? (std::max)(360.0f, height_ - 130) : 360.0f;
        const float rowHeight = (gridHeight - 100) / 6;
        const float x = 12, y = 8, editorX = wide ? x + gridWidth + 24 : x;
        const float editorY = wide ? y : y + gridHeight + 70, editorWidth = wide ? width - gridWidth - 24 : width;
        Json result = {{"version", 1}, {"panel_id", "calendar"}, {"runtime_id", runtime_}, {"revision", revision_},
            {"title", "Calendar"}, {"width", 1160}, {"height", 600},
            {"content_height", wide ? (std::max)(gridHeight + 70, 490.0f) : gridHeight + 560}, {"widgets", Json::array()}};
        auto& nodes = result["widgets"];
        const auto node = [&](std::string id, const char* type, std::string text, float px, float py, float w, float h,
                              uint32_t color, Json action = Json::object(), uint64_t valueRevision = 0) {
            nodes.push_back({{"id", std::move(id)}, {"type", type}, {"text", std::move(text)}, {"rect", {px, py, w, h}},
                {"color", color}, {"action", std::move(action)}, {"value_revision", valueRevision}});
        };
        const char* months[] = {"January","February","March","April","May","June","July","August","September","October","November","December"};
        node("month", "label", std::string(months[unsigned(month_.month()) - 1]) + " " + std::to_string(int(month_.year())),
             x, y, gridWidth, 24, Colors::TextPrimary);
        const float nav = (std::min)(100.0f, (gridWidth - 24) / 3);
        node("previous-month", "button", "< Month", x, y + 32, nav, 28, Colors::WidgetBg, {{"op", "month"}, {"offset", -1}});
        node("today", "button", "Today", x + (gridWidth - nav) / 2, y + 32, nav, 28, Colors::WidgetBg, {{"op", "today"}});
        node("next-month", "button", "Month >", x + gridWidth - nav, y + 32, nav, 28, Colors::WidgetBg, {{"op", "month"}, {"offset", 1}});
        const char* days[] = {"Mon","Tue","Wed","Thu","Fri","Sat","Sun"};
        for (int i = 0; i < 7; ++i)
            node("weekday-" + std::to_string(i), "label", days[i], x + i * gridWidth / 7, y + 76, gridWidth / 7, 24, Colors::TextSecondary);
        for (size_t i = 0; i < 42; ++i) {
            const auto date = calendarGridStart(month_) + std::chrono::days{int(i)};
            auto label = std::to_string(unsigned(std::chrono::year_month_day{date}.day()));
            if (counts_[i] && gridWidth / 7 >= 64) label += " (" + std::to_string(counts_[i]) + ")";
            node("day-" + std::to_string(i), "button", label, x + float(i % 7) * (gridWidth + 10) / 7,
                y + 100 + float(i / 7) * rowHeight, (gridWidth - 60) / 7, rowHeight - 10,
                date == day_ ? Colors::Primary : Colors::WidgetBg, {{"op", "day"}, {"day", date.time_since_epoch().count()}});
        }
        node("selected-day", "label", "Selected: " + formatCalendarTime(milliseconds(day_)).substr(0,10) + " UTC",
             x, y + 110 + 6 * rowHeight, gridWidth, 24, Colors::TextPrimary);
        const auto label = [&](const char* id, float offset, const std::string& text) {
            node(id, "label", text, editorX, editorY + offset, editorWidth, 24, Colors::TextPrimary);
        };
        label("header", 0, "Saved local calendar | UTC");
        label("title-label", 30, "Title"); label("start-label", 86, "Start (UTC)"); label("end-label", 142, "End (UTC)");
        for (const auto& field : std::array<std::pair<const char*, float>,3>{{{"title",52},{"start",108},{"end",164}}}) {
            const auto& draft = drafts_.at(field.first);
            node(field.first, "input", draft.value, editorX, editorY + field.second, editorWidth, 28,
                Colors::TextPrimary, Json::object(), draft.revision);
        }
        node("add", "button", "Add event", editorX, editorY + 210, 120, 28, Colors::WidgetBg, {{"op", "add"}});
        label("status", 254, status_);
        auto selected = std::find_if(entries_.begin(), entries_.end(), [&](const auto& e) { return e.at("id") == selected_; });
        if (selected == entries_.end()) label("empty", 296, entries_.empty() ? "No events on this day." : "Select an event.");
        else {
            label("event-index", 290, "Event " + std::to_string(selected - entries_.begin() + 1) + " of " + std::to_string(entries_.size()));
            label("event-title", 316, selected->at("title").get<std::string>());
            label("event-start", 348, "Start: " + formatCalendarTime(selected->at("start_utc_ms").get<int64_t>()) + " UTC");
            label("event-end", 374, "End: " + formatCalendarTime(selected->at("end_utc_ms").get<int64_t>()) + " UTC");
            node("delete", "button", editorWidth < 400 ? "Delete" : "Delete event",
                editorX + 2 * (editorWidth + 12) / 3, editorY + 426, (editorWidth - 24) / 3, 28, Colors::WidgetBg,
                {{"op", "delete"}, {"id", selected->at("id")}, {"revision", selected->at("revision")}});
        }
        if (!entries_.empty()) {
            auto index = size_t{0};
            if (selected != entries_.end()) index = size_t(selected - entries_.begin());
            const size_t previous = index > 0 ? index - 1 : 0, next = (std::min)(index + 1, entries_.size() - 1);
            node("previous", "button", "Previous", editorX, editorY + 426, (editorWidth - 24) / 3, 28,
                Colors::WidgetBg, {{"op", "select"}, {"id", entries_[previous].at("id")}});
            node("next", "button", "Next", editorX + (editorWidth + 12) / 3, editorY + 426, (editorWidth - 24) / 3, 28,
                Colors::WidgetBg, {{"op", "select"}, {"id", entries_[next].at("id")}});
        }
        return result;
    }
};
}

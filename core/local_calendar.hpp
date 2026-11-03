#pragma once
#include "scheduling_api.h"
#include "calendar_file.hpp"
#include <nlohmann/json.hpp>
#include <memory>
#include <algorithm>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace grim::scheduling {
// Local calendar with optional durable storage. No alarm capability.
class LocalCalendar {
    friend class SchedulingService;
    struct Event {
        std::string id, calendar, title, description, zone;
        int64_t start, end;
        uint64_t revision;
        std::string kind="event", delivery="not_requested";
        GrimScheduleEvent view() const {
            return {id.c_str(), calendar.c_str(), title.c_str(), description.c_str(),
                zone.c_str(), start, end, revision};
        }
    };
    std::map<std::string, Event> events;
    std::unique_ptr<CalendarFile> file;
    void save(const std::map<std::string,Event>& records, uint64_t revision) {
        if(!file) return;
        nlohmann::json rows=nlohmann::json::array();
        for(const auto& [id,e]:records) rows.push_back({{"id",id},{"calendar_id",e.calendar},
            {"title",e.title},{"description",e.description},{"timezone",e.zone},
            {"start_utc_ms",e.start},{"end_utc_ms",e.end},{"revision",e.revision},{"kind",e.kind},{"delivery",e.delivery}});
        file->write(nlohmann::json{{"schema_version",2},{"sequence",revision},{"events",rows}}.dump(2));
    }
    void load() {
        if(!std::filesystem::exists(file->location())) return;
        std::ifstream input(file->location());
        const auto data=nlohmann::json::parse(input);
        if((data.at("schema_version")!=1 && data.at("schema_version")!=2) || !data.at("sequence").is_number_unsigned() || !data.at("events").is_array()) throw std::runtime_error("Invalid calendar schema");
        sequence=data.at("sequence").get<uint64_t>();
        for(const auto& row:data.at("events")) {
            if(!row.at("revision").is_number_unsigned() || !row.at("start_utc_ms").is_number_integer() || !row.at("end_utc_ms").is_number_integer()) throw std::runtime_error("Invalid event numbers");
            for(const auto* field:{"start_utc_ms","end_utc_ms"})
                if(row.at(field).is_number_unsigned() && row.at(field).get<uint64_t>()>uint64_t(INT64_MAX)) throw std::runtime_error("Event timestamp overflow");
            Event e{row.at("id").get<std::string>(),row.at("calendar_id").get<std::string>(),
                row.at("title").get<std::string>(),row.at("description").get<std::string>(),row.at("timezone").get<std::string>(),
                row.at("start_utc_ms").get<int64_t>(),row.at("end_utc_ms").get<int64_t>(),row.at("revision").get<uint64_t>()};
            e.kind=row.value("kind",std::string("event"));e.delivery=row.value("delivery",std::string("not_requested"));
            if((e.kind!="event" && e.kind!="timer" && e.kind!="alarm" && e.kind!="reminder") ||
               (e.delivery!="not_requested" && e.delivery!="pending" && e.delivery!="dispatched")) throw std::runtime_error("Invalid scheduling metadata");
            if(e.id.empty() || e.calendar.empty() || e.title.empty() || e.zone!="Etc/UTC" || e.end<=e.start || !e.revision || e.revision>sequence ||
                e.start < -62135596800000LL || e.end > 253402300800000LL ||
                e.id.find('\0')!=std::string::npos || e.calendar.find('\0')!=std::string::npos || e.title.find('\0')!=std::string::npos || e.description.find('\0')!=std::string::npos)
                throw std::runtime_error("Invalid calendar event");
            if(!events.emplace(e.id,e).second) throw std::runtime_error("Duplicate event id");
        }
    }
    uint64_t sequence = 0; // Never reuse revisions after delete/recreate.
    static bool present(const char* s) { return s && *s; }
    static GrimScheduleResult authorize(void* ctx, uint32_t caps) noexcept {
        return (caps & ~(GRIM_SCHEDULE_CALENDAR | (static_cast<LocalCalendar*>(ctx)->file ? GRIM_SCHEDULE_PERSISTENT : 0))) ? GRIM_SCHEDULE_UNSUPPORTED : GRIM_SCHEDULE_OK;
    }
    static GrimScheduleResult put(void* ctx, const GrimScheduleEvent* e, uint64_t* out) noexcept {
        try {
            if (!ctx || !e || !out || !present(e->id) || !present(e->calendar_id) ||
                !present(e->title) || !e->description || !present(e->timezone) ||
                e->end_utc_ms <= e->start_utc_ms || e->start_utc_ms < -62135596800000LL || e->end_utc_ms > 253402300800000LL) return GRIM_SCHEDULE_INVALID;
            // No bundled timezone database yet: explicitly support UTC only.
            if (std::string(e->timezone) != "Etc/UTC") return GRIM_SCHEDULE_UNSUPPORTED;
            auto& self = *static_cast<LocalCalendar*>(ctx);
            auto it = self.events.find(e->id);
            if (it == self.events.end() ? e->revision != 0 : e->revision != it->second.revision)
                return GRIM_SCHEDULE_CONFLICT;
            if (self.sequence == (std::numeric_limits<uint64_t>::max)()) return GRIM_SCHEDULE_FAILED;
            Event value{e->id, e->calendar_id, e->title, e->description, e->timezone,
                e->start_utc_ms, e->end_utc_ms, self.sequence + 1};
            auto candidate=self.events;
            const auto key=value.id;
            candidate.insert_or_assign(key, std::move(value));
            self.save(candidate,self.sequence+1);
            self.events.swap(candidate);
            *out = ++self.sequence;
            return GRIM_SCHEDULE_OK;
        } catch (...) { return GRIM_SCHEDULE_FAILED; }
    }
    static GrimScheduleResult remove(void* ctx, const char* id, uint64_t revision) noexcept {
        try {
            if (!ctx || !present(id) || !revision) return GRIM_SCHEDULE_INVALID;
            auto& events = static_cast<LocalCalendar*>(ctx)->events;
            auto it = events.find(id);
            if (it == events.end()) return GRIM_SCHEDULE_NOT_FOUND;
            if (it->second.revision != revision) return GRIM_SCHEDULE_CONFLICT;
            auto& self=*static_cast<LocalCalendar*>(ctx);
            auto candidate=events;
            candidate.erase(id);
            self.save(candidate,self.sequence);
            events.swap(candidate);
            return GRIM_SCHEDULE_OK;
        } catch (...) { return GRIM_SCHEDULE_FAILED; }
    }
    static GrimScheduleResult query(void* ctx, const char* calendar, int64_t from, int64_t until,
        GrimScheduleEventVisitor visit, void* user) noexcept {
        try {
            if (!ctx || !present(calendar) || from >= until || !visit) return GRIM_SCHEDULE_INVALID;
            std::vector<const Event*> matches;
            for (const auto& pair : static_cast<LocalCalendar*>(ctx)->events) {
                const auto& e = pair.second;
                if (e.calendar == calendar && e.start < until && e.end > from) matches.push_back(&e);
            }
            std::sort(matches.begin(), matches.end(), [](auto a, auto b) {
                return a->start == b->start ? a->id < b->id : a->start < b->start;
            });
            for (auto e : matches) { auto view = e->view(); if (visit(user, &view)) break; }
            return GRIM_SCHEDULE_OK;
        } catch (...) { return GRIM_SCHEDULE_FAILED; }
    }
public:
    static GrimScheduleResult create(uint32_t version, GrimScheduleProviderV1* out, const std::filesystem::path& path = {}) noexcept {
        if (!out) return GRIM_SCHEDULE_INVALID;
        if (version != GRIM_SCHEDULE_ABI_V1 || out->abi_version != version ||
            out->struct_size < sizeof(*out)) return GRIM_SCHEDULE_UNSUPPORTED;
        try {
            auto owned=std::make_unique<LocalCalendar>();
            if(!path.empty()) {owned->file=std::make_unique<CalendarFile>(std::filesystem::absolute(path));owned->load();}
            auto* context=owned.get();
            *out = {GRIM_SCHEDULE_ABI_V1, sizeof(*out), uint32_t(GRIM_SCHEDULE_CALENDAR | (context->file ? GRIM_SCHEDULE_PERSISTENT : 0)),
                context->file ? "grim.local.durable" : "grim.local.session", context, authorize, query, put, remove, nullptr, nullptr,
                [](void* p) { delete static_cast<LocalCalendar*>(p); }};
            owned.release();
            return GRIM_SCHEDULE_OK;
        } catch (...) { return GRIM_SCHEDULE_FAILED; }
    }
};
}

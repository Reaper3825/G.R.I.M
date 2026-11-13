#pragma once
#include "local_calendar.hpp"
#include <mutex>
#include <chrono>
#include <random>
#include <sstream>
namespace grim::scheduling {
class SchedulingService {
    std::mutex mutex;
    GrimScheduleProviderV1 provider{};
    std::filesystem::path path;
    std::string deliveryError;
    int64_t lastPoll=0;
    using Json=nlohmann::json;
    LocalCalendar& store() {
        if(!provider.context) {
            provider.abi_version=1;provider.struct_size=sizeof(provider);
            if(LocalCalendar::create(1,&provider,path.empty()?calendarStorePath():path)!=GRIM_SCHEDULE_OK)
                throw std::runtime_error("Saved calendar unavailable, locked or invalid");
        }
        return *static_cast<LocalCalendar*>(provider.context);
    }
    static Json record(const LocalCalendar::Event& e) {
        return {{"id",e.id},{"kind",e.kind},{"calendar_id",e.calendar},{"title",e.title},{"description",e.description},
            {"timezone",e.zone},{"start_utc_ms",e.start},{"end_utc_ms",e.end},{"revision",e.revision},
            {"delivery_status",e.delivery},{"device","local_host"}};
    }
    static std::string str(const Json& j,const char* key) {
        const auto value=j.at(key).get<std::string>();
        if(value.empty() || value.size()>4096 || value.find('\0')!=std::string::npos) throw std::invalid_argument(key);
        return value;
    }
    static int64_t number(const Json& j,const char* key) {
        const auto& v=j.at(key);
        if(!v.is_number_integer() || (v.is_number_unsigned() && v.get<uint64_t>()>uint64_t(INT64_MAX))) throw std::invalid_argument(key);
        return v.get<int64_t>();
    }
    static Json error(const char* code,const std::string& message) {return {{"success",false},{"error",code},{"message",message}};}
public:
    explicit SchedulingService(std::filesystem::path location={}):path(location.empty()?calendarStorePath():std::filesystem::absolute(location)) {}
    ~SchedulingService(){if(provider.context) provider.destroy(provider.context);}
    struct Lease {
        std::unique_lock<std::mutex> lock;
        GrimScheduleProviderV1& provider;
        ~Lease() { if (provider.context) { provider.destroy(provider.context); provider = {}; } }
    };
    Lease acquire(){std::unique_lock lock(mutex);store();return {std::move(lock),provider};}
    Json execute(const std::string& op,const Json& args,int64_t now) {
        try {
            auto lease = acquire();
            if(!args.is_object()) return error("invalid_arguments","Expected a JSON object");
            // Reject misspellings and undeclared fields instead of silently ignoring them.
            std::vector<std::string> allowed;
            if(op=="create" || op=="update") allowed={"id","kind","title","description","start_utc_ms","end_utc_ms","duration_seconds","revision"};
            else if(op=="get" || op=="cancel") allowed={"id","revision"};
            else if(op=="list") allowed={"kind","from_utc_ms","until_utc_ms","limit","offset"};
            else if(op=="status") allowed={"id"};
            else return error("unsupported_operation",op);
            for(auto it=args.begin();it!=args.end();++it)
                if(std::find(allowed.begin(),allowed.end(),it.key())==allowed.end()) return error("invalid_arguments","Unknown field: "+it.key());
            auto& s=store();
            if(op=="status" && !args.contains("id")) return {{"success",true},{"persistent",true},{"supported_kinds",{"event","timer","alarm","reminder"}},
                {"device","local_host"},{"delivery","console_notifications"},{"last_poll_utc_ms",lastPoll},{"delivery_error",deliveryError},{"native_calendar_sync",false},{"background_delivery",false},{"overdue_policy","dispatch_on_next_provider_poll"}};
            if(op=="list") {
                auto from=args.contains("from_utc_ms")?number(args,"from_utc_ms"):INT64_MIN;
                auto until=args.contains("until_utc_ms")?number(args,"until_utc_ms"):INT64_MAX;
                auto limit=args.contains("limit")?number(args,"limit"):50;
                auto offset=args.contains("offset")?number(args,"offset"):0;
                if(from>=until || limit<1 || limit>200 || offset<0) return error("invalid_arguments","Invalid range or pagination");
                std::string kind=args.contains("kind")?str(args,"kind"):"";
                if(!kind.empty() && kind!="event" && kind!="timer" && kind!="alarm" && kind!="reminder") return error("invalid_arguments","Unknown kind");
                std::vector<const LocalCalendar::Event*> rows;
                for(const auto& [id,e]:s.events) if(e.start<until && e.end>from && (kind.empty() || e.kind==kind)) rows.push_back(&e);
                std::sort(rows.begin(),rows.end(),[](auto a,auto b){return a->start==b->start?a->id<b->id:a->start<b->start;});
                Json items=Json::array();for(size_t i=size_t(offset);i<rows.size() && items.size()<size_t(limit);++i) items.push_back(record(*rows[i]));
                return {{"success",true},{"items",items},{"total",rows.size()},{"offset",offset}};
            }
            std::string id;
            if(op=="create") {
                if(args.contains("revision")) return error("invalid_arguments","Create does not accept revision");
                if(args.contains("id")) id=str(args,"id");
                else {std::random_device r;std::ostringstream text;text<<"schedule-"<<std::hex;for(int i=0;i<4;++i) text<<r()<<'-';id=text.str();}
                if(s.events.count(id)) return error("conflict","ID already exists; retrieve it before retrying");
            } else {id=str(args,"id");if(!s.events.count(id)) return error("not_found","Scheduled item not found");}
            if(op=="get" || op=="status") return {{"success",true},{"item",record(s.events.at(id))}};
            auto candidate=s.events;
            if(op!="create") {
                if(number(args,"revision")<=0 || uint64_t(number(args,"revision"))!=s.events.at(id).revision) return error("conflict","Fetch the current revision before modifying");
            }
            if(op=="cancel") {const bool dispatched=candidate.at(id).delivery=="dispatched";candidate.erase(id);s.save(candidate,s.sequence);s.events.swap(candidate);return {{"success",true},{"id",id},{"cancelled",true},{"already_dispatched",dispatched}};}
            if(s.sequence>=uint64_t(INT64_MAX)) return error("unavailable","Revision space exhausted");
            LocalCalendar::Event e=op=="create"?LocalCalendar::Event{id,"local","","","Etc/UTC",0,0,0}:s.events.at(id);
            if(op=="create") {e.kind=str(args,"kind");e.title=str(args,"title");}
            if(args.contains("kind") && op=="update") return error("invalid_arguments","Item kind cannot change");
            if(args.contains("title")) e.title=str(args,"title");
            if(e.title.size()>120 || e.title.find_first_not_of(" \t\r\n")==std::string::npos) return error("invalid_arguments","Title must contain 1..120 nonblank bytes");
            if(args.contains("description")) {e.description=args.at("description").get<std::string>();if(e.description.size()>4096 || e.description.find('\0')!=std::string::npos) throw std::invalid_argument("description");}
            if(e.kind!="event" && e.kind!="timer" && e.kind!="alarm" && e.kind!="reminder") return error("invalid_arguments","Unknown item kind");
            bool changedTime=op=="create" || args.contains("start_utc_ms") || args.contains("duration_seconds");
            if(e.kind=="timer") {
                if(args.contains("start_utc_ms") || args.contains("end_utc_ms")) return error("invalid_arguments","Timers use duration_seconds");
                if(op=="create" || args.contains("duration_seconds")) {
                    auto duration=number(args,"duration_seconds");if(duration<1 || duration>31536000) return error("invalid_arguments","Duration must be 1..31536000 seconds");
                    e.start=now+duration*1000;e.end=e.start+1;
                }
            } else {
                if(args.contains("duration_seconds")) return error("invalid_arguments","Duration is only for timers");
                if(op=="create" || args.contains("start_utc_ms")) e.start=number(args,"start_utc_ms");
                if(e.kind=="event") {if(op=="create" || args.contains("end_utc_ms")) e.end=number(args,"end_utc_ms");}
                else {if(args.contains("end_utc_ms")) return error("invalid_arguments","Alarm/reminder uses only start_utc_ms");if(e.start>=253402300800000LL) return error("invalid_arguments","Time out of range");e.end=e.start+1;}
            }
            if(e.start< -62135596800000LL || e.end>253402300800000LL || e.end<=e.start) return error("invalid_arguments","Invalid UTC interval");
            if(e.kind!="event" && changedTime && e.start<=now) return error("invalid_arguments","Delivery time must be in the future");
            if(changedTime) e.delivery=e.kind=="event"?"not_requested":"pending";
            e.revision=s.sequence+1;candidate[id]=e;s.save(candidate,e.revision);s.events.swap(candidate);s.sequence=e.revision;
            return {{"success",true},{"item",record(e)}};
        } catch(const Json::exception& e) {return error("invalid_arguments",e.what());}
          catch(const std::invalid_argument& e) {return error("invalid_arguments",e.what());}
          catch(const std::exception& e) {return error("unavailable",e.what());}
    }
    Json resourceState() {
        auto lease = acquire();
        Json items = Json::array();
        for (const auto& [id, event] : store().events) { if (items.size() == 200) break; items.push_back(record(event)); }
        return {{"items", items}, {"total", store().events.size()}, {"last_poll_utc_ms", lastPoll}, {"delivery_error", deliveryError}};
    }
    std::vector<std::string> takeDue(int64_t now) {
        try {
            auto lease = acquire();
            lastPoll=now;
            auto& s=store();auto candidate=s.events;auto revision=s.sequence;std::vector<std::string> messages;
            for(auto& [id,e]:candidate) if(e.delivery=="pending" && e.start<=now) {
                if(revision>=uint64_t(INT64_MAX)) return {};
                e.delivery="dispatched";e.revision=++revision;messages.push_back(e.kind+": "+e.title);
            }
            if(!messages.empty()){s.save(candidate,revision);s.events.swap(candidate);s.sequence=revision;}
            deliveryError.clear();return messages;
        } catch(const std::exception& e) {std::lock_guard lock(mutex);deliveryError=e.what();return {};}
    }
};
inline int64_t scheduleNow(){return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();}
}

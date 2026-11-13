#ifndef GRIM_SCHEDULING_API_H
#define GRIM_SCHEDULING_API_H

#include <stdint.h>
#include <stddef.h>

/* Standalone provider contract; independent of the host plugin table.
 * All calls are synchronous and serialized by the caller. No exceptions may
 * cross this boundary. Strings are UTF-8, NUL-terminated and borrowed only for
 * the duration of a call. Providers copy retained input. */
#ifdef __cplusplus
extern "C" {
#endif

#define GRIM_SCHEDULE_ABI_V1 1u
typedef int32_t GrimScheduleResult;
enum {
    GRIM_SCHEDULE_OK = 0,
    GRIM_SCHEDULE_INVALID = 1,
    GRIM_SCHEDULE_NOT_FOUND = 2,
    GRIM_SCHEDULE_CONFLICT = 3,
    GRIM_SCHEDULE_UNSUPPORTED = 4,
    GRIM_SCHEDULE_PERMISSION_DENIED = 5,
    GRIM_SCHEDULE_UNAVAILABLE = 6,
    GRIM_SCHEDULE_FAILED = 7
};
enum {
    GRIM_SCHEDULE_CALENDAR = 1u << 0,
    GRIM_SCHEDULE_ALARMS = 1u << 1,
    GRIM_SCHEDULE_PERSISTENT = 1u << 2
};

/* V1 supports single timed events only. UTC milliseconds since Unix epoch;
 * end is exclusive and must exceed start. timezone is an IANA display zone;
 * it never changes the supplied instants. All-day/recurrence require a future
 * negotiated extension, not conversion to fixed 24-hour durations. */
typedef struct GrimScheduleEvent {
    const char* id;
    const char* calendar_id;
    const char* title;
    const char* description;
    const char* timezone;
    int64_t start_utc_ms;
    int64_t end_utc_ms;
    uint64_t revision;
} GrimScheduleEvent;

typedef struct GrimScheduleAlarm {
    const char* id;
    const char* event_id; /* Optional association; empty for standalone alarm. */
    const char* title;
    int64_t fire_utc_ms;
} GrimScheduleAlarm;

/* Callback records are valid only during callback. Must not reenter provider.
 * Nonzero return requests early stop (successful query). */
typedef int32_t (*GrimScheduleEventVisitor)(void*, const GrimScheduleEvent*);

typedef struct GrimScheduleProviderV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t capabilities;
    const char* provider_id; /* Stable unique name, valid until destroy. */
    void* context;

    /* Never prompt from other operations. Explicit user action invokes this.
     * requested capabilities must be a subset of advertised capabilities. */
    GrimScheduleResult (*authorize)(void*, uint32_t requested_capabilities);

    /* Query events overlapping [from, until), ordered by start then id.
     * calendar_id is required. Invalid intervals return INVALID.
     * On error, callers discard any records already delivered. */
    GrimScheduleResult (*query)(void*, const char* calendar_id,
        int64_t from_utc_ms, int64_t until_utc_ms,
        GrimScheduleEventVisitor, void* visitor_context);

    /* Caller supplies stable nonempty id. revision=0 creates; nonzero updates
     * only if current revision matches, otherwise CONFLICT. Success yields a
     * new nonzero revision through out_revision. Failure leaves state intact.
     * IDs are unique within this provider, across calendars. */
    GrimScheduleResult (*put_event)(void*, const GrimScheduleEvent*,
        uint64_t* out_revision);
    GrimScheduleResult (*remove_event)(void*, const char* id,
        uint64_t expected_revision);

    /* Idempotent replacement by alarm id; cancellation of missing id succeeds.
     * Success means accepted by provider, not proof of delivery. */
    GrimScheduleResult (*put_alarm)(void*, const GrimScheduleAlarm*);
    GrimScheduleResult (*cancel_alarm)(void*, const char* id);

    /* Release provider resources on its creating thread after calls drain.
     * Does not delete persisted events or cancel accepted native alarms. */
    void (*destroy)(void*);
} GrimScheduleProviderV1;

/* Future plugin entry point signature. Host supplies zeroed table with version
 * and size set. Factory fills it only on success; no allocation crosses ABI.
 * Unsupported version/undersized table returns UNSUPPORTED. */
typedef GrimScheduleResult (*GrimScheduleCreateProvider)(
    uint32_t requested_version, GrimScheduleProviderV1* out_provider);

#ifdef __cplusplus
}
#endif
#endif

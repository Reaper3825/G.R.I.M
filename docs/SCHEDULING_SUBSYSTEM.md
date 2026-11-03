# Scheduling subsystem interface

Status: interface implemented in `core/scheduling_api.h`; API v3.15 exposes
`create_local_calendar` under SYSTEM permission. The header-only local provider
in `core/local_calendar.hpp` implements volatile sessions and an optional durable store.
The Calendar panel opens a durable store and provides a month grid and daily agenda.
The host frame loop polls due local notifications once per second. Native
integration and delivery while GRIM is closed are not implemented.

The portable contract uses fixed-width values, opaque provider context and
borrowed UTF-8 strings. It can be included from C or C++. The ABI header requires no platform SDK or third-party dependency. The local
implementation uses the existing nlohmann JSON dependency and OS file locking. GrimPluginAPI consumes one reserved slot; its overall size is unchanged.
The ABI assumes matching architecture and default C calling convention.

## Provider lifecycle

The eventual host loader negotiates ABI version and table size, validates all
mandatory members, and retains the library until destroy completes. authorize
and destroy are mandatory. Calendar capability requires query, put_event and
remove_event. Alarm capability requires put_alarm and cancel_alarm. Unsupported
capability functions may be null; hosts must check capabilities before calling.
Unknown capability bits are ignored. Factories must not partially publish tables
on failure. Providers catch internal exceptions and return FAILED.

All operations run serially on the creating thread. Native UI authorization
must be initiated explicitly; an adapter may return UNAVAILABLE while an
asynchronous platform authorization flow is pending. No callbacks survive calls.
Non-optional string fields must be non-null; ids, calendar ids, titles and
timezone must be nonempty. Descriptions and optional associations may be empty.
Adapters reject invalid or unsupported zones rather than silently replacing them.
Removal requires a nonzero matching revision; missing records return NOT_FOUND.
Revision overflow must fail without changing the event.

## Calendar and alarm separation

Calendar providers own events. Alarm providers own delivery requests. A future
coordinator owns the mapping between provider ids, local event ids and alarm ids.
Deleting or moving an event does not implicitly cancel a separately submitted
alarm. The coordinator must reconcile these operations and surface partial
failure; no distributed transaction is promised. Native identifiers stay inside
adapter storage. Local identifiers must remain stable across retries and restarts.

V1 deliberately supports only single timed events and absolute alarm instants.
Recurring and all-day events need explicit local-date, timezone and daylight
saving policies in a future extension. Queries use half-open overlap semantics:
event.start < until && event.end > from.

## Planned adapters

- Local provider: durable event records are implemented. Alarm delivery remains
  future work. Persistence means records survive normal restart; it does not
  promise background execution or power-loss durability.
- Apple calendar provider: EventKit for calendar events and calendar permissions.
- Apple alarm provider: AlarmKit for native alarm scheduling on supported Apple
  targets, with separate authorization and availability checks.
- Windows provider: select the calendar/account integration independently from
  the native notification mechanism; advertise only implemented capabilities.

Apple references: https://developer.apple.com/documentation/eventkit and
https://developer.apple.com/documentation/alarmkit.

## Integration and validation next steps

The local provider and panel are implemented. Tests cover
overlap boundaries, stale revisions, invalid input, deterministic ordering,
alarm replacement/cancellation, restart recovery and provider unload. Then add
host-owned provider discovery and a separately versioned plugin service bridge.
Do not route calendar operations through existing countdown timers or scheduled
agent goals: these have different ownership and delivery semantics.

## Using the local provider

Require host API >= 3.15 before reading create_local_calendar. Initialize a
GrimScheduleProviderV1 with abi_version=GRIM_SCHEDULE_ABI_V1 and
struct_size=sizeof(GrimScheduleProviderV1), then call the factory with that version.
Check the returned status before using the table. Each factory call returns a new
isolated calendar; keep its table for the session. Use caller-chosen calendar ids
and event ids, and timezone `Etc/UTC`. Other zones return UNSUPPORTED until a
validated timezone database is integrated. No authorization prompt is needed for
this memory-only provider. A granted session remains usable until destroyed.

Serialize all calls on the creating thread. Destroy the session before plugin
shutdown; do not retain its context or table afterward. Session destruction loses
all events. This is an explicit low-level plugin facility, not an agent tool.
The schedule_service API shares a separate durable host service with the panel.
The isolated create_local_calendar factory remains unchanged.

Validation: `tests/run_local_calendar.ps1` builds and runs only the focused local
provider tests with the existing MSVC toolchain. It does not build GRIM.

## Calendar panel and saved event records

Open Calendar from the console toolbar or show panel key `calendar`. The Monday-first
month grid includes adjacent-month days, Previous/Next Month, Today, selected-day
highlight and per-day event counts. Select a day to browse its agenda and prefill
09:00–10:00 UTC for a new event. Events spanning midnight appear on each overlapping
day; an exclusive midnight end does not count on the following day. Previous/Next
browse that day's events; Delete event persists removal.

The panel opens a host-owned durable provider at:
- Windows: `%LOCALAPPDATA%/GRIM/calendar/events.json`
- macOS: `$HOME/Library/Application Support/GRIM/calendar/events.json`
- Linux: `$XDG_DATA_HOME/grim/calendar/events.json`, falling back to
  `$HOME/.local/share/grim/calendar/events.json`

The schema contains `schema_version: 2`, a monotonic revision `sequence`, and an
`events` array. Every event saves id, calendar_id, title, description, timezone,
start_utc_ms, end_utc_ms, revision, kind, and delivery state. Version 1 files load
as events with no delivery request; the next successful write upgrades to version 2. IDs remain stable after restart. Local
records support civil years 1–9999 (end may be exclusive midnight at year 10000).
Only Etc/UTC is currently supported. No history/audit log is maintained.

Mutations write a complete candidate snapshot to a sibling `.tmp` file, close it,
and replace the original before publishing the in-memory change. Failed commits
leave memory unchanged. A sibling `.lock` file is held with an OS exclusive lock
for the provider lifetime; a second writer fails to open. Locks release on process
exit. Temporary files are not promoted automatically. Malformed, duplicate-id or
unsupported-schema data causes opening to fail without overwriting the original.
The panel reports an error instead of falling back to an empty volatile calendar.
This protects normal restart behavior, not all power-loss/filesystem failure cases.

The plugin factory still creates isolated memory-only sessions; it does not grant
filesystem access or expose this file. Closing the panel retains its session;
closing GRIM releases the lock and preserves successfully saved events. Native
calendar synchronization and native alarm delivery remain unimplemented. Local
console notifications are supported through the schedule service.

`tests/run_local_calendar.ps1` verifies restart recovery, persisted removal,
revision continuity, exclusive writers, failed-save rollback, corrupt-file
preservation, date validation and month grid boundaries. `tests/check_calendar_ui.ps1`
runs MSVC syntax-only diagnostics on the panel without building or running GRIM.

## Model tools

See [Scheduling tools](SCHEDULING_TOOLS.md). API v3.16 exposes schedule_service;
schedule_plugin registers six Model-classified JSON tools with ToolRegistry.
The shared SchedulingService owns the durable provider and serializes the panel,
model calls and delivery polling with a mutex. This local provider has no native
thread-affine objects; the shared service may operate it from different threads
under that lock. Other adapters retain their own threading requirements.

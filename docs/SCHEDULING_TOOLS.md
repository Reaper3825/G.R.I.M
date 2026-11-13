# Scheduling tool contract

The schedule_plugin supplies six Model tools through normal plugin registration
and ToolRegistry discovery. API v8 is required. All inputs are JSON objects;
unknown fields are rejected. All outputs contain `success`; failures add `error`
and `message`. The full JSON is returned as structured_output_json, not only a
human-readable success message. Mutations declare needs_confirmation; read tools
are informational. These are metadata for the existing caller policy, not an
additional confirmation implementation in the plugin.

| Tool | Required arguments | Optional arguments |
|---|---|---|
| schedule.create | kind, title; event: start_utc_ms and end_utc_ms; timer: duration_seconds; alarm/reminder: start_utc_ms | id, description |
| schedule.get | id | none |
| schedule.list | none | kind, from_utc_ms, until_utc_ms, limit, offset |
| schedule.update | id, revision | title, description, start_utc_ms, end_utc_ms, duration_seconds (as appropriate to kind) |
| schedule.cancel | id, revision | none |
| schedule.status | none | id |

Kinds are event, timer, alarm, reminder. Times are UTC Unix milliseconds; the
caller must resolve natural language dates and timezone ambiguity before calling.
Events require end > start and do not automatically notify. Alarm and reminder
start is the due instant; their stored end is start + 1 ms. Timers require an
integer duration of 1..31536000 seconds and persist the calculated UTC due time.
They follow wall-clock adjustments rather than a monotonic elapsed-time clock.
Changing timer duration restarts it from the call time. Delivery times must be
in the future when created or changed. Changing title/description alone does not
rearm an already dispatched item. Item kind is immutable. All tools use the local
calendar also displayed in the UI; native accounts are not yet available.

Create returns item with stable id and revision. Supplying a caller-generated id
allows retry reconciliation: conflict means retrieve that id and inspect it,
not create another copy blindly. Get/status(id) return item. Update returns its
new revision. Fetch current revision before update/cancel; conflicts never mutate
records. Cancel removes the record and reports whether it was already dispatched.
Cancellation cannot withdraw a notification handed to the host. Missing id yields
not_found, including repeated cancellation. IDs/revisions must come from observed
results rather than being guessed.

List returns items, total and offset, ordered by start then id. Time filters use
half-open overlap [from, until). Defaults: 50 records, offset zero; maximum 200.
Pagination is not a snapshot across concurrent edits. Use narrow ranges for
large calendars. Records include title, description, UTC interval, timezone,
kind, revision, device (`local_host`) and delivery_status.

Status without id reports supported kinds, storage persistence, local console
delivery, last poll time (zero before first poll), last delivery error, no native
calendar sync, no background delivery, and the overdue policy.

## Delivery and persistence semantics

The plugin serializes tool calls and notification polls. It opens and locks the
store per operation, rereading disk under that lock. The host frame loop pumps a
generic one-second callback; the plugin selects due records, saves their dispatch
state, and sends copied notification text to the host UI sink.
`dispatched` means handed to this path, not user acknowledgment or OS delivery.
A crash between saving dispatch state and displaying the message can lose that
notification; it is not retried. Failed saves leave items pending for a later poll.

Overdue pending items dispatch on the next poll after restart. Closed/suspended
GRIM cannot deliver on time. AlarmKit, native calendar services, snooze, recurring
schedules, device routing and acknowledgments are not implemented. Event records
and delivery states survive normal restart in the versioned calendar file. There
is no separate audit log or cancellation tombstone.

Plugin unload removes callbacks and resources but does not cancel saved records.
The private calendar implementation and storage live in `plugins/schedule_plugin`;
there is no calendar factory or scheduling service in the host API. Calendar UI
operations resolve `schedule.*` through ToolRegistry and report unavailability
when the provider is absent. The provider publishes `items` and `delivery`
resources of kind `schedule.state`; items is a summary capped at 200 records.

## Validation

`tests/run_schedule_plugin.ps1` exercises all six handlers, structured model
results, supported kinds, revisions, due boundaries, cancellation and persistence.
`tests/run_local_calendar.ps1` checks the private store implementation.
`tests/run_plugin_shutdown.ps1 -BuildSchedulingFixtures` builds isolated DLLs and
tests real registration, resource updates, callback delivery, failed replacement,
successful reload, countdown cancellation, durable schedule survival and UI client
unavailability after removal. UI implementation receives a syntax check.
These tests do not launch GRIM or touch the user's calendar store.

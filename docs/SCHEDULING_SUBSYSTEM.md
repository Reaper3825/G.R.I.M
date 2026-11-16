# Scheduling provider ownership

The host owns registration, generation call gates, system tick dispatch and
copied UI notifications. Schedule owns calendar storage and due-item logic.

`schedule_plugin` owns six `schedule.*` tools, persistent UTC records and a
one-second delivery poll driven by host system ticks. It requires SYSTEM, FILESYSTEM and UI permissions.
API v14 makes Calendar a native `UIPanel` subclass owned by Schedule. It directly
owns buttons, inputs, scrolling, layout and actions and uses the provider service.
Constructor registration is staged during provider initialization. After commit,
the host attaches that same panel and publishes `panel:calendar` under Schedule.
Unload detaches it synchronously before Schedule destroys it and the DLL unloads.
UI-plugin unload leaves Calendar intact. Visible Calendar refresh runs every
500 ms in its UI-thread update; storage operations are synchronous.

The implementation headers `scheduling_service.hpp`, `local_calendar.hpp`,
`calendar_file.hpp` and `scheduling_api.h` live inside the scheduling plugin. The
C table is private implementation detail, not a host API or plugin loader contract.
SchedulingService opens the store and holds its OS lock only for an operation,
then destroys that store instance. Each operation rereads disk under the lock,
so replacement generations cannot overwrite newer records from a stale cache.
The store path is resolved once when the provider is created. Staged initialization
publishes unready resources without opening or mutating durable storage.

The existing versioned calendar format, stable IDs, optimistic revisions, atomic
save behavior and storage location are retained. Resources `items` and `delivery`
of kind `schedule.state` describe records and delivery state. The item summary is
capped at 200; paginated tools expose the full set. A resource publication failure
after a durable mutation is reported separately without falsely failing the saved
operation and encouraging duplicate retries.

Unload withdraws tools, resources and callbacks. Durable records survive, and
pending overdue records dispatch on the next poll after reload. Dispatch is saved
before notification output; it is not an acknowledgment and a crash in between
can lose output. There is no delivery while GRIM is closed, native calendar sync,
recurrence, snooze, remote routing or background scheduler.

See [tool and persistence contract](SCHEDULING_TOOLS.md) for inputs, outputs,
time semantics, limitations and focused validation commands. API v8 removes the
old host-specific slots completely; rebuild host and plugins together.

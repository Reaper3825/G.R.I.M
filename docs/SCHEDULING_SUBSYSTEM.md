# Scheduling provider ownership

API v8 loads timers and scheduling as ordinary plugin providers. The host owns
registration, generation call gates, generic callback pumping and copied UI
notification output. It owns no countdown queue, calendar store, schedule
operation dispatcher or due-item logic.

`timers_plugin` owns volatile steady-clock countdowns through `set_timer` and
publishes `timers.queue`. Its 100 ms callback checks expiry. Unload/replacement
cancels pending countdowns. It requires SYSTEM and UI permissions.

`schedule_plugin` owns six `schedule.*` tools, persistent UTC records and a
one-second delivery callback. It requires SYSTEM, FILESYSTEM and UI permissions.
The Calendar panel invokes these tools through ToolRegistry; no provider pointer
or store instance is held by the panel. Absent providers produce an unavailable
result. Generic host callbacks are pumped from the application frame loop.

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

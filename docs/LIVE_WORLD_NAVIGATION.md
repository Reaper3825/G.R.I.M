# Live-world navigation read model

`LiveWorld::navigation()` returns a typed `LiveNavigationSnapshot` representing
the reconciled live topology. The session uses it for hub descriptions and
navigation command discovery. `navigationJson()` in
`ui/world_lab/live_world_navigation.hpp` provides its versioned JSON representation.
This read model is independent of TextWorld, authored world files, and capture.

World Lab supplies `gamesDirectory/live_navigation.json` to the live view's
session. With the default games directory this is
`resources/models/games/live_navigation.json`. The first opening of Live creates
the file. While that live view is updating, provider registration, replacement,
and unload are reconciled automatically and publish a fresh snapshot. No export
button is required. This is a view-owned session, not a background registry service:
when the view is inactive the file is the last observed snapshot and refreshes
when the view resumes. No file is generated merely by running an offline script.

The version-1 envelope contains:

- `format`: `grim-live-navigation-snapshot`.
- `version`: `1`.
- `session_id`: identity of this live-world instance, not a persistent environment ID.
- `hub_id`: the permanent hub's room ID.
- `rooms`: all connected navigation rooms, including the hub. Each contains
  `room_id`, `parent_id` (empty for the hub), `display_name`, and resolved `path`.
  Provider rooms also contain `provider` with `type` (`builtin` or `plugin`),
  exact registration `name`, and current generation `instance`.

Consumers should identify provider destinations by `(provider.type, provider.name)`.
Room IDs and paths are resolved within the current session; do not reconstruct
paths from provider names. Replacements retain the room but update generation
metadata. Unloads remove the room from subsequent snapshots. The snapshot makes
no promise that its providers are still connected when an offline consumer reads it.

`LiveNavigationStore` owns persistence and can be omitted by other session users.
It compares content before writing, so movement, tool listings, and resource state
refreshes do not rewrite navigation metadata. Missing files are recreated. It uses
the existing atomic JSON writer, preserving the previous complete file on failure.
Errors are exposed to callers and displayed in the live view; retries are throttled
to five seconds and use the latest snapshot. It exports no resource readings,
tool contracts, handlers, or action history.

Validation: `tests/run_live_world.ps1` exercises automatic publication, generation
replacement, room removal, unchanged-file suppression, deleted-file recovery,
and isolated persistence failure alongside the existing host lifecycle tests.

# Live-world navigation export

World Lab writes `gamesDirectory/live_navigation.json` when the Live session
opens for the first time. The default location is
`resources/models/games/live_navigation.json`. The file describes the room map
built from the committed startup providers.

The version-1 envelope contains the following fields.

- `format` is `grim-live-navigation-snapshot`.
- `version` is `1`.
- `session_id` identifies the Live session.
- `hub_id` identifies System Hub.
- `rooms` contains each room's `room_id`, `parent_id`, `display_name`, and
  `path`. The hub has an empty `parent_id`. Provider rooms include the
  provider's `type`, exact registration `name`, and startup `instance`.

Offline consumers use `(provider.type, provider.name)` to identify a destination
and use the exported paths to navigate. Room IDs and paths are resolved for the
session. The file records startup membership. A provider can retire while its
room remains in that session.

The session writes the file atomically once and shows any write error in the
Live view. To regenerate the file, open a new Live session. Movement, tools,
and resources are managed in memory.

The [navigation course generator](../scripts/textworld/NAVIGATION_GOALS.md)
consumes this export.

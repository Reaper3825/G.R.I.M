# Live System World

World Lab's **Live** button opens a host-owned session. System Hub at `/` has a
child room for every provider committed by the startup plugin batch, including
providers with no tools. The session builds the room map once. **World editor**
returns to the authored world. Reopening Live resumes the same session.

## Startup and updates

When Live first opens, the host supplies the startup providers and their latest
tool and resource lists. It attaches update delivery under the same registry
lock used to copy the initial contents. The lists contain descriptors only.
The Live session owns its copies and has no registry dependency.

The host coalesces committed changes by provider instance and delivers the
latest changed collections on the UI thread. Tool metadata, usage statistics,
and withdrawals flow through the same path as resource publication,
replacement, and removal. Delivery continues while Live is hidden. The session
uses the provider instance as its update ID. Retirement clears that instance's
tools and resources. Its room and the player's location stay in place. A provider
loaded after startup has no room in the current session.

## Navigation and inspection

The room paths use readable provider names, with a trailing `Plugin` label
omitted and a numeric suffix added for path collisions. Use `go /UI` to enter a
room and `back` to return to the hub before entering another room. Clicking a
room previews its resources. **Available actions** lists navigation and
inspection commands for the current location.

Use `tools` to list model tools in the room, `open tool_catalog` to browse tools
across providers, and `inspect <tool_id>` to read a tool contract. Live session
commands inspect tools without executing them.

Use `inspect <object name or resource_id>` to read published resource state in
the current room. Use `inspect-resource <resource_id>` for ambiguous names or
tool-ID collisions. Resource tiles and inspection commands share a formatter.
[System Information resources](SYSTEM_INFO_RESOURCES.md) shows the hardware
readings and capture-status format.

State-only resource changes refresh the observation without adding transcript
entries. Additions, removals, and identity changes in the current room add
short entries. An inspection becomes stale when the inspected object changes
or is removed. The session retains the recorded inspection in its history.

The first opening exports the fixed room list for offline navigation authoring.
See [Live-world navigation](LIVE_WORLD_NAVIGATION.md) for its format.

## Validation

- `tests/run_live_world.ps1` exercises startup rooms, provider-scoped updates,
  navigation export, and resource inspection.
- `tests/run_ui_input_foundation.ps1 -WorldLab` exercises the map, navigation
  controls, resource grid, and delivery while the view is hidden.
- `tests/run_plugin_shutdown.ps1 -BuildSystemInfoFixture` exercises startup
  membership, tool metadata and withdrawal, and plugin resource delivery.

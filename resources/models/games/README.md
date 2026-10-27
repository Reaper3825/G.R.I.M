# World Lab games

Save editable games here as `*.grimworld.json`. World Lab detects these files
in its Saved games dropdown; selecting one opens it for editing and playtesting.
The list refreshes automatically while the tab is open. Subfolders are not scanned.
Compiled TextWorld playtest files remain temporary.

`grim_internal.grimworld.json` is the **GRIM** environment. It starts in the
System Hub at `/`, with `/perception`, `/tools`, `/computer`, and `/memory` as
direct child rooms. The five rooms contain 20 fixed inspection stations copied
from the GRIM room assets. Use `enter perception` (or `go /perception`),
`examine digital capture`, and `back` to explore. It is a sandbox environment
with no attached Goals, live subsystem bindings, or recorded solutions.

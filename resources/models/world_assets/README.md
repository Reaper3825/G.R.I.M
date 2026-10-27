# World assets

The Training Control Assets tab stores independent room and object templates as
`*.grimasset.json` in `rooms/` and `objects/`. Rooms include their contents; objects
include nested contents. Inserting an asset makes an independent copy with fresh
IDs and unique in-game names. Original passages and goals are not copied.

Use Save as asset in World Lab to capture a selection, then Save in Assets.
Use Add to world to insert a room, or select a destination room/container/supporter
in World Lab before inserting an object. Save the destination game afterward.

Delete asset moves the file to `.trash/`; restore it to its original subfolder to
recover it. Editing the library name renames the browser entry; Save copy creates
a separate file. Existing games never change when a library asset is edited.

## GRIM subsystem rooms

The room library includes **GRIM - System Hub**, **GRIM - Perception**,
**GRIM - Tools**, **GRIM - Computer**, and **GRIM - Memory**. Each has a room
description and fixed, inspectable reference stations grounded in the subsystem
interfaces. Stations use the existing `supporter` kind so they cannot be taken
as inventory. Use the current TextWorld `examine <name>` command to inspect them.
Their content is authored reference material; it does not expose live readings,
execute tools, control the desktop, or read/write actual memory.

Insert the rooms into World Lab, then select the System Hub and enable path
navigation to make it `/`. Assign the subsystem rooms `/perception`, `/tools`,
`/computer`, and `/memory` through their Room path fields. Assets do not persist
world-specific addresses, connections, Goals, or player state. Set the player
start and save the assembled world separately.

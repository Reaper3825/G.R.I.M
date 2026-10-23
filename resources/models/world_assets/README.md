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

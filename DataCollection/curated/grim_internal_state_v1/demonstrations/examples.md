# GRIM internal goal-following demonstrations

Nine successful attempts. Catalog feedback uses the explicit empty-model-tool fixture.
Input and environment observations are context; assistant spans are supervision targets.
This is an authoring preview, not configured tokenizer output.

## grim_internal_state_v1_computer_closed

**Prompt:** Go to the computer room. Do not open the tool catalog.

**Target state:** You are in the computer room with the tool catalog closed.

**Success criteria:**

- You are in the computer room.

- The tool catalog is closed.


**Constraints:**

- Do not open the tool catalog.


**Initial environment — context**

```text
Description:
You are in GRIM's system hub, an orientation space for its internal subsystems. The perception, tools, computer, and memory rooms are connected beneath this hub.

Inventory:
You are carrying nothing.

Location path:
/
```


**Determine — supervised**

```text
Move to the computer room without opening the tool catalog.
```


**Execute — supervised**

```text
enter computer
```


**Environment — context**

```text
Description:
You are in the computer subsystem, a representation of the digital workspace and the controls used to interact with it. Perception supplies observations of that workspace, while computer controls can change it.

Inventory:
You are carrying nothing.

Location path:
/computer
```


**Answer — supervised**

```text
I am in the computer room and did not open the tool catalog.
```

## grim_internal_state_v1_memory_closed

**Prompt:** Go to the memory room. Do not open the tool catalog.

**Target state:** You are in the memory room with the tool catalog closed.

**Success criteria:**

- You are in the memory room.

- The tool catalog is closed.


**Constraints:**

- Do not open the tool catalog.


**Initial environment — context**

```text
Description:
You are in GRIM's system hub, an orientation space for its internal subsystems. The perception, tools, computer, and memory rooms are connected beneath this hub.

Inventory:
You are carrying nothing.

Location path:
/
```


**Determine — supervised**

```text
Move to the memory room without opening the tool catalog.
```


**Execute — supervised**

```text
enter memory
```


**Environment — context**

```text
Description:
You are in the memory subsystem, where information is retained and retrieved for later use. Observation, retrieval, and durable recording are separate operations.

Inventory:
You are carrying nothing.

Location path:
/memory
```


**Answer — supervised**

```text
I am in the memory room and did not open the tool catalog.
```

## grim_internal_state_v1_perception_closed

**Prompt:** Go to the perception room. Do not open the tool catalog.

**Target state:** You are in the perception room with the tool catalog closed.

**Success criteria:**

- You are in the perception room.

- The tool catalog is closed.


**Constraints:**

- Do not open the tool catalog.


**Initial environment — context**

```text
Description:
You are in GRIM's system hub, an orientation space for its internal subsystems. The perception, tools, computer, and memory rooms are connected beneath this hub.

Inventory:
You are carrying nothing.

Location path:
/
```


**Determine — supervised**

```text
Move to the perception room without opening the tool catalog.
```


**Execute — supervised**

```text
enter perception
```


**Environment — context**

```text
Description:
You are in the perception subsystem, where sensory inputs become observations that reasoning can use. Computer control and durable memory storage belong to other subsystems.

Inventory:
You are carrying nothing.

Location path:
/perception
```


**Answer — supervised**

```text
I am in the perception room and did not open the tool catalog.
```

## grim_internal_state_v1_tools_closed

**Prompt:** Go to the tools room. Do not open the tool catalog.

**Target state:** You are in the tools room with the tool catalog closed.

**Success criteria:**

- You are in the tools room.

- The tool catalog is closed.


**Constraints:**

- Do not open the tool catalog.


**Initial environment — context**

```text
Description:
You are in GRIM's system hub, an orientation space for its internal subsystems. The perception, tools, computer, and memory rooms are connected beneath this hub.

Inventory:
You are carrying nothing.

Location path:
/
```


**Determine — supervised**

```text
Move to the tools room without opening the tool catalog.
```


**Execute — supervised**

```text
enter tools
```


**Environment — context**

```text
Description:
You are in the tools subsystem, where available commands are described and requests are prepared for execution. A tool catalog stands here.

Inventory:
You are carrying nothing.

Location path:
/tools
```


**Answer — supervised**

```text
I am in the tools room and did not open the tool catalog.
```

## grim_internal_state_v1_tools_open

**Prompt:** Go to the tools room and open the tool catalog. Stay in the tools room.

**Target state:** You are in the tools room with the tool catalog open.

**Success criteria:**

- You are in the tools room.

- The tool catalog is open.


**Initial environment — context**

```text
Description:
You are in GRIM's system hub, an orientation space for its internal subsystems. The perception, tools, computer, and memory rooms are connected beneath this hub.

Inventory:
You are carrying nothing.

Location path:
/
```


**Determine — supervised**

```text
Open the tool catalog in the tools room and remain there.
```


**Execute — supervised**

```text
enter tools
```


**Environment — context**

```text
Description:
You are in the tools subsystem, where available commands are described and requests are prepared for execution. A tool catalog stands here.

Inventory:
You are carrying nothing.

Location path:
/tools
```


**Execute — supervised**

```text
open tool catalog
```


**Environment — context**

```text
Feedback:
Tool catalog:
No model tools are currently available.

Description:
You are in the tools subsystem, where available commands are described and requests are prepared for execution. A tool catalog stands here.

Inventory:
You are carrying nothing.

Location path:
/tools
```


**Answer — supervised**

```text
The tool catalog is open, and I am in the tools room.
```

## grim_internal_state_v1_hub_open

**Prompt:** Open the tool catalog in the tools room, then return to the System Hub. Leave the catalog open.

**Target state:** You are in the System Hub with the tool catalog open.

**Success criteria:**

- You are in the System Hub.

- The tool catalog is open.


**Initial environment — context**

```text
Description:
You are in GRIM's system hub, an orientation space for its internal subsystems. The perception, tools, computer, and memory rooms are connected beneath this hub.

Inventory:
You are carrying nothing.

Location path:
/
```


**Determine — supervised**

```text
Open the tool catalog in the tools room, then return to the System Hub.
```


**Execute — supervised**

```text
enter tools
```


**Environment — context**

```text
Description:
You are in the tools subsystem, where available commands are described and requests are prepared for execution. A tool catalog stands here.

Inventory:
You are carrying nothing.

Location path:
/tools
```


**Execute — supervised**

```text
open tool catalog
```


**Environment — context**

```text
Feedback:
Tool catalog:
No model tools are currently available.

Description:
You are in the tools subsystem, where available commands are described and requests are prepared for execution. A tool catalog stands here.

Inventory:
You are carrying nothing.

Location path:
/tools
```


**Execute — supervised**

```text
back
```


**Environment — context**

```text
Description:
You are in GRIM's system hub, an orientation space for its internal subsystems. The perception, tools, computer, and memory rooms are connected beneath this hub.

Inventory:
You are carrying nothing.

Location path:
/
```


**Answer — supervised**

```text
The tool catalog is open, and I am back in the System Hub.
```

## grim_internal_state_v1_computer_open

**Prompt:** Open the tool catalog in the tools room, then go to the computer room. Leave the catalog open.

**Target state:** You are in the computer room with the tool catalog open.

**Success criteria:**

- You are in the computer room.

- The tool catalog is open.


**Initial environment — context**

```text
Description:
You are in GRIM's system hub, an orientation space for its internal subsystems. The perception, tools, computer, and memory rooms are connected beneath this hub.

Inventory:
You are carrying nothing.

Location path:
/
```


**Determine — supervised**

```text
Open the tool catalog in the tools room, then move to the computer room.
```


**Execute — supervised**

```text
enter tools
```


**Environment — context**

```text
Description:
You are in the tools subsystem, where available commands are described and requests are prepared for execution. A tool catalog stands here.

Inventory:
You are carrying nothing.

Location path:
/tools
```


**Execute — supervised**

```text
open tool catalog
```


**Environment — context**

```text
Feedback:
Tool catalog:
No model tools are currently available.

Description:
You are in the tools subsystem, where available commands are described and requests are prepared for execution. A tool catalog stands here.

Inventory:
You are carrying nothing.

Location path:
/tools
```


**Execute — supervised**

```text
back
```


**Environment — context**

```text
Description:
You are in GRIM's system hub, an orientation space for its internal subsystems. The perception, tools, computer, and memory rooms are connected beneath this hub.

Inventory:
You are carrying nothing.

Location path:
/
```


**Execute — supervised**

```text
enter computer
```


**Environment — context**

```text
Description:
You are in the computer subsystem, a representation of the digital workspace and the controls used to interact with it. Perception supplies observations of that workspace, while computer controls can change it.

Inventory:
You are carrying nothing.

Location path:
/computer
```


**Answer — supervised**

```text
The tool catalog is open, and I am in the computer room.
```

## grim_internal_state_v1_memory_open

**Prompt:** Open the tool catalog in the tools room, then go to the memory room. Leave the catalog open.

**Target state:** You are in the memory room with the tool catalog open.

**Success criteria:**

- You are in the memory room.

- The tool catalog is open.


**Initial environment — context**

```text
Description:
You are in GRIM's system hub, an orientation space for its internal subsystems. The perception, tools, computer, and memory rooms are connected beneath this hub.

Inventory:
You are carrying nothing.

Location path:
/
```


**Determine — supervised**

```text
Open the tool catalog in the tools room, then move to the memory room.
```


**Execute — supervised**

```text
enter tools
```


**Environment — context**

```text
Description:
You are in the tools subsystem, where available commands are described and requests are prepared for execution. A tool catalog stands here.

Inventory:
You are carrying nothing.

Location path:
/tools
```


**Execute — supervised**

```text
open tool catalog
```


**Environment — context**

```text
Feedback:
Tool catalog:
No model tools are currently available.

Description:
You are in the tools subsystem, where available commands are described and requests are prepared for execution. A tool catalog stands here.

Inventory:
You are carrying nothing.

Location path:
/tools
```


**Execute — supervised**

```text
back
```


**Environment — context**

```text
Description:
You are in GRIM's system hub, an orientation space for its internal subsystems. The perception, tools, computer, and memory rooms are connected beneath this hub.

Inventory:
You are carrying nothing.

Location path:
/
```


**Execute — supervised**

```text
enter memory
```


**Environment — context**

```text
Description:
You are in the memory subsystem, where information is retained and retrieved for later use. Observation, retrieval, and durable recording are separate operations.

Inventory:
You are carrying nothing.

Location path:
/memory
```


**Answer — supervised**

```text
The tool catalog is open, and I am in the memory room.
```

## grim_internal_state_v1_perception_open

**Prompt:** Open the tool catalog in the tools room, then go to the perception room. Leave the catalog open.

**Target state:** You are in the perception room with the tool catalog open.

**Success criteria:**

- You are in the perception room.

- The tool catalog is open.


**Initial environment — context**

```text
Description:
You are in GRIM's system hub, an orientation space for its internal subsystems. The perception, tools, computer, and memory rooms are connected beneath this hub.

Inventory:
You are carrying nothing.

Location path:
/
```


**Determine — supervised**

```text
Open the tool catalog in the tools room, then move to the perception room.
```


**Execute — supervised**

```text
enter tools
```


**Environment — context**

```text
Description:
You are in the tools subsystem, where available commands are described and requests are prepared for execution. A tool catalog stands here.

Inventory:
You are carrying nothing.

Location path:
/tools
```


**Execute — supervised**

```text
open tool catalog
```


**Environment — context**

```text
Feedback:
Tool catalog:
No model tools are currently available.

Description:
You are in the tools subsystem, where available commands are described and requests are prepared for execution. A tool catalog stands here.

Inventory:
You are carrying nothing.

Location path:
/tools
```


**Execute — supervised**

```text
back
```


**Environment — context**

```text
Description:
You are in GRIM's system hub, an orientation space for its internal subsystems. The perception, tools, computer, and memory rooms are connected beneath this hub.

Inventory:
You are carrying nothing.

Location path:
/
```


**Execute — supervised**

```text
enter perception
```


**Environment — context**

```text
Description:
You are in the perception subsystem, where sensory inputs become observations that reasoning can use. Computer control and durable memory storage belong to other subsystems.

Inventory:
You are carrying nothing.

Location path:
/perception
```


**Answer — supervised**

```text
The tool catalog is open, and I am in the perception room.
```

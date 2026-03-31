# Device Communication System — Implementation Plan

> **Scope:** WebSocket-based device registry with UI panel. Phase 1 of the broader "GRIM talks to any user device" system.

---

## 1. Architecture Overview

Phase 1 is a **hub-authoritative** device directory running inside `GRIM.exe`.

- `DeviceRegistry` owns **persistent device metadata and hub policy**.
- `SessionManager` owns **runtime session truth**.
- `DeviceCommServer` owns **WebSocket I/O, protocol parsing, and orchestration**.
- `UIControlPanel` reads a **joined snapshot** from the server. It does **not** read the raw registry directly.

Use the repo’s existing **`uWebSockets`** dependency for the C++ server. Do **not** add a second C++ WebSocket library for this subsystem.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  GRIM.exe                                                                │
│                                                                          │
│  ┌────────────────────┐        ┌──────────────────────────────────────┐  │
│  │  UIControlPanel    │        │  DeviceCommServer                   │  │
│  │  title: "Control" │◄──────►│  ws://localhost:11437               │  │
│  │                    │        │                                      │  │
│  │  • device list     │        │  ┌──────────────────────────────┐    │  │
│  │  • pair / unpair   │        │  │ DeviceRegistry               │    │  │
│  │  • permissions     │        │  │ persistent metadata + policy │    │  │
│  │  • availability    │        │  └──────────────────────────────┘    │  │
│  │  • status          │        │                                      │  │
│  │                    │        │  ┌──────────────────────────────┐    │  │
│  │  reads snapshots   │        │  │ SessionManager               │    │  │
│  │  not raw registry  │        │  │ runtime sessions + heartbeat │    │  │
│  │                    │        │  └──────────────────────────────┘    │  │
│  └────────────────────┘        └──────────────────┬───────────────────┘  │
│                                                   │ WebSocket             │
└───────────────────────────────────────────────────┼───────────────────────┘
                                    │
           ┌────────────────────────────────────┼─────────────────────────────────┐
           │                                    │                                 │
    ┌────────▼────────┐                  ┌────────▼────────┐               ┌────────▼────────┐
    │ Laptop Agent    │                  │ Phone Agent     │               │ Desktop Agent   │
    │ grim_device_    │                  │ (future client) │               │ (future client) │
    │ agent.py        │                  │                 │               │                 │
    └─────────────────┘                  └─────────────────┘               └─────────────────┘
```

---

## 2. Directory Structure

All new source still lives under `control/`, but the device subsystem is kept separate from training control.

```
control/
├── CMakeLists.txt                         # Existing training-control project (unchanged unless shared headers are added)
├── training_control_server.cpp            # Existing
├── training_controller.hpp                # Existing
├── ...                                    # Existing
│
└── devices/                               # NEW — device communication subsystem
    ├── README.md                          # Subsystem docs
    │
    ├── protocol/                          # JSON wire message definitions only
    │   ├── device_protocol.hpp            # Message names, error codes, protocol constants
    │   └── device_messages.hpp            # Wire payload structs (request/response only)
    │
    ├── registry/                          # Persistent hub-owned device directory
    │   ├── device_record.hpp              # Persistent device metadata + hub policy
    │   ├── device_snapshot.hpp            # Joined read model for UI/routing
    │   ├── device_registry.hpp            # CRUD + persistence
    │   └── device_registry.cpp            # JSON load/save
    │
    ├── session/                           # Runtime-only live state
    │   ├── device_session.hpp             # Session/runtime state only
    │   ├── session_manager.hpp            # Session lifecycle + heartbeat timeout
    │   └── session_manager.cpp            # Implementation
    │
    ├── server/                            # WebSocket server + orchestration
    │   ├── device_comm_server.hpp         # DeviceCommServer class
    │   └── device_comm_server.cpp         # uWebSockets handlers, JSON dispatch
    │
    └── client/                            # Reference client for user devices
        └── grim_device_agent.py           # Lightweight Python agent
```

**Phase 1 protocol choice:** JSON over WebSocket text frames only.  
**Not in scope for Phase 1:** FlatBuffers schema files for device comm.

FlatBuffers remain in the existing training-control subsystem only.

---

## 3. Data Model

### 3.1 Persistent registry entry — `DeviceRecord`

**File:** `control/devices/registry/device_record.hpp`

```cpp
enum class PairingState {
        Pending = 0,   // Seen by hub, not trusted for routing
        Paired  = 1,   // Approved by user, eligible for routing if online
        Blocked = 2    // Explicitly denied; connection is rejected
};

struct DeviceRecord {
        std::string device_id;                 // Primary key: "dev_laptop_01"
        std::string device_name;               // Display name: "Austin Laptop"
        std::string device_type;               // laptop | desktop | phone | tablet | server | iot
        std::string platform;                  // windows | macos | linux | ios | android
        std::string platform_version;          // "11", "macOS 15", "Ubuntu 24.04"
        std::string architecture;              // x86_64 | arm64 | armv7 | ...
        std::string agent_version;             // Device agent version
        int         protocol_version = 1;      // Supported hub protocol version

        PairingState pairing_state = PairingState::Pending;
        std::vector<std::string> roles;                // Broad routing labels: ["laptop", "primary_ui"]
        std::vector<std::string> allowed_actions;      // Hub-granted actions: ["display_text", "show_notification"]
        std::vector<std::string> declared_capabilities; // Device-reported support

        int64_t registered_at = 0;             // Persisted
        int64_t last_seen_at = 0;              // Persisted; updated from runtime heartbeats
};
```

**Persisted fields:** everything in `DeviceRecord`.  
**Not allowed inside `DeviceRecord`:** `online`, `session_id`, socket handle, heartbeat timer state, current availability.

**Ownership rules:**

- **Device-reported and updateable on re-registration:**
    - `device_name`
    - `device_type`
    - `platform`
    - `platform_version`
    - `architecture`
    - `agent_version`
    - `protocol_version`
    - `declared_capabilities`
- **Hub-owned and never writable by the device agent:**
    - `pairing_state`
    - `roles`
    - `allowed_actions`
    - `registered_at`
    - `last_seen_at`

### 3.2 Runtime session state — `DeviceSession`

**File:** `control/devices/session/device_session.hpp`

```cpp
struct DeviceSession {
        std::string session_id;                // Runtime-only: "sess_123"
        std::string device_id;                 // Which record this session belongs to
        std::string remote_address;            // Peer address for diagnostics
        int64_t     connected_at = 0;          // Unix timestamp
        int64_t     last_heartbeat_at = 0;     // Unix timestamp
        std::vector<std::string> available_capabilities; // What is enabled/available right now
};
```

**Runtime-only fields:** everything in `DeviceSession`.  
**Persisted:** nothing.  
`SessionManager` owns all `DeviceSession` instances.

### 3.3 Joined view for UI and routing — `DeviceSnapshot`

**File:** `control/devices/registry/device_snapshot.hpp`

```cpp
struct DeviceSnapshot {
        DeviceRecord record;                   // Persisted metadata + policy
        bool online = false;                   // Derived from SessionManager
        std::string session_id;                // Runtime-only view field
        int64_t connected_at = 0;              // Runtime-only view field
        int64_t last_heartbeat_at = 0;         // Runtime-only view field
        std::vector<std::string> available_capabilities; // Runtime-only view field
        std::vector<std::string> routeable_actions;      // Derived intersection
};
```

`UIControlPanel` displays `DeviceSnapshot`, not `DeviceRecord`.

### 3.4 Capability vs availability vs permission

The plan must keep these separate:

- `declared_capabilities` = what the device says it supports in general
- `available_capabilities` = what is actually available **right now** in the live session
- `allowed_actions` = what the hub is willing to send to this device

Routing uses this exact rule:

- If `pairing_state == Paired` and the device is online:
    - `routeable_actions = declared_capabilities ∩ available_capabilities ∩ allowed_actions`
- Otherwise:
    - `routeable_actions = []`

### 3.5 Example snapshot shown in the `control` UI

This preserves the spirit of the original JSON row, but the live fields are now coming from runtime state instead of polluting the registry record.

```json
{
    "device_id": "dev_laptop_01",
    "device_name": "Austin Laptop",
    "device_type": "laptop",
    "platform": "windows",
    "platform_version": "11",
    "architecture": "x86_64",
    "pairing_state": "paired",
    "online": true,
    "session_id": "sess_123",
    "last_seen_at": 1774931250,
    "declared_capabilities": [
        "display_text",
        "show_notification",
        "receive_file"
    ],
    "available_capabilities": [
        "display_text",
        "show_notification"
    ],
    "allowed_actions": [
        "display_text",
        "show_notification"
    ],
    "roles": [
        "laptop",
        "primary_ui"
    ],
    "agent_version": "0.1.0",
    "protocol_version": 1
}
```

---

## 4. DeviceRegistry API

**File:** `control/devices/registry/device_registry.hpp`

`DeviceRegistry` stores persistent records only. It does **not** know whether a socket is open.

```cpp
namespace GRIM::Devices {

class DeviceRegistry {
public:
        explicit DeviceRegistry(const std::string& persist_path);

        // Create or refresh the device-reported profile.
        // This updates only client-owned fields and preserves hub policy.
        bool upsertDeviceProfile(
                const std::string& device_id,
                const std::string& device_name,
                const std::string& device_type,
                const std::string& platform,
                const std::string& platform_version,
                const std::string& architecture,
                const std::string& agent_version,
                int protocol_version,
                const std::vector<std::string>& declared_capabilities);

        bool removeDevice(const std::string& device_id);            // Hub-only action
        std::optional<DeviceRecord> getDevice(const std::string& device_id) const;
        std::vector<DeviceRecord> getAllDevices() const;

        bool setPairingState(const std::string& device_id, PairingState state);
        bool setRoles(const std::string& device_id, const std::vector<std::string>& roles);
        bool setAllowedActions(const std::string& device_id, const std::vector<std::string>& actions);
        void touchLastSeen(const std::string& device_id, int64_t timestamp);

        void save();
        void load();

private:
        std::string persist_path_;
        std::unordered_map<std::string, DeviceRecord> devices_;
        mutable std::shared_mutex mutex_;
};

} // namespace GRIM::Devices
```

### Registry rules

- `device_id` is the primary key.
- First registration creates a record with:
    - `pairing_state = Pending`
    - `roles = [device_type]`
    - `allowed_actions = []`
- Re-registration updates the device-reported profile only.
- Re-registration never overwrites `pairing_state`, `roles`, or `allowed_actions`.
- `removeDevice()` is a hub/UI action only. Devices do not delete their own registry entry over the wire.
- Save is atomic: write temp file, then rename.

---

## 5. SessionManager

**File:** `control/devices/session/session_manager.hpp`

`SessionManager` owns runtime truth. The registry never becomes a socket table.

```cpp
namespace GRIM::Devices {

class SessionManager {
public:
        explicit SessionManager(int timeout_seconds = 30);

        std::string openSession(
                const std::string& device_id,
                const std::string& remote_address,
                const std::vector<std::string>& available_capabilities,
                int64_t now);

        void refreshHeartbeat(
                const std::string& session_id,
                const std::vector<std::string>& available_capabilities,
                int64_t now);

        void closeSession(const std::string& session_id);
        std::vector<std::string> sweepExpiredSessions(int64_t now); // Returns expired device_ids

        std::optional<DeviceSession> getBySessionId(const std::string& session_id) const;
        std::optional<DeviceSession> getByDeviceId(const std::string& device_id) const;
        std::vector<DeviceSession> getAllSessions() const;

private:
        int timeout_seconds_;
        std::unordered_map<std::string, DeviceSession> sessions_by_id_;
        std::unordered_map<std::string, std::string> device_to_session_id_;
        mutable std::mutex mutex_;
};

} // namespace GRIM::Devices
```

### Session rules

- One active session per `device_id` in v1.
- A new registration for an already-online `device_id` replaces the old session.
- `online = true` is derived from the presence of an active session.
- `available_capabilities` live only in the session.
- `SessionManager` never writes JSON files and never stores hub policy.

---

## 6. WebSocket Protocol

**Port:** `11437`  
**Transport library:** existing repo dependency `uWebSockets`  
**Message format:** JSON text frames only  
**Phase 1 protocol serialization:** JSON only — no FlatBuffers in this subsystem

### 6.1 Message envelope

Every application message uses the same top-level shape.

```json
{
    "type": "register_device",
    "request_id": "req_001",
    "protocol_version": 1,
    "payload": {}
}
```

### 6.2 Client → Server messages

#### `register_device`

Sent immediately after the socket opens.

```json
{
    "type": "register_device",
    "request_id": "req_001",
    "protocol_version": 1,
    "payload": {
        "device_id": "dev_laptop_01",
        "device_name": "Austin Laptop",
        "device_type": "laptop",
        "platform": "windows",
        "platform_version": "11",
        "architecture": "x86_64",
        "agent_version": "0.1.0",
        "declared_capabilities": [
            "display_text",
            "show_notification",
            "receive_file"
        ],
        "available_capabilities": [
            "display_text",
            "show_notification"
        ]
    }
}
```

**Server behavior:**

1. Validate protocol version.
2. Validate required fields.
3. Upsert the persistent device profile in `DeviceRegistry`.
4. Reject immediately if the record is `Blocked`.
5. Open or replace the runtime session in `SessionManager`.
6. Return `register_ack`.

#### `heartbeat`

Sent every **10 seconds**.

```json
{
    "type": "heartbeat",
    "request_id": "req_002",
    "protocol_version": 1,
    "payload": {
        "device_id": "dev_laptop_01",
        "available_capabilities": [
            "display_text",
            "show_notification"
        ]
    }
}
```

**Server behavior:**

- Verify that the socket session is already bound to this `device_id`.
- Refresh `last_heartbeat_at` and `available_capabilities` in `SessionManager`.
- Update persisted `last_seen_at` in `DeviceRegistry`.
- Return `heartbeat_ack`.

### 6.3 Server → Client messages

#### `register_ack`

```json
{
    "type": "register_ack",
    "request_id": "req_001",
    "protocol_version": 1,
    "payload": {
        "device_id": "dev_laptop_01",
        "session_id": "sess_123",
        "pairing_state": "pending",
        "roles": ["laptop"],
        "allowed_actions": [],
        "heartbeat_interval_sec": 10
    }
}
```

#### `policy_update`

Sent whenever the hub changes pairing state, roles, or allowed actions.

```json
{
    "type": "policy_update",
    "protocol_version": 1,
    "payload": {
        "device_id": "dev_laptop_01",
        "pairing_state": "paired",
        "roles": ["laptop", "primary_ui"],
        "allowed_actions": ["display_text", "show_notification"]
    }
}
```

#### `heartbeat_ack`

```json
{
    "type": "heartbeat_ack",
    "request_id": "req_002",
    "protocol_version": 1,
    "payload": {
        "server_time": 1774931250
    }
}
```

#### `device_removed`

Sent if the hub deletes the record while the device is online.

```json
{
    "type": "device_removed",
    "protocol_version": 1,
    "payload": {
        "device_id": "dev_laptop_01"
    }
}
```

#### `error`

```json
{
    "type": "error",
    "request_id": "req_002",
    "protocol_version": 1,
    "payload": {
        "code": "device_id_mismatch",
        "message": "heartbeat device_id does not match bound session",
        "close_connection": true
    }
}
```

### 6.4 Unpaired / blocked behavior

The behavior must be deterministic.

| Pairing state | Allowed client messages | Eligible for routing | Can receive hub commands | Notes |
|---------------|-------------------------|----------------------|--------------------------|-------|
| `Pending` | `register_device`, `heartbeat` | No | No | Session may stay open for presence only |
| `Paired` | `register_device`, `heartbeat` | Yes, if online and `routeable_actions` is non-empty | Not in Phase 1, yes in later phases | Normal trusted device |
| `Blocked` | None after reject | No | No | Server returns `error` and closes connection |

Additional rules:

- A `Pending` device can identify itself and maintain presence only.
- A `Pending` device cannot receive commands, cannot receive files, cannot mutate pairing state, cannot mutate roles, cannot mutate permissions, and is never considered a route target.
- Any message type outside the allowed set for the current pairing state returns `error` and closes the connection.
- A `Blocked` device never gets a runtime session.

---

## 7. DeviceCommServer

**File:** `control/devices/server/device_comm_server.hpp`

`DeviceCommServer` is the only place that deals with raw sockets and JSON frames.

```cpp
struct DeviceCommConfig {
    bool enabled = true;
    std::string bind_address = "127.0.0.1";
    int port = 11437;
    int max_connections = 32;
    int heartbeat_interval_sec = 10;
    int session_timeout_sec = 30;
    std::string registry_path = "control/devices/device_registry.json";
};
```

```cpp
namespace GRIM::Devices {

class DeviceCommServer {
public:
        explicit DeviceCommServer(const DeviceCommConfig& config);

        void start();
        void stop();
        bool isRunning() const;

        // Read model for UI
        std::vector<DeviceSnapshot> listDeviceSnapshots() const;
        std::optional<DeviceSnapshot> getDeviceSnapshot(const std::string& device_id) const;

        // Hub policy controls used by UIControlPanel
        bool setPairingState(const std::string& device_id, PairingState state);
        bool setRoles(const std::string& device_id, const std::vector<std::string>& roles);
        bool setAllowedActions(const std::string& device_id, const std::vector<std::string>& actions);
        bool removeDevice(const std::string& device_id);

private:
        void handleRegister(/* ws */, const nlohmann::json& payload, const std::string& remote_address);
        void handleHeartbeat(/* ws */, const nlohmann::json& payload);
        void handleSocketClose(const std::string& session_id, int64_t now);
        void pushPolicyUpdate(const std::string& device_id);
        std::vector<DeviceSnapshot> buildSnapshots() const;

        DeviceCommConfig config_;
        std::atomic<bool> running_{false};
        DeviceRegistry registry_;
        SessionManager session_mgr_;
};

} // namespace GRIM::Devices
```

### Server responsibility boundaries

- `DeviceCommServer`
    - owns WebSocket open/message/close callbacks
    - parses and validates JSON
    - enforces protocol version and message allowlists
    - translates wire payloads into registry/session updates
    - pushes `policy_update` / `device_removed` back to connected agents
- `SessionManager`
    - owns `session_id`
    - owns online/offline truth
    - owns current `available_capabilities`
- `DeviceRegistry`
    - owns persistent records only
    - owns pairing state, roles, allowed actions, and timestamps
    - does **not** store sockets or session IDs

### Snapshot composition

`listDeviceSnapshots()` joins `DeviceRegistry` + `SessionManager`.

That is the only model the UI should consume.

---

## 8. UI Panel — `UIControlPanel`

**File:** `ui/ui_control_panel.hpp` / `ui/ui_control_panel.cpp`

Subclass `UIPanel` and title it exactly `"Control"`.

Phase 1 keeps the same general UI direction, but the panel displays joined `DeviceSnapshot` rows instead of pretending the registry record contains live socket state.

### 8.1 Tabs

| Tab | Purpose | Phase |
|-----|---------|-------|
| **Devices** | Device registry browser, pairing state, permissions, live availability | **Phase 1** |
| Commands | Send commands to paired devices | Phase 2 |
| Files | File transfer | Phase 3 |
| Logs | Device comm diagnostics | Phase 3 |

### 8.2 Devices tab layout

```
┌─ Control ──────────────────────────────────────────────────────────────┐
│  ● ● ●                           Control                              │
│  ┌──────────┬──────────┬──────────┬──────────┐                         │
│  │ Devices  │ Commands │  Files   │   Logs   │                         │
│  └──────────┴──────────┴──────────┴──────────┘                         │
│                                                                        │
│  Connected: 2 / 3 devices                      Pending: 1              │
│  ───────────────────────────────────────────────────────────────────── │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 🟢 Austin Laptop      paired     windows 11     x86_64          │  │
│  │    Supports: display_text, show_notification, receive_file      │  │
│  │    Available now: display_text, show_notification               │  │
│  │    Allowed: display_text, show_notification                     │  │
│  │    Roles: laptop, primary_ui    Session: sess_123               │  │
│  │    Last seen: just now              [Unpair] [Permissions]      │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │ 🔴 Old Tablet         pending    android 14    arm64            │  │
│  │    Supports: display_text                                      │  │
│  │    Available now: —                                            │  │
│  │    Allowed: —                                                   │  │
│  │    Roles: tablet            Last seen: 3 days ago              │  │
│  │                                 [Pair] [Block] [Remove]        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  Device server: ws://127.0.0.1:11437   ● Running                       │
└────────────────────────────────────────────────────────────────────────┘
```

### 8.3 UI behavior

- Poll `DeviceCommServer::listDeviceSnapshots()` every **1 second**.
- Show pairing state as `pending`, `paired`, or `blocked`.
- Show **both**:
    - `Supports` = `declared_capabilities`
    - `Available now` = `available_capabilities`
- Show `Allowed` = hub-granted `allowed_actions`.
- `Pair` changes `pairing_state` to `Paired`.
- `Unpair` changes `pairing_state` back to `Pending`.
- `Block` changes `pairing_state` to `Blocked` and disconnects the device if online.
- `Permissions` opens a lightweight editor for `allowed_actions` and `roles`.
    - `allowed_actions` must be a subset of `declared_capabilities`.
    - No extra framework. A simple modal or side pane is enough.

---

## 9. Reference Client — `grim_device_agent.py`

Minimal Python script that connects to GRIM’s device server and maintains presence.

```text
Usage:
    python grim_device_agent.py --host 192.168.1.100 --port 11437 \
            --device-id dev_laptop_01 --device-name "Austin Laptop" \
            --device-type laptop --platform windows --platform-version 11 \
            --architecture x86_64 --agent-version 0.1.0 \
            --declared-capabilities display_text,show_notification,receive_file \
            --available-capabilities display_text,show_notification
```

**Dependency:** `websockets` (`pip install websockets`)

### Client behavior

1. Connect to `ws://<host>:<port>`
2. Send `register_device`
3. Store returned `session_id`
4. Send `heartbeat` every 10 seconds with current `available_capabilities`
5. Handle `policy_update`, `device_removed`, and `error`
6. Exit cleanly on disconnect or Ctrl+C

The Python client does **not** send pairing state, roles, or allowed actions. Those are hub-owned.

---

## 10. Integration Points

### 10.1 `main.cpp`

```cpp
#include "ui/ui_control_panel.hpp"
#include "control/devices/server/device_comm_server.hpp"

auto controlPanel = std::make_shared<UIControlPanel>(deviceCommServer);
controlPanel->setVisible(false);
UIRoot::get().addPanel(controlPanel);
```

### 10.2 Startup / config loading

Device comm config is loaded by the main GRIM app startup/config path.

**Do not route this through `GRIM-text` training startup files** such as `Phase1_Startup.cu`.

### 10.3 Build system

The device subsystem source lives under `control/devices/`, but it is compiled into the main `GRIM` target because:

- the UI panel lives in the main app
- the device server runs inside `GRIM.exe`
- the panel needs direct access to the in-process server API

That means:

- add `control/devices/*.cpp` to the main build
- reuse existing `uWebSockets`, `nlohmann_json`, and existing logging/config utilities
- do **not** add `ixwebsocket`
- do **not** create a second standalone device daemon for v1

---

## 11. Security and policy rules

| Concern | Rule |
|---------|------|
| Unauthorized registration | First contact creates or refreshes a `Pending` record only. No trust is implied. |
| Client tries to self-grant trust | Rejected by design. Pairing state, roles, and allowed actions are hub-only fields. |
| Client claims capabilities it should not use | Claims go to `declared_capabilities`; routing still requires `allowed_actions` and live availability. |
| Session spoofing | Every heartbeat must match the `device_id` already bound to that `session_id`; mismatch returns `error` and closes the socket. |
| Local network exposure | Default bind address is `127.0.0.1`. Wider bind must be explicit in config. |
| Device already blocked | Return `error(code="blocked_device")` and close immediately. |
| Stale sessions | Session timeout after 30 seconds without heartbeat; session removed, record stays. |

---

## 12. `ai_config.json` Additions

```json
{
    "device_comm": {
        "enabled": true,
        "port": 11437,
        "bind_address": "127.0.0.1",
        "max_connections": 32,
        "heartbeat_interval_sec": 10,
        "session_timeout_sec": 30,
        "registry_path": "control/devices/device_registry.json"
    }
}
```

---

## 13. Implementation Order

| Step | Task | Files | Depends On |
|------|------|-------|------------|
| 1 | Define `PairingState`, `DeviceRecord`, `DeviceSession`, `DeviceSnapshot` | `device_record.hpp`, `device_session.hpp`, `device_snapshot.hpp` | — |
| 2 | Implement `DeviceRegistry` persistence and hub-policy updates | `device_registry.hpp/cpp` | Step 1 |
| 3 | Implement `SessionManager` runtime ownership and timeout sweep | `session_manager.hpp/cpp` | Step 1 |
| 4 | Define JSON protocol constants, payload structs, validation helpers | `device_protocol.hpp`, `device_messages.hpp` | Step 1 |
| 5 | Implement `DeviceCommServer` using existing `uWebSockets` | `device_comm_server.hpp/cpp` | Steps 2, 3, 4 |
| 6 | Implement `grim_device_agent.py` reference client | `client/grim_device_agent.py` | Step 5 |
| 7 | Implement `UIControlPanel` using joined `DeviceSnapshot` rows | `ui/ui_control_panel.hpp/cpp` | Step 5 |
| 8 | Integrate startup, config loading, and main build wiring | `main.cpp`, main CMake/config loader | Steps 5, 7 |
| 9 | Add policy editor for `allowed_actions` and `roles` in the panel | `ui/ui_control_panel.cpp` | Step 7 |

Steps 1–4 are pure structure and can be tested without a live socket.  
Steps 6 and 7 can proceed in parallel once Step 5 exists.

---

## 14. Future Phases

Keep future scope narrow and attached to this architecture.

| Phase | Feature | Notes |
|-------|---------|-------|
| 2 | Hub → device command dispatch | Uses `allowed_actions` and `routeable_actions` already defined here |
| 3 | File transfer | Uses the same registry/session split; add file-specific actions only |
| 3 | Device comm logs tab | Show connection history, rejects, and heartbeat failures |

---

## 15. Dependencies

| Package | Purpose | Status |
|---------|---------|--------|
| `uwebsockets` | C++ WebSocket server | Already in repo dependencies |
| `nlohmann-json` | JSON protocol + registry persistence | Already in repo dependencies |
| `websockets` | Python reference client | Add to Python environment only |

**Do not add for Phase 1:** `ixwebsocket`, FlatBuffers device schemas, extra RPC frameworks.

---

## 16. Testing Strategy

| Level | What | How |
|-------|------|-----|
| Unit | `DeviceRegistry` create/update/load/save | Verify hub-owned fields survive re-registration unchanged |
| Unit | `SessionManager` session replacement + timeout | One device ID → one active session; stale sessions expire cleanly |
| Unit | Snapshot composition | `routeable_actions` must equal declared ∩ available ∩ allowed, or empty when not paired/offline |
| Integration | Pending device registration | Python client connects, record appears as `Pending`, online=true, no routeable actions |
| Integration | Pair / unpair / block flows | UI changes policy, client receives `policy_update`, blocked client is disconnected |
| Integration | Heartbeat mismatch rejection | Wrong `device_id` for bound session returns error and closes socket |
| UI | Device rows show supports vs available vs allowed | Manual visual verification |

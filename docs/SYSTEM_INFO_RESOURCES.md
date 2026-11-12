# System Information resources

The System Information plugin owns native detection, snapshot state, resource
construction, command handling, and cleanup. During initialization it captures
once and stages copied resources through the generic `replace_resources` API.
The framework commits the provider, tools, and resources atomically. No host
hardware service or per-plugin hardware state remains in `plugin_api_impl`.
The provider declares SYSTEM permission and supplies resources of kind
`system_info.snapshot`:

| Resource ID | Component | Data |
| --- | --- | --- |
| `system` | `system` | `os_name`, `arch`, `wifi_connected`, `gpu_count`, `monitor_count`, `desktop` |
| `cpu` | `cpu` | `cpu_cores` |
| `memory` | `memory` | `ram_total_mb` |
| `gpu_<device_index>` | `gpu` | `device_index`, `name`, `vram_mb`, `driver_version`, `has_cuda`, `has_metal`, `has_rocm` |
| `monitor_<index>` | `monitor` | `monitor_index`, `x`, `y`, `width`, `height`, `is_primary` |

`desktop` contains `origin_x`, `origin_y`, `width`, and `height` in virtual desktop
coordinates. Memory and VRAM values are capacities in MB, not current usage.
Audio device names, voice backend flags, and the Whisper model suggestion are
excluded from these objects. The shared inventory codec and existing `system_info`
text response retain their compatibility behavior.

Each resource state has this envelope:

```json
{
  "format": "grim-system-info-resource",
  "version": 1,
  "component": "cpu",
  "data": {"cpu_cores": 12},
  "capture_time_ns": 123000000000,
  "error": null,
  "stale": false
}
```

The timestamp is monotonic nanoseconds, not wall time. Every object in a batch
shares the capture timestamp and latest error/stale state. On failure, all last
successful objects, data, and timestamps are retained and marked stale. Before
the first successful capture, only System, CPU, and Memory exist, with null data
and timestamp. Device objects appear after enumeration. Success clears errors
and atomically replaces the entire batch, removing devices no longer present.

Identity includes the owning provider, `provider_instance`, and resource ID.
Each object uses `runtime_id: 1`. GPU keys follow the plugin detector's
device index; monitor keys follow enumeration order. These indices identify
enumeration slots, not persistent physical device identifiers across topology
changes. Replacement and unload retire the provider's whole batch; an old
generation cannot republish into a replacement. No plugin callbacks or C++
records cross the ABI.

Calling `system_info` captures fresh hardware and atomically refreshes all objects.
The plugin serializes captures and owns last-good state. Detection failure during
initialization publishes the fixed objects with explicit unavailable state; a
publication/registration failure rejects the candidate. A failed command refresh
publishes stale state before returning the existing command exception response.
There is no periodic polling or worker thread. Shutdown destroys plugin state
after the host drains callbacks; replacement starts with an independent cache.

`hardware_detection.cpp` is compiled into the DLL. It uses native OS queries and
optional dynamic CUDA driver discovery (no CUDA toolkit build dependency).
Windows DXGI/WMI, COM objects, and driver modules have scoped lifetimes. No host
startup detector, audio service, hardware global, or C++ host function is called.
Audio capture is outside this provider. The separate startup inventory used by
bootstrap is unchanged and is never consulted or mutated by the plugin.

API v6 removes `get_hardware_snapshot`, `register_hardware_resource`, and unused
legacy host system/CPU/memory query slots. No retired fields preserve their old
positions in the table. Older DLLs are rejected; deployment needs
a matching rebuilt host and plugins. The resource and tool result schemas remain
unchanged. Linux/macOS paths require native validation on those platforms.

The live world projects registered objects into the System Information room
automatically. `inspect CPU`, `inspect cpu`, and other exact object names or IDs
read published state without triggering detection. Names are ASCII case
insensitive and must identify one resource in the current provider room.
`inspect-resource cpu` remains the explicit object-ID route for ambiguous names
or collisions with tool IDs. Tool catalog access does not grant remote resource
access. Provider changes are reconciled before inspection.

Inspection feedback and resource tile clicks use readable labels and units:

```text
CPU
Logical cores: 12
Status: Current
```

Failed captures with previous data show the readings with
`Status: Stale - latest capture failed.` and a capture error. Before any successful
capture, inspection reports `CPU information unavailable` and `Status: Unavailable`
without inventing a core count. System, Memory, GPU, and Monitor use the same
name/fields/status presentation, including Wi-Fi and signed desktop positions.
Negative capability flags are shown as `Not reported`, and empty GPU driver
strings as `Unavailable`.

The session observation and history entry retain the complete resource descriptor
under `resource_inspection`, including its original structured `state`, provider
generation, runtime identity, and capture timestamp. This field is cleared on the
next command or when its inspected resource changes or disappears; unrelated
refreshes retain it. Prior history entries retain their original evidence.
State refreshes update the projection without appending repeated room snapshots.
Human feedback does not serialize that host envelope.
Non-hardware resources retain their existing inspection presentation.

Authored room and
standalone templates use matching bindings for System, CPU, Memory, GPU 0, and
Monitor 0; templates store no machine readings. Live device counts come from
enumeration, independently of the authored examples. The former aggregate
`hardware` object is no longer published; replace its binding with the relevant
component bindings in existing saved worlds.

This exposes topology and capabilities, not live utilization. The separate MMO
resource signal and resource admission systems remain independent of these
objects. The unused `hardware/resource_values.*` monitor and its randomized GPU
fallback have been removed.

## Validation

`tests/run_plugin_shutdown.ps1 -BuildApplicationFixture -BuildSystemInfoFixture`
builds isolated DLL fixtures. Hardware fixture implementation is linked into the
system-info DLL, not the host. Tests cover topology, fresh capture, formatting,
standard/unknown failures, stale retention, recovery, device removal, read-only
world inspection, independent state after reload, and unload cleanup. Generic
provider tests cover failed publication and retired-generation rejection.

`tests/run_system_info_native.ps1` links the real native DLL in a test directory
and runs a read-only native detector smoke test without the host runtime.

# Plugin-owned world projection feasibility

Date: November 5, 2026. Status: provider-room live foundation implemented;
explicit resource projection remains proposed.

The host-owned live backend is documented in [LIVE_WORLD.md](LIVE_WORLD.md).
It publishes provider identity with tools atomically, supports zero-tool providers,
and connects/disconnects provider rooms without recompiling TextWorld. It exposes
read-only tool inspection. The resource declarations, asset resolution, typed
state and executable action adapters discussed below remain future work. Pinned
compiled episodes and the host-owned live session are separate backend paths.

## Finding

The native plugin lifecycle already provides staging, provider identity, generation
checks, atomic tool replacement, and generation-scoped removal. Extend that
publication transaction with explicit resource declarations and project the world
from its committed snapshot. Do not discover available resources by scanning the
asset library, matching descriptions, or inferring internal services from tools.
The existing authored plugin room files are templates, not an availability source.

## Existing integration points

- `core/plugin_api_impl.cpp`: `PluginContext::pending`, `api_register_tools`,
  `commitPluginTools`, and `beginPluginShutdown` provide the staging/commit/removal
  boundary. The host copies metadata and owns provider identity and generation.
- `MMO/Core/ToolRegistry.cpp`: `replaceProvider` validates a candidate before
  swapping tools and aliases under one mutex. A rejected replacement retains the
  previous generation; `unregisterInstance` removes only the specified generation.
- `core/plugin_api.hpp`: v3.16 ends its service table with 16 reserved pointer
  slots. A versioned resource-registration service can consume one while retaining
  table size and old field offsets. This needs ABI assertions and compatibility
  tests; do not append resource fields to `GrimPluginInfo` without versioning.
- `ui/world_lab/world_asset.*`: templates support metadata, independent insertion,
  fresh world IDs and preserved stable resource keys. No registration connection
  or automatic availability filtering currently exists.
- `ui/world_lab/textworld_client.cpp`: the client already supplies a filtered live
  tool catalog from `getModelTools`. That is discovery integration, not resource
  registration or a native inventory registry object.

## Proposed ownership and publication

Add a provider-level registration record alongside tools, containing host-assigned
provider type/name/instance, optional room-template reference, and an explicit list
of resource declarations. A resource declaration contains a provider-scoped stable
key, resource type, versioned object-template reference, and authored initial state
or an explicit host observation source. Resource declarations are data, not plugin
callbacks retained by the world. Internal request queues are absent unless explicitly
registered as public resources. Do not infer new resources from tool names.

The plugin supplies its declarations during init. The host validates and stages
both tools and resources. Successful initialization publishes one coherent provider
snapshot and revision; failure publishes neither. Replacement and unload update
both together before notifying consumers. Two independently updated registries or
listener-only reconciliation would expose mixed generations and are insufficient.
Support providers with zero tools: today's tool-only registry cannot represent
that provider's existence. Late registration/unregistration must obey the same
transaction/version rules. Host built-ins remain a distinct provider type.

Room identity uses provider name within provider type; the loaded instance is the
execution generation guard. The room projection takes committed providers as its
input, obtains tools from their existing registrations, and instantiates only their
explicit resources. A room template must not independently introduce resource
children that bypass those declarations: room layout and resource membership are
separate. Asset references resolve by stable asset ID plus version, with duplicate
and missing definitions rejected; filenames and search rank are not identity.
Copy resolved asset contents into the snapshot so subsequent library edits cannot
silently change a running episode. The library currently lacks such an ID index,
so this resolver must be added rather than assumed.

## World and execution boundaries

The available-world projection follows load/reload/unload. Each new episode pins
a coherent provider/tool/resource/asset snapshot. Existing compiled TextWorld
sessions retain their pinned world instead of mutating rooms mid-attempt. A stale
production generation fails closed; retaining descriptor copies does not retain
unloaded native code. Sandbox replay requires retained versioned transition
semantics, which are not implemented yet. Registry revision and snapshot must be
captured under one lock; separate version/getAllTools calls are not an atomic read.

Keep existing usage and permission rules. Rooms do not grant access to Debug,
System, or Ignore tools. The inventory registry is a read-only virtual projection,
not a second executable registry or a portable copy of native handlers.

Metadata is not behavior: `initial_state` currently does not add TextWorld facts,
visibility predicates, transitions, or verification rules. Implement typed state
and shared result projection separately. Sandbox dispatch still returns its
explicit not-implemented error; production remains the default. No fallback from
sandbox to production is permissible.

## Evidence and implementation order

A focused host-only probe against the real `ToolRegistry.cpp` passed provider
projection, Model filtering, rejected-replacement rollback, successful replacement,
pinned descriptor-copy preservation, stale invocation rejection, and old/new
instance-scoped unload. It invoked no native plugin, UI operation, model or trainer.
Probe source and runner: `.codex-build/plugin-world-feasibility/`. This establishes
the existing tool lifecycle behavior only; atomic resource publication, zero-tool
providers, ABI compatibility and TextWorld integration remain untested/unbuilt.

1. Add provider/resource base records and atomic snapshot/publication operations,
   with rollback, zero-tool-provider and generation-removal tests.
2. Add the versioned C registration service, copied validation and lifecycle wiring;
   verify legacy plugin layouts and failed-init/reload rollback using host fixtures.
3. Add stable asset-reference resolution and provider-to-world projection. Test
   that library-only objects, including panel-request templates, never appear.
4. Have plugins explicitly declare their exposed resources. Replace the manually
   authored availability catalog; preserve reusable templates where appropriate.
5. Integrate projected rooms and the virtual inventory registry into episode
   preparation, then implement typed state/actions and sandbox execution.

The first end-to-end acceptance case should load a UI fixture exposing console
only, verify settings and panel requests are absent even if templates exist, reload
it exposing settings instead, verify new episodes reflect that change, reject a
malformed replacement without changing the available world, and unload the provider
without leaving a room or resource behind. Pinned episodes retain their snapshot
and cannot execute against a replacement production generation.

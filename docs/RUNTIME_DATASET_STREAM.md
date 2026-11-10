# Runtime dataset stream

`bootstrap/dataset_bootstrap.cpp` starts a process-owned `DatasetStream` after
configuration loading, before runtime services and DataHub. Shutdown joins its
worker after UI teardown. `dataset_services_bootstrap.cpp` also creates the
process-owned collection pipeline, structurer, Hugging Face client, and generator
job service. Bootstrap initializes these services without automatically starting
a collection/generation job or running training.

The source directory is `paths.grim_text.training_data`; the model-store path
uses `paths.grim_text.model_store`. Both resolve relative to the GRIM root.
The stream watches `mass_dataset.jsonl`, `concept_blocks.fb`, legacy
`concept_blocks.jsonl`, and `curriculum_registry.json` every second. It also
exposes `refresh()` for an immediate reload after a maintenance operation.
Missing files initially represent an empty catalog; the reader creates no files.

Runtime consumers use the service without constructing a UI:

```cpp
#include "bootstrap/dataset_bootstrap.hpp"

auto stream = GRIM::DataCollection::runtimeDatasetStream();
uint64_t cursor = 0; // Each consumer keeps its own cursor between polls.
GRIM::DataCollection::DatasetStream::Snapshot snapshot;
if (stream->pullLatest(snapshot, cursor)) {
    // Read snapshot.catalog here, or retain snapshot for an ongoing operation.
    const auto courses = snapshot.catalog->getCourses();
}
```

Snapshots contain a read-only `DatasetTarget` and a monotonically increasing
process-local revision. Readers can retain old snapshots while new ones publish.
This is a latest-version stream, not a durable log of every individual edit.
`latest()` returns the current version; before the first successful load its
catalog is null and revision is zero. `lastError()` reports reload failure.
Failed loads retain the last valid snapshot and retry on subsequent polls.
No legacy migration is performed by this reader.

`runtimeDatasetCatalog()` supplies the authoring client used by DataHub and
World Lab. It owns no separate mutable catalog. Each operation copies the latest
immutable catalog under a serialized writer lock, performs the existing validated
persistence operation, and publishes the successful result immediately. Stale
revision writes and invalid external updates reject authoring. Failed operations
discard their in-memory candidate. UI drafts remain local and unsaved drafts
are not published. The active model selection/assignments are shared process
state and are retained when source files reload.
DataHub's lightweight client pins an immutable view during each interaction;
list indices cannot silently switch to replacement rows midway through an edit.
Editor saves carry the revision originally loaded and reject stale content.

World Lab obtains Goal selections, imports, registry membership, and ConceptBlock
details directly from this service. It never looks up or creates DataHub.
Goal payloads, collision checks, evidence handling, and preview rendering retain
their existing contracts.

Headless authoring and creation are available without a panel:

```cpp
auto catalog = GRIM::DataCollection::runtimeDatasetCatalog();
catalog->addConceptBlock(block);

#include "bootstrap/dataset_services_bootstrap.hpp"
auto services = GRIM::DataCollection::runtimeDatasetServices();
GRIM::DataCollection::DatasetCreationService::Request request;
request.seed = 1337;
request.attempts = 25;
services->creation->start(request);
// Poll status(), pause()/resume(), or saveQueue(added) from any client.
```

Generator requests pin their recipe/source version, perform model fills on a
service worker, and retain their queue independently of UI visibility/lifetime.
`reloadAssets()` reloads the configured directory's `LexiconPieces.json` and
`GeneratorDataSource.json`; failed reloads retain established assets. UI generator
controls submit requests and display status. Source-bound validation/assembly and
explicit queue-save behavior remain unchanged. Queues survive panel destruction
but are in memory, not durable across process exit. Shutdown joins the worker,
waiting for any in-flight adapter call's configured timeout.

The collection pipeline uses the configured training-data directory directly and
commits final writes through the same writer lock. It publishes a refreshed
catalog immediately after writing. A mismatched output destination is rejected.
Pipeline shutdown requests stop and joins its worker before runtime teardown;
closing DataHub does not stop it. UI search/preview/download presentation and its
download queue remain in DataHub; this change does not introduce a headless
download-queue API.

For maintenance, write a complete replacement file and atomically replace the
destination, using one external writer. In-process catalog and pipeline writes
share the service lock. Reload checks modification times and sizes before
and after reading, rejecting detected concurrent changes. This is not a
transaction across multiple files: update referenced blocks before publishing
their registry references. Invalid JSON/typed rows, invalid ConceptBlocks, failed
registry loads reject publication. Stale course-to-block references remain in
the browsable registry so one missing block cannot empty all dropdowns. Exact
block lookup reports missing IDs when selected; loading never silently removes
membership or invents replacement content. Files
changed without changing their size or timestamp are not detected automatically.

Host-only verification: `tests/run_dataset_stream.ps1`, including catalog CRUD,
Goal access/import, stale-write rejection, pipeline publication, generator jobs
without UI ticks, asset reload failure, snapshot retention, and shutdown. This
does not build or run GRIM or training/runtime targets.

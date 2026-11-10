#pragma once

#include "DataCollection/dataset_stream.hpp"
#include "DataCollection/dataset_catalog.hpp"

void bootstrapDatasetSubsystem();
void shutdownDatasetSubsystem();

namespace GRIM::DataCollection {
// Available after bootstrap, independently of DataHub. Consumers retain the
// stream/snapshots by shared ownership; shutdown stops refresh, not pinned reads.
std::shared_ptr<DatasetStream> runtimeDatasetStream();
std::shared_ptr<DatasetCatalog> runtimeDatasetCatalog();
}

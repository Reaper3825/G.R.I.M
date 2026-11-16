#pragma once
#include "core/plugin_api.hpp"
#include "application_catalog.hpp"
#include <string_view>
GrimResult dispatchApplicationTool(std::string_view id, const GrimToolCall* call,
                                   GrimToolResultWriter* writer, GRIM::Applications::Catalog& catalog);

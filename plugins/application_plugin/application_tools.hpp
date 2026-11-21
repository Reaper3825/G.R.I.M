#pragma once
#include "core/plugin_api.hpp"
#include "application_catalog.hpp"
#include <string_view>
GrimToolReply dispatchApplicationTool(std::string_view id, const GrimToolCall* call,
                                   GRIM::Applications::Catalog& catalog);

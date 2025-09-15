#pragma once
#include "commands_core.hpp"

// ------------------------------------------------------------
// Filesystem command declarations
// ------------------------------------------------------------
CommandResult cmdShowPwd(const std::string& arg);
CommandResult cmdChangeDir(const std::string& arg);
CommandResult cmdListDir(const std::string& arg);
CommandResult cmdMakeDir(const std::string& arg);
CommandResult cmdRemoveFile(const std::string& arg);

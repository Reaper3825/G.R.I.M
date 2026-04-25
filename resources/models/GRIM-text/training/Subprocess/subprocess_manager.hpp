#pragma once

// subprocess_manager
//
// Primitive coordinator for child processes spawned at the start of training.
// Designed as a foundation: each subprocess is described by a tiny descriptor,
// the manager spawns it as an external executable, waits for it to exit, and
// reads a structured status file the child wrote on its way out.
//
// Rule 20: there are NO fallback paths. A missing executable, a missing status
// file, a non-zero exit with no error_message, or a malformed status file all
// throw std::runtime_error with a precise message identifying the subprocess.
//
// Rule 26: this header is the entire public surface. Do not add convenience
// wrappers or compatibility shims. Add a new descriptor + spawn call site for
// each new subprocess.

#include <string>
#include <vector>

#include "subprocess_status.hpp"

namespace GRIMText {
namespace Subprocess {

// Descriptor for one subprocess invocation. Filled in by the per-subprocess
// wrapper (e.g. tokenizer_subprocess.cpp), then handed to spawn_and_wait().
struct subprocess_request {
    // Human-readable identifier ("train_tokenizer"). Used in errors and logs.
    std::string name;

    // Absolute path to the executable to run. Required.
    std::string executable_path;

    // Argv to pass (NOT including argv[0]; manager prepends executable_path).
    std::vector<std::string> arguments;

    // Absolute path the child will write its status JSON to. The manager
    // deletes any pre-existing file at this path before spawning so a stale
    // success cannot be mistaken for a fresh one.
    std::string status_file_path;
};

// Resolve the absolute path to a sibling executable in the same directory as
// the currently-running process. On Windows, automatically appends ".exe" if
// the file at the bare name does not exist.
//
// Throws if the current executable's directory cannot be determined or no
// matching file is found.
std::string resolve_sibling_executable(const std::string& base_name);

// Spawn the subprocess described by `req`, wait for it to exit, and read the
// status file it wrote.
//
// Returns a populated subprocess_result. Throws on infrastructure failure
// (cannot fork, cannot read status file, status JSON malformed). A subprocess
// that exited cleanly but reported failure via the status file returns a
// subprocess_result with outcome=error and a populated error_message - this is
// not an exception.
subprocess_result spawn_and_wait(const subprocess_request& req);

} // namespace Subprocess
} // namespace GRIMText

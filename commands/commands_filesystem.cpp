#include "commands_filesystem.hpp"
#include "console_history.hpp"

extern std::filesystem::path g_currentDir;

// ====================================================
// pwd → show current directory
// ====================================================
CommandResult cmdShowPwd([[maybe_unused]] const std::string& arg) {
    return {
        true,                                                   // success
        "[FS] Current directory: " + g_currentDir.string(),     // message
        "ERR_NONE",                                             // errorCode
        "summary",                                              // category
        "Current directory shown",                              // voice
        Colors::Cyan                                            // color
    };
}

// ====================================================
// cd <dir> → change directory
// ====================================================
CommandResult cmdChangeDir(const std::string& arg) {
    if (arg.empty()) {
        return {
            false,                              // success
            "[FS] Usage: cd <directory>",       // message
            "ERR_FS_NO_ARGUMENT",               // errorCode
            "error",                            // category
            "Directory name required",          // voice
            Colors::Red                         // color
        };
    }

    std::filesystem::path newPath = g_currentDir / arg;
    if (!std::filesystem::exists(newPath)) {
        return {
            false,                                                  // success
            "[FS] Directory does not exist: " + arg,                // message
            "ERR_FS_NOT_FOUND",                                     // errorCode
            "error",                                                // category
            "Directory not found",                                  // voice
            Colors::Red                                             // color
        };
    }

    g_currentDir = std::filesystem::canonical(newPath);
    return {
        true,                                                       // success
        "[FS] Changed directory to: " + g_currentDir.string(),     // message
        "ERR_NONE",                                                 // errorCode
        "routine",                                                  // category
        "Directory changed",                                        // voice
        Colors::Green                                               // color
    };
}

// ====================================================
// ls → list directory contents
// ====================================================
CommandResult cmdListDir([[maybe_unused]] const std::string& arg) {
    std::string output = "[FS] Contents:\n";
    for (const auto& entry : std::filesystem::directory_iterator(g_currentDir)) {
        output += " - " + entry.path().filename().string() + "\n";
    }

    return {
        true,                               // success
        output,                             // message
        "ERR_NONE",                         // errorCode
        "summary",                          // category
        "Directory contents listed",        // voice
        Colors::Cyan                        // color
    };
}

// ====================================================
// mkdir <dir> → make directory
// ====================================================
CommandResult cmdMakeDir(const std::string& arg) {
    if (arg.empty()) {
        return {
            false,                             
            "[FS] Usage: mkdir <directory>",    
            "ERR_FS_NO_ARGUMENT",               
            "error",                            
            "Directory name required",          
            Colors::Red                         
        };
    }

    std::filesystem::path newDir = g_currentDir / arg;
    if (std::filesystem::create_directory(newDir)) {
        return {
            true,                                              
            "[FS] Directory created: " + newDir.string(),       
            "ERR_NONE",                                         
            "routine",                                          
            "Directory created",                              
            Colors::Green
        };
    } else {
        return {
            false,                                                      // success
            "[FS] Failed to create directory: " + newDir.string(),     // message
            "ERR_FS_CREATE_FAILED",                                     // errorCode
            "error",                                                    // category
            "Failed to create directory",                               // voice
            Colors::Red                                                 // color
        };
    }
}

// ====================================================
// rm <file> → remove file
// ====================================================
CommandResult cmdRemoveFile(const std::string& arg) {
    if (arg.empty()) {
        return {
            false,                          // success
            "[FS] Usage: rm <file>",        // message
            "ERR_FS_NO_ARGUMENT",           // errorCode
            "error",                        // category
            "File name required",           // voice
            Colors::Red                     // color
        };
    }

    std::filesystem::path file = g_currentDir / arg;
    if (!std::filesystem::exists(file)) {
        return {
            false,                                
            "[FS] File not found: " + arg,        
            "ERR_FS_NOT_FOUND",                   
            "error",                             
            "File not found",                   
            Colors::Red                          
        };
    }

    if (std::filesystem::remove(file)) {
        return {
            true,                                   
            "[FS] Removed: " + file.string(),       
            "ERR_NONE",                           
            "routine",                             
            "File removed",                      
            Colors::Green                           
        };
    } else {
        return {
            false,                                       
            "[FS] Failed to remove: " + file.string(),    
            "ERR_FS_REMOVE_FAILED",                         
            "error",                                        
            "Failed to remove file",                        
            Colors::Red                                    
        };
    }
}

#include "core/crash_dump.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dbghelp.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace GRIM {
namespace {

std::string timestampForFile()
{
    SYSTEMTIME time{};
    GetLocalTime(&time);

    std::array<char, 64> buffer{};
    std::snprintf(buffer.data(), buffer.size(),
                  "%04hu%02hu%02hu_%02hu%02hu%02hu_%03hu",
                  time.wYear,
                  time.wMonth,
                  time.wDay,
                  time.wHour,
                  time.wMinute,
                  time.wSecond,
                  time.wMilliseconds);
    return buffer.data();
}

std::filesystem::path crashBasePath()
{
    std::filesystem::path logsDir = std::filesystem::current_path() / "logs";
    std::error_code error;
    std::filesystem::create_directories(logsDir, error);
    return logsDir / ("GRIM_crash_" + timestampForFile());
}

void writeMiniDump(const std::filesystem::path& dumpPath, EXCEPTION_POINTERS* exceptionPointers)
{
    HANDLE file = CreateFileW(dumpPath.wstring().c_str(),
                              GENERIC_WRITE,
                              0,
                              nullptr,
                              CREATE_ALWAYS,
                              FILE_ATTRIBUTE_NORMAL,
                              nullptr);
    if (file == INVALID_HANDLE_VALUE)
        return;

    MINIDUMP_EXCEPTION_INFORMATION exceptionInfo{};
    exceptionInfo.ThreadId = GetCurrentThreadId();
    exceptionInfo.ExceptionPointers = exceptionPointers;
    exceptionInfo.ClientPointers = FALSE;

    const MINIDUMP_TYPE dumpType = static_cast<MINIDUMP_TYPE>(
        MiniDumpWithFullMemoryInfo |
        MiniDumpWithHandleData |
        MiniDumpWithThreadInfo |
        MiniDumpWithUnloadedModules);

    MiniDumpWriteDump(GetCurrentProcess(),
                      GetCurrentProcessId(),
                      file,
                      dumpType,
                      &exceptionInfo,
                      nullptr,
                      nullptr);
    CloseHandle(file);
}

void writeStackFrame(std::ofstream& out, HANDLE process, DWORD64 address)
{
    HMODULE module = nullptr;
    std::array<char, MAX_PATH> moduleName{};
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(address),
                           &module) &&
        module) {
        GetModuleFileNameA(module, moduleName.data(), static_cast<DWORD>(moduleName.size()));
    }

    std::array<unsigned char, sizeof(SYMBOL_INFO) + MAX_SYM_NAME> symbolBuffer{};
    auto* symbol = reinterpret_cast<SYMBOL_INFO*>(symbolBuffer.data());
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
    symbol->MaxNameLen = MAX_SYM_NAME;

    DWORD64 displacement = 0;
    if (SymFromAddr(process, address, &displacement, symbol)) {
        out << "  0x" << std::hex << address << std::dec
            << " " << symbol->Name << "+0x" << std::hex << displacement << std::dec;
    } else {
        out << "  0x" << std::hex << address << std::dec << " <unresolved>";
    }

    if (moduleName[0] != '\0')
        out << " [" << moduleName.data() << "]";
    out << '\n';
}

void writeCrashText(const std::filesystem::path& textPath,
                    const std::filesystem::path& dumpPath,
                    EXCEPTION_POINTERS* exceptionPointers)
{
    std::ofstream out(textPath, std::ios::out | std::ios::trunc);
    if (!out.is_open())
        return;

    EXCEPTION_RECORD* record = exceptionPointers ? exceptionPointers->ExceptionRecord : nullptr;
    CONTEXT* sourceContext = exceptionPointers ? exceptionPointers->ContextRecord : nullptr;

    out << "GRIM native crash report\n";
    out << "Dump: " << dumpPath.string() << "\n";
    out << "ProcessId: " << GetCurrentProcessId() << "\n";
    out << "ThreadId: " << GetCurrentThreadId() << "\n";
    if (record) {
        out << "ExceptionCode: 0x" << std::hex << record->ExceptionCode << std::dec << "\n";
        out << "ExceptionAddress: 0x" << std::hex
            << reinterpret_cast<uintptr_t>(record->ExceptionAddress) << std::dec << "\n";
        if (record->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && record->NumberParameters >= 2) {
            out << "AccessViolationOperation: "
                << (record->ExceptionInformation[0] == 0 ? "read" :
                    (record->ExceptionInformation[0] == 1 ? "write" : "execute")) << "\n";
            out << "AccessViolationAddress: 0x" << std::hex
                << record->ExceptionInformation[1] << std::dec << "\n";
        }
    }

    if (!sourceContext) {
        out << "No exception context available.\n";
        return;
    }

    HANDLE process = GetCurrentProcess();
    SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME | SYMOPT_LOAD_LINES);
    SymInitialize(process, nullptr, TRUE);

    CONTEXT context = *sourceContext;
    STACKFRAME64 frame{};
#if defined(_M_X64)
    DWORD machineType = IMAGE_FILE_MACHINE_AMD64;
    frame.AddrPC.Offset = context.Rip;
    frame.AddrPC.Mode = AddrModeFlat;
    frame.AddrFrame.Offset = context.Rbp;
    frame.AddrFrame.Mode = AddrModeFlat;
    frame.AddrStack.Offset = context.Rsp;
    frame.AddrStack.Mode = AddrModeFlat;
#elif defined(_M_IX86)
    DWORD machineType = IMAGE_FILE_MACHINE_I386;
    frame.AddrPC.Offset = context.Eip;
    frame.AddrPC.Mode = AddrModeFlat;
    frame.AddrFrame.Offset = context.Ebp;
    frame.AddrFrame.Mode = AddrModeFlat;
    frame.AddrStack.Offset = context.Esp;
    frame.AddrStack.Mode = AddrModeFlat;
#else
    DWORD machineType = 0;
#endif

    out << "Stack:\n";
    if (machineType == 0) {
        out << "  Stack walking is unsupported for this architecture.\n";
        SymCleanup(process);
        return;
    }

    HANDLE thread = GetCurrentThread();
    for (int frameIndex = 0; frameIndex < 64; ++frameIndex) {
        if (!StackWalk64(machineType,
                         process,
                         thread,
                         &frame,
                         &context,
                         nullptr,
                         SymFunctionTableAccess64,
                         SymGetModuleBase64,
                         nullptr)) {
            break;
        }
        if (frame.AddrPC.Offset == 0)
            break;
        writeStackFrame(out, process, frame.AddrPC.Offset);
    }

    SymCleanup(process);
}

LONG WINAPI unhandledExceptionFilter(EXCEPTION_POINTERS* exceptionPointers)
{
    const std::filesystem::path basePath = crashBasePath();
    const std::filesystem::path dumpPath = basePath.string() + ".dmp";
    const std::filesystem::path textPath = basePath.string() + ".txt";

    writeMiniDump(dumpPath, exceptionPointers);
    writeCrashText(textPath, dumpPath, exceptionPointers);

    std::fprintf(stderr, "[CRASH][GRIM] Wrote crash dump: %s\n", dumpPath.string().c_str());
    std::fprintf(stderr, "[CRASH][GRIM] Wrote crash stack: %s\n", textPath.string().c_str());
    std::fflush(stderr);

    return EXCEPTION_EXECUTE_HANDLER;
}

} // namespace

void InstallCrashDumpHandler()
{
    SetUnhandledExceptionFilter(unhandledExceptionFilter);
}

} // namespace GRIM

#else

namespace GRIM {

void InstallCrashDumpHandler() {}

} // namespace GRIM

#endif

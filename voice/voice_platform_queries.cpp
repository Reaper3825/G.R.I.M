#include "voice_platform_queries.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <sapi.h>
#include <sphelper.h>
#pragma comment(lib, "sapi.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "oleaut32.lib")
#pragma comment(lib, "shlwapi.lib")
#endif

namespace GRIM::VoiceCommandServices {
nlohmann::json paths() {
    return {{"test_wav", "resources/test.wav"}, {"embeddings", "D:/G.R.I.M/resources/voices/embeddings"},
        {"temp", "D:/G.R.I.M/resources/tts_out/temp"}, {"cache", "D:/G.R.I.M/resources/tts_out/cache"},
        {"cache_index", "D:/G.R.I.M/resources/tts_out/cache_index.json"}};
}
nlohmann::json listSapiVoices() {
    nlohmann::json result{{"com_ok", false}, {"enumerated", false}, {"count", 0}, {"voices", nlohmann::json::array()}};
#ifdef _WIN32
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (FAILED(hr)) return result;
    result["com_ok"] = true;
    IEnumSpObjectTokens* enumeration = nullptr;
    ULONG count = 0;
    hr = SpEnumTokens(SPCAT_VOICES, nullptr, nullptr, &enumeration);
    if (SUCCEEDED(hr) && enumeration) {
        enumeration->GetCount(&count);
        result["enumerated"] = true; result["count"] = count;
        for (ULONG i = 0; i < count; ++i) {
            ISpObjectToken* token = nullptr;
            if (SUCCEEDED(enumeration->Next(1, &token, nullptr)) && token) {
                WCHAR* description = nullptr;
                if (SUCCEEDED(SpGetDescription(token, &description)) && description) {
                    char buffer[512]; size_t converted = 0;
                    wcstombs_s(&converted, buffer, sizeof(buffer), description, _TRUNCATE);
                    result["voices"].push_back(std::string(buffer));
                    CoTaskMemFree(description);
                }
                token->Release();
            }
        }
    }
    if (enumeration) enumeration->Release();
    CoUninitialize();
#endif
    return result;
}
nlohmann::json sapiOutputDevice() {
    nlohmann::json result{{"com_ok", false}, {"voice_created", false}, {"has_output", false}, {"description", nullptr}};
#ifdef _WIN32
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (FAILED(hr)) return result;
    result["com_ok"] = true;
    ISpVoice* voice = nullptr;
    hr = CoCreateInstance(CLSID_SpVoice, nullptr, CLSCTX_ALL, IID_ISpVoice, reinterpret_cast<void**>(&voice));
    if (FAILED(hr) || !voice) { CoUninitialize(); return result; }
    result["voice_created"] = true;
    ISpObjectToken* output = nullptr;
    hr = voice->GetOutputObjectToken(&output);
    if (SUCCEEDED(hr) && output) {
        result["has_output"] = true;
        WCHAR* description = nullptr;
        if (SUCCEEDED(SpGetDescription(output, &description)) && description) {
            char buffer[512]; size_t converted = 0;
            wcstombs_s(&converted, buffer, sizeof(buffer), description, _TRUNCATE);
            result["description"] = std::string(buffer);
            CoTaskMemFree(description);
        }
        output->Release();
    }
    voice->Release(); CoUninitialize();
#endif
    return result;
}
}

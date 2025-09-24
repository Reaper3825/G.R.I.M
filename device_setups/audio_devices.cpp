#include "audio_devices.hpp"
#include <iostream>

#ifdef _WIN32
    #include <windows.h>
    #include <mmdeviceapi.h>
    #include <functiondiscoverykeys_devpkey.h>
    #pragma comment(lib, "ole32.lib")
#endif

std::vector<std::wstring> getPlaybackDevices() {
    std::vector<std::wstring> devices;
#ifdef _WIN32
    IMMDeviceEnumerator* pEnum = nullptr;
    IMMDeviceCollection* pDevices = nullptr;

    if (FAILED(CoInitialize(nullptr))) return devices;

    if (SUCCEEDED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                                   __uuidof(IMMDeviceEnumerator), (void**)&pEnum))) {
        if (SUCCEEDED(pEnum->EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE, &pDevices))) {
            UINT count;
            pDevices->GetCount(&count);
            for (UINT i = 0; i < count; i++) {
                IMMDevice* pDevice = nullptr;
                IPropertyStore* pStore = nullptr;
                if (SUCCEEDED(pDevices->Item(i, &pDevice))) {
                    if (SUCCEEDED(pDevice->OpenPropertyStore(STGM_READ, &pStore))) {
                        PROPVARIANT varName;
                        PropVariantInit(&varName);
                        if (SUCCEEDED(pStore->GetValue(PKEY_Device_FriendlyName, &varName))) {
                            devices.push_back(varName.pwszVal);
                            PropVariantClear(&varName);
                        }
                        pStore->Release();
                    }
                    pDevice->Release();
                }
            }
            pDevices->Release();
        }
        pEnum->Release();
    }

    CoUninitialize();
#endif
    return devices;
}

void promptForAudioDevice() {
#ifdef _WIN32
    auto devices = getPlaybackDevices();
    if (devices.empty()) {
        std::cout << "[Audio] No playback devices found\n";
        return;
    }

    std::wcout << L"Available playback devices:\n";
    for (size_t i = 0; i < devices.size(); i++) {
        std::wcout << L"  [" << i << L"] " << devices[i] << std::endl;
    }

    std::cout << "Select output device by number: ";
    int choice;
    std::cin >> choice;

    if (choice >= 0 && choice < (int)devices.size()) {
        std::wcout << L"[Audio] You selected: " << devices[choice] << std::endl;
        // 🔹 TODO: implement actual default device switching via IPolicyConfig
    } else {
        std::cout << "[Audio] Invalid selection, keeping current default.\n";
    }
#endif
}

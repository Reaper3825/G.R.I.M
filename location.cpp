#include "location.hpp"
#include "logger.hpp"
#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <windows.h>
#include <Wlanapi.h>
#pragma comment(lib, "wlanapi.lib")
#endif

// =========================================================
// Globals
// =========================================================
LocationInfo g_location;
bool g_wifiConnected = false;

// =========================================================
// GeoIP location fetch (ip-api.com)
// =========================================================
bool fetchLocationByIP(LocationInfo& loc) {
    try {
        LOG_DEBUG("Location", "Starting IP geolocation request...");

        cpr::Session session;
        session.SetOption(cpr::Redirect{true});
        session.SetUrl(cpr::Url{"http://ip-api.com/json/"});
        session.SetTimeout(cpr::Timeout{5000});

        cpr::Response r = session.Get();

        if (r.error) {
            LOG_ERROR("Location",
                std::string("CPR error: ") +
                r.error.message + " (code " + std::to_string((int)r.error.code) + ")");
            return false;
        }

        if (r.status_code != 200) {
            LOG_ERROR("Location",
                "Location fetch failed: HTTP " + std::to_string(r.status_code));
            return false;
        }

        nlohmann::json j = nlohmann::json::parse(r.text, nullptr, false);
        if (j.is_discarded()) {
            LOG_ERROR("Location", "Invalid JSON in IP API response.");
            return false;
        }

        if (!j.contains("status") || j["status"] != "success") {
            std::string msg = j.contains("message") ? j["message"].get<std::string>() : "unknown";
            LOG_ERROR("Location", "GeoIP lookup failed: " + msg);
            return false;
        }

        loc.ip      = j.value("query", "");
        loc.country = j.value("country", "");
        loc.region  = j.value("regionName", "");
        loc.city    = j.value("city", "");
        loc.postal  = j.value("zip", "");
        loc.lat     = j.value("lat", 0.0);
        loc.lon     = j.value("lon", 0.0);
        loc.isp     = j.value("isp", "");

        LOG_DEBUG("Location", "GeoIP success: " + loc.fullAddress());
        return true;
    }
    catch (const nlohmann::json::exception& e) {
        LOG_ERROR("Location", std::string("JSON parse exception: ") + e.what());
    }
    catch (const std::exception& e) {
        LOG_ERROR("Location", std::string("CPR/Network exception: ") + e.what());
    }
    catch (...) {
        LOG_ERROR("Location", "Unknown fatal exception in fetchLocationByIP.");
    }

    return false;
}

// =========================================================
// Wi-Fi connection detection
// =========================================================
#ifdef _WIN32
bool detectWifiConnected() {
    HANDLE hClient = nullptr;
    DWORD dwMaxClient = 2;
    DWORD dwCurVersion = 0;
    DWORD dwResult = WlanOpenHandle(dwMaxClient, NULL, &dwCurVersion, &hClient);
    if (dwResult != ERROR_SUCCESS) return false;

    PWLAN_INTERFACE_INFO_LIST pIfList = nullptr;
    dwResult = WlanEnumInterfaces(hClient, NULL, &pIfList);
    if (dwResult != ERROR_SUCCESS) {
        WlanCloseHandle(hClient, NULL);
        return false;
    }

    bool connected = false;
    for (int i = 0; i < (int)pIfList->dwNumberOfItems; i++) {
        WLAN_INTERFACE_INFO info = pIfList->InterfaceInfo[i];
        if (info.isState == wlan_interface_state_connected) {
            connected = true;
            break;
        }
    }

    if (pIfList != nullptr) WlanFreeMemory(pIfList);
    WlanCloseHandle(hClient, NULL);
    return connected;
}
#else
bool detectWifiConnected() {
    // Stub for non-Windows — always returns false
    return false;
}
#endif

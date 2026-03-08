// Geolocation and network connectivity state.
// Extracted from the old system_detect.hpp — that file is dead.
//======================================================//
#pragma once
#include <string>

// =========================================================
// Wi-Fi connectivity flag (set during bootstrap)
// =========================================================
extern bool g_wifiConnected;

// =========================================================
// Geolocation information (populated via IP-API during bootstrap)
// =========================================================
struct LocationInfo {
    std::string ip;
    std::string country;
    std::string region;
    std::string city;
    std::string postal;
    double lat = 0.0;
    double lon = 0.0;
    std::string isp;

    std::string shortAddress() const {
        if (!city.empty() && !region.empty()) return city + ", " + region;
        if (!city.empty()) return city;
        if (!region.empty()) return region;
        return country.empty() ? "Unknown" : country;
    }

    std::string fullAddress() const {
        std::string out;
        if (!city.empty()) out += city;
        if (!region.empty()) { if (!out.empty()) out += ", "; out += region; }
        if (!postal.empty()) { if (!out.empty()) out += " "; out += postal; }
        if (!country.empty()) { if (!out.empty()) out += ", "; out += country; }
        return out.empty() ? "Unknown location" : out;
    }
};

extern LocationInfo g_location;

// =========================================================
// Bootstrap helpers
// =========================================================

// Detect Wi-Fi connectivity (Windows WLAN API / stubbed on other platforms).
bool detectWifiConnected();

// Fetch geolocation by public IP (ip-api.com). Only call if Wi-Fi connected.
bool fetchLocationByIP(LocationInfo& loc);

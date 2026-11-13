#pragma once

#include <string>
#include <vector>

namespace GRIM { namespace Perception { namespace Physical {

// One IPv4 address bound to this machine.
struct LocalNicAddress {
    std::string interface_name; // e.g. "en0", "Ethernet"
    std::string ipv4;           // bare address, no port
    bool        is_loopback = false;
    bool        is_link_local = false; // 169.254.x.x
};

// Cross-platform NIC enumeration. Implemented per-OS in
//   PhysicalNicScan_macos.mm
//   PhysicalNicScan_win32.cpp
//   PhysicalNicScan_linux.cpp
//
// Throws std::runtime_error on failure with a detailed message including the
// underlying OS error (Rule 20: fail loud).
std::vector<LocalNicAddress> ScanLocalNicAddresses();

}}} // namespace GRIM::Perception::Physical

// Windows implementation of PhysicalNicScan.hpp — IP Helper API.

#include "PhysicalNicScan.hpp"
#include "PhysicalEnvironmentLogTag.hpp"
#include "logger.hpp"

// clang-format off
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
// clang-format on

#include <stdexcept>
#include <string>
#include <vector>

#pragma comment(lib, "iphlpapi.lib")
#pragma comment(lib, "ws2_32.lib")

namespace GRIM { namespace Perception { namespace Physical {

std::vector<LocalNicAddress> ScanLocalNicAddresses() {
    ULONG buf_len = 16 * 1024;
    std::vector<unsigned char> buffer(buf_len);
    PIP_ADAPTER_ADDRESSES head = nullptr;

    DWORD rc = ERROR_BUFFER_OVERFLOW;
    for (int attempt = 0; attempt < 3 && rc == ERROR_BUFFER_OVERFLOW; ++attempt) {
        head = reinterpret_cast<PIP_ADAPTER_ADDRESSES>(buffer.data());
        rc = GetAdaptersAddresses(AF_INET,
                                  GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST
                                      | GAA_FLAG_SKIP_DNS_SERVER,
                                  nullptr, head, &buf_len);
        if (rc == ERROR_BUFFER_OVERFLOW) {
            buffer.assign(buf_len, 0);
        }
    }

    if (rc != NO_ERROR) {
        throw std::runtime_error(
            std::string("ScanLocalNicAddresses(win32): GetAdaptersAddresses failed rc=")
            + std::to_string(rc));
    }

    std::vector<LocalNicAddress> out;
    for (PIP_ADAPTER_ADDRESSES adapter = head; adapter != nullptr; adapter = adapter->Next) {
        if (adapter->OperStatus != IfOperStatusUp) continue;

        const std::string iface_name = adapter->AdapterName ? adapter->AdapterName : "";
        const bool is_loopback = (adapter->IfType == IF_TYPE_SOFTWARE_LOOPBACK);

        for (PIP_ADAPTER_UNICAST_ADDRESS ua = adapter->FirstUnicastAddress;
             ua != nullptr; ua = ua->Next) {
            if (ua->Address.lpSockaddr == nullptr) continue;
            if (ua->Address.lpSockaddr->sa_family != AF_INET) continue;

            char buf[INET_ADDRSTRLEN] = {0};
            auto* sin = reinterpret_cast<sockaddr_in*>(ua->Address.lpSockaddr);
            if (inet_ntop(AF_INET, &sin->sin_addr, buf, sizeof(buf)) == nullptr) {
                continue;
            }

            LocalNicAddress entry;
            entry.interface_name = iface_name;
            entry.ipv4           = buf;
            entry.is_loopback    = is_loopback;
            entry.is_link_local  = (entry.ipv4.rfind("169.254.", 0) == 0);
            out.push_back(std::move(entry));
        }
    }

    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
              "ScanLocalNicAddresses(win32): found " + std::to_string(out.size())
              + " IPv4 interfaces");
    return out;
}

}}} // namespace

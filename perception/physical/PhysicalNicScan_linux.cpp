// Linux implementation of PhysicalNicScan.hpp — getifaddrs(3).

#include "PhysicalNicScan.hpp"
#include "PhysicalEnvironmentLogTag.hpp"
#include "logger.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <stdexcept>
#include <string>

namespace GRIM { namespace Perception { namespace Physical {

std::vector<LocalNicAddress> ScanLocalNicAddresses() {
    struct ifaddrs* head = nullptr;
    if (getifaddrs(&head) != 0 || head == nullptr) {
        throw std::runtime_error(
            std::string("ScanLocalNicAddresses(linux): getifaddrs failed errno=")
            + std::to_string(errno));
    }

    std::vector<LocalNicAddress> out;
    for (struct ifaddrs* ifa = head; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr) continue;
        if (ifa->ifa_addr->sa_family != AF_INET) continue;
        if ((ifa->ifa_flags & IFF_UP) == 0) continue;

        char buf[INET_ADDRSTRLEN] = {0};
        auto* sin = reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr);
        if (inet_ntop(AF_INET, &sin->sin_addr, buf, sizeof(buf)) == nullptr) {
            continue;
        }

        LocalNicAddress entry;
        entry.interface_name = ifa->ifa_name ? ifa->ifa_name : "";
        entry.ipv4           = buf;
        entry.is_loopback    = (ifa->ifa_flags & IFF_LOOPBACK) != 0;
        entry.is_link_local  = (entry.ipv4.rfind("169.254.", 0) == 0);
        out.push_back(std::move(entry));
    }

    freeifaddrs(head);
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
              "ScanLocalNicAddresses(linux): found " + std::to_string(out.size())
              + " IPv4 interfaces");
    return out;
}

}}} // namespace

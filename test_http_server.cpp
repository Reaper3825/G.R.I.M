#include <iostream>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")

#define CPPHTTPLIB_OPENSSL_SUPPORT 0
#include "httplib.h"

int main() {
    std::cout << "Initializing Winsock..." << std::endl;
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed!" << std::endl;
        return 1;
    }
    
    std::cout << "Creating server..." << std::endl;
    httplib::Server server;
    
    server.Get("/test", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("Hello", "text/plain");
    });
    
    std::cout << "About to listen on 127.0.0.1:8888..." << std::endl;
    std::cout.flush();
    std::cerr.flush();
    
    bool result = server.listen("127.0.0.1", 8888);
    
    std::cout << "Listen returned: " << result << std::endl;
    WSACleanup();
    return 0;
}

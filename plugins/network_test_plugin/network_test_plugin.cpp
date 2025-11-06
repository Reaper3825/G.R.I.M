

#include "../core/plugin.hpp"
#include "../core/plugin_api.hpp"
#include "../commands/commands_core.hpp"
#include "../logger.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>
#include <sstream>
#include <ctime>
#include <cstdlib>

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #define closesocket close
#endif

namespace GRIM {
namespace NetworkTest {

    std::atomic<bool> testRunning{false};
    std::vector<std::thread> workers;
    std::atomic<uint64_t> requestsSent{0};
    std::atomic<uint64_t> requestsSucceeded{0};
    std::atomic<uint64_t> requestsFailed{0};
    std::atomic<uint64_t> totalResponseTime{0}; // microseconds
    std::atomic<uint64_t> httpStatusCodes[6]{0}; // 1xx, 2xx, 3xx, 4xx, 5xx, other

    struct TestConfig {
        std::string targetIP;
        std::vector<std::string> targetIPs; // Multiple targets
        uint16_t targetPort = 80;
        int threadCount = 1;
        int requestsPerThread = 100;
        int durationSeconds = 0; // If > 0, run for duration instead of request count
        int delayMs = 0;
        int delayVariance = 0; // Random variance ±ms
        int rampUpSeconds = 0; // Gradually increase threads
        int burstSize = 0; // If > 0, send bursts
        int burstPauseMs = 0;
        int bandwidthLimitKBps = 0; // 0 = unlimited
        std::string protocol = "TCP"; // TCP, HTTP, UDP
        std::string httpMethod = "GET"; // GET, POST, PUT
        std::string httpHeaders; // Custom headers separated by |
        std::string httpBody; // For POST/PUT
        std::string payloadFile; // File with random payloads
        std::string targetListFile; // File with target IPs
        std::string outputFile; // CSV/JSON output
        bool keepAlive = false; // HTTP keep-alive
        bool randomPayload = false;
    };

    void initializeWinsock() {
#ifdef _WIN32
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            LOG_ERROR("NetworkTest", "WSAStartup failed");
        }
#endif
    }

    void cleanupWinsock() {
#ifdef _WIN32
        WSACleanup();
#endif
    }

    bool sendTcpRequest(const std::string& ip, uint16_t port) {
        int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (sock < 0) {
            return false;
        }

        struct sockaddr_in serverAddr;
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(port);
        
        if (inet_pton(AF_INET, ip.c_str(), &serverAddr.sin_addr) <= 0) {
            closesocket(sock);
            return false;
        }

        // Set socket timeout
        struct timeval tv;
        tv.tv_sec = 5;
        tv.tv_usec = 0;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));

        bool success = (connect(sock, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) >= 0);
        
        closesocket(sock);
        return success;
    }

    int extractHttpStatusCode(const std::string& response) {
        if (response.size() < 12) return 0;
        size_t pos = response.find("HTTP/");
        if (pos == std::string::npos) return 0;
        pos = response.find(' ', pos);
        if (pos == std::string::npos) return 0;
        return std::atoi(response.substr(pos + 1, 3).c_str());
    }

    void updateStatusCodeCounter(int statusCode) {
        if (statusCode >= 100 && statusCode < 200) httpStatusCodes[0]++;
        else if (statusCode >= 200 && statusCode < 300) httpStatusCodes[1]++;
        else if (statusCode >= 300 && statusCode < 400) httpStatusCodes[2]++;
        else if (statusCode >= 400 && statusCode < 500) httpStatusCodes[3]++;
        else if (statusCode >= 500 && statusCode < 600) httpStatusCodes[4]++;
        else httpStatusCodes[5]++;
    }

    bool sendHttpRequest(const std::string& ip, uint16_t port, const TestConfig& config, int64_t* responseTimeUs = nullptr) {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (sock < 0) {
            return false;
        }

        struct sockaddr_in serverAddr;
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(port);
        
        if (inet_pton(AF_INET, ip.c_str(), &serverAddr.sin_addr) <= 0) {
            closesocket(sock);
            return false;
        }

        struct timeval tv;
        tv.tv_sec = 5;
        tv.tv_usec = 0;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));

        if (connect(sock, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
            closesocket(sock);
            return false;
        }

        // Build HTTP request
        std::ostringstream request;
        request << config.httpMethod << " / HTTP/1.1\r\n";
        request << "Host: " << ip << "\r\n";
        
        // Custom headers
        if (!config.httpHeaders.empty()) {
            std::istringstream headerStream(config.httpHeaders);
            std::string header;
            while (std::getline(headerStream, header, '|')) {
                request << header << "\r\n";
            }
        }
        
        if (config.keepAlive) {
            request << "Connection: keep-alive\r\n";
        } else {
            request << "Connection: close\r\n";
        }
        
        // Body for POST/PUT
        if (!config.httpBody.empty()) {
            request << "Content-Length: " << config.httpBody.length() << "\r\n";
            request << "\r\n";
            request << config.httpBody;
        } else {
            request << "\r\n";
        }
        
        std::string reqStr = request.str();
        bool sendSuccess = (send(sock, reqStr.c_str(), reqStr.length(), 0) >= 0);
        
        // Read response to get status code
        char buffer[4096];
        std::string response;
        int bytesRead = recv(sock, buffer, sizeof(buffer) - 1, 0);
        if (bytesRead > 0) {
            buffer[bytesRead] = '\0';
            response = buffer;
            int statusCode = extractHttpStatusCode(response);
            updateStatusCodeCounter(statusCode);
        }
        
        closesocket(sock);
        
        auto endTime = std::chrono::high_resolution_clock::now();
        if (responseTimeUs) {
            *responseTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
        }
        
        return sendSuccess;
    }

    bool sendUdpRequest(const std::string& ip, uint16_t port, const std::string& payload) {
        int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (sock < 0) {
            return false;
        }

        struct sockaddr_in serverAddr;
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(port);
        
        if (inet_pton(AF_INET, ip.c_str(), &serverAddr.sin_addr) <= 0) {
            closesocket(sock);
            return false;
        }

        const char* data = payload.empty() ? "STRESS_TEST_PAYLOAD" : payload.c_str();
        bool success = (sendto(sock, data, strlen(data), 0, 
                              (struct sockaddr*)&serverAddr, sizeof(serverAddr)) >= 0);
        
        closesocket(sock);
        return success;
    }

    void workerThread(TestConfig config, int threadId) {
        LOG_DEBUG("NetworkTest", "Worker thread " + std::to_string(threadId) + " started");
        
        // Ramp-up delay
        if (config.rampUpSeconds > 0) {
            int delayMs = (threadId * config.rampUpSeconds * 1000) / config.threadCount;
            std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
        }
        
        std::srand(std::time(nullptr) + threadId);
        auto startTime = std::chrono::steady_clock::now();
        
        int requestCount = 0;
        while (testRunning) {
            // Check duration-based exit
            if (config.durationSeconds > 0) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - startTime).count();
                if (elapsed >= config.durationSeconds) break;
            } else if (requestCount >= config.requestsPerThread) {
                break;
            }
            
            // Select target (round-robin if multiple)
            std::string targetIP = config.targetIP;
            if (!config.targetIPs.empty()) {
                targetIP = config.targetIPs[requestCount % config.targetIPs.size()];
            }
            
            bool success = false;
            int64_t responseTime = 0;
            
            if (config.protocol == "UDP") {
                std::string payload = config.randomPayload ? 
                    std::to_string(std::rand()) : config.httpBody;
                success = sendUdpRequest(targetIP, config.targetPort, payload);
            } else if (config.protocol == "HTTP") {
                success = sendHttpRequest(targetIP, config.targetPort, config, &responseTime);
            } else {
                success = sendTcpRequest(targetIP, config.targetPort);
            }
            
            requestsSent++;
            requestCount++;
            if (success) {
                requestsSucceeded++;
                if (responseTime > 0) {
                    totalResponseTime += responseTime;
                }
            } else {
                requestsFailed++;
            }
            
            // Burst mode
            if (config.burstSize > 0 && requestCount % config.burstSize == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(config.burstPauseMs));
            }
            
            // Delay with variance
            if (config.delayMs > 0 || config.delayVariance > 0) {
                int actualDelay = config.delayMs;
                if (config.delayVariance > 0) {
                    actualDelay += (std::rand() % (config.delayVariance * 2)) - config.delayVariance;
                    if (actualDelay < 0) actualDelay = 0;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(actualDelay));
            }
            
            // Bandwidth throttling
            if (config.bandwidthLimitKBps > 0) {
                // Simple throttle: sleep based on data sent
                int bytesPerRequest = 100; // Approximate
                int sleepMs = (bytesPerRequest * 1000) / (config.bandwidthLimitKBps * 1024);
                std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
            }
        }
        
        LOG_DEBUG("NetworkTest", "Worker thread " + std::to_string(threadId) + " completed");
    }

    void stopTest() {
        if (!testRunning) {
            return;
        }
        
        LOG_DEBUG("NetworkTest", "Stopping network test...");
        testRunning = false;
        
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        
        workers.clear();
        cleanupWinsock();
    }

    CommandResult cmdNetworkTest(const std::string& arg) {
        // Parse arguments with extended options
        std::istringstream iss(arg);
        TestConfig config;
        
        if (!(iss >> config.targetIP)) {
            return CommandResult{
                false,
                "❌ NETWORK STRESS TESTER (Advanced)\n\n"
                "⚠️  WARNING: Use ONLY on authorized systems\n\n"
                "Usage: networktest <IP|@file> <port> [options...]\n\n"
                "Basic Options:\n"
                "  threads=N        Number of threads (default: 1)\n"
                "  requests=N       Requests per thread (default: 100)\n"
                "  duration=N       Run for N seconds (overrides requests)\n"
                "  delay=N          Delay between requests in ms (default: 0)\n"
                "  variance=N       Random delay variance ±N ms\n"
                "  protocol=X       TCP/HTTP/UDP (default: TCP)\n\n"
                "HTTP Options:\n"
                "  method=X         GET/POST/PUT (default: GET)\n"
                "  headers=X        Custom headers (pipe-separated: User-Agent: Bot|Cookie: x=y)\n"
                "  body=X           Request body for POST/PUT\n"
                "  keepalive        Enable HTTP keep-alive\n\n"
                "Advanced:\n"
                "  rampup=N         Ramp up threads over N seconds\n"
                "  burst=N          Send N requests then pause\n"
                "  burstpause=N     Pause N ms between bursts\n"
                "  bandwidth=N      Limit bandwidth to N KB/s\n"
                "  random           Use random payloads\n"
                "  output=file.csv  Export results to CSV/JSON\n\n"
                "Examples:\n"
                "  networktest 127.0.0.1 8080 threads=10 requests=1000\n"
                "  networktest 192.168.1.100 80 protocol=HTTP method=POST body=test\n"
                "  networktest @targets.txt 443 duration=60 protocol=HTTP\n"
                "  networktest 10.0.0.1 53 protocol=UDP threads=50 burst=100 burstpause=1000",
                "",
                "information"
            };
        }
        
        // Check if target is a file (starts with @)
        if (config.targetIP[0] == '@') {
            config.targetListFile = config.targetIP.substr(1);
            // Load IPs from file
            std::ifstream file(config.targetListFile);
            if (!file.is_open()) {
                return CommandResult{false, "Could not open target file: " + config.targetListFile, "", "error"};
            }
            std::string line;
            while (std::getline(file, line)) {
                if (!line.empty() && line[0] != '#') {
                    config.targetIPs.push_back(line);
                }
            }
            if (config.targetIPs.empty()) {
                return CommandResult{false, "No valid targets in file", "", "error"};
            }
            config.targetIP = config.targetIPs[0]; // Use first as default
        }
        
        iss >> config.targetPort;
        
        // Parse key=value options
        std::string option;
        while (iss >> option) {
            if (option == "keepalive") {
                config.keepAlive = true;
            } else if (option == "random") {
                config.randomPayload = true;
            } else {
                size_t eqPos = option.find('=');
                if (eqPos != std::string::npos) {
                    std::string key = option.substr(0, eqPos);
                    std::string value = option.substr(eqPos + 1);
                    
                    if (key == "threads") config.threadCount = std::stoi(value);
                    else if (key == "requests") config.requestsPerThread = std::stoi(value);
                    else if (key == "duration") config.durationSeconds = std::stoi(value);
                    else if (key == "delay") config.delayMs = std::stoi(value);
                    else if (key == "variance") config.delayVariance = std::stoi(value);
                    else if (key == "protocol") config.protocol = value;
                    else if (key == "method") config.httpMethod = value;
                    else if (key == "headers") config.httpHeaders = value;
                    else if (key == "body") config.httpBody = value;
                    else if (key == "rampup") config.rampUpSeconds = std::stoi(value);
                    else if (key == "burst") config.burstSize = std::stoi(value);
                    else if (key == "burstpause") config.burstPauseMs = std::stoi(value);
                    else if (key == "bandwidth") config.bandwidthLimitKBps = std::stoi(value);
                    else if (key == "output") config.outputFile = value;
                }
            }
        }
        
        // Validate IP address (skip if using file)
        if (config.targetListFile.empty()) {
            struct sockaddr_in sa;
            if (inet_pton(AF_INET, config.targetIP.c_str(), &(sa.sin_addr)) != 1) {
                return CommandResult{false, "Invalid IP address format", "", "error"};
            }
        }
        
        // Stop any existing test
        if (testRunning) {
            stopTest();
        }
        
        // Reset counters
        requestsSent = 0;
        requestsSucceeded = 0;
        requestsFailed = 0;
        totalResponseTime = 0;
        for (int i = 0; i < 6; ++i) httpStatusCodes[i] = 0;
        
        // Display configuration
        std::ostringstream msg;
        msg << "⚠️  NETWORK STRESS TEST ⚠️\n\n";
        if (!config.targetIPs.empty()) {
            msg << "Targets: " << config.targetIPs.size() << " IPs from " << config.targetListFile << "\n";
        } else {
            msg << "Target: " << config.targetIP << ":" << config.targetPort << "\n";
        }
        msg << "Protocol: " << config.protocol;
        if (config.protocol == "HTTP") {
            msg << " (" << config.httpMethod << ")";
        }
        msg << "\nThreads: " << config.threadCount;
        if (config.rampUpSeconds > 0) {
            msg << " (ramp up: " << config.rampUpSeconds << "s)";
        }
        msg << "\n";
        
        if (config.durationSeconds > 0) {
            msg << "Duration: " << config.durationSeconds << " seconds\n";
        } else {
            msg << "Requests per thread: " << config.requestsPerThread << "\n";
            msg << "Total requests: " << (config.threadCount * config.requestsPerThread) << "\n";
        }
        
        if (config.delayMs > 0 || config.delayVariance > 0) {
            msg << "Delay: " << config.delayMs << "ms";
            if (config.delayVariance > 0) msg << " ±" << config.delayVariance << "ms";
            msg << "\n";
        }
        
        if (config.burstSize > 0) {
            msg << "Burst mode: " << config.burstSize << " requests, " 
                << config.burstPauseMs << "ms pause\n";
        }
        
        if (config.bandwidthLimitKBps > 0) {
            msg << "Bandwidth limit: " << config.bandwidthLimitKBps << " KB/s\n";
        }
        
        msg << "\nStarting test...\n";
        LOG_DEBUG("NetworkTest", msg.str());
        
        // Initialize networking
        initializeWinsock();
        
        // Start test
        testRunning = true;
        LOG_DEBUG("NetworkTest", "Starting network stress test...");
        
        auto startTime = std::chrono::steady_clock::now();
        
        for (int i = 0; i < config.threadCount; ++i) {
            workers.emplace_back(workerThread, config, i);
        }
        
        // Monitor progress
        std::thread monitor([config, startTime]() {
            while (testRunning) {
                std::this_thread::sleep_for(std::chrono::seconds(2));
                
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - startTime).count();
                
                std::ostringstream status;
                status << "Progress: " << requestsSent.load() << " sent, "
                       << requestsSucceeded.load() << " succeeded, "
                       << requestsFailed.load() << " failed (Elapsed: " << elapsed << "s)";
                LOG_DEBUG("NetworkTest", status.str());
            }
        });
        
        // Wait for completion in background
        std::thread completion([config, startTime, mon = std::move(monitor)]() mutable {
            for (auto& worker : workers) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
            
            testRunning = false;
            if (mon.joinable()) {
                mon.join();
            }
            
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - startTime).count();
            
            std::ostringstream final;
            final << "\n✅ NETWORK STRESS TEST COMPLETED\n\n"
                  << "Total requests sent: " << requestsSent.load() << "\n"
                  << "Successful: " << requestsSucceeded.load() << "\n"
                  << "Failed: " << requestsFailed.load() << "\n"
                  << "Duration: " << (elapsed / 1000.0) << " seconds\n"
                  << "Requests/sec: " << (requestsSent.load() * 1000.0 / elapsed) << "\n";
            
            if (totalResponseTime.load() > 0 && requestsSucceeded.load() > 0) {
                double avgResponseMs = (totalResponseTime.load() / 1000.0) / requestsSucceeded.load();
                final << "Avg Response Time: " << avgResponseMs << " ms\n";
            }
            
            // HTTP status code breakdown
            if (config.protocol == "HTTP") {
                final << "\nHTTP Status Codes:\n";
                if (httpStatusCodes[0] > 0) final << "  1xx: " << httpStatusCodes[0].load() << "\n";
                if (httpStatusCodes[1] > 0) final << "  2xx: " << httpStatusCodes[1].load() << "\n";
                if (httpStatusCodes[2] > 0) final << "  3xx: " << httpStatusCodes[2].load() << "\n";
                if (httpStatusCodes[3] > 0) final << "  4xx: " << httpStatusCodes[3].load() << "\n";
                if (httpStatusCodes[4] > 0) final << "  5xx: " << httpStatusCodes[4].load() << "\n";
                if (httpStatusCodes[5] > 0) final << "  Other: " << httpStatusCodes[5].load() << "\n";
            }
            
            LOG_DEBUG("NetworkTest", final.str());
            
            // Export to file if specified
            if (!config.outputFile.empty()) {
                std::ofstream outFile(config.outputFile);
                if (outFile.is_open()) {
                    bool isJson = config.outputFile.find(".json") != std::string::npos;
                    
                    if (isJson) {
                        outFile << "{\n";
                        outFile << "  \"total_requests\": " << requestsSent.load() << ",\n";
                        outFile << "  \"successful\": " << requestsSucceeded.load() << ",\n";
                        outFile << "  \"failed\": " << requestsFailed.load() << ",\n";
                        outFile << "  \"duration_seconds\": " << (elapsed / 1000.0) << ",\n";
                        outFile << "  \"requests_per_second\": " << (requestsSent.load() * 1000.0 / elapsed) << ",\n";
                        if (totalResponseTime.load() > 0 && requestsSucceeded.load() > 0) {
                            outFile << "  \"avg_response_ms\": " << ((totalResponseTime.load() / 1000.0) / requestsSucceeded.load()) << ",\n";
                        }
                        outFile << "  \"http_status_codes\": {\n";
                        outFile << "    \"1xx\": " << httpStatusCodes[0].load() << ",\n";
                        outFile << "    \"2xx\": " << httpStatusCodes[1].load() << ",\n";
                        outFile << "    \"3xx\": " << httpStatusCodes[2].load() << ",\n";
                        outFile << "    \"4xx\": " << httpStatusCodes[3].load() << ",\n";
                        outFile << "    \"5xx\": " << httpStatusCodes[4].load() << ",\n";
                        outFile << "    \"other\": " << httpStatusCodes[5].load() << "\n";
                        outFile << "  }\n}\n";
                    } else {
                        // CSV format
                        outFile << "Metric,Value\n";
                        outFile << "Total Requests," << requestsSent.load() << "\n";
                        outFile << "Successful," << requestsSucceeded.load() << "\n";
                        outFile << "Failed," << requestsFailed.load() << "\n";
                        outFile << "Duration (seconds)," << (elapsed / 1000.0) << "\n";
                        outFile << "Requests/sec," << (requestsSent.load() * 1000.0 / elapsed) << "\n";
                        if (totalResponseTime.load() > 0 && requestsSucceeded.load() > 0) {
                            outFile << "Avg Response (ms)," << ((totalResponseTime.load() / 1000.0) / requestsSucceeded.load()) << "\n";
                        }
                        outFile << "HTTP 1xx," << httpStatusCodes[0].load() << "\n";
                        outFile << "HTTP 2xx," << httpStatusCodes[1].load() << "\n";
                        outFile << "HTTP 3xx," << httpStatusCodes[2].load() << "\n";
                        outFile << "HTTP 4xx," << httpStatusCodes[3].load() << "\n";
                        outFile << "HTTP 5xx," << httpStatusCodes[4].load() << "\n";
                        outFile << "HTTP Other," << httpStatusCodes[5].load() << "\n";
                    }
                    outFile.close();
                    LOG_DEBUG("NetworkTest", "Results exported to: " + config.outputFile);
                }
            }
            
            cleanupWinsock();
        });
        completion.detach();
        
        return CommandResult{
            true,
            "Network stress test started. Check logs for progress.\nUse 'networktest_stop' to abort.",
            "",
            "action"
        };
    }

    CommandResult cmdNetworkTestStop(const std::string& arg) {
        if (!testRunning) {
            return CommandResult{false, "No network test is currently running", "", "information"};
        }
        
        stopTest();
        
        return CommandResult{
            true,
            "Network test stopped.\nFinal stats: " + std::to_string(requestsSent.load()) + " sent, " +
            std::to_string(requestsSucceeded.load()) + " succeeded, " +
            std::to_string(requestsFailed.load()) + " failed",
            "",
            "action"
        };
    }

} // namespace NetworkTest
} // namespace GRIM

// ============================================================================
// New Plugin API Implementation
// ============================================================================

static const GrimPluginAPI* g_api = nullptr;

// Command handler wrappers for new API
static GrimCommandResult networkTestHandler(const char* input, void* user_data) {
    CommandResult result = GRIM::NetworkTest::cmdNetworkTest(input ? input : "");
    
    GrimCommandResult apiResult;
    apiResult.success = result.success;
    apiResult.message = result.message.c_str();
    apiResult.error_code = result.errorCode.empty() ? nullptr : result.errorCode.c_str();
    apiResult.category = result.category.c_str();
    apiResult.color_rgb = (result.color.r << 16) | (result.color.g << 8) | result.color.b;
    
    return apiResult;
}

static GrimCommandResult networkTestStopHandler(const char* input, void* user_data) {
    CommandResult result = GRIM::NetworkTest::cmdNetworkTestStop(input ? input : "");
    
    GrimCommandResult apiResult;
    apiResult.success = result.success;
    apiResult.message = result.message.c_str();
    apiResult.error_code = result.errorCode.empty() ? nullptr : result.errorCode.c_str();
    apiResult.category = result.category.c_str();
    apiResult.color_rgb = (result.color.r << 16) | (result.color.g << 8) | result.color.b;
    
    return apiResult;
}

// Plugin metadata
extern "C" PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
    static GrimPluginInfo info;
    info.api_version = GRIM_API_VERSION;
    info.name = "Network Test Plugin";
    info.version = "1.0.0";
    info.author = "GRIM";
    info.description = "Advanced network stress testing and load generation tool";
    info.required_permissions = GRIM_PERM_NETWORK;
    return &info;
}

// Initialize plugin
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if (!api) {
        return GRIM_ERROR_INVALID_PARAM;
    }
    
    if (api->api_version < GRIM_API_VERSION) {
        return GRIM_ERROR_API_VERSION_MISMATCH;
    }
    
    g_api = api;
    g_api->log(GRIM_LOG_INFO, "Network Test Plugin initializing...");
    
    // Register commands
    GrimResult res = g_api->register_command("networktest", networkTestHandler, nullptr);
    if (res != GRIM_OK) {
        g_api->log(GRIM_LOG_ERROR, "Failed to register 'networktest' command");
        return res;
    }
    
    res = g_api->register_command("networktest_stop", networkTestStopHandler, nullptr);
    if (res != GRIM_OK) {
        g_api->log(GRIM_LOG_ERROR, "Failed to register 'networktest_stop' command");
        g_api->unregister_command("networktest");
        return res;
    }
    
    g_api->log(GRIM_LOG_INFO, "Network Test Plugin loaded successfully");
    return GRIM_OK;
}

// Shutdown plugin
extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {
    if (g_api) {
        // Stop any running test
        GRIM::NetworkTest::stopTest();
        
        g_api->unregister_command("networktest");
        g_api->unregister_command("networktest_stop");
        g_api->log(GRIM_LOG_INFO, "Network Test Plugin unloaded");
    }
}

#pragma once

#include <string>
#include <iostream>
#include "httplib.h"
#include "flatbuffers/flatbuffers.h"
#include "data_collection_protocol_generated.h"

namespace GRIM {
namespace DataCollection {

// Client-side status info (not conflicting with FlatBuffer types)
struct ClientCollectionStatus {
    bool isCollecting = false;
    float progress = 0.0f;
    std::string phase;
    std::string error;
    bool valid = false;  // Flag to indicate if this status is valid (not from failed HTTP request)
};

struct ClientCollectionResult {
    bool success = false;
    std::string message;
    std::string error;
};

class DataCollectionClient {
public:
    DataCollectionClient(const std::string& host = "localhost", int port = 11437)
        : host_(host), port_(port) {
        std::cout << "[DataCollectionClient] Initialized for " << host << ":" << port << std::endl;
    }

    bool isServerRunning() {
        try {
            httplib::Client client(host_, port_);
            client.set_connection_timeout(0, 50000);   // 50ms connection timeout
            client.set_read_timeout(0, 100000);        // 100ms read timeout
            
            auto res = client.Get("/health");
            return res && res->status == 200;
        } catch (...) {
            return false;
        }
    }

    ClientCollectionResult startCollection(const std::string& mode = "full") {
        ClientCollectionResult result;

        std::cout << "\n[DataCollectionClient] >>> HTTP POST REQUEST INITIATED <<<" << std::endl;
        std::cout << "[DataCollectionClient]     FROM: DataCollectionClient::startCollection()" << std::endl;
        std::cout << "[DataCollectionClient]     TO: " << host_ << ":" << port_ << "/api/collection/start" << std::endl;
        std::cout << "[DataCollectionClient]     MODE: " << mode << std::endl;

        try {
            httplib::Client client(host_, port_);
            client.set_read_timeout(10); // Quick response since collection runs async

            std::cout << "[DataCollectionClient] Building FlatBuffer request..." << std::endl;
            flatbuffers::FlatBufferBuilder builder(256);
            auto modeStr = builder.CreateString(mode);
            auto request = GRIM::DataCollection::CreateDataCollectionRequest(builder, modeStr);
            builder.Finish(request);

            std::cout << "[DataCollectionClient] Sending HTTP POST (size: " << builder.GetSize() << " bytes)..." << std::endl;
            auto res = client.Post("/api/collection/start",
                                  reinterpret_cast<const char*>(builder.GetBufferPointer()),
                                  builder.GetSize(),
                                  "application/octet-stream");

            if (!res) {
                std::cout << "[DataCollectionClient] ERROR: No response from server" << std::endl;
                result.error = "Failed to contact data collection server";
                return result;
            }

            std::cout << "[DataCollectionClient] <<< HTTP RESPONSE RECEIVED <<<" << std::endl;
            std::cout << "[DataCollectionClient]     STATUS: " << res->status << std::endl;

            if (res->status == 202 || res->status == 200) {
                auto response = flatbuffers::GetRoot<GRIM::DataCollection::DataCollectionResponse>(
                    reinterpret_cast<const uint8_t*>(res->body.data())
                );

                result.success = response->success();
                result.message = response->message() ? response->message()->str() : "";
                result.error = response->error() ? response->error()->str() : "";

                std::cout << "[DataCollectionClient]     SUCCESS: " << (result.success ? "true" : "false") << std::endl;
                std::cout << "[DataCollectionClient]     MESSAGE: " << result.message << std::endl;
            } else {
                result.error = "Server returned status: " + std::to_string(res->status);
                std::cout << "[DataCollectionClient]     ERROR: " << result.error << std::endl;
            }

            return result;
        } catch (const std::exception& e) {
            std::cout << "[DataCollectionClient] EXCEPTION: " << e.what() << std::endl;
            result.error = std::string("Exception: ") + e.what();
            return result;
        }
    }

    ClientCollectionStatus getStatus() {
        ClientCollectionStatus status;

        try {
            httplib::Client client(host_, port_);
            client.set_connection_timeout(0, 50000);   // 50ms connection timeout (was 100ms)
            client.set_read_timeout(0, 100000);        // 100ms read timeout (was 200ms)

            auto res = client.Get("/api/collection/status");

            if (!res || res->status != 200) {
                std::cout << "[DataCollectionClient] getStatus() - No response or bad status" << std::endl;
                status.valid = false;  // Mark status as invalid
                return status;
            }

            status.valid = true;  // Mark status as valid

            // Parse JSON response manually (simple fields)
            std::string body = res->body;
            // Only log status on changes, not every poll
            static std::string lastStatus;
            if (body != lastStatus) {
                std::cout << "[DataCollectionClient] Status changed: " << body << std::endl;
                lastStatus = body;
            }
            
            // Extract status
            size_t statusPos = body.find("\"status\":\"");
            if (statusPos != std::string::npos) {
                statusPos += 10;
                size_t endPos = body.find("\"", statusPos);
                std::string statusStr = body.substr(statusPos, endPos - statusPos);
                status.isCollecting = (statusStr == "collecting");
            }

            // Extract progress
            size_t progressPos = body.find("\"progress\":");
            if (progressPos != std::string::npos) {
                progressPos += 11;
                size_t endPos = body.find_first_of(",}", progressPos);
                std::string progressStr = body.substr(progressPos, endPos - progressPos);
                status.progress = std::stof(progressStr);
            }

            // Extract phase
            size_t phasePos = body.find("\"phase\":\"");
            if (phasePos != std::string::npos) {
                phasePos += 9;
                size_t endPos = body.find("\"", phasePos);
                status.phase = body.substr(phasePos, endPos - phasePos);
            }

            // Extract error
            size_t errorPos = body.find("\"error\":\"");
            if (errorPos != std::string::npos) {
                errorPos += 9;
                size_t endPos = body.find("\"", errorPos);
                status.error = body.substr(errorPos, endPos - errorPos);
            }

        } catch (...) {
            // Return invalid status on error
            status.valid = false;
        }

        return status;
    }

    bool stopCollection() {
        try {
            httplib::Client client(host_, port_);
            client.set_read_timeout(5);

            auto res = client.Post("/api/collection/stop", "", "application/json");
            return res && res->status == 200;
        } catch (...) {
            return false;
        }
    }

    bool shutdownServer() {
        std::cout << "[DataCollectionClient] Sending shutdown request to server..." << std::endl;
        try {
            httplib::Client client(host_, port_);
            client.set_read_timeout(5);

            auto res = client.Post("/shutdown", "", "application/json");
            bool success = res && res->status == 200;
            
            if (success) {
                std::cout << "[DataCollectionClient] Server shutdown initiated successfully" << std::endl;
            } else {
                std::cout << "[DataCollectionClient] Server shutdown request failed" << std::endl;
            }
            
            return success;
        } catch (const std::exception& e) {
            std::cout << "[DataCollectionClient] Exception during shutdown: " << e.what() << std::endl;
            return false;
        }
    }

private:
    std::string host_;
    int port_;
};

} // namespace DataCollection
} // namespace GRIM

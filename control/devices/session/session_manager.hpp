#pragma once

#include "device_session.hpp"
#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace GRIM {

// ─────────────────────────────────────────────────────────
//  SessionManager — runtime device session tracking
//
//  Thread-safe.  NOT persisted — sessions exist only while
//  the connection is live.  Stale sessions are dropped
//  when heartbeat_timeout_sec elapses without a heartbeat.
// ─────────────────────────────────────────────────────────

class SessionManager {
public:
    explicit SessionManager(int heartbeat_timeout_sec = 30);

    void addSession(const DeviceSession& session);
    void removeSession(const std::string& session_id);
    void removeByDeviceId(const std::string& device_id);

    void updateHeartbeat(const std::string& device_id,
                         const std::vector<std::string>& capabilities);
    void setTransferState(const std::string& device_id, TransferState state);

    const DeviceSession* findByDeviceId(const std::string& device_id) const;
    bool isOnline(const std::string& device_id) const;

    std::vector<DeviceSession> listOnlineSessions() const;

    // Drop sessions that haven't sent a heartbeat within timeout
    std::vector<std::string> purgeStale();

private:
    int heartbeat_timeout_sec_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, DeviceSession> sessions_by_device_id_;
};

} // namespace GRIM

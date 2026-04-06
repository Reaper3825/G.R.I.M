#include "session_manager.hpp"
#include "logger.hpp"

namespace GRIM {

static constexpr const char* TAG = "SessionManager";

SessionManager::SessionManager(int heartbeat_timeout_sec)
    : heartbeat_timeout_sec_(heartbeat_timeout_sec) {}

void SessionManager::addSession(const DeviceSession& session) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (sessions_by_device_id_.count(session.device_id))
        throw std::runtime_error(
            "SessionManager: device '" + session.device_id + "' already has an active session");

    sessions_by_device_id_[session.device_id] = session;
    LOG_DEBUG(TAG, "Session added for device: " + session.device_id);
}

void SessionManager::removeSession(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto it = sessions_by_device_id_.begin(); it != sessions_by_device_id_.end(); ++it) {
        if (it->second.session_id == session_id) {
            std::string device_id = it->second.device_id;
            sessions_by_device_id_.erase(it);
            LOG_DEBUG(TAG, "Session removed for device: " + device_id);
            return;
        }
    }
}

void SessionManager::removeByDeviceId(const std::string& device_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto erased = sessions_by_device_id_.erase(device_id);
    if (erased)
        LOG_DEBUG(TAG, "Session removed for device: " + device_id);
}

void SessionManager::updateHeartbeat(const std::string& device_id,
                                     const std::vector<std::string>& capabilities) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sessions_by_device_id_.find(device_id);
    if (it == sessions_by_device_id_.end())
        throw std::runtime_error("SessionManager::updateHeartbeat: no session for device '" + device_id + "'");

    it->second.last_heartbeat_at = std::chrono::steady_clock::now();
    it->second.available_capabilities = capabilities;
}

void SessionManager::setTransferState(const std::string& device_id, TransferState state) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sessions_by_device_id_.find(device_id);
    if (it == sessions_by_device_id_.end())
        throw std::runtime_error("SessionManager::setTransferState: no session for device '" + device_id + "'");

    it->second.transfer_state = state;
}

const DeviceSession* SessionManager::findByDeviceId(const std::string& device_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sessions_by_device_id_.find(device_id);
    if (it == sessions_by_device_id_.end()) return nullptr;
    return &it->second;
}

bool SessionManager::isOnline(const std::string& device_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sessions_by_device_id_.count(device_id) > 0;
}

std::vector<DeviceSession> SessionManager::listOnlineSessions() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<DeviceSession> result;
    result.reserve(sessions_by_device_id_.size());
    for (const auto& [id, session] : sessions_by_device_id_) {
        result.push_back(session);
    }
    return result;
}

std::vector<std::string> SessionManager::purgeStale() {
    std::lock_guard<std::mutex> lock(mutex_);
    auto now = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(heartbeat_timeout_sec_);

    std::vector<std::string> purged;
    for (auto it = sessions_by_device_id_.begin(); it != sessions_by_device_id_.end(); ) {
        if (now - it->second.last_heartbeat_at > timeout) {
            purged.push_back(it->second.device_id);
            LOG_DEBUG(TAG, "Purging stale session for device: " + it->second.device_id);
            it = sessions_by_device_id_.erase(it);
        } else {
            ++it;
        }
    }
    return purged;
}

} // namespace GRIM

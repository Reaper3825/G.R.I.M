#pragma once

#include "device_record.hpp"
#include "../session/device_session.hpp"
#include <algorithm>
#include <string>
#include <vector>

namespace GRIM {

// ─────────────────────────────────────────────────────────
//  DeviceSnapshot — joined read-only view for UI consumption
//
//  Built by combining DeviceRecord (persistent) with
//  DeviceSession (runtime).  UI MUST only read snapshots,
//  never raw records or sessions.
// ─────────────────────────────────────────────────────────

struct DeviceSnapshot {
    // From DeviceRecord
    std::string              device_id;
    std::string              device_name;
    DeviceType               device_type;
    Platform                 platform;
    PairingState             pairing_state;
    std::string              pairing_code;
    uint32_t                 roles;
    std::vector<std::string> allowed_actions;
    std::string              registered_at;
    std::string              last_seen_at;

    // From DeviceSession (runtime)
    bool                     is_online = false;
    std::vector<std::string> available_capabilities;
    TransferState            transfer_state = TransferState::Idle;

    // Derived: actions that can actually be routed to this device right now
    std::vector<std::string> routeable_actions;

    // ─── Factory ─────────────────────────────────────────

    static DeviceSnapshot fromRecord(const DeviceRecord& rec) {
        DeviceSnapshot snap;
        snap.device_id       = rec.device_id;
        snap.device_name     = rec.device_name;
        snap.device_type     = rec.device_type;
        snap.platform        = rec.platform;
        snap.pairing_state   = rec.pairing_state;
        snap.pairing_code    = rec.pairing_code;
        snap.roles           = rec.roles;
        snap.allowed_actions = rec.allowed_actions;
        snap.registered_at   = rec.registered_at;
        snap.last_seen_at    = rec.last_seen_at;
        snap.is_online       = false;
        return snap;
    }

    static DeviceSnapshot fromJoin(const DeviceRecord& rec,
                                   const DeviceSession& sess) {
        DeviceSnapshot snap = fromRecord(rec);
        snap.is_online              = true;
        snap.available_capabilities = sess.available_capabilities;
        snap.transfer_state         = sess.transfer_state;

        // routeable = allowed ∩ available (only if paired + online)
        if (rec.pairing_state == PairingState::Paired) {
            for (const auto& action : rec.allowed_actions) {
                if (std::find(sess.available_capabilities.begin(),
                              sess.available_capabilities.end(),
                              action) != sess.available_capabilities.end()) {
                    snap.routeable_actions.push_back(action);
                }
            }
        }
        return snap;
    }
};

} // namespace GRIM

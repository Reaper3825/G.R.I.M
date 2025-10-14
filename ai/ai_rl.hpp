#pragma once
#include <nlohmann/json.hpp>

namespace GRIM::RL {

bool init();                                      // starts rl_bridge.py
nlohmann::json getAction(const nlohmann::json&);  // sends obs → gets action
void shutdown();                                  // stops rl_bridge.py

} // namespace GRIM::RL

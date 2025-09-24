#pragma once
#include <string>
#include <vector>

// Returns list of playback (output) device names
std::vector<std::wstring> getPlaybackDevices();

// Prompts the user to select a playback device at startup
// If switching is implemented, this will set the chosen device as default
void promptForAudioDevice();

#include <iostream>
#include <portaudio.h>

int main() {
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << "\n";
        return 1;
    }

    int numDevices = Pa_GetDeviceCount();
    if (numDevices < 0) {
        std::cerr << "ERROR: Pa_GetDeviceCount returned " << numDevices << "\n";
        Pa_Terminate();
        return 1;
    }

    std::cout << "=== PortAudio Device List ===\n";
    std::cout << "Found " << numDevices << " devices total\n\n";

    for (int i = 0; i < numDevices; i++) {
        const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(i);
        if (!deviceInfo) continue;

        const PaHostApiInfo* hostApiInfo = Pa_GetHostApiInfo(deviceInfo->hostApi);

        std::cout << "Device #" << i << ": " << deviceInfo->name
                  << "  (Host API: " << hostApiInfo->name << ")\n";
        std::cout << "  Max input channels : " << deviceInfo->maxInputChannels << "\n";
        std::cout << "  Max output channels: " << deviceInfo->maxOutputChannels << "\n";
        std::cout << "  Default sample rate: " << deviceInfo->defaultSampleRate << "\n";
        std::cout << "  Latency (input/output): "
                  << deviceInfo->defaultLowInputLatency << " / "
                  << deviceInfo->defaultLowOutputLatency << " sec\n";

        if (i == Pa_GetDefaultInputDevice())
            std::cout << "  *** Default INPUT device ***\n";
        if (i == Pa_GetDefaultOutputDevice())
            std::cout << "  *** Default OUTPUT device ***\n";

        std::cout << "-------------------------------------------\n\n";
    }

    Pa_Terminate();
    return 0;
}

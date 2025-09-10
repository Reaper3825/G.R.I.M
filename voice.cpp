// Voice integration entry point
#include <iostream>
#include <vector>
#include <string>
#include <portaudio.h>
#include "whisper.h"

// Sample rate for Whisper
#define SAMPLE_RATE 16000
#define FRAMES_PER_BUFFER 512

// Struct for holding audio data
struct AudioData {
    std::vector<float> buffer;
    bool ready = false;
};

// PortAudio callback
static int recordCallback(const void *input,
                          void *output,
                          unsigned long frameCount,
                          const PaStreamCallbackTimeInfo* timeInfo,
                          PaStreamCallbackFlags statusFlags,
                          void *userData) {
    AudioData *data = (AudioData*) userData;
    const float *in = (const float*) input;

    if (in) {
        data->buffer.insert(data->buffer.end(), in, in + frameCount);
    }

    return paContinue;
}

int runVoiceDemo(const std::string &modelPath) {
    // Load Whisper model
    struct whisper_context *ctx = whisper_init_from_file(modelPath.c_str());
    if (!ctx) {
        std::cerr << "[ERROR] Failed to load Whisper model: " << modelPath << std::endl;
        return 1;
    }

    // Init PortAudio
    if (Pa_Initialize() != paNoError) {
        std::cerr << "[ERROR] Failed to initialize PortAudio" << std::endl;
        whisper_free(ctx);
        return 1;
    }

    AudioData audio;
    PaStream *stream;

    if (Pa_OpenDefaultStream(&stream,
                             1, // input channels
                             0, // output channels
                             paFloat32,
                             SAMPLE_RATE,
                             FRAMES_PER_BUFFER,
                             recordCallback,
                             &audio) != paNoError) {
        std::cerr << "[ERROR] Failed to open audio stream" << std::endl;
        Pa_Terminate();
        whisper_free(ctx);
        return 1;
    }

    std::cout << "[INFO] Recording for 5 seconds... speak now!" << std::endl;
    Pa_StartStream(stream);
    Pa_Sleep(5000); // record for 5 seconds
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    if (audio.buffer.empty()) {
        std::cerr << "[ERROR] No audio captured" << std::endl;
        whisper_free(ctx);
        return 1;
    }

    std::cout << "[INFO] Transcribing..." << std::endl;

    // Set Whisper params
    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_progress = true;
    params.print_realtime = true;

    if (whisper_full(ctx, params, audio.buffer.data(), audio.buffer.size()) != 0) {
        std::cerr << "[ERROR] Whisper failed during transcription" << std::endl;
        whisper_free(ctx);
        return 1;
    }

    // Print result
    int n_segments = whisper_full_n_segments(ctx);
    std::cout << "[RESULT] Transcription:" << std::endl;
    for (int i = 0; i < n_segments; i++) {
        std::cout << whisper_full_get_segment_text(ctx, i) << std::endl;
    }

    whisper_free(ctx);
    return 0;
}

// Entry point for GRIM integration (for now standalone)
int main() {
    std::string modelPath = "external/whisper.cpp/models/ggml-small.bin";
    return runVoiceDemo(modelPath);
}

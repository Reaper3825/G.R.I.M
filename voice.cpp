#include "voice.hpp"
#include <iostream>
#include <vector>
#include <portaudio.h>
#include "whisper.h"

#define SAMPLE_RATE   16000
#define FRAMES_PER_BUFFER 512

struct AudioData {
    std::vector<float> buffer;
    bool ready = false;
};

static int recordCallback(const void *input,
                          void * /*output*/,
                          unsigned long frameCount,
                          const PaStreamCallbackTimeInfo* /*timeInfo*/,
                          PaStreamCallbackFlags /*statusFlags*/,
                          void *userData) {
    AudioData *data = reinterpret_cast<AudioData*>(userData);
    const float *in = reinterpret_cast<const float*>(input);

    if (input != nullptr) {
        data->buffer.insert(data->buffer.end(), in, in + frameCount);
    }

    if (data->buffer.size() >= SAMPLE_RATE * 5) { // 5 seconds
        data->ready = true;
        return paComplete;
    }

    return paContinue;
}

std::string runVoiceDemo(const std::string &modelPath) {
    std::string transcript;

    // Initialize Whisper
    struct whisper_context *ctx = whisper_init_from_file(modelPath.c_str());
    if (!ctx) {
        std::cerr << "Failed to initialize Whisper model: " << modelPath << std::endl;
        return "[ERROR] Failed to load model";
    }

    // Initialize PortAudio
    if (Pa_Initialize() != paNoError) {
        std::cerr << "Failed to initialize PortAudio" << std::endl;
        whisper_free(ctx);
        return "[ERROR] Failed to init PortAudio";
    }

    AudioData data;
    PaStream *stream;

    if (Pa_OpenDefaultStream(&stream,
                             1, // mono input
                             0, // no output
                             paFloat32,
                             SAMPLE_RATE,
                             FRAMES_PER_BUFFER,
                             recordCallback,
                             &data) != paNoError) {
        std::cerr << "Failed to open audio stream" << std::endl;
        Pa_Terminate();
        whisper_free(ctx);
        return "[ERROR] Failed to open audio stream";
    }

    std::cout << "Recording... speak into your mic (5 sec window)" << std::endl;
    Pa_StartStream(stream);

    while (!data.ready) {
        Pa_Sleep(100);
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    std::cout << "Transcribing..." << std::endl;

    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    if (whisper_full(ctx, params, data.buffer.data(), data.buffer.size()) != 0) {
        std::cerr << "Whisper transcription failed" << std::endl;
        whisper_free(ctx);
        return "[ERROR] Transcription failed";
    }

    int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        transcript += whisper_full_get_segment_text(ctx, i);
        transcript += " ";
    }

    whisper_free(ctx);

    // Trim trailing space
    if (!transcript.empty() && transcript.back() == ' ') {
        transcript.pop_back();
    }

    return transcript;
}

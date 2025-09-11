#include "voice_stream.hpp"
#include "nlp.hpp"
#include "console_history.hpp"
#include "ui_helpers.hpp"   // for ui_set_textbox()
#include <whisper.h>
#include <portaudio.h>
#include "commands.hpp"
#include "timer.hpp"
#include "ai.hpp"
#include <filesystem>

#include <iostream>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <csignal>
#include <chrono>

namespace fs = std::filesystem;

#define SAMPLE_RATE       16000
#define FRAMES_PER_BUFFER 512
#define CHUNK_SIZE        (SAMPLE_RATE / 2) // 0.5 seconds of audio

struct AudioData {
    std::vector<float> buffer;
    std::mutex mutex;
    bool ready = false;
};

static AudioData g_audio;
static std::atomic<bool> g_running{false};
static std::string g_partial; // evolving transcript for textbox

// ---------------- Signal Handling ----------------

static void handleSigint(int) {
    g_running = false;
}

// ---------------- PortAudio Callback ----------------

static int recordCallback(const void *input,
                          void * /*output*/,
                          unsigned long frameCount,
                          const PaStreamCallbackTimeInfo* /*timeInfo*/,
                          PaStreamCallbackFlags /*statusFlags*/,
                          void * /*userData*/) {
    const float *in = static_cast<const float*>(input);
    if (in) {
        std::lock_guard<std::mutex> lock(g_audio.mutex);
        g_audio.buffer.insert(g_audio.buffer.end(), in, in + frameCount);
        g_audio.ready = true;

        std::cout << "[DEBUG][VoiceStream] Captured " << frameCount
                  << " frames (" << g_audio.buffer.size() << " total)\n";
    }
    return paContinue;
}

// ---------------- Silence Detection ----------------

static bool isSilence(const std::vector<float> &pcm) {
    if (pcm.empty()) return true;
    double energy = 0.0;
    for (float s : pcm) energy += s * s;
    energy /= pcm.size();
    return energy < 1e-4; // threshold can be tuned
}

// ---------------- Main Voice Stream ----------------

void runVoiceStream(whisper_context *ctx,
                    ConsoleHistory* history,
                    std::vector<Timer>& timers,
                    nlohmann::json& longTermMemory,
                    NLP& nlp) {

    g_running = true;
    g_partial.clear();

    // Setup Ctrl+C handler
    signal(SIGINT, handleSigint);

    // Init PortAudio
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        history->push("[VoiceStream] ERROR: Failed to initialize PortAudio", sf::Color::Red);
        return;
    }

    PaStream *stream = nullptr;
    err = Pa_OpenDefaultStream(&stream,
                               1, // input channels
                               0, // output channels
                               paFloat32,
                               SAMPLE_RATE,
                               FRAMES_PER_BUFFER,
                               recordCallback,
                               nullptr);
    if (err != paNoError || !stream) {
        history->push("[VoiceStream] ERROR: Could not open microphone stream", sf::Color::Red);
        Pa_Terminate();
        return;
    }

    err = Pa_StartStream(stream);
    if (err != paNoError) {
        history->push("[VoiceStream] ERROR: Could not start microphone stream", sf::Color::Red);
        Pa_CloseStream(stream);
        Pa_Terminate();
        return;
    }

    history->push("[VoiceStream] Listening... (Ctrl+C to stop)", sf::Color(0, 200, 255));
    std::cout << "[VoiceStream] Listening... speak into your mic (Ctrl+C to stop)\n";

    // Whisper params
    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_realtime    = false;
    params.print_progress    = false;
    params.translate         = false;
    params.no_context        = true;
    params.single_segment    = true;
    params.no_timestamps     = true;
    params.token_timestamps  = false;
    params.max_tokens        = 64;

    auto lastSpeechTime = std::chrono::steady_clock::now();

    // Worker thread for Whisper decoding
    std::thread worker([&]() {
        while (g_running) {
            std::vector<float> pcm;

            {
                std::lock_guard<std::mutex> lock(g_audio.mutex);
                if (!g_audio.ready) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    continue;
                }
                pcm.swap(g_audio.buffer);
                g_audio.ready = false;
            }

            std::cout << "[DEBUG][VoiceStream] Processing " << pcm.size() << " samples\n";

            // Process audio in slices
            while (pcm.size() >= CHUNK_SIZE) {
                std::vector<float> slice(pcm.begin(), pcm.begin() + CHUNK_SIZE);
                pcm.erase(pcm.begin(), pcm.begin() + CHUNK_SIZE);

                std::cout << "[DEBUG][VoiceStream] Sending slice of " << slice.size() << " samples to Whisper\n";

                if (whisper_full(ctx, params, slice.data(), slice.size()) == 0) {
                    const int n_segments = whisper_full_n_segments(ctx);
                    std::string transcript;
                    for (int i = 0; i < n_segments; i++) {
                        transcript += whisper_full_get_segment_text(ctx, i);
                    }

                    if (!transcript.empty()) {
                        g_partial = transcript;
                        ui_set_textbox(g_partial); // update UI input field
                        std::cout << "[VoiceStream] Transcript: " << g_partial << std::endl;
                        lastSpeechTime = std::chrono::steady_clock::now();
                    }
                }
            }

            // Silence timeout check
            auto now = std::chrono::steady_clock::now();
            auto silenceMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastSpeechTime).count();
            if (!g_partial.empty() && silenceMs > 1200) {
                std::cout << "[DEBUG][VoiceStream] Silence detected (" << silenceMs << " ms), finalizing command.\n";

                // Finalize transcript
                history->push("[VoiceStream Heard] " + g_partial, sf::Color::Yellow);

                // Parse transcript and dispatch as command
                Intent intent = nlp.parse(g_partial);
                std::cout << "[DEBUG][VoiceStream] Intent parsed: name='" << intent.name
                          << "', matched=" << intent.matched
                          << ", slots=" << intent.slots.size()
                          << ", score=" << intent.score << "\n";

                if (intent.matched) {
                    auto currentDir = fs::current_path();
                    handleCommand(intent, g_partial, currentDir,
                                  timers, longTermMemory, nlp, *history);
                } else {
                    history->push("[VoiceStream] Sorry, I didn’t understand: " + g_partial,
                                  sf::Color::Red);
                }

                g_partial.clear();
                ui_set_textbox(""); // clear input box
                std::cout << "[VoiceStream] Command committed.\n";
            }
        }
    });

    // Block until stopped
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // Shutdown
    worker.join();
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    history->push("[VoiceStream] Stopped.", sf::Color(0, 200, 255));
    std::cout << "[VoiceStream] Stopped.\n";
}

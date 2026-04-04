// Apple TTS engine — AVSpeechSynthesizer wrapper
// macOS 10.14+ / iOS 7+
#if defined(__APPLE__)

#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#include "voice_tts_apple.hpp"
#include "logger.hpp"
#include <atomic>
#include <mutex>
#include <condition_variable>

// =============================================================================
// Delegate to track utterance completion
// =============================================================================
@interface GRIMSpeechDelegate : NSObject <AVSpeechSynthesizerDelegate>
@property (atomic, assign) BOOL finished;
@property (atomic, assign) BOOL cancelled;
@end

@implementation GRIMSpeechDelegate {
    std::mutex _mtx;
    std::condition_variable _cv;
}

- (void)speechSynthesizer:(AVSpeechSynthesizer*)synth
 didFinishSpeechUtterance:(AVSpeechUtterance*)utterance {
    (void)synth; (void)utterance;
    self.finished = YES;
    std::lock_guard<std::mutex> lk(_mtx);
    _cv.notify_all();
}

- (void)speechSynthesizer:(AVSpeechSynthesizer*)synth
didCancelSpeechUtterance:(AVSpeechUtterance*)utterance {
    (void)synth; (void)utterance;
    self.finished = YES;
    self.cancelled = YES;
    std::lock_guard<std::mutex> lk(_mtx);
    _cv.notify_all();
}

- (void)waitUntilFinished {
    std::unique_lock<std::mutex> lk(_mtx);
    _cv.wait(lk, [self]{ return self.finished == YES; });
}

@end

// =============================================================================
// Static state
// =============================================================================
static AVSpeechSynthesizer* s_synth = nil;
static GRIMSpeechDelegate*  s_delegate = nil;
static std::atomic<bool>    s_ready{false};
static std::atomic<bool>    s_speaking{false};
static std::string          s_voiceId;   // chosen voice identifier

namespace AppleTTS {

bool init() {
    if (s_ready.load()) return true;

    @autoreleasepool {
        s_synth = [[AVSpeechSynthesizer alloc] init];
        s_delegate = [[GRIMSpeechDelegate alloc] init];
        s_synth.delegate = s_delegate;

        // Pick the best available English voice.
        // Prefer "enhanced" or "premium" quality voices when installed.
        AVSpeechSynthesisVoice* bestVoice = nil;
        int bestQuality = -1;

        for (AVSpeechSynthesisVoice* v in [AVSpeechSynthesisVoice speechVoices]) {
            if (![[v language] hasPrefix:@"en"]) continue;

            int q = 0;
            if (v.quality == AVSpeechSynthesisVoiceQualityEnhanced) q = 2;
            else if (v.quality == AVSpeechSynthesisVoiceQualityPremium) q = 3;
            else q = 1; // default quality

            if (q > bestQuality) {
                bestQuality = q;
                bestVoice = v;
            }
        }

        if (!bestVoice) {
            bestVoice = [AVSpeechSynthesisVoice voiceWithLanguage:@"en-US"];
        }

        if (bestVoice) {
            s_voiceId = [[bestVoice identifier] UTF8String];
            std::string name = [[bestVoice name] UTF8String];
            std::string lang = [[bestVoice language] UTF8String];
            LOG_DEBUG("AppleTTS", "Selected voice: " + name +
                      " (" + lang + ") quality=" + std::to_string(bestQuality));
        } else {
            LOG_ERROR("AppleTTS", "No English voice found on this system");
            return false;
        }

        s_ready.store(true);
        LOG_DEBUG("AppleTTS", "AVSpeechSynthesizer initialized");
        return true;
    }
}

void shutdown() {
    if (!s_ready.load()) return;

    @autoreleasepool {
        if (s_synth && [s_synth isSpeaking]) {
            [s_synth stopSpeakingAtBoundary:AVSpeechBoundaryImmediate];
        }
        s_synth = nil;
        s_delegate = nil;
        s_ready.store(false);
        s_speaking.store(false);
        LOG_DEBUG("AppleTTS", "Shutdown complete");
    }
}

bool isReady() {
    return s_ready.load();
}

bool speak(const std::string& text, float speed, float pitch) {
    if (!s_ready.load()) {
        LOG_ERROR("AppleTTS", "speak() called before init()");
        return false;
    }
    if (text.empty()) return true;

    @autoreleasepool {
        NSString* nsText = [NSString stringWithUTF8String:text.c_str()];
        AVSpeechUtterance* utterance = [[AVSpeechUtterance alloc] initWithString:nsText];

        // Voice
        if (!s_voiceId.empty()) {
            AVSpeechSynthesisVoice* voice =
                [AVSpeechSynthesisVoice voiceWithIdentifier:
                    [NSString stringWithUTF8String:s_voiceId.c_str()]];
            if (voice) utterance.voice = voice;
        }

        // Speed: AVSpeechUtterance rate is 0.0–1.0, default ~0.5.
        // Map our multiplier (1.0 = normal) into that range.
        float baseRate = AVSpeechUtteranceDefaultSpeechRate;
        float clampedRate = baseRate * speed;
        if (clampedRate < AVSpeechUtteranceMinimumSpeechRate)
            clampedRate = AVSpeechUtteranceMinimumSpeechRate;
        if (clampedRate > AVSpeechUtteranceMaximumSpeechRate)
            clampedRate = AVSpeechUtteranceMaximumSpeechRate;
        utterance.rate = clampedRate;

        // Pitch: 0.5–2.0 range, 1.0 = default.
        float clampedPitch = pitch;
        if (clampedPitch < 0.5f) clampedPitch = 0.5f;
        if (clampedPitch > 2.0f) clampedPitch = 2.0f;
        utterance.pitchMultiplier = clampedPitch;

        // Reset delegate state
        s_delegate.finished = NO;
        s_delegate.cancelled = NO;

        s_speaking.store(true);
        [s_synth speakUtterance:utterance];

        // Block until finished (delegate callback fires)
        [s_delegate waitUntilFinished];
        s_speaking.store(false);

        return !s_delegate.cancelled;
    }
}

bool isSpeaking() {
    return s_speaking.load();
}

} // namespace AppleTTS

#endif // __APPLE__

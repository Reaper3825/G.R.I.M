#pragma once
#include <cstddef>
#include <cstdint>


namespace GRIM{
namespace Sound {
    struct sound {
        float LSB; // Least Significant Bit depth || the smalled change in amplitude that can be represented
        float MSB; // Most Significant Bit depth || the largest change in amplitude that can be represented
        float BitDepth; // in decibels || dynamic range
        float Wavelength; // in meters
        float Volume; // normalized comparitively to other sounds in system
        float Amplitude; // in decibels
        float Frequency; // in hertz
        float sample_rate; // in samples per second
        float Format; // PCM format
        float QuantNoise; // quantization noise level
        float Latency; // in milliseconds
        int Channels; // number of audio channels
        int BufferSize; // size of audio buffer in bytes
    };
    void SampleSound(sound& s); // Entry point for sound sample data
    float getLatency(sound& s); // in milliseconds
    float getBitDepth(sound& s); // in decibels
    float getLSB(sound& s); // Least Significant Bit depth
    float getMSB(sound& s); // Most Significant Bit depth
    float getWavelength(sound& s); // in meters
    float getVolume(sound& s); // normalized comparitively to other sounds in system
    float getFrequency(sound& s); // in hertz
    float getAmplitude(sound& s); // in decibels
    float getSampleRate(sound& s); // in samples per second
    float getFormat(sound& s); // PCM format
    float getQuantNoise(sound& s); // quantization noise level
    int getChannels(sound& s); // number of audio channels
    int getBufferSize(sound& s); // size of audio buffer in bytes
    } // namespace Sound
} // namespace GRIM
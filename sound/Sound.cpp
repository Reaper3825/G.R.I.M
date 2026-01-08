#include <iostream>
#include <cmath>
#include "Sound.hpp"
namespace GRIM{
namespace Sound {
    sound Sample;
    float getLatency(sound& s){
        s.Latency = 10.0f + (s.BufferSize / (s.sample_rate * s.Channels * (s.BitDepth / 8))) * 1000.0f; // in milliseconds
        return s.Latency;
    }
        float getLSB( sound& s){
            s.LSB = 1.0f / (std::pow(2.0f, s.BitDepth) - 1.0f); // Least Significant Bit depth
        return s.LSB;
    }
    float getMSB( sound& s){
        s.MSB = 1.0f; // Most Significant Bit depth
        return s.MSB;
    }
    float getBitDepth( sound& s){
        s.BitDepth = 20.0f * log10f(s.MSB / s.LSB); // in decibels
        return s.BitDepth;
    }
    float getWavelength(sound& s){
        s.Wavelength = 343.0f / s.Frequency; // in meters (assuming speed of sound in air at 20°C)
        return s.Wavelength;
    }
    float getVolume(sound& s){
        s.Volume = s.Amplitude / 100.0f; // normalized comparitively to other sounds in system
        return s.Volume;
    }
    float getFrequency(sound& s){

        return s.Frequency;
    }
    float getAmplitude(sound& s){
        s.Amplitude = 20.0f * log10f(s.Volume); // in decibels
        return s.Amplitude;
    }
    float getSampleRate(sound& s){
        s.sample_rate = 44100.0f; // in samples per second
        return s.sample_rate;
    }
    float getFormat(sound& s){
        return s.Format;
    }
    float getQuantNoise(sound& s){
        return s.QuantNoise;
    }
    int getChannels(sound& s){
        return s.Channels;
    }
    int getBufferSize(sound& s){
        return s.BufferSize;
    }
    void SampleSound(sound& s){
        get
        // Placeholder implementation
        std::cout << "Sampling sound with frequency: " << s.Frequency << " Hz" << std::endl;
    }

} // namespace Sound
} // namespace GRIM
#pragma once

namespace calc {

//variables
float time;
float distance;
float mass;
float velocity;
float acceleration;
float force;
float energy;
float power;
float faces;
float surface_area;
float gravity;
float weight;
float frequency;

float MassToWeight(float);
float CalculateFrequency(float);
float CalculateForce(float, float);
float CalculateAcceleration(float initialVelocity, float finalVelocity, float time);
//functions
}
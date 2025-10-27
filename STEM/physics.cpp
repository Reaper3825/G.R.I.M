#include "physics.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>


float weight = 0.0f;
float gravity = 9.81f;
float frequency = 0.0f;
float force = 0.0f;

float calc::MassToWeight(float mass)
{
    calc::weight = mass * calc::gravity;
    return calc::weight;
}

float calc::CalculateFrequency(float period)
{
    calc::frequency = 1.0f / period;
    return calc::frequency;
}

float calc::CalculateForce(float mass, float acceleration)
{
    calc::force = mass * acceleration;
    return calc::force;
}

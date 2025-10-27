#include "physics.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

float weight = 0.0f;
float gravity = 9.81f;
int main() {
calc::MassToWeight(10.0f);
if(calc::weight != 0.0f) {

        std::cout << calc::weight << std::endl;
}
    return 0;
}

float calc::MassToWeight(float mass)
{
    calc::weight = mass * gravity;
    return calc::weight;
}
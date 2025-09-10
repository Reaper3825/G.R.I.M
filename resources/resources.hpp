#pragma once
#include <string>
#include <SFML/Graphics.hpp>
#include "console_history.hpp"
#pragma message("Using resources.hpp from project root")
#error "THIS IS THE CORRECT resources.hpp"


std::string getResourcePath();
std::string loadTextResource(const std::string& filename, int argc, char** argv);
std::string findAnyFontInResources(int argc, char** argv, ConsoleHistory* history);

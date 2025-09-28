#pragma once

#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/Image.hpp>
#include <SFML/Graphics/Shader.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <windows.h>

// ===========================================================
// Resource + overlay helpers
// ===========================================================
bool loadResources(sf::Texture& diffuse,
                   sf::Texture& opacity,
                   sf::Shader& shader);

void updateOverlay(HWND hwnd, const sf::Texture& texture);
// Overload: accept an sf::Image directly to avoid copying textures across
// contexts (callers that already have an Image can pass it to skip
// an extra copyToImage()).
void updateOverlay(HWND hwnd, const sf::Image& image);

// ===========================================================
// New helper: create scaled sprite
// ===========================================================
sf::Sprite createScaledSprite(const sf::Texture& tex, sf::Vector2u windowSize);

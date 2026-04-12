#pragma once

// Icon codepoint constants for use with drawText().
// These map to Unicode Private Use Area codepoints used by popular icon fonts:
//   - FontAwesome (Free/Pro)
//   - Material Design Icons
//   - Nerd Fonts (patched fonts that include all of the above)
//
// Usage:
//   #include "ui/icon_codepoints.hpp"
//   renderer.drawText(pos, ICON_GEAR " Settings", color);
//
// To use icons, call loadIconFont() with a compatible .ttf after setFont():
//   renderer.setFont("DejaVuSans.ttf", 16);
//   renderer.loadIconFont("fa-solid-900.ttf");
//
// Or use a Nerd Font patched TTF (all icons baked into one font file):
//   renderer.setFont("DejaVuSansM Nerd Font.ttf", 16);
//   // No separate loadIconFont() needed — icons are in the main font.

// Helper: encode a Unicode codepoint as a UTF-8 string literal.
// For codepoints in the BMP (U+0000..U+FFFF), 3-byte UTF-8.
// For supplementary (U+10000+), 4-byte UTF-8.
// These macros produce compile-time string literals.

// FontAwesome Solid (0xF000 range) — most common icons
#define ICON_FA_HOME           "\xEF\x80\x95"  // U+F015
#define ICON_FA_GEAR           "\xEF\x80\x93"  // U+F013
#define ICON_FA_COG            "\xEF\x80\x93"  // U+F013 (alias)
#define ICON_FA_SEARCH         "\xEF\x80\x82"  // U+F002
#define ICON_FA_CHECK          "\xEF\x80\x8C"  // U+F00C
#define ICON_FA_XMARK          "\xEF\x80\x8D"  // U+F00D
#define ICON_FA_TIMES          "\xEF\x80\x8D"  // U+F00D (alias)
#define ICON_FA_PLUS           "\xEF\x81\xA7"  // U+F067
#define ICON_FA_MINUS          "\xEF\x81\xA8"  // U+F068
#define ICON_FA_STAR           "\xEF\x80\x85"  // U+F005
#define ICON_FA_USER           "\xEF\x80\x87"  // U+F007
#define ICON_FA_TRASH          "\xEF\x87\xB8"  // U+F1F8
#define ICON_FA_PENCIL         "\xEF\x8C\x83"  // U+F303
#define ICON_FA_FOLDER         "\xEF\x81\xBB"  // U+F07B
#define ICON_FA_FOLDER_OPEN    "\xEF\x81\xBC"  // U+F07C
#define ICON_FA_FILE           "\xEF\x85\x9B"  // U+F15B
#define ICON_FA_DOWNLOAD       "\xEF\x80\x99"  // U+F019
#define ICON_FA_UPLOAD         "\xEF\x82\x93"  // U+F093
#define ICON_FA_SAVE           "\xEF\x83\x87"  // U+F0C7
#define ICON_FA_COPY           "\xEF\x83\x85"  // U+F0C5
#define ICON_FA_PASTE          "\xEF\x83\xAA"  // U+F0EA
#define ICON_FA_PLAY           "\xEF\x81\x8B"  // U+F04B
#define ICON_FA_PAUSE          "\xEF\x81\x8C"  // U+F04C
#define ICON_FA_STOP           "\xEF\x81\x8D"  // U+F04D
#define ICON_FA_REFRESH        "\xEF\x80\xA1"  // U+F021
#define ICON_FA_SPINNER        "\xEF\x84\x90"  // U+F110
#define ICON_FA_WARNING        "\xEF\x81\xB1"  // U+F071
#define ICON_FA_INFO_CIRCLE    "\xEF\x81\x9A"  // U+F05A
#define ICON_FA_QUESTION       "\xEF\x84\xA8"  // U+F128
#define ICON_FA_EXCLAMATION    "\xEF\x84\xAA"  // U+F12A
#define ICON_FA_BUG            "\xEF\x86\x88"  // U+F188
#define ICON_FA_TERMINAL       "\xEF\x84\xA0"  // U+F120
#define ICON_FA_CODE           "\xEF\x84\xA1"  // U+F121
#define ICON_FA_DATABASE       "\xEF\x87\x80"  // U+F1C0
#define ICON_FA_CHART_LINE     "\xEF\x88\x81"  // U+F201
#define ICON_FA_CHART_BAR      "\xEF\x82\x80"  // U+F080
#define ICON_FA_CUBE           "\xEF\x86\xB2"  // U+F1B2
#define ICON_FA_CUBES          "\xEF\x86\xB3"  // U+F1B3
#define ICON_FA_MICROCHIP      "\xEF\x8B\x9B"  // U+F2DB
#define ICON_FA_BRAIN          "\xEF\x97\x83"  // U+F5DC
#define ICON_FA_ROBOT          "\xEF\x95\x84"  // U+F544
#define ICON_FA_SERVER         "\xEF\x88\xB3"  // U+F233
#define ICON_FA_NETWORK        "\xEF\x86\xB5"  // U+F1B5
#define ICON_FA_WIFI           "\xEF\x87\xAB"  // U+F1EB
#define ICON_FA_BOLT           "\xEF\x83\xA7"  // U+F0E7
#define ICON_FA_WRENCH         "\xEF\x82\xAD"  // U+F0AD
#define ICON_FA_SLIDERS        "\xEF\x87\x9E"  // U+F1DE
#define ICON_FA_EYE            "\xEF\x81\xAE"  // U+F06E
#define ICON_FA_EYE_SLASH      "\xEF\x81\xB0"  // U+F070
#define ICON_FA_LOCK           "\xEF\x80\xA3"  // U+F023
#define ICON_FA_UNLOCK         "\xEF\x82\x9C"  // U+F09C
#define ICON_FA_BELL           "\xEF\x83\xB3"  // U+F0F3
#define ICON_FA_CLOCK          "\xEF\x80\x97"  // U+F017
#define ICON_FA_CALENDAR       "\xEF\x81\xB3"  // U+F073
#define ICON_FA_COMMENTS       "\xEF\x82\x86"  // U+F086
#define ICON_FA_ENVELOPE       "\xEF\x83\xA0"  // U+F0E0
#define ICON_FA_LINK           "\xEF\x83\x81"  // U+F0C1
#define ICON_FA_ARROW_UP       "\xEF\x81\xA2"  // U+F062
#define ICON_FA_ARROW_DOWN     "\xEF\x81\xA3"  // U+F063
#define ICON_FA_ARROW_LEFT     "\xEF\x81\xA0"  // U+F060
#define ICON_FA_ARROW_RIGHT    "\xEF\x81\xA1"  // U+F061
#define ICON_FA_CHEVRON_UP     "\xEF\x81\xB7"  // U+F077
#define ICON_FA_CHEVRON_DOWN   "\xEF\x81\xB8"  // U+F078
#define ICON_FA_CHEVRON_LEFT   "\xEF\x81\x93"  // U+F053
#define ICON_FA_CHEVRON_RIGHT  "\xEF\x81\x94"  // U+F054
#define ICON_FA_BARS           "\xEF\x83\x89"  // U+F0C9
#define ICON_FA_ELLIPSIS_H     "\xEF\x85\x81"  // U+F141
#define ICON_FA_ELLIPSIS_V     "\xEF\x85\x82"  // U+F142
#define ICON_FA_EXPAND         "\xEF\x81\xA5"  // U+F065
#define ICON_FA_COMPRESS       "\xEF\x81\xA6"  // U+F066
#define ICON_FA_POWER_OFF      "\xEF\x80\x91"  // U+F011
#define ICON_FA_CIRCLE         "\xEF\x84\x91"  // U+F111
#define ICON_FA_SQUARE         "\xEF\x83\x88"  // U+F0C8

// Convenience aliases for GRIM UI
#define ICON_SETTINGS          ICON_FA_GEAR
#define ICON_CLOSE             ICON_FA_XMARK
#define ICON_OK                ICON_FA_CHECK
#define ICON_TRAINING          ICON_FA_CHART_LINE
#define ICON_CONSOLE           ICON_FA_TERMINAL
#define ICON_STORAGE           ICON_FA_DATABASE
#define ICON_AI                ICON_FA_BRAIN
#define ICON_NETWORK           ICON_FA_NETWORK
#define ICON_MENU              ICON_FA_BARS
#define ICON_EXPAND            ICON_FA_EXPAND
#define ICON_COLLAPSE          ICON_FA_COMPRESS

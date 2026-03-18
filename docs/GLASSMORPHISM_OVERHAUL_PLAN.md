# Glassmorphism Panel Layout Overhaul Plan

> **Scope:** All files under `ui/` — panels, widgets, draw helpers, content panels.  
> **DO NOT TOUCH:** `popup_ui/` directory, `resources/popup.*` files.

---

## Reference Analysis

Six reference images studied (dark dashboard UIs). Key design patterns:

1. **Panel backgrounds are OPAQUE dark surfaces** (~94% opacity). They're dark cards (`#202030` to `#282840` range) sitting on a deep background (`#0E0E1A`). The "glass" comes from **subtle edge highlights** and **card elevation**, NOT transparency.

2. **Card elevation via drop shadows.** Panels cast a subtle `0x30000000` shadow offset by a few pixels. Inner content areas (graphs, log areas) use a *slightly darker* shade to create visual nesting — outer card `#252535`, inner area `#1A1A28`.

3. **Borders are BARELY VISIBLE.** ~9% white (`0x18FFFFFF`) on top/left edge as a paper-thin "light catch". Bottom/right shadow at ~6% black (`0x10000000`). Nothing bright.

4. **Title bars BLEND into the card.** No separate title bar rectangle. Title text sits at the top of the card with padding — just bigger/brighter text, no background band.

5. **Warm orange/amber accent** (`#E8A840`) is dominant — used for active states, graph lines, progress fills, key values. Already in our palette.

6. **Inner sections use dividers, not tinted backgrounds.** Faint horizontal lines (`0x10FFFFFF`) separate sections. No colored section header rectangles.

7. **Chrome controls are barely visible** — small, dim squares that brighten slightly on hover.

---

## Current vs Target

| Element | Current Value | Target Value | Reason |
|---------|--------------|--------------|--------|
| Panel bg | `0xCC1A1A2E` (80% α) | `0xF0202030` (94% α) | Refs show nearly opaque dark cards |
| Title bar | `0xD0202038` separate rect | **Remove** — just pad title text | Refs show no title bar separation |
| Top/left border | `0x60FFFFFF` (37% white) | `0x18FFFFFF` (9% white) | Paper-thin edge catch |
| Bottom/right border | `0x20000000` | `0x10000000` | Even subtler shadow edge |
| Panel drop shadow | None | `0x30000000` offset `{3,3}` | Creates card elevation |
| Widget bg | `0xD9222238` (85% α) | `0xF0282840` (94% α) | Solid feel |
| Widget bg hover | `0xE62A2A42` | `0xF5303048` | Slight lift |
| Widget bg pressed | `0xF0303050` | `0xFF383858` | Fully opaque on press |
| Scrollbox / recessed areas | `0xE6141422` | `0xF01A1A28` | Opaque recessed area |
| Slider track | `0xD91A1A30` | `0xF01A1A28` | Match recessed areas |
| Glass highlight | `0x18FFFFFF` | `0x0AFFFFFF` | Near-invisible top catch |
| Inner glow | `0x0CFFFFFF` | `0x06FFFFFF` | Almost nothing |
| Chrome buttons | `0x30FFFFFF` | `0x15FFFFFF` | More subtle |
| Chrome hover | `0x50FFFFFF` | `0x25FFFFFF` | Slight brighten only |
| Section headers | Colored tint rects | Text + `0x10FFFFFF` divider | No colored bg blocks |
| Content area bgs | Various transparent | `0xF01A1A28` (new `ContentAreaBg`) | Consistent recessed look |

---

## New Theme Tokens to Add

```cpp
constexpr uint32_t ContentAreaBg  = 0xF01A1A28;  // Recessed inner areas (graphs, logs, code)
constexpr uint32_t CardSurface    = 0xF0252535;   // Card surface (slightly lighter than ContentArea)
constexpr uint32_t DividerLine    = 0x10FFFFFF;   // Section dividers within cards
constexpr uint32_t PanelShadow    = 0x30000000;   // Drop shadow behind panels
```

---

## Implementation Phases

### Phase 1 — Theme Tokens (`ui/ui_theme.hpp`)

Update all background/border/highlight token values per the table above.  
Add `ContentAreaBg`, `CardSurface`, `DividerLine`, `PanelShadow`.

### Phase 2 — Panel Chrome (`ui/ui_panel.hpp`, `ui/ui_panel.cpp`)

1. Update default `bgColor` in hpp: `0xCC1A1A2E` → `0xF0202030`
2. In `drawOverlay()`:
   - **Add drop shadow**: `drawRect({pos.x+3, pos.y+3}, size, PanelShadow)` before panel bg
   - **Remove title bar rectangle** (`0xD0202038` draw). Keep title text with padding.
   - **Remove title bar glass highlight** (`0x30FFFFFF` 1px line under title bar)
   - **Dim all border draws**: top/left → `0x18FFFFFF`, bottom/right → `0x10000000`, left → `0x12FFFFFF`
   - **Dim chrome buttons**: base `0x15FFFFFF`, hover `0x25FFFFFF`
   - **Dim resize grip**: `0x20FFFFFF` (was `0x50FFFFFF`)

### Phase 3 — Widget Surfaces

Files: `ui_button.hpp/.cpp`, `ui_slider.cpp`, `ui_toggle.cpp`, `ui_dropdown.cpp`, `ui_inputbox.cpp`, `ui_textarea.cpp`, `ui_scrollbox.cpp`

For each widget:
- Background: `0xD9222238` → `0xF0282840`
- Hover: `0xE62A2A42` → `0xF5303048`
- Pressed/active: `0xF0303050` → `0xFF383858`
- Border highlights: `0x60FFFFFF` / `0x40FFFFFF` → `0x18FFFFFF` / `0x12FFFFFF`
- Border shadows: `0x20000000` → `0x10000000`
- Inner glass highlight lines: `0x30FFFFFF` → `0x0AFFFFFF`
- Focus rings: Keep `0xFF6B8CFF` (accent color is fine)
- Recessed areas (track, input bg): use `0xF01A1A28`

### Phase 4 — Content Panels

Files: `console_panel.cpp`, `ui_training_panel.cpp`, `ui_osint_results.cpp`, `ui_settings_menu.cpp`, `ui_model_panel.cpp`

**DO NOT TOUCH:** `popup_ui/*` (sprite popup)

For each panel:
- Content area bgs (`0xD90E0E1A`, `0xFF0A0A0A`) → `0xF01A1A28` (ContentAreaBg)
- Border draws `0x60FFFFFF` → `0x18FFFFFF`
- Border draws `0x30FFFFFF` → `0x0CFFFFFF`
- Border draws `0x40FFFFFF` → `0x12FFFFFF`
- Border draws `0x20000000` → `0x10000000`
- Old-style borders `0xFF303030` → `0x10FFFFFF`
- Section headers: remove tinted background rects, use text + faint divider line
- Scrollbar tracks: `0xD9141422` → `0xF01A1A28`
- Scrollbar thumbs: `0x40FFFFFF`/`0x60FFFFFF`/`0x80FFFFFF` → `0x15FFFFFF`/`0x20FFFFFF`/`0x30FFFFFF`

### Phase 5 — Draw Helpers & Graphs

Files: `ui_draw_helpers.hpp`, `ui_graph.hpp/.cpp`, `ui_progress_bar.hpp`

- `drawSectionHeader()`: Remove tinted background rect. Draw title text + faint divider line below.
- `drawDivider()`: Default color → `DividerLine` token (`0x10FFFFFF`)
- `drawWidgetBackground()`: Use new opaque bg/border values
- Graph backgrounds: → `ContentAreaBg`
- Grid lines: `0x30FFFFFF` → `0x10FFFFFF`
- Axis lines: `0x50FFFFFF` → `0x15FFFFFF`

### Phase 6 — Constants (`ui/ui_constants.hpp`)

- Update `kColorBackground`, `kColorTitleBar`, `kColorInputBar` to match new opaque values

---

## Rendering Constraints

Our `OverlayRenderer` supports:
- `drawRect(pos, size, color)` — filled rectangles
- `drawText(pos, text, color)` — text rendering  
- `drawLine(start, end, color, thickness)` — lines
- `pushClipRect` / `popClipRect` — clipping
- `blurRegion(x, y, w, h, radius)` — **NEW** (see Phase 7 below)

**Cannot do:** rounded corners, gradients, textures.

The "glassmorphism" effect is achieved through:
- Nearly-opaque dark card backgrounds with subtle shade variation
- Paper-thin 1px edge highlights (top/left white, bottom/right black)
- Drop shadows behind panels for elevation
- Clean spacing between cards
- **Frosted blur** on the overlay pixel buffer behind each panel

---

## Phase 7 — Frosted Blur Effect (`overlay_renderer.hpp`, `overlay_renderer.cpp`, `ui_panel.cpp`)

### How It Works

`OverlayRenderer` is a **CPU software renderer** — it writes to a `uint32_t*` pixel buffer and composites via `UpdateLayeredWindow`. There's no GPU pipeline here.

**Approach:** Before each panel draws its background rect, blur the pixels already rendered in that panel's rectangular region (other panels, background elements). A 2-pass separable box blur (horizontal → temp → vertical → back) approximates Gaussian blur.

**What it blurs:** Other overlay elements drawn earlier in the same frame (panels behind panels, background fill). It does NOT blur the desktop or applications behind the overlay window — that would require `BitBlt` screen capture which has latency and reliability issues.

### Memory Cost

| Component | Size |
|-----------|------|
| Current pixel buffer (1920×1080) | **8.3 MB** (already allocated) |
| Temp buffer for blur pass (largest panel region, ~600×500) | **~1.2 MB** |
| **Total additional** | **~1.2 MB** |

Full-screen worst case: 8.3 MB additional. Per-panel blur regions: **1–2 MB** total.

### Performance Cost & Mitigations

Radius-8 separable box blur on a 600×500 panel region:
- 300K pixels × 17 taps × 2 passes ≈ 10M operations
- **~2–4ms per panel** on a modern CPU
- 5 visible panels: **10–20ms/frame** — too steep for 60fps

**Mitigations (implement at least one):**
1. **Half-resolution blur** — downscale 2× before blurring → 4× cheaper → ~1ms/panel
2. **Cache the blur** — only re-blur when panel moves or content behind it changes (dirty flag)
3. **Smaller radius (4–6)** — still looks frosted, much cheaper per pass
4. **Only blur when panels overlap** — skip blur for panels with nothing behind them

Recommended combo: half-res + radius 5 + dirty caching ≈ **2–3ms total per frame** for all panels.

### Implementation

**`overlay_renderer.hpp`** — add:
```cpp
void blurRegion(int x, int y, int w, int h, int radius = 6);
```
Plus a temp buffer member:
```cpp
std::vector<uint32_t> m_blurTemp;
```

**`overlay_renderer.cpp`** — add:
1. Allocate `m_blurTemp` in `init()` (sized to largest expected panel, or lazily on first call)
2. `blurRegion()` implementation: 2-pass separable box blur operating on `m_pixels` within the given rect, using `m_blurTemp` as scratch. Respects clip rect. Premultiplied alpha aware.

**`ui_panel.cpp`** — in `drawOverlay()`, before the panel background `drawRect`:
```cpp
// Frosted blur behind panel
renderer.blurRegion((int)pos.x, (int)pos.y, (int)size.x, (int)size.y, 6);
```

### New Theme Token

```cpp
constexpr int BlurRadius = 6;  // Blur kernel radius for frosted glass effect
```

---

## File Edit Order

| # | File | Changes |
|---|------|---------|
| 1 | `ui/ui_theme.hpp` | Token values + new tokens |
| 2 | `ui/ui_panel.hpp` | Default bgColor |
| 3 | `ui/ui_panel.cpp` | Shadow, title bar removal, border dims |
| 4 | `ui/ui_draw_helpers.hpp` | Section header, divider, widget bg helpers |
| 5 | `ui/ui_constants.hpp` | Base color constants |
| 6 | `ui/ui_button.hpp` + `.cpp` | Bg/border opacity |
| 7 | `ui/ui_slider.cpp` | Bg/border/track opacity |
| 8 | `ui/ui_toggle.cpp` | Bg/border opacity |
| 9 | `ui/ui_dropdown.cpp` | Bg/border/scrollbar opacity |
| 10 | `ui/ui_scrollbox.cpp` | Bg/border/thumb opacity |
| 11 | `ui/ui_inputbox.cpp` | Bg/border opacity |
| 12 | `ui/ui_textarea.cpp` | Bg/border opacity |
| 13 | `ui/ui_progress_bar.hpp` | Bg opacity |
| 14 | `ui/ui_graph.hpp` + `.cpp` | Bg/grid/axis opacity |
| 15 | `ui/console_panel.cpp` | Content areas, borders, dividers |
| 16 | `ui/ui_training_panel.cpp` | Content areas, borders, sections |
| 17 | `ui/ui_settings_menu.cpp` | Borders, section styling |
| 18 | `ui/ui_osint_results.cpp` | Content areas, borders |
| 19 | `ui/ui_model_panel.cpp` | Content areas, borders |

**EXCLUDED:** `popup_ui/*`, `resources/popup.*`

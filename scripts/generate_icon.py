#!/usr/bin/env python3
"""Generate VibeBoy app icon - a stylized Game Boy Color in macOS icon style."""

from PIL import Image, ImageDraw, ImageFilter
import math
import os

def rounded_rect(draw, xy, radius, fill=None, outline=None, width=1):
    """Draw a rounded rectangle."""
    x0, y0, x1, y1 = xy
    r = radius
    # Main body
    draw.rectangle([x0 + r, y0, x1 - r, y1], fill=fill)
    draw.rectangle([x0, y0 + r, x1, y1 - r], fill=fill)
    # Corners
    draw.pieslice([x0, y0, x0 + 2*r, y0 + 2*r], 180, 270, fill=fill)
    draw.pieslice([x1 - 2*r, y0, x1, y0 + 2*r], 270, 360, fill=fill)
    draw.pieslice([x0, y1 - 2*r, x0 + 2*r, y1], 90, 180, fill=fill)
    draw.pieslice([x1 - 2*r, y1 - 2*r, x1, y1], 0, 90, fill=fill)
    if outline:
        # Top edge
        draw.arc([x0, y0, x0 + 2*r, y0 + 2*r], 180, 270, fill=outline, width=width)
        draw.arc([x1 - 2*r, y0, x1, y0 + 2*r], 270, 360, fill=outline, width=width)
        draw.arc([x0, y1 - 2*r, x0 + 2*r, y1], 90, 180, fill=outline, width=width)
        draw.arc([x1 - 2*r, y1 - 2*r, x1, y1], 0, 90, fill=outline, width=width)
        draw.line([x0 + r, y0, x1 - r, y0], fill=outline, width=width)
        draw.line([x0 + r, y1, x1 - r, y1], fill=outline, width=width)
        draw.line([x0, y0 + r, x0, y1 - r], fill=outline, width=width)
        draw.line([x1, y0 + r, x1, y1 - r], fill=outline, width=width)


def make_rounded_mask(size, radius):
    """Create a rounded rectangle mask for the macOS icon shape."""
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    rounded_rect(draw, [0, 0, size[0]-1, size[1]-1], radius, fill=255)
    return mask


def draw_icon(size):
    """Draw the VibeBoy icon at the given size."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    s = size  # shorthand

    # macOS icon corner radius (~22.37% of size, Apple's standard)
    icon_radius = int(s * 0.2237)

    # === Background: gradient teal ===
    bg = Image.new('RGBA', (s, s), (0, 0, 0, 0))
    bg_draw = ImageDraw.Draw(bg)
    # Draw gradient manually - top lighter, bottom darker (GBC teal)
    for y in range(s):
        t = y / s
        r = int(0 + t * 10)
        g = int(155 + t * (-30))
        b = int(145 + t * (-20))
        bg_draw.line([(0, y), (s, y)], fill=(r, g, b, 255))

    # Apply rounded rect mask
    mask = make_rounded_mask((s, s), icon_radius)
    bg.putalpha(mask)
    img = Image.alpha_composite(img, bg)
    draw = ImageDraw.Draw(img)

    # === Subtle border/edge highlight ===
    border_img = Image.new('RGBA', (s, s), (0, 0, 0, 0))
    border_draw = ImageDraw.Draw(border_img)
    rounded_rect(border_draw, [0, 0, s-1, s-1], icon_radius, outline=(255, 255, 255, 40), width=max(1, s//128))
    border_mask = make_rounded_mask((s, s), icon_radius)
    img = Image.alpha_composite(img, border_img)
    draw = ImageDraw.Draw(img)

    # === Game Boy body (inner device shape) ===
    margin = int(s * 0.12)
    body_x0 = margin
    body_y0 = int(s * 0.06)
    body_x1 = s - margin
    body_y1 = int(s * 0.94)
    body_r = int(s * 0.08)

    # Body shadow
    shadow = Image.new('RGBA', (s, s), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_offset = max(1, s // 64)
    rounded_rect(shadow_draw,
                 [body_x0 + shadow_offset, body_y0 + shadow_offset*2,
                  body_x1 + shadow_offset, body_y1 + shadow_offset*2],
                 body_r, fill=(0, 0, 0, 60))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(1, s//40)))
    img = Image.alpha_composite(img, shadow)
    draw = ImageDraw.Draw(img)

    # Body fill - slight gradient for depth
    body = Image.new('RGBA', (s, s), (0, 0, 0, 0))
    body_draw = ImageDraw.Draw(body)
    # Light gray body like real GBC
    rounded_rect(body_draw, [body_x0, body_y0, body_x1, body_y1], body_r,
                 fill=(220, 218, 225, 255))
    # Add subtle vertical gradient
    for y in range(body_y0, body_y1):
        t = (y - body_y0) / (body_y1 - body_y0)
        overlay_alpha = int(20 * t)
        body_draw.line([(body_x0, y), (body_x1, y)], fill=(0, 0, 0, overlay_alpha))

    body_clip = make_rounded_mask((s, s), icon_radius)
    body.putalpha(body_clip)
    img = Image.alpha_composite(img, body)
    draw = ImageDraw.Draw(img)

    # === Upper body color strip (GBC had colored top) ===
    strip_y1 = int(body_y0 + (body_y1 - body_y0) * 0.52)
    strip = Image.new('RGBA', (s, s), (0, 0, 0, 0))
    strip_draw = ImageDraw.Draw(strip)
    # Teal/purple upper portion
    rounded_rect(strip_draw, [body_x0, body_y0, body_x1, strip_y1], body_r,
                 fill=(88, 80, 140, 255))
    # Cut off bottom corners (make them square since they're mid-body)
    strip_draw.rectangle([body_x0, strip_y1 - body_r, body_x1, strip_y1], fill=(88, 80, 140, 255))

    strip_clip = make_rounded_mask((s, s), icon_radius)
    strip.putalpha(strip_clip)
    img = Image.alpha_composite(img, strip)
    draw = ImageDraw.Draw(img)

    # === Screen bezel (dark inset) ===
    screen_margin_x = int(s * 0.07)
    scr_x0 = body_x0 + screen_margin_x
    scr_y0 = body_y0 + int(s * 0.06)
    scr_x1 = body_x1 - screen_margin_x
    scr_y1 = strip_y1 - int(s * 0.04)
    scr_r = int(s * 0.04)

    # Screen bezel
    rounded_rect(draw, [scr_x0, scr_y0, scr_x1, scr_y1], scr_r, fill=(40, 35, 55, 255))

    # === Actual screen (green-ish LCD) ===
    lcd_pad = int(s * 0.03)
    lcd_x0 = scr_x0 + lcd_pad + int(s * 0.02)
    lcd_y0 = scr_y0 + lcd_pad
    lcd_x1 = scr_x1 - lcd_pad - int(s * 0.02)
    lcd_y1 = scr_y1 - lcd_pad
    lcd_r = max(1, int(s * 0.015))

    # LCD background - classic green-ish tint
    rounded_rect(draw, [lcd_x0, lcd_y0, lcd_x1, lcd_y1], lcd_r, fill=(135, 170, 100, 255))

    # === Pixel art pattern on screen (subtle) ===
    # Draw a tiny pixel art smiley/pattern to suggest game content
    lcd_w = lcd_x1 - lcd_x0
    lcd_h = lcd_y1 - lcd_y0
    pixel_size = max(1, lcd_w // 20)

    if size >= 64:
        # Simple pixel art: a small character silhouette
        dark_lcd = (75, 120, 60, 180)

        # Draw a simple pixel art pattern - abstract game-like shapes
        pattern = [
            # A small sprite-like shape in the center
            (8, 4), (9, 4), (10, 4), (11, 4),
            (7, 5), (8, 5), (9, 5), (10, 5), (11, 5), (12, 5),
            (7, 6), (8, 6), (9, 6), (10, 6), (11, 6), (12, 6),
            (8, 7), (9, 7), (10, 7), (11, 7),
            (9, 8), (10, 8),
            # Eyes
            (8, 5), (11, 5),
            # Some ground blocks
            (2, 14), (3, 14), (4, 14), (5, 14), (6, 14),
            (14, 14), (15, 14), (16, 14), (17, 14),
            (2, 15), (3, 15), (4, 15), (5, 15), (6, 15),
            (14, 15), (15, 15), (16, 15), (17, 15),
            # Clouds
            (3, 2), (4, 2), (5, 2),
            (14, 3), (15, 3), (16, 3),
        ]

        for px, py in pattern:
            x = lcd_x0 + int(px * pixel_size)
            y = lcd_y0 + int(py * pixel_size)
            if x + pixel_size <= lcd_x1 and y + pixel_size <= lcd_y1:
                draw.rectangle([x, y, x + pixel_size - 1, y + pixel_size - 1], fill=dark_lcd)

    # === Screen glow effect ===
    glow = Image.new('RGBA', (s, s), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    rounded_rect(glow_draw, [lcd_x0, lcd_y0, lcd_x1, lcd_y1], lcd_r, fill=(180, 210, 140, 30))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=max(1, s//50)))
    img = Image.alpha_composite(img, glow)
    draw = ImageDraw.Draw(img)

    # === Power LED (small red dot) ===
    led_x = scr_x0 + int(s * 0.02)
    led_y = scr_y0 + int((scr_y1 - scr_y0) * 0.12)
    led_r_size = max(1, int(s * 0.012))
    draw.ellipse([led_x - led_r_size, led_y - led_r_size,
                  led_x + led_r_size, led_y + led_r_size], fill=(220, 40, 40, 255))
    # LED glow
    if size >= 64:
        led_glow = Image.new('RGBA', (s, s), (0, 0, 0, 0))
        lg_draw = ImageDraw.Draw(led_glow)
        lg_draw.ellipse([led_x - led_r_size*3, led_y - led_r_size*3,
                        led_x + led_r_size*3, led_y + led_r_size*3], fill=(255, 50, 50, 40))
        led_glow = led_glow.filter(ImageFilter.GaussianBlur(radius=max(1, s//80)))
        img = Image.alpha_composite(img, led_glow)
        draw = ImageDraw.Draw(img)

    # === D-Pad ===
    dpad_cx = body_x0 + int((body_x1 - body_x0) * 0.32)
    dpad_cy = strip_y1 + int((body_y1 - strip_y1) * 0.35)
    dpad_arm_w = int(s * 0.05)
    dpad_arm_l = int(s * 0.07)
    dpad_color = (55, 55, 65, 255)

    # Horizontal arm
    draw.rectangle([dpad_cx - dpad_arm_l, dpad_cy - dpad_arm_w,
                    dpad_cx + dpad_arm_l, dpad_cy + dpad_arm_w], fill=dpad_color)
    # Vertical arm
    draw.rectangle([dpad_cx - dpad_arm_w, dpad_cy - dpad_arm_l,
                    dpad_cx + dpad_arm_w, dpad_cy + dpad_arm_l], fill=dpad_color)

    # === A and B buttons ===
    btn_r = int(s * 0.045)
    btn_a_cx = body_x0 + int((body_x1 - body_x0) * 0.75)
    btn_a_cy = dpad_cy - int(s * 0.03)
    btn_b_cx = btn_a_cx - int(s * 0.10)
    btn_b_cy = dpad_cy + int(s * 0.03)

    # A button (reddish-purple like GBC)
    btn_a_color = (160, 50, 70, 255)
    draw.ellipse([btn_a_cx - btn_r, btn_a_cy - btn_r,
                  btn_a_cx + btn_r, btn_a_cy + btn_r], fill=btn_a_color)
    # B button
    btn_b_color = (160, 50, 70, 255)
    draw.ellipse([btn_b_cx - btn_r, btn_b_cy - btn_r,
                  btn_b_cx + btn_r, btn_b_cy + btn_r], fill=btn_b_color)

    # Button highlights
    if size >= 64:
        hl_r = max(1, btn_r // 3)
        hl_off = max(1, btn_r // 4)
        draw.ellipse([btn_a_cx - hl_r - hl_off, btn_a_cy - hl_r - hl_off,
                      btn_a_cx + hl_r - hl_off, btn_a_cy + hl_r - hl_off],
                     fill=(255, 255, 255, 50))
        draw.ellipse([btn_b_cx - hl_r - hl_off, btn_b_cy - hl_r - hl_off,
                      btn_b_cx + hl_r - hl_off, btn_b_cy + hl_r - hl_off],
                     fill=(255, 255, 255, 50))

    # === Start / Select buttons ===
    ss_y = dpad_cy + int(s * 0.14)
    ss_w = int(s * 0.055)
    ss_h = int(s * 0.02)
    ss_gap = int(s * 0.04)
    ss_color = (170, 168, 175, 255)

    cx = (body_x0 + body_x1) // 2
    # Select
    sel_x = cx - ss_gap
    rounded_rect(draw, [sel_x - ss_w, ss_y - ss_h, sel_x + ss_w, ss_y + ss_h],
                 ss_h, fill=ss_color)
    # Start
    sta_x = cx + ss_gap
    rounded_rect(draw, [sta_x - ss_w, ss_y - ss_h, sta_x + ss_w, ss_y + ss_h],
                 ss_h, fill=ss_color)

    # === Speaker grille (bottom right) ===
    if size >= 32:
        spk_x0 = body_x0 + int((body_x1 - body_x0) * 0.55)
        spk_y0 = strip_y1 + int((body_y1 - strip_y1) * 0.65)
        spk_spacing = max(2, int(s * 0.025))
        spk_dot_r = max(1, int(s * 0.006))
        spk_color = (185, 183, 190, 255)

        for row in range(4):
            for col in range(4):
                # Diagonal pattern like real GBC
                if (row + col) % 2 == 0:
                    continue
                dx = spk_x0 + col * spk_spacing
                dy = spk_y0 + row * spk_spacing
                draw.ellipse([dx - spk_dot_r, dy - spk_dot_r,
                              dx + spk_dot_r, dy + spk_dot_r], fill=spk_color)

    # === "COLOR" text label above screen ===
    if size >= 128:
        label_y = scr_y0 - int(s * 0.02)
        label_x = (scr_x0 + scr_x1) // 2
        # Draw tiny colored dots spelling suggestion of "COLOR"
        dot_r = max(1, int(s * 0.005))
        dot_spacing = int(s * 0.015)
        colors_rainbow = [
            (220, 50, 50, 255),    # Red
            (220, 150, 30, 255),   # Orange
            (50, 180, 50, 255),    # Green
            (50, 80, 200, 255),    # Blue
            (150, 50, 180, 255),   # Purple
        ]
        start_x = label_x - (len(colors_rainbow) - 1) * dot_spacing // 2
        for i, c in enumerate(colors_rainbow):
            dx = start_x + i * dot_spacing
            draw.ellipse([dx - dot_r, label_y - dot_r, dx + dot_r, label_y + dot_r], fill=c)

    # === Final: apply macOS icon mask ===
    final_mask = make_rounded_mask((s, s), icon_radius)
    img.putalpha(final_mask)

    return img


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    resources_dir = os.path.join(project_dir, "resources")
    os.makedirs(resources_dir, exist_ok=True)

    iconset_dir = os.path.join(resources_dir, "AppIcon.iconset")
    os.makedirs(iconset_dir, exist_ok=True)

    # Generate all required sizes
    sizes = {
        "icon_16x16.png": 16,
        "icon_16x16@2x.png": 32,
        "icon_32x32.png": 32,
        "icon_32x32@2x.png": 64,
        "icon_128x128.png": 128,
        "icon_128x128@2x.png": 256,
        "icon_256x256.png": 256,
        "icon_256x256@2x.png": 512,
        "icon_512x512.png": 512,
        "icon_512x512@2x.png": 1024,
    }

    # Cache rendered sizes to avoid re-rendering
    cache = {}
    for name, sz in sizes.items():
        if sz not in cache:
            print(f"Rendering {sz}x{sz}...")
            cache[sz] = draw_icon(sz)
        path = os.path.join(iconset_dir, name)
        cache[sz].save(path)
        print(f"  Saved {name}")

    # Convert to icns using iconutil
    icns_path = os.path.join(resources_dir, "AppIcon.icns")
    print(f"\nConverting to .icns...")
    ret = os.system(f'iconutil -c icns "{iconset_dir}" -o "{icns_path}"')
    if ret == 0:
        print(f"Created {icns_path}")
    else:
        print(f"iconutil failed (exit {ret}). .iconset is at {iconset_dir}")

    return ret == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

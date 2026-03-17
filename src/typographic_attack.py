"""Typographic attack implementation.

Adds misleading text overlays to images to test whether Gemini Embedding 2
shifts its embedding representation based on the overlaid text.
"""

import io
from PIL import Image, ImageDraw, ImageFont


def _get_font(size=40):
    """Get a font, falling back to default if no TTF available."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def _bytes_to_image(data):
    return Image.open(io.BytesIO(data))


def _image_to_bytes(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def add_typographic_text(image_bytes, text, font_size=60, position="center",
                         color=(0, 0, 0), bg_box=False, bg_box_color=(255, 255, 255),
                         repeat=1):
    """Add typographic text overlay to an image.

    Args:
        image_bytes: Source image as bytes.
        text: Text to overlay.
        font_size: Font size in pixels.
        position: One of 'center', 'top', 'bottom', 'top-left', 'top-right',
                  'bottom-left', 'bottom-right', 'scattered'.
        color: RGB tuple for text color.
        bg_box: Whether to draw a background box behind text.
        bg_box_color: Background box color.
        repeat: Number of times to repeat the text.

    Returns:
        Modified image as bytes.
    """
    img = _bytes_to_image(image_bytes).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _get_font(font_size)
    w, h = img.size

    display_text = " ".join([text] * repeat) if repeat > 1 else text
    bbox = draw.textbbox((0, 0), display_text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    positions = {
        "center": ((w - tw) // 2, (h - th) // 2),
        "top": ((w - tw) // 2, 10),
        "bottom": ((w - tw) // 2, h - th - 10),
        "top-left": (10, 10),
        "top-right": (w - tw - 10, 10),
        "bottom-left": (10, h - th - 10),
        "bottom-right": (w - tw - 10, h - th - 10),
    }

    if position == "scattered":
        # Place text at multiple locations
        locs = [
            (10, 10),
            (w - tw - 10, 10),
            ((w - tw) // 2, (h - th) // 2),
            (10, h - th - 10),
            (w - tw - 10, h - th - 10),
        ]
        for loc in locs:
            if bg_box:
                draw.rectangle(
                    [loc[0] - 2, loc[1] - 2, loc[0] + tw + 2, loc[1] + th + 2],
                    fill=bg_box_color
                )
            draw.text(loc, text, fill=color, font=font)
    else:
        pos = positions.get(position, positions["center"])
        if bg_box:
            draw.rectangle(
                [pos[0] - 2, pos[1] - 2, pos[0] + tw + 2, pos[1] + th + 2],
                fill=bg_box_color
            )
        draw.text(pos, display_text, fill=color, font=font)

    return _image_to_bytes(img)


def generate_attack_variants(image_bytes, target_text):
    """Generate multiple attack variants with different parameters.

    Args:
        image_bytes: Source image bytes.
        target_text: Misleading text to overlay.

    Returns:
        Dict mapping variant name to attacked image bytes.
    """
    variants = {}

    # Vary font size
    for size_name, size in [("small", 24), ("medium", 48), ("large", 80)]:
        key = f"size_{size_name}"
        variants[key] = add_typographic_text(
            image_bytes, target_text, font_size=size, position="center",
            color=(0, 0, 0)
        )

    # Vary position
    for pos in ["center", "top", "bottom", "top-left", "scattered"]:
        key = f"pos_{pos}"
        variants[key] = add_typographic_text(
            image_bytes, target_text, font_size=48, position=pos,
            color=(0, 0, 0)
        )

    # Vary color (against white background)
    for color_name, color in [("black", (0, 0, 0)), ("red", (220, 30, 30)),
                               ("white_on_bg", (200, 200, 200)),
                               ("blue", (30, 30, 220))]:
        key = f"color_{color_name}"
        variants[key] = add_typographic_text(
            image_bytes, target_text, font_size=48, position="center",
            color=color
        )

    # With background box
    variants["with_bg_box"] = add_typographic_text(
        image_bytes, target_text, font_size=48, position="center",
        color=(0, 0, 0), bg_box=True
    )

    # Repeated text
    for rep in [2, 3]:
        key = f"repeat_{rep}x"
        variants[key] = add_typographic_text(
            image_bytes, target_text, font_size=36, position="center",
            color=(0, 0, 0), repeat=rep
        )

    return variants


def add_multi_text_overlay(image_bytes, text_position_pairs, font_size=36,
                           color=(0, 0, 0)):
    """Add multiple different text strings at specified positions on an image.

    Useful for constructing adversarial hub images with keywords from
    different domains placed at different locations.

    Args:
        image_bytes: Source image as bytes.
        text_position_pairs: List of (text, position) tuples. Position is one of
            'center', 'top', 'bottom', 'top-left', 'top-right',
            'bottom-left', 'bottom-right'.
        font_size: Font size in pixels.
        color: RGB tuple for text color.

    Returns:
        Modified image as bytes.
    """
    img = _bytes_to_image(image_bytes).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _get_font(font_size)
    w, h = img.size

    for text, position in text_position_pairs:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        positions = {
            "center": ((w - tw) // 2, (h - th) // 2),
            "top": ((w - tw) // 2, 10),
            "bottom": ((w - tw) // 2, h - th - 10),
            "top-left": (10, 10),
            "top-right": (w - tw - 10, 10),
            "bottom-left": (10, h - th - 10),
            "bottom-right": (w - tw - 10, h - th - 10),
        }
        pos = positions.get(position, positions["center"])
        draw.text(pos, text, fill=color, font=font)

    return _image_to_bytes(img)


def create_document_with_hidden_text(title, body_text, hidden_text,
                                      text_color=(250, 250, 250),
                                      size=(512, 512)):
    """Create a document-like image with hidden/near-invisible text.

    Simulates a document screenshot with barely visible injected text
    for retrieval poisoning experiments.

    Args:
        title: Document title.
        body_text: Main body text.
        hidden_text: Text to hide in the document.
        text_color: Color of hidden text (near background color = hard to see).
        size: Image size.

    Returns:
        Image bytes (PNG).
    """
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)

    title_font = _get_font(28)
    body_font = _get_font(16)
    hidden_font = _get_font(14)

    # Draw title
    draw.text((20, 20), title, fill=(0, 0, 0), font=title_font)

    # Draw body text (word wrap)
    y = 70
    words = body_text.split()
    line = ""
    for word in words:
        test_line = f"{line} {word}".strip()
        bbox = draw.textbbox((0, 0), test_line, font=body_font)
        if bbox[2] - bbox[0] > size[0] - 40:
            draw.text((20, y), line, fill=(30, 30, 30), font=body_font)
            y += 22
            line = word
        else:
            line = test_line
    if line:
        draw.text((20, y), line, fill=(30, 30, 30), font=body_font)
        y += 22

    # Draw hidden text (near-invisible)
    draw.text((20, y + 20), hidden_text, fill=text_color, font=hidden_font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

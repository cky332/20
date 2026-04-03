"""Typographic attack implementation.

Adds misleading text overlays to images to test whether Gemini Embedding 2
shifts its embedding representation based on the overlaid text.

Supports configurable opacity, text outlines, tiled text placement,
dominant color extraction for low-contrast attacks, and font size
as a percentage of image height.

Includes FigStep-style typographic image generation (Gong et al., AAAI 2025):
renders paraphrased statements with numbered indices as standalone images.
"""

import io
import textwrap
from collections import Counter

import numpy as np
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


def get_dominant_color(image_bytes, n_colors=5):
    """Extract the dominant color from an image using pixel sampling.

    Downsamples the image and finds the most common color cluster.

    Args:
        image_bytes: Source image as bytes.
        n_colors: Number of color buckets for quantization.

    Returns:
        RGB tuple of the dominant color.
    """
    img = _bytes_to_image(image_bytes).convert("RGB")
    # Downsample for speed
    img = img.resize((64, 64), Image.LANCZOS)
    pixels = list(img.getdata())
    # Quantize: round each channel to nearest bucket
    bucket_size = 256 // n_colors
    quantized = []
    for r, g, b in pixels:
        qr = (r // bucket_size) * bucket_size + bucket_size // 2
        qg = (g // bucket_size) * bucket_size + bucket_size // 2
        qb = (b // bucket_size) * bucket_size + bucket_size // 2
        quantized.append((min(qr, 255), min(qg, 255), min(qb, 255)))
    counter = Counter(quantized)
    dominant = counter.most_common(1)[0][0]
    return dominant


def _draw_text_with_outline(draw, pos, text, font, fill, outline_color, outline_width):
    """Draw text with an outline/stroke effect."""
    x, y = pos
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, fill=outline_color, font=font)
    draw.text(pos, text, fill=fill, font=font)


def add_typographic_text(image_bytes, text, font_size=60, font_size_pct=None,
                         position="center", color=(0, 0, 0),
                         bg_box=False, bg_box_color=(255, 255, 255),
                         repeat=1, opacity=1.0,
                         outline_color=None, outline_width=0):
    """Add typographic text overlay to an image.

    Args:
        image_bytes: Source image as bytes.
        text: Text to overlay.
        font_size: Font size in pixels (ignored if font_size_pct is set).
        font_size_pct: Font size as fraction of image height (e.g., 0.10 = 10%).
        position: One of 'center', 'top', 'bottom', 'top-left', 'top-right',
                  'bottom-left', 'bottom-right', 'top-center', 'bottom-center',
                  'scattered', 'tiled'.
        color: RGB tuple for text color.
        bg_box: Whether to draw a background box behind text.
        bg_box_color: Background box color.
        repeat: Number of times to repeat the text.
        opacity: Text opacity from 0.0 (transparent) to 1.0 (opaque).
        outline_color: Optional RGB tuple for text outline/stroke.
        outline_width: Width of text outline in pixels.

    Returns:
        Modified image as bytes.
    """
    img = _bytes_to_image(image_bytes).convert("RGBA")
    w, h = img.size

    # Compute font size from percentage if specified
    if font_size_pct is not None:
        font_size = max(8, int(h * font_size_pct))

    font = _get_font(font_size)

    # Create transparent overlay for opacity support
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    alpha = int(opacity * 255)
    fill_color = (*color, alpha)
    outline_fill = (*outline_color, alpha) if outline_color else None

    display_text = " ".join([text] * repeat) if repeat > 1 else text
    bbox = draw.textbbox((0, 0), display_text, font=font)
    tw, th_text = bbox[2] - bbox[0], bbox[3] - bbox[1]

    if position == "tiled":
        # Tile text across entire image
        pad_x, pad_y = max(10, tw // 4), max(10, th_text // 2)
        y_pos = 5
        while y_pos < h:
            x_pos = 5
            while x_pos < w:
                if outline_color and outline_width > 0:
                    _draw_text_with_outline(draw, (x_pos, y_pos), text, font,
                                            fill_color, outline_fill, outline_width)
                else:
                    draw.text((x_pos, y_pos), text, fill=fill_color, font=font)
                x_pos += tw + pad_x
            y_pos += th_text + pad_y
    elif position == "scattered":
        locs = [
            (10, 10),
            (w - tw - 10, 10),
            ((w - tw) // 2, (h - th_text) // 2),
            (10, h - th_text - 10),
            (w - tw - 10, h - th_text - 10),
        ]
        for loc in locs:
            if bg_box:
                box_fill = (*bg_box_color, alpha) if len(bg_box_color) == 3 else bg_box_color
                draw.rectangle(
                    [loc[0] - 2, loc[1] - 2, loc[0] + tw + 2, loc[1] + th_text + 2],
                    fill=box_fill
                )
            if outline_color and outline_width > 0:
                _draw_text_with_outline(draw, loc, text, font,
                                        fill_color, outline_fill, outline_width)
            else:
                draw.text(loc, text, fill=fill_color, font=font)
    else:
        positions = {
            "center": ((w - tw) // 2, (h - th_text) // 2),
            "top": ((w - tw) // 2, 10),
            "bottom": ((w - tw) // 2, h - th_text - 10),
            "top-left": (10, 10),
            "top-right": (w - tw - 10, 10),
            "bottom-left": (10, h - th_text - 10),
            "bottom-right": (w - tw - 10, h - th_text - 10),
            "top-center": ((w - tw) // 2, 10),
            "bottom-center": ((w - tw) // 2, h - th_text - 10),
        }
        pos = positions.get(position, positions["center"])
        if bg_box:
            box_fill = (*bg_box_color, alpha) if len(bg_box_color) == 3 else bg_box_color
            draw.rectangle(
                [pos[0] - 2, pos[1] - 2, pos[0] + tw + 2, pos[1] + th_text + 2],
                fill=box_fill
            )
        if outline_color and outline_width > 0:
            _draw_text_with_outline(draw, pos, display_text, font,
                                    fill_color, outline_fill, outline_width)
        else:
            draw.text(pos, display_text, fill=fill_color, font=font)

    # Composite overlay onto base image
    img = Image.alpha_composite(img, overlay)
    img = img.convert("RGB")
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


# ---------------------------------------------------------------------------
# FigStep-style typographic image generation (Gong et al., AAAI 2025)
# ---------------------------------------------------------------------------

def _get_freemono_font(size=50):
    """Get FreeMono Bold font as used in FigStep paper, with fallback."""
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf", size
        )
    except (OSError, IOError):
        return _get_font(size)


def paraphrase_to_figstep_statement(target_label, prefix="Steps to"):
    """Convert a target label to a FigStep-format declarative statement.

    Follows the FigStep paper's paraphrase step: transforms a topic into
    a declarative statement beginning with "Steps to", "List of", or
    "Methods to".

    Args:
        target_label: Target concept (e.g., "fire truck").
        prefix: One of "Steps to", "List of", "Methods to".

    Returns:
        Paraphrased statement string.
    """
    prefix_templates = {
        "Steps to": f"Steps to identify a {target_label}.",
        "List of": f"List of features of a {target_label}.",
        "Methods to": f"Methods to recognize a {target_label}.",
    }
    return prefix_templates.get(prefix, f"{prefix} {target_label}.")


def create_figstep_image(statement, numbered_items=3, font_size=50,
                         size=(512, 512), bg_color=(255, 255, 255),
                         text_color=(0, 0, 0)):
    """Create a FigStep-style typographic image.

    Renders a paraphrased declarative statement followed by numbered
    indices (e.g., "1.\\n2.\\n3.") on a solid background, mimicking the
    FigStep paper's typography step.

    Args:
        statement: The paraphrased declarative statement text.
        numbered_items: Number of numbered indices to append (default 3).
        font_size: Font size in pixels (paper uses 80 at 760px; 50 at 512px).
        size: Image dimensions (width, height).
        bg_color: Background color RGB tuple.
        text_color: Text color RGB tuple.

    Returns:
        Image bytes (PNG).
    """
    img = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(img)
    font = _get_freemono_font(font_size)
    w, h = size
    margin = 30

    # Compute character width for wrapping
    sample_bbox = draw.textbbox((0, 0), "M", font=font)
    char_w = sample_bbox[2] - sample_bbox[0]
    max_chars = max(1, (w - 2 * margin) // char_w)

    # Wrap the statement text
    wrapped_lines = textwrap.wrap(statement, width=max_chars)

    # Add numbered indices
    for i in range(1, numbered_items + 1):
        wrapped_lines.append(f"{i}.")

    # Draw lines
    line_bbox = draw.textbbox((0, 0), "Mg", font=font)
    line_height = (line_bbox[3] - line_bbox[1]) + 8
    y = margin
    for line in wrapped_lines:
        if y + line_height > h - margin:
            break
        draw.text((margin, y), line, fill=text_color, font=font)
        y += line_height

    return _image_to_bytes(img)

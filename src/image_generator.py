"""Programmatic test image generation using Pillow.

Generates simple but recognizable images for embedding experiments
without requiring external datasets.
"""

import io
import math
from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.insert(0, ".")
from config import IMAGE_SIZE


def _get_font(size=40):
    """Get a font, falling back to default if no TTF available."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert a PIL Image to bytes."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def bytes_to_image(data: bytes) -> Image.Image:
    """Convert bytes to a PIL Image."""
    return Image.open(io.BytesIO(data))


def generate_solid_color_image(color, size=None):
    """Generate a solid color image.

    Args:
        color: RGB tuple or color name.
        size: (width, height) tuple.

    Returns:
        Image bytes (PNG).
    """
    size = size or IMAGE_SIZE
    img = Image.new("RGB", size, color)
    return image_to_bytes(img)


def generate_shape_image(shape_type, shape_color, bg_color=(255, 255, 255), size=None):
    """Generate an image with a geometric shape.

    Args:
        shape_type: One of 'circle', 'rectangle', 'triangle', 'star'.
        shape_color: RGB tuple for the shape.
        bg_color: RGB tuple for background.
        size: (width, height) tuple.

    Returns:
        Image bytes (PNG).
    """
    size = size or IMAGE_SIZE
    img = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(img)
    w, h = size
    margin = int(min(w, h) * 0.15)

    if shape_type == "circle":
        draw.ellipse(
            [margin, margin, w - margin, h - margin],
            fill=shape_color
        )
    elif shape_type == "rectangle":
        draw.rectangle(
            [margin, margin, w - margin, h - margin],
            fill=shape_color
        )
    elif shape_type == "triangle":
        points = [
            (w // 2, margin),
            (margin, h - margin),
            (w - margin, h - margin),
        ]
        draw.polygon(points, fill=shape_color)
    elif shape_type == "star":
        cx, cy = w // 2, h // 2
        outer_r = min(w, h) // 2 - margin
        inner_r = outer_r * 0.4
        points = []
        for i in range(10):
            angle = math.pi / 2 + i * math.pi / 5
            r = outer_r if i % 2 == 0 else inner_r
            points.append((cx + r * math.cos(angle), cy - r * math.sin(angle)))
        draw.polygon(points, fill=shape_color)

    return image_to_bytes(img)


def generate_scene_image(scene_type, size=None):
    """Generate a simple scene image.

    Args:
        scene_type: One of 'sky', 'forest', 'ocean', 'sunset', 'city'.
        size: (width, height) tuple.

    Returns:
        Image bytes (PNG).
    """
    size = size or IMAGE_SIZE
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    w, h = size

    if scene_type == "sky":
        # Blue gradient with white clouds
        for y in range(h):
            blue = int(135 + (y / h) * 70)
            draw.line([(0, y), (w, y)], fill=(100, 150, min(255, blue)))
        # Sun
        draw.ellipse([w - 150, 30, w - 50, 130], fill=(255, 223, 0))
        # Clouds
        for cx, cy in [(100, 80), (250, 50), (380, 90)]:
            draw.ellipse([cx - 40, cy - 20, cx + 40, cy + 20], fill=(255, 255, 255))
            draw.ellipse([cx - 20, cy - 35, cx + 20, cy + 5], fill=(255, 255, 255))

    elif scene_type == "forest":
        # Green background
        draw.rectangle([0, 0, w, h], fill=(34, 120, 50))
        # Ground
        draw.rectangle([0, h * 3 // 4, w, h], fill=(80, 50, 20))
        # Trees
        for tx in range(50, w, 100):
            # Trunk
            draw.rectangle([tx - 10, h // 2, tx + 10, h * 3 // 4], fill=(100, 60, 20))
            # Canopy
            draw.polygon([(tx, h // 4), (tx - 50, h // 2), (tx + 50, h // 2)],
                         fill=(0, 150, 0))

    elif scene_type == "ocean":
        # Sky
        draw.rectangle([0, 0, w, h // 2], fill=(135, 206, 235))
        # Water
        for y in range(h // 2, h):
            blue = int(0 + ((y - h // 2) / (h // 2)) * 100)
            draw.line([(0, y), (w, y)], fill=(0, 50, 150 + min(105, blue)))

    elif scene_type == "sunset":
        for y in range(h):
            ratio = y / h
            r = int(255 * (1 - ratio * 0.5))
            g = int(100 + 80 * (1 - ratio))
            b = int(50 + 100 * ratio)
            draw.line([(0, y), (w, y)], fill=(r, min(255, g), min(255, b)))
        # Sun
        draw.ellipse([w // 2 - 60, h // 3 - 60, w // 2 + 60, h // 3 + 60],
                     fill=(255, 69, 0))

    elif scene_type == "city":
        draw.rectangle([0, 0, w, h], fill=(40, 40, 60))
        # Buildings
        import random
        rng = random.Random(42)
        for bx in range(0, w, 60):
            bh = rng.randint(h // 4, h * 3 // 4)
            bw = rng.randint(30, 55)
            color = (rng.randint(60, 120), rng.randint(60, 120), rng.randint(80, 140))
            draw.rectangle([bx, h - bh, bx + bw, h], fill=color)
            # Windows
            for wy in range(h - bh + 10, h - 10, 20):
                for wx in range(bx + 5, bx + bw - 5, 15):
                    draw.rectangle([wx, wy, wx + 8, wy + 12],
                                   fill=(255, 255, 150) if rng.random() > 0.3 else (50, 50, 50))

    return image_to_bytes(img)


def generate_labeled_image(label, bg_color=(255, 255, 255), text_color=(0, 0, 0), size=None):
    """Generate an image with a text label centered on it.

    Args:
        label: Text to display.
        bg_color: Background color.
        text_color: Text color.
        size: Image size.

    Returns:
        Image bytes (PNG).
    """
    size = size or IMAGE_SIZE
    img = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(img)
    font = _get_font(48)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size[0] - tw) // 2
    y = (size[1] - th) // 2
    draw.text((x, y), label, fill=text_color, font=font)
    return image_to_bytes(img)


# Pre-defined test image sets for experiments
SOURCE_IMAGES = {
    "red_circle": lambda: generate_shape_image("circle", (220, 30, 30), (255, 255, 255)),
    "blue_rectangle": lambda: generate_shape_image("rectangle", (30, 30, 220), (255, 255, 255)),
    "green_triangle": lambda: generate_shape_image("triangle", (30, 180, 30), (255, 255, 255)),
    "yellow_star": lambda: generate_shape_image("star", (255, 215, 0), (255, 255, 255)),
    "ocean_scene": lambda: generate_scene_image("ocean"),
    "forest_scene": lambda: generate_scene_image("forest"),
    "sunset_scene": lambda: generate_scene_image("sunset"),
    "city_scene": lambda: generate_scene_image("city"),
}

TARGET_LABELS = [
    "fire truck",
    "iPod",
    "banana",
    "airplane",
    "pizza",
    "laptop computer",
    "basketball",
    "christmas tree",
]

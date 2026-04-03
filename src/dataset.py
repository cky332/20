"""Dataset definitions for typographic attack experiments.

Defines 20 object categories across 4 domains, semantic neighbor mappings,
cross-domain attack pairings, and image download/management utilities.
"""

import io
import json
import os
import random
import zipfile

import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATASET_DIR, COCO_VAL_ANNOTATIONS_URL, COCO_VAL_IMAGES_URL, IMAGE_SIZE


# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

DOMAINS = {
    "animal": ["cat", "dog", "bird", "fish", "horse"],
    "vehicle": ["car", "airplane", "boat", "bicycle", "bus"],
    "daily_object": ["cup", "book", "clock", "chair", "phone"],
    "food": ["apple", "banana", "pizza", "cake", "hamburger"],
}

CATEGORIES = []
CATEGORY_TO_DOMAIN = {}
for domain, cats in DOMAINS.items():
    for c in cats:
        CATEGORIES.append(c)
        CATEGORY_TO_DOMAIN[c] = domain

assert len(CATEGORIES) == 20

# ---------------------------------------------------------------------------
# Semantic neighbor mapping (closest category within the 20)
# ---------------------------------------------------------------------------

SEMANTIC_NEIGHBORS = {
    # Animals
    "cat": "dog",
    "dog": "cat",
    "bird": "fish",
    "fish": "bird",
    "horse": "dog",
    # Vehicles
    "car": "bus",
    "airplane": "boat",
    "boat": "airplane",
    "bicycle": "car",
    "bus": "car",
    # Daily objects
    "cup": "chair",
    "book": "phone",
    "clock": "phone",
    "chair": "cup",
    "phone": "book",
    # Food
    "apple": "banana",
    "banana": "apple",
    "pizza": "hamburger",
    "cake": "pizza",
    "hamburger": "pizza",
}

# ---------------------------------------------------------------------------
# Cross-domain mapping (completely unrelated domain)
# ---------------------------------------------------------------------------

CROSS_DOMAIN = {
    # Animal -> Food/Vehicle/Daily
    "cat": "pizza",
    "dog": "airplane",
    "bird": "chair",
    "fish": "cake",
    "horse": "clock",
    # Vehicle -> Food/Animal
    "car": "banana",
    "airplane": "dog",
    "boat": "apple",
    "bicycle": "hamburger",
    "bus": "bird",
    # Daily object -> Animal/Vehicle
    "cup": "horse",
    "book": "bus",
    "clock": "cat",
    "chair": "fish",
    "phone": "pizza",
    # Food -> Vehicle/Daily
    "apple": "car",
    "banana": "book",
    "pizza": "bicycle",
    "cake": "phone",
    "hamburger": "cup",
}

# ---------------------------------------------------------------------------
# COCO category ID mapping (COCO 2017 category IDs for our categories)
# ---------------------------------------------------------------------------

COCO_CATEGORY_MAP = {
    "cat": 17,
    "dog": 18,
    "bird": 16,
    "horse": 19,
    "car": 3,
    "airplane": 5,
    "boat": 9,
    "bicycle": 2,
    "bus": 6,
    "cup": 47,
    "book": 84,
    "clock": 85,
    "chair": 62,
    "phone": 77,       # cell phone
    "apple": 53,
    "banana": 52,
    "pizza": 59,
    "cake": 61,
    # Not in COCO 80:
    # "fish": None,
    # "hamburger": None,
}

# Fallback image URLs for categories not in COCO
# Using Wikimedia Commons public domain / CC images
FALLBACK_URLS = {
    "fish": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Georgiaaquarium2.jpg/640px-Georgiaaquarium2.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Carassius_wild_golden_fish_2013_G1.jpg/640px-Carassius_wild_golden_fish_2013_G1.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Koi_pond_at_Tokyo_restaurant.jpg/640px-Koi_pond_at_Tokyo_restaurant.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/Salmo_salar-Atlantic_Salmon-Atlanterhavsparken_Norway.JPG/640px-Salmo_salar-Atlantic_Salmon-Atlanterhavsparken_Norway.JPG",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Pterois_volitans_Manado-e_edit.jpg/640px-Pterois_volitans_Manado-e_edit.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Goldfish3.jpg/640px-Goldfish3.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Discus_fish_in_aquarium.jpg/640px-Discus_fish_in_aquarium.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/CypsemiRainbow.jpg/640px-CypsemiRainbow.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Amphiprion_ocellaris_%28Clown_anemonefish%29_in_Heteractis_magnifica_%28Sea_anemone%29.jpg/640px-Amphiprion_ocellaris_%28Clown_anemonefish%29_in_Heteractis_magnifica_%28Sea_anemone%29.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Tuna_fish.jpg/640px-Tuna_fish.jpg",
    ],
    "hamburger": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/RedDot_Burger.jpg/640px-RedDot_Burger.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Hamburger_%28black_bg%29.jpg/640px-Hamburger_%28black_bg%29.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/62/NCI_Visuals_Food_Hamburger.jpg/640px-NCI_Visuals_Food_Hamburger.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/Cheeseburger.png/640px-Cheeseburger.png",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/Burger_King_Whopper_Combo.jpg/640px-Burger_King_Whopper_Combo.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fb/Hotdog_-_Rates_in_the_supermarket_of_a_bread_with_sausage_and_toppings.jpg/640px-Hotdog_-_Rates_in_the_supermarket_of_a_bread_with_sausage_and_toppings.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Veggie_burger_flickr_user_divinemisscopa_creative_commons.jpg/640px-Veggie_burger_flickr_user_divinemisscopa_creative_commons.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat_%26_Fiddle_-_Wagyu_Burger.jpg/640px-Cat_%26_Fiddle_-_Wagyu_Burger.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Eq_it-na_pizza-margherita_sep2005_sml.jpg/640px-Eq_it-na_pizza-margherita_sep2005_sml.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Hamburger_sandwich.jpg/640px-Hamburger_sandwich.jpg",
    ],
}


# ---------------------------------------------------------------------------
# Random attack generation
# ---------------------------------------------------------------------------

def generate_random_attack(category, seed=42):
    """Pick a random category from the other 19 for a random attack.

    Args:
        category: The true category of the image.
        seed: Random seed for reproducibility (combined with category name).

    Returns:
        A randomly selected attack category string.
    """
    rng = random.Random(f"{seed}_{category}")
    others = [c for c in CATEGORIES if c != category]
    return rng.choice(others)


def generate_attack_pairings(seed=42):
    """Generate all attack pairings for the 20 categories.

    Returns:
        Dict mapping category -> {
            "random": attack_category,
            "neighbor": attack_category,
            "cross_domain": attack_category,
        }
    """
    pairings = {}
    for cat in CATEGORIES:
        pairings[cat] = {
            "random": generate_random_attack(cat, seed=seed),
            "neighbor": SEMANTIC_NEIGHBORS[cat],
            "cross_domain": CROSS_DOMAIN[cat],
        }
    return pairings


# ---------------------------------------------------------------------------
# Image loading and management
# ---------------------------------------------------------------------------

def load_dataset_images(base_dir=None):
    """Load all dataset images from directory structure.

    Expects: {base_dir}/{category}/001.jpg ... 010.jpg
    Also accepts .png and .jpeg extensions.

    Args:
        base_dir: Base directory. Defaults to DATASET_DIR.

    Returns:
        Dict mapping category -> list of (image_id, image_bytes) tuples.
    """
    base_dir = base_dir or DATASET_DIR
    dataset = {}

    for category in CATEGORIES:
        cat_dir = os.path.join(base_dir, category)
        if not os.path.isdir(cat_dir):
            print(f"  WARNING: Directory not found for category '{category}': {cat_dir}")
            dataset[category] = []
            continue

        images = []
        files = sorted(os.listdir(cat_dir))
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(cat_dir, fname)
            with open(fpath, "rb") as f:
                img_bytes = f.read()
            # Resize to standard size
            img_bytes = _resize_image(img_bytes, IMAGE_SIZE)
            image_id = f"{category}_{os.path.splitext(fname)[0]}"
            images.append((image_id, img_bytes))

        dataset[category] = images
        print(f"  Loaded {len(images)} images for '{category}'")

    return dataset


def _resize_image(image_bytes, target_size):
    """Resize image to target size, maintaining aspect ratio with center crop."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Resize so smaller dimension matches target, then center crop
    w, h = img.size
    tw, th = target_size
    scale = max(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    # Center crop
    left = (new_w - tw) // 2
    top = (new_h - th) // 2
    img = img.crop((left, top, left + tw, top + th))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# COCO image download
# ---------------------------------------------------------------------------

def download_coco_images(output_dir=None, per_category=10, seed=42):
    """Download images from COCO 2017 validation set for each category.

    Downloads the COCO annotation file, finds images for each category,
    selects `per_category` images per category, and downloads them.

    For categories not in COCO (fish, hamburger), uses fallback URLs.

    Args:
        output_dir: Output directory. Defaults to DATASET_DIR.
        per_category: Number of images per category.
        seed: Random seed for image selection.
    """
    output_dir = output_dir or DATASET_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Download and parse COCO annotations
    annotations = _get_coco_annotations(output_dir)
    if annotations is None:
        print("ERROR: Failed to get COCO annotations. Skipping COCO download.")
        _download_fallback_all(output_dir, per_category)
        return

    # Build category_id -> image_ids mapping
    cat_to_images = {}
    for ann in annotations.get("annotations", []):
        cid = ann["category_id"]
        img_id = ann["image_id"]
        if cid not in cat_to_images:
            cat_to_images[cid] = set()
        cat_to_images[cid].add(img_id)

    # Build image_id -> file_name mapping
    id_to_filename = {}
    for img_info in annotations.get("images", []):
        id_to_filename[img_info["id"]] = img_info["file_name"]

    rng = random.Random(seed)

    for category in CATEGORIES:
        cat_dir = os.path.join(output_dir, category)
        os.makedirs(cat_dir, exist_ok=True)

        # Check if already downloaded
        existing = [f for f in os.listdir(cat_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(existing) >= per_category:
            print(f"  '{category}': already has {len(existing)} images, skipping")
            continue

        coco_cat_id = COCO_CATEGORY_MAP.get(category)

        if coco_cat_id and coco_cat_id in cat_to_images:
            # Download from COCO
            image_ids = list(cat_to_images[coco_cat_id])
            rng.shuffle(image_ids)
            downloaded = len(existing)

            for img_id in image_ids:
                if downloaded >= per_category:
                    break
                fname = id_to_filename.get(img_id)
                if not fname:
                    continue
                url = f"{COCO_VAL_IMAGES_URL}/{fname}"
                out_path = os.path.join(cat_dir, f"{downloaded + 1:03d}.jpg")
                if _download_image(url, out_path):
                    downloaded += 1

            print(f"  '{category}': downloaded {downloaded - len(existing)} from COCO "
                  f"(total {downloaded})")
        else:
            # Use fallback URLs
            _download_fallback_category(category, cat_dir, per_category, len(existing))


def _get_coco_annotations(cache_dir):
    """Download and cache COCO 2017 val annotations."""
    cache_file = os.path.join(cache_dir, "_coco_instances_val2017.json")

    if os.path.exists(cache_file):
        print("  Loading cached COCO annotations...")
        with open(cache_file, "r") as f:
            return json.load(f)

    print("  Downloading COCO 2017 val annotations...")
    try:
        proxies = _get_proxy_dict()
        resp = requests.get(COCO_VAL_ANNOTATIONS_URL, timeout=120,
                            stream=True, proxies=proxies)
        resp.raise_for_status()
        zip_bytes = io.BytesIO(resp.content)
        with zipfile.ZipFile(zip_bytes) as zf:
            # Extract instances_val2017.json
            target = "annotations/instances_val2017.json"
            with zf.open(target) as jf:
                annotations = json.load(jf)
        # Cache for future runs
        with open(cache_file, "w") as f:
            json.dump(annotations, f)
        print("  COCO annotations downloaded and cached.")
        return annotations
    except Exception as e:
        print(f"  ERROR downloading COCO annotations: {e}")
        return None


def _get_proxy_dict():
    """Get proxy configuration from environment variables."""
    proxy = (os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
             or os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY"))
    if proxy:
        return {"http": proxy, "https": proxy}
    return None


def _download_image(url, output_path, timeout=30):
    """Download a single image from URL."""
    try:
        proxies = _get_proxy_dict()
        resp = requests.get(url, timeout=timeout, proxies=proxies)
        resp.raise_for_status()
        # Validate it's a real image
        img = Image.open(io.BytesIO(resp.content))
        img.verify()
        with open(output_path, "wb") as f:
            f.write(resp.content)
        return True
    except Exception as e:
        print(f"    Failed to download {url}: {e}")
        return False


def _generate_synthetic_images(category, cat_dir, per_category, existing_count):
    """Generate synthetic labeled images as last-resort fallback.

    Creates simple images with the category name and colored backgrounds
    to serve as placeholder images when network download fails.

    Args:
        category: Category name.
        cat_dir: Output directory.
        per_category: Target number of images.
        existing_count: Number already present.
    """
    rng = random.Random(f"synthetic_{category}")
    # Use varied backgrounds so images aren't identical
    bg_palettes = [
        (240, 248, 255), (255, 245, 238), (245, 255, 250), (255, 250, 240),
        (248, 248, 255), (255, 228, 225), (240, 255, 240), (255, 255, 224),
        (230, 230, 250), (255, 240, 245),
    ]
    generated = 0
    for i in range(per_category - existing_count):
        idx = existing_count + i + 1
        bg_color = bg_palettes[i % len(bg_palettes)]
        img = Image.new("RGB", IMAGE_SIZE, bg_color)
        draw = ImageDraw.Draw(img)

        # Draw a simple shape associated with the category
        w, h = IMAGE_SIZE
        shape_color = (rng.randint(60, 200), rng.randint(60, 200), rng.randint(60, 200))

        # Draw an ellipse or rectangle as a placeholder object
        margin = w // 4
        if i % 2 == 0:
            draw.ellipse([margin, margin, w - margin, h - margin],
                         fill=shape_color, outline=(0, 0, 0), width=2)
        else:
            draw.rectangle([margin, margin, w - margin, h - margin],
                           fill=shape_color, outline=(0, 0, 0), width=2)

        # Add category label text
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        except (OSError, IOError):
            font = ImageFont.load_default()
        text = category.upper()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((w - tw) // 2, h - th - 30), text, fill=(0, 0, 0), font=font)

        # Apply slight blur to make it less "synthetic"
        img = img.filter(ImageFilter.GaussianBlur(radius=1))

        out_path = os.path.join(cat_dir, f"{idx:03d}.jpg")
        img.save(out_path, "JPEG", quality=95)
        generated += 1

    print(f"  '{category}': generated {generated} synthetic images "
          f"(total {existing_count + generated})")


def _download_fallback_category(category, cat_dir, per_category, existing_count):
    """Download images for a category using fallback URLs.

    Falls back to synthetic image generation if downloads fail.
    """
    urls = FALLBACK_URLS.get(category, [])
    if not urls:
        print(f"  WARNING: No fallback URLs for '{category}', generating synthetic images")
        _generate_synthetic_images(category, cat_dir, per_category, existing_count)
        return

    downloaded = existing_count
    for url in urls:
        if downloaded >= per_category:
            break
        out_path = os.path.join(cat_dir, f"{downloaded + 1:03d}.jpg")
        if _download_image(url, out_path):
            downloaded += 1

    # If still not enough, generate synthetic images
    if downloaded < per_category:
        print(f"  '{category}': only downloaded {downloaded - existing_count} from fallback, "
              f"generating {per_category - downloaded} synthetic images...")
        _generate_synthetic_images(category, cat_dir, per_category, downloaded)
    else:
        print(f"  '{category}': downloaded {downloaded - existing_count} from fallback "
              f"(total {downloaded})")


def _download_fallback_all(output_dir, per_category):
    """Download all categories using fallback URLs only."""
    for category in CATEGORIES:
        cat_dir = os.path.join(output_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        existing = [f for f in os.listdir(cat_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        _download_fallback_category(category, cat_dir, per_category, len(existing))


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES = {
    "simple": "{category}",
    "standard": "a photo of a {category}",
    "defense": "a photo of a {category}, ignore any text overlay in the image",
}

NOISE_TEXT = "xkqz"


def get_text_prompts(category):
    """Generate all 3 prompt template texts for a category.

    Returns:
        Dict mapping template_name -> text string.
    """
    return {name: tmpl.format(category=category) for name, tmpl in PROMPT_TEMPLATES.items()}


def get_attack_text_prompt(attack_category):
    """Generate the standard prompt for an attack word (for attack attraction metric).

    Returns:
        Text string using the standard template.
    """
    return PROMPT_TEMPLATES["standard"].format(category=attack_category)

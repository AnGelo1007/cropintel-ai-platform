from pathlib import Path
import argparse
import random
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CROPS = ["corn", "eggplant", "pepper", "rice", "tomato"]
HEALTHS = ["healthy", "unhealthy"]


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def list_images(folder: Path):
    return [p for p in folder.iterdir() if is_image_file(p)]


def safe_open_image(path: Path):
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        print(f"  Skipped unreadable image: {path.name} ({e})")
        return None


def random_augment(img: Image.Image) -> Image.Image:
    # Small random rotation
    angle = random.uniform(-18, 18)
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)

    # Random horizontal flip
    if random.random() < 0.5:
        img = ImageOps.mirror(img)

    # Random vertical flip (less frequent)
    if random.random() < 0.15:
        img = ImageOps.flip(img)

    # Random brightness
    if random.random() < 0.8:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))

    # Random contrast
    if random.random() < 0.8:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))

    # Random color
    if random.random() < 0.7:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(0.85, 1.15))

    # Slight sharpness change
    if random.random() < 0.5:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(random.uniform(0.9, 1.3))

    # Occasional blur
    if random.random() < 0.15:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))

    # Random crop + resize back
    if random.random() < 0.6:
        w, h = img.size
        crop_ratio = random.uniform(0.88, 0.98)
        new_w = int(w * crop_ratio)
        new_h = int(h * crop_ratio)

        if new_w < w and new_h < h:
            left = random.randint(0, w - new_w)
            top = random.randint(0, h - new_h)
            img = img.crop((left, top, left + new_w, top + new_h))
            img = img.resize((w, h), Image.Resampling.LANCZOS)

    return img


def next_augmented_filename(folder: Path, stem: str, counter: int) -> Path:
    return folder / f"{stem}_aug_{counter:04d}.jpg"


def augment_folder_to_target(folder: Path, target: int):
    originals = list_images(folder)
    current = len(originals)

    print(f"{folder}: current={current}")

    if current == 0:
        print("  Skipped: no images found.")
        return

    if current >= target:
        print(f"  Already at or above target ({target}).")
        return

    needed = target - current
    print(f"  Creating {needed} augmented images...")

    generated = 0
    counter = 1

    # Make sure counter starts after any existing augmented files
    existing_names = {p.name for p in folder.iterdir() if p.is_file()}
    while generated < needed:
        src_path = random.choice(originals)
        img = safe_open_image(src_path)
        if img is None:
            continue

        aug_img = random_augment(img)

        while True:
            out_path = next_augmented_filename(folder, src_path.stem, counter)
            counter += 1
            if out_path.name not in existing_names and not out_path.exists():
                break

        try:
            aug_img.save(out_path, format="JPEG", quality=95)
            existing_names.add(out_path.name)
            generated += 1
        except Exception as e:
            print(f"  Failed to save {out_path.name}: {e}")

    print(f"  Done. New total should be at least {target}.")


def validate_structure(dataset_root: Path):
    missing = []

    for crop in CROPS:
        crop_dir = dataset_root / crop
        if not crop_dir.exists():
            missing.append(str(crop_dir))
            continue

        for health in HEALTHS:
            sub = crop_dir / health
            if not sub.exists():
                missing.append(str(sub))

    if missing:
        raise FileNotFoundError(
            "Missing required dataset folders:\n" + "\n".join(f" - {m}" for m in missing)
        )


def main():
    parser = argparse.ArgumentParser(
        description="Augment crop health dataset folders until each reaches target image count."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to dataset_crops root folder"
    )
    parser.add_argument(
        "--target",
        type=int,
        required=True,
        help="Target number of images per healthy/unhealthy folder"
    )

    args = parser.parse_args()

    dataset_root = Path(args.input).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Input folder not found: {dataset_root}")

    validate_structure(dataset_root)

    print(f"\nUsing dataset root: {dataset_root}")
    print(f"Target per folder: {args.target}\n")

    for crop in CROPS:
        for health in HEALTHS:
            folder = dataset_root / crop / health
            augment_folder_to_target(folder, args.target)

    print("\nAugmentation complete.")


if __name__ == "__main__":
    main()
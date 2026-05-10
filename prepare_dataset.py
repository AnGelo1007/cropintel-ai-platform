import argparse
import os
import random
import shutil
from pathlib import Path

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def collect_images(folder: Path):
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]

def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test folders.")
    parser.add_argument("--input", required=True, help="Input dataset root, e.g. dataset_crops")
    parser.add_argument("--output", required=True, help="Output dataset root, e.g. dataset_split")
    parser.add_argument("--train", type=float, default=0.70, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--test", type=float, default=0.15, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if abs((args.train + args.val + args.test) - 1.0) > 1e-6:
        raise ValueError("train + val + test must equal 1.0")

    input_root = Path(args.input)
    output_root = Path(args.output)

    if not input_root.exists():
        raise FileNotFoundError(f"Input folder not found: {input_root}")

    random.seed(args.seed)

    classes = [p for p in input_root.iterdir() if p.is_dir()]
    if not classes:
        raise RuntimeError("No class folders found in input dataset.")

    for split in ["train", "val", "test"]:
        safe_mkdir(output_root / split)

    for class_dir in classes:
        images = collect_images(class_dir)
        if len(images) < 2:
            print(f"Skipping {class_dir.name}: not enough images")
            continue

        random.shuffle(images)
        total = len(images)
        train_n = int(total * args.train)
        val_n = int(total * args.val)
        test_n = total - train_n - val_n

        split_map = {
            "train": images[:train_n],
            "val": images[train_n:train_n + val_n],
            "test": images[train_n + val_n:]
        }

        for split, files in split_map.items():
            class_out = output_root / split / class_dir.name
            safe_mkdir(class_out)
            for file_path in files:
                shutil.copy2(file_path, class_out / file_path.name)

        print(f"{class_dir.name}: total={total}, train={train_n}, val={val_n}, test={test_n}")

    print(f"\nDone. Dataset split saved to: {output_root}")

if __name__ == "__main__":
    main()

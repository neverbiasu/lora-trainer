#!/usr/bin/env python3
"""
Intelligent crop Fern character images to 512x512.
SAFETY: Reads from input, writes to output ONLY. Never modifies source.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

try:
    import numpy as np
    from PIL import Image
except ImportError:
    print("Install dependencies: pip install pillow numpy")
    sys.exit(1)


def detect_face_region(img_array: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect face/character region in image using skin color detection.
    Returns (left, top, right, bottom) or None if not found.
    """
    try:
        h, w = img_array.shape[:2]

        if w < 512 or h < 512:
            return None

        rgb = img_array[:, :, :3]

        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        skin_mask = (r > 95) & (g > 40) & (b > 20) & (r > g) & (r > b) & (abs(int(r) - int(g)) > 15)

        if np.sum(skin_mask) < 100:
            return None

        coords = np.where(skin_mask)
        if len(coords[0]) == 0:
            return None

        top = max(0, np.min(coords[0]) - 50)
        bottom = min(h, np.max(coords[0]) + 100)
        left = max(0, np.min(coords[1]) - 50)
        right = min(w, np.max(coords[1]) + 50)

        return (left, top, right, bottom)

    except Exception:
        return None


def smart_crop_to_512(img: Image.Image) -> Optional[Image.Image]:
    """
    Intelligently crop to 512x512 focusing on character.
    """
    w, h = img.size

    if w < 512 or h < 512:
        return None

    img_array = np.array(img)

    face_region = detect_face_region(img_array)

    if face_region:
        left, top, right, bottom = face_region
        region_w = right - left
        region_h = bottom - top

        if region_w > 50 and region_h > 50:
            cx = (left + right) // 2
            cy = (top + bottom) // 2

            crop_left = max(0, min(cx - 256, w - 512))
            crop_top = max(0, min(cy - 256, h - 512))
            crop_right = crop_left + 512
            crop_bottom = crop_top + 512

            return img.crop((crop_left, crop_top, crop_right, crop_bottom))

    crop_left = (w - 512) // 2
    crop_top = (h - 512) // 2
    crop_right = crop_left + 512
    crop_bottom = crop_top + 512

    return img.crop((crop_left, crop_top, crop_right, crop_bottom))


def process_dataset(
    input_dir: str = "./examples/fern/raw",
    output_dir: str = "./examples/fern/dataset_512",
    target_size: int = 512,
    create_captions: bool = True,
) -> int:
    """
    Process images: smart crop to 512x512, convert to PNG.
    SAFETY: Input files are NEVER modified.

    Args:
        input_dir: Source directory (NOT modified)
        output_dir: Target directory for processed images
        target_size: Target size (512x512)
        create_captions: Generate caption files

    Returns:
        Number of successfully processed images
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return 0

    output_path.mkdir(parents=True, exist_ok=True)

    images = (
        list(input_path.glob("*.jpg"))
        + list(input_path.glob("*.jpeg"))
        + list(input_path.glob("*.JPG"))
        + list(input_path.glob("*.JPEG"))
        + list(input_path.glob("*.png"))
        + list(input_path.glob("*.PNG"))
        + list(input_path.glob("*.webp"))
        + list(input_path.glob("*.WEBP"))
    )

    if not images:
        print(f"❌ No images found in {input_dir}")
        return 0

    images = sorted(images)

    print("=" * 70)
    print("🖼️  Fern Image Cropper (512x512 Smart Crop)")
    print("=" * 70)
    print(f"📂 Input:  {input_path.absolute()} (READ ONLY - NOT MODIFIED)")
    print(f"📂 Output: {output_path.absolute()}")
    print(f"📊 Images: {len(images)}")
    print()

    captions = [
        "fern, girl, elf, purple hair, illustration, frieren",
        "fern elf mage, frieren beyond journey's end, fantasy",
        "fern character, anime illustration style, magic",
        "fern, purple long hair, elf, serious expression, frieren",
        "character illustration, fern anime, elf mage",
        "fern frieren, purple hair girl, anime character",
    ]

    processed = 0
    failed = 0

    for idx, img_path in enumerate(images, 1):
        try:
            img = Image.open(img_path)

            if img.mode != "RGB":
                img = img.convert("RGB")

            w, h = img.size
            print(
                f"[{idx:2d}/{len(images)}] {img_path.name:20s} ({w:4d}x{h:4d}) ", end="", flush=True
            )

            if w < 512 or h < 512:
                print("❌ Too small (< 512x512)")
                failed += 1
                continue

            cropped = smart_crop_to_512(img)

            if cropped is None:
                print("❌ Crop failed")
                failed += 1
                continue

            output_img = output_path / f"fern_{processed:03d}.png"
            cropped.save(output_img, quality=95)

            if create_captions:
                caption = captions[processed % len(captions)]
                caption_file = output_img.with_suffix(".txt")
                caption_file.write_text(caption, encoding="utf-8")

            processed += 1
            print("✅")

        except Exception as e:
            print(f"❌ {e}")
            failed += 1

    print()
    print("=" * 70)
    print(f"✅ Successfully processed: {processed}/{len(images)}")
    if failed > 0:
        print(f"❌ Failed: {failed}")

    print()
    print(f"📂 Output location: {output_path.absolute()}/")
    print(f"📝 Original images SAFE at: {input_path.absolute()}/")

    if create_captions:
        print(f"📋 Generated {processed} caption files")

    if processed > 0:
        print()
        print("🎉 Ready for training!")
        print("   lora-trainer --config examples/config_fern_test.yaml \\")
        print(f"                --dataset {output_dir} \\")
        print("                --run-dir ./runs/test_fern")

    return processed


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Smart crop Fern images to 512x512 (input NOT modified)"
    )
    parser.add_argument(
        "--input",
        default="./examples/fern/raw",
        help="Input directory (read only)",
    )
    parser.add_argument(
        "--output",
        default="./examples/fern/dataset_512",
        help="Output directory for cropped images",
    )
    parser.add_argument(
        "--no-captions",
        action="store_true",
        help="Don't create caption files",
    )

    args = parser.parse_args()

    process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        create_captions=not args.no_captions,
    )


if __name__ == "__main__":
    main()

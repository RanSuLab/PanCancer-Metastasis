import os
import datetime
import numpy as np
from PIL import Image
import openslide


def is_valid_patch(region, save_dir, slide_name, patch_size, index, x, y, coordinates):
    image_rgb = region.convert("RGB")
    image_gray = region.convert("L")
    image_array = np.array(image_gray)

    # Threshold-based background filtering
    background_ratio = np.sum(image_array < 200) / (patch_size * patch_size)
    dark_ratio = np.sum(image_array < 30) / (patch_size * patch_size)

    if background_ratio > 0.5 and dark_ratio < 0.3:
        patch_name = f"{slide_name}_{index + 1}.jpg"
        image_rgb.save(os.path.join(save_dir, patch_name))
        coordinates.append([x, y])
        return True
    return False


def remove_empty_folders(base_dir, min_patches=1):
    deleted_count = 0
    print("Deleting empty folders:")

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        patch_count = len(os.listdir(folder_path))
        if patch_count < min_patches:
            os.rmdir(folder_path)
            print(f"Removed: {folder}")
            deleted_count += 1

    print(f"Total deleted folders: {deleted_count}")
    return deleted_count


def extract_patches_from_slide(slide, patch_size, save_dir, slide_name, suffix, coord_dir):
    patch_output_dir = os.path.join(save_dir, f"{slide_name}+{suffix}")
    os.makedirs(patch_output_dir, exist_ok=True)
    os.makedirs(coord_dir, exist_ok=True)

    width, height = slide.size
    coordinates = []
    patch_count = 0

    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            region = slide.crop((x, y, x + patch_size, y + patch_size))
            if is_valid_patch(region, patch_output_dir, slide_name, patch_size, patch_count, x, y, coordinates):
                patch_count += 1

    np.save(os.path.join(coord_dir, f"{slide_name}_{suffix}.npy"), np.array(coordinates))
    return patch_count


def process_wsi_folder(img_dir, save_dir, patch_size, coord_dir, level=1):
    svs_files = [f for f in os.listdir(img_dir) if f.endswith(".svs")]
    for i, filename in enumerate(svs_files):
        start_time = datetime.datetime.now()
        slide_path = os.path.join(img_dir, filename)
        slide_name = os.path.splitext(filename)[0]

        slide = openslide.open_slide(slide_path)
        print(f"Processing ({i + 1}/{len(svs_files)}): {slide_name}")

        img = slide.read_region((0, 0), level, slide.level_dimensions[level])
        num_patches = extract_patches_from_slide(img, patch_size, save_dir, slide_name, "AA", coord_dir)

        elapsed = (datetime.datetime.now() - start_time).seconds
        print(f"Extracted {num_patches} patches in {elapsed}s.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract patches from WSI files.")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to input WSI directory.")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save patch images.")
    parser.add_argument("--coord_dir", type=str, required=True, help="Path to save coordinates.")
    parser.add_argument("--patch_size", type=int, default=512, help="Patch size (default: 512).")
    parser.add_argument("--min_patches", type=int, default=1, help="Minimum patch count to keep a folder.")

    args = parser.parse_args()

    process_wsi_folder(args.img_dir, args.save_dir, args.patch_size, args.coord_dir)
    remove_empty_folders(args.save_dir, args.min_patches)

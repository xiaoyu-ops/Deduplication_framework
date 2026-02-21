import os
import random
from pathlib import Path

# Configuration
SOURCE_DIR = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\imagenet_bloated"
DEST_DIR = r"D:\Deduplication_framework\2026_new_experiment\datasets\test_10k_subset"
TARGET_COUNT = 10000

def create_subset():
    source_path = Path(SOURCE_DIR)
    dest_path = Path(DEST_DIR)

    if not source_path.exists():
        print(f"Error: Source directory {SOURCE_DIR} does not exist.")
        return

    dest_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Scan for images
    print(f"Scanning {SOURCE_DIR}...")
    all_images = []
    for root, _, files in os.walk(source_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                all_files_path = os.path.join(root, file)
                all_images.append(all_files_path)
    
    total_found = len(all_images)
    print(f"Found {total_found} images.")

    if total_found == 0:
        return

    # 2. Random sample
    selected_images = random.sample(all_images, min(TARGET_COUNT, total_found))
    print(f"Selected {len(selected_images)} images for the subset.")

    # 3. Create Hard Links
    success_count = 0
    clean_count = 0
    
    # Clean existing destination first? No, just add to it or let it fail if exists.
    # Actually, better to start fresh to ensure exactly 10k.
    for item in dest_path.iterdir():
        if item.is_file():
            try:
                os.unlink(item)
                clean_count += 1
            except:
                pass
    if clean_count > 0:
        print(f"Cleaned {clean_count} files from previous test.")
        
    print("Creating hard links...")
    for src in selected_images:
        filename = os.path.basename(src)
        # Handle duplicates in filename if flattening directory structure
        # Since os.walk goes deep, images in different folders might have same name.
        # We'll prepend parent folder name to be safe.
        src_path = Path(src)
        parent_name = src_path.parent.name
        new_filename = f"{parent_name}_{filename}"
        
        dst = dest_path / new_filename
        
        try:
            os.link(src, dst)
            success_count += 1
        except FileExistsError:
            pass
        except Exception as e:
            print(f"Failed to link {src}: {e}")
            
    print(f"Successfully created {success_count} hard links in {DEST_DIR}")

if __name__ == "__main__":
    create_subset()

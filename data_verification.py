import os
import random
import hashlib
from PIL import Image
import matplotlib.pyplot as plt

def get_all_images(folder_path):
    # Recursively get all image paths from a directory.
    images = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(valid_extensions):
                images.append(os.path.join(root, file))
    return images

def visual_check(real_path, fake_path):
    # Plot 10 random images from each folder.
    print("\n3.3 Visual Check")
    real_images = get_all_images(real_path)
    fake_images = get_all_images(fake_path)

    if len(real_images) < 10 or len(fake_images) < 10:
        print("Not enough images for visual check (need at least 10 in each).")
        return

    selected_real = random.sample(real_images, 10)
    selected_fake = random.sample(fake_images, 10)

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle("Visual Check: Top 2 rows Real, Bottom 2 rows Fake")

    # Plot Real images
    for i, img_path in enumerate(selected_real):
        row = i // 5
        col = i % 5
        try:
            img = Image.open(img_path)
            axes[row, col].imshow(img)
            axes[row, col].set_title("Real")
            axes[row, col].axis('off')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    # Plot Fake images
    for i, img_path in enumerate(selected_fake):
        row = (i // 5) + 2
        col = i % 5
        try:
            img = Image.open(img_path)
            axes[row, col].imshow(img)
            axes[row, col].set_title("Fake")
            axes[row, col].axis('off')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    plt.tight_layout()
    plt.show()

def check_corrupt_files(folder_path):
    # Open every image and check for errors.
    print(f"\nChecking for corrupt files in {folder_path}")
    images = get_all_images(folder_path)
    corrupt_count = 0
    total_images = len(images)
    print(f"Found {total_images} images.")
    
    for i, img_path in enumerate(images):
        if (i + 1) % 50 == 0:
            print(f"Checking {i + 1}/{total_images}...", end='\r')
            
        try:
            with Image.open(img_path) as img:
                img.verify() # Verify it's an image
        except (IOError, SyntaxError) as e:
            print(f"\nCorrupt file found: {img_path}. Deleting...")
            try:
                os.remove(img_path)
                corrupt_count += 1
            except OSError as os_err:
                print(f"Failed to delete {img_path}: {os_err}")
    print(f"\nFinished checking. Removed {corrupt_count} corrupt files.")

def check_duplicates(folder_paths):
    # Check for duplicates across directories using MD5 hashing.
    print("\nChecking for duplicates")
    hashes = {}
    duplicates = []
    
    for folder in folder_paths:
        images = get_all_images(folder)
        print(f"Hashing images in {folder}...")
        total_images = len(images)
        
        for i, img_path in enumerate(images):
            if (i + 1) % 50 == 0:
                print(f"Hashing {i + 1}/{total_images}...", end='\r')
                
            try:
                with Image.open(img_path) as img:
                    # Calculate hash
                    img_hash = hashlib.md5(img.tobytes()).hexdigest()
                    
                    if img_hash in hashes:
                        duplicates.append((img_path, hashes[img_hash]))
                    else:
                        hashes[img_hash] = img_path
            except Exception:
                continue # Skip corrupt files (handled by other function)
        print(f"Finished hashing {total_images} images in {folder}.")

    if duplicates:
        print(f"Found {len(duplicates)} duplicates!")
        for dup in duplicates:
            print(f"Duplicate: '{dup[0]}' is same as '{dup[1]}'")
            try:
                os.remove(dup[0])
                print(f"Deleted duplicate: '{dup[0]}'")
            except OSError as os_err:
                print(f"Failed to delete {dup[0]}: {os_err}")
    else:
        print("No duplicates found.")

def main():
    # Define the paths to your image folders here
    real_folder = r"/home/jakko/Documents/Kool/5.Semester/IDS/Project/IDS-AI-Image-detection/archive/REAL"
    fake_folder = r"/home/jakko/Documents/Kool/5.Semester/IDS/Project/IDS-AI-Image-detection/archive/FAKE"

    if not os.path.exists(real_folder):
        print(f"Error: The folder '{real_folder}' does not exist. Please check the path in the script.")
        return
    if not os.path.exists(fake_folder):
        print(f"Error: The folder '{fake_folder}' does not exist. Please check the path in the script.")
        return

    # 3.3 Visual Check
    visual_check(real_folder, fake_folder)

    # 3.4 Verifying data quality
    # Corrupt Files
    check_corrupt_files(real_folder)
    check_corrupt_files(fake_folder)

    # Duplicates
    check_duplicates([real_folder, fake_folder])

if __name__ == "__main__":
    main()

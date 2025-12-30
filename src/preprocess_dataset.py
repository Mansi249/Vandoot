import os
import cv2  # If this fails, run: pip install opencv-python

# CONFIGURATION
# ---------------------------------------------------------
SOURCE_FOLDER = "VanDoot_Patent_Dataset"
OUTPUT_FOLDER = "VanDoot_Ready_96x96"
TARGET_SIZE = 96  # ESP32-CAM limit
CLASSES = ["0_fire", "1_human", "2_animal", "3_empty"]
# ---------------------------------------------------------

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print(f"--- STARTING DAY 1 PRE-PROCESSING ---")
print(f"Target Size: {TARGET_SIZE}x{TARGET_SIZE}")

total_processed = 0

for category in CLASSES:
    src_path = os.path.join(SOURCE_FOLDER, category)
    dst_path = os.path.join(OUTPUT_FOLDER, category)
    
    # Create destination sub-folder
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    
    # Check if source exists
    if not os.path.exists(src_path):
        print(f"[ERROR] Missing folder: {src_path}. Did you create it?")
        continue

    print(f"\nProcessing Class: {category}...")
    files = os.listdir(src_path)
    count = 0
    
    for filename in files:
        try:
            # 1. Read Image
            file_path = os.path.join(src_path, filename)
            img = cv2.imread(file_path)
            
            if img is None:
                continue # Skip unreadable files
            
            # 2. Resize to 96x96 (Crucial for ESP32)
            img_resized = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))
            
            # 3. Save to new folder
            cv2.imwrite(os.path.join(dst_path, filename), img_resized)
            count += 1
            
        except Exception as e:
            print(f"Skipped {filename}: {e}")
            
    print(f" -> Successfully converted {count} images.")
    total_processed += count

print(f"\n--- DONE! ---")
print(f"Total images ready for training: {total_processed}")
print(f"Your 'Day 2' dataset is located in: {OUTPUT_FOLDER}")
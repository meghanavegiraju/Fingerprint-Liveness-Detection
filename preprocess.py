import os
import random
import shutil
from PIL import Image

# -----------------------------
# Configuration
# -----------------------------
SOURCE_DIR = "data"
OUTPUT_DIR = "dataset"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".bmp", ".png")

random.seed(42)

# -----------------------------
# Process each class
# -----------------------------
classes = ["live", "spoof"]

for cls in classes:

    source_path = os.path.join(SOURCE_DIR, cls)

    images = [
        f for f in os.listdir(source_path)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    random.shuffle(images)

    total = len(images)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    split_data = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    for split_name, image_list in split_data.items():

        output_path = os.path.join(OUTPUT_DIR, split_name, cls)
        os.makedirs(output_path, exist_ok=True)

        for image_name in image_list:

            image_path = os.path.join(source_path, image_name)

            image = Image.open(image_path)

            image = image.convert("RGB")

            image = image.resize((224, 224))

            save_path = os.path.join(output_path, image_name)

            image.save(save_path)

print("====================================")
print("Dataset preprocessing completed!")
print("Images resized to 224 x 224")
print("Dataset split into:")
print("Train : 70%")
print("Validation : 15%")
print("Test : 15%")
print("====================================")
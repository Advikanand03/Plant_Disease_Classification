import os
import re

import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def clean_class_name(class_name):
    """
    Clean class names by removing trailing numbers and underscores
    Examples:
    - 'anthracnose3102' -> 'anthracnose'
    - 'gumosis_1714' -> 'gumosis'
    - 'leaf miner3466' -> 'leaf miner'
    - 'bacterial blight3241' -> 'bacterial blight'
    - 'healthy5877' -> 'healthy'
    """
    cleaned = re.sub(r'[_]?\d+$', '', class_name)
    return cleaned.strip()


def load_ccmt_dataset_separate(base_dir):
    """
    Load images from CCMT dataset keeping train and test sets separate.
    Returns two dataframes: train_df and test_df.
    """
    train_image_paths = []
    train_labels = []
    test_image_paths = []
    test_labels = []
    train_skipped = 0
    test_skipped = 0
    valid_extensions = ('.jpg', '.png', '.jpeg', '.bmp', '.webp', '.JPG', '.PNG', '.JPEG')

    print(f"\n📂 Loading CCMT dataset from: {base_dir}")
    print("Keeping train_set and test_set separate...")

    plant_categories = [d for d in os.listdir(base_dir)
                       if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith('.')]

    print(f"Found plant categories: {plant_categories}")

    name_mapping = {}

    for plant in plant_categories:
        plant_path = os.path.join(base_dir, plant)

        if os.path.exists(os.path.join(plant_path, 'train_set')):
            train_path = os.path.join(plant_path, 'train_set')
            print(f"\n  Processing {plant}/train_set...")

            disease_folders = [d for d in os.listdir(train_path)
                               if os.path.isdir(os.path.join(train_path, d)) and not d.startswith('.')]

            for disease_folder in disease_folders:
                disease_path = os.path.join(train_path, disease_folder)
                disease_name = clean_class_name(disease_folder)
                label = f"{plant}_{disease_name}"

                if disease_folder != disease_name:
                    name_mapping[f"{plant}_{disease_folder}"] = label

                count = 0
                for img_name in os.listdir(disease_path):
                    if img_name.startswith('.'):
                        train_skipped += 1
                        continue

                    if img_name.lower().endswith(valid_extensions):
                        img_path = os.path.join(disease_path, img_name)
                        try:
                            with Image.open(img_path) as img:
                                img.verify()

                            train_image_paths.append(img_path)
                            train_labels.append(label)
                            count += 1
                        except Exception:
                            train_skipped += 1

                if count > 0:
                    print(f"    {label}: Added {count} training images (from {disease_folder})")

        if os.path.exists(os.path.join(plant_path, 'test_set')):
            test_path = os.path.join(plant_path, 'test_set')
            print(f"\n  Processing {plant}/test_set...")

            disease_folders = [d for d in os.listdir(test_path)
                               if os.path.isdir(os.path.join(test_path, d)) and not d.startswith('.')]

            for disease_folder in disease_folders:
                disease_path = os.path.join(test_path, disease_folder)
                disease_name = clean_class_name(disease_folder)
                label = f"{plant}_{disease_name}"

                if disease_folder != disease_name:
                    name_mapping[f"{plant}_{disease_folder}"] = label

                count = 0
                for img_name in os.listdir(disease_path):
                    if img_name.startswith('.'):
                        test_skipped += 1
                        continue

                    if img_name.lower().endswith(valid_extensions):
                        img_path = os.path.join(disease_path, img_name)
                        try:
                            with Image.open(img_path) as img:
                                img.verify()

                            test_image_paths.append(img_path)
                            test_labels.append(label)
                            count += 1
                        except Exception:
                            test_skipped += 1

                if count > 0:
                    print(f"    {label}: Added {count} test images (from {disease_folder})")

    print(f"\n✅ Dataset loading complete!")
    print(f"   Training images: {len(train_image_paths)}")
    print(f"   Test images: {len(test_image_paths)}")
    print(f"   Total classes in training: {len(set(train_labels))}")
    print(f"   Total classes in test: {len(set(test_labels))}")
    print(f"   Training files skipped: {train_skipped}")
    print(f"   Test files skipped: {test_skipped}")

    print(f"\n🔄 Example class name mappings:")
    for i, (original, cleaned) in enumerate(list(name_mapping.items())[:10]):
        print(f"   {original} -> {cleaned}")

    train_df = pd.DataFrame({'image_path': train_image_paths, 'label': train_labels})
    test_df = pd.DataFrame({'image_path': test_image_paths, 'label': test_labels})

    return train_df, test_df

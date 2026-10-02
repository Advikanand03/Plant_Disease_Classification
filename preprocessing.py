import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class PlantDiseaseDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

        self.classes = sorted(dataframe['label'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}

        self.valid_indices = []
        print("Validating images...")
        for idx in range(len(self.dataframe)):
            img_path = self.dataframe.iloc[idx]['image_path']
            try:
                with Image.open(img_path) as img:
                    img.verify()
                img = Image.open(img_path).convert('RGB')
                self.valid_indices.append(idx)
            except Exception as e:
                print(f"Warning: Skipping corrupted image {img_path}: {e}")

        print(f"Kept {len(self.valid_indices)} out of {len(self.dataframe)} images")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        img_path = self.dataframe.iloc[actual_idx]['image_path']
        label = self.dataframe.iloc[actual_idx]['label']

        try:
            img = Image.open(img_path).convert('RGB')

            if self.transform:
                img = self.transform(img)

            label_idx = self.class_to_idx[label]
            return img, label_idx

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img = torch.zeros(3, 224, 224)
            label_idx = 0
            return img, label_idx


def get_transforms():
    """
    Basic data augmentation transforms
    Training: Basic augmentations
    Test: Only resize and normalize
    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform
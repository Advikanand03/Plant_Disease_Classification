import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_loader import load_ccmt_dataset_separate
from src.evaluation import evaluate_model
from src.model_training import create_loss_optimizer_scheduler, create_model, train_model
from src.preprocessing import PlantDiseaseDataset, get_transforms
from src.visualization import plot_confusion_matrix, plot_training_curves, plot_tsne_features


def main():
    """Main training pipeline"""
    print("🌿 Plant Disease Classification - MobileNetV4-Conv-Medium")
    print("=" * 60)

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n📂 Loading CCMT dataset...")

    base_dir = '/kaggle/input/crop-pest-and-disease-dataset/Dataset for Crop Pest and Disease Detection/CCMT Dataset-Augmented'

    train_df, test_df = load_ccmt_dataset_separate(base_dir)

    print(f"\n📊 Dataset Summary:")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Training classes: {train_df['label'].nunique()}")
    print(f"  Test classes: {test_df['label'].nunique()}")

    train_classes = set(train_df['label'].unique())
    test_classes = set(test_df['label'].unique())
    missing_in_train = test_classes - train_classes

    if missing_in_train:
        print(f"\n⚠️ Warning: {len(missing_in_train)} classes in test set not found in training:")
        for cls in missing_in_train:
            print(f"  - {cls}")

    train_transform, test_transform = get_transforms()

    train_dataset = PlantDiseaseDataset(train_df, transform=train_transform)
    test_dataset = PlantDiseaseDataset(test_df, transform=test_transform)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"\n🏷️ Number of classes: {num_classes}")
    print(f"Classes: {class_names}")

    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    print("\n🤖 Creating model...")
    model = create_model(num_classes, device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion, optimizer, scheduler = create_loss_optimizer_scheduler(model)

    num_epochs = 25

    model, train_losses, train_accs = train_model(
        model, train_loader, criterion, optimizer,
        scheduler, num_epochs, device
    )

    print("\n🧪 Evaluating on test set...")
    results = evaluate_model(model, test_loader, class_names, device)

    print("\n📊 Test Results:")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1_score']:.4f}")

    print("\n📝 Classification Report:")
    print(results['classification_report'])

    print("\n📈 Generating visualizations...")
    plot_training_curves(train_losses, train_accs)
    plot_confusion_matrix(results['confusion_matrix'], class_names)

    print("Creating t-SNE visualization (this may take a minute)...")
    try:
        plot_tsne_features(model, test_loader, class_names, device)
    except Exception as e:
        print(f"t-SNE visualization skipped due to: {e}")
        print("This is normal if dataset is very large or has issues")

    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'num_classes': num_classes,
        'train_results': {
            'train_losses': train_losses,
            'train_accuracies': train_accs
        },
        'test_results': results
    }, 'final_results.pth')

    print("\n✅ Results saved as 'final_results.pth'")
    print("=" * 60)
    print("🎉 Training and evaluation complete!")

    return model, results


if __name__ == "__main__":
    model, results = main()

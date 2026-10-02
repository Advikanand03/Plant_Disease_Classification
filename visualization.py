import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def plot_training_curves(train_losses, train_accs):
    """
    Plot training curves (loss and accuracy)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, label='Training Loss', linewidth=2, color='blue')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_accs, label='Training Accuracy', linewidth=2, color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names):
    """
    Plot confusion matrix
    """
    plt.figure(figsize=(10, 8))

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()


def plot_tsne_features(model, dataloader, class_names, device='cuda'):
    """
    Create t-SNE plot of feature representations
    """
    from sklearn.manifold import TSNE
    import matplotlib.cm as cm

    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            features = model.forward_features(images)
            features = features.mean([2, 3])

            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())

    all_features = np.concatenate(features_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(all_features)

    plt.figure(figsize=(10, 8))
    colors = cm.rainbow(np.linspace(0, 1, len(class_names)))

    for i, class_name in enumerate(class_names):
        indices = np.where(all_labels == i)[0]
        if len(indices) > 0:
            plt.scatter(features_2d[indices, 0], features_2d[indices, 1],
                       c=[colors[i]], label=class_name, alpha=0.6)

    plt.title('t-SNE Visualization of Feature Space')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

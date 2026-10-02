import torch
import torch.nn as nn
import torch.optim as optim
import timm


def create_model(num_classes, device):
    model_name = 'mobilenetv3_large_100'
    print(f"Loading {model_name}...")

    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes
    )
    model = model.to(device)
    return model


def create_loss_optimizer_scheduler(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
    return criterion, optimizer, scheduler


def train_model(model, train_loader, criterion, optimizer,
                scheduler, num_epochs=25, device='cuda'):
    """
    Simple training function without validation
    """
    train_losses = []
    train_accuracies = []

    print(f"\nStarting training for {num_epochs} epochs...")
    print("Epoch\tTrain Loss\tTrain Acc")
    print("-" * 30)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        scheduler.step()

        print(f"{epoch+1:2d}/{num_epochs}\t{train_loss:.4f}\t\t{train_acc:.2f}%")

        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_accuracy': train_acc,
            }, f'checkpoint_epoch_{epoch+1}.pth')

    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'train_accuracy': train_acc,
    }, 'final_model.pth')

    print(f"\nTraining Complete! Final Training Accuracy: {train_acc:.2f}%")

    return model, train_losses, train_accuracies

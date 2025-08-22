import os
import torch.nn as nn
import torch
import torch.optim as optim
from src.models.cnn import CNN
from src.dataa.loader import get_dataloader
import matplotlib.pyplot as plt

def training_loop(train_loader, model, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch, (image, label) in enumerate(train_loader):
        image = image.to(torch.device(device))
        label = label.to(torch.device(device))
        
        optimizer.zero_grad()
        
        outputs = model(image)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        
        if batch % 5 == 0:
            print(f'Batch {batch}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    return epoch_loss, epoch_accuracy


def validation_loop(test_loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch, (image , label) in  enumerate(test_loader):
            image = image.to(torch.device(device))
            label = label.to(torch.device(device))
            
            outputs = model(image)
            loss = criterion(outputs, label)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            
            if batch % 5 == 0:
                print(f'Validation Batch {batch}/{len(test_loader)}, Loss: {loss.item():.4f}')
            
    epoch_loss = running_loss / len(test_loader)
    epoch_accuracy = 100 * correct / total
    return epoch_loss, epoch_accuracy


def epoch_loop(EPOCHS, model , train_loader, test_loader, optimizer, criterion,device):
    train_losses = []
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        
        train_loss, train_accuracy = training_loop(train_loader, model, optimizer, criterion, device)
        val_loss, val_accuracy = validation_loop(test_loader, model, criterion, device)
        
        train_losses.append(train_loss)
        validation_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(val_accuracy)
        
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            print(f'New best validation accuracy: {best_val_acc:.2f}%')
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
    print('Training complete.')
    return train_losses, validation_losses, train_accuracies, validation_accuracies

def main():
    
    epochs = 150
    
    
    train_dir = "data/processed/train"
    test_dir = "data/processed/test"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    torch.set_default_device(DEVICE)
    train_loader, test_loader, class_names = get_dataloader(train_dir, test_dir, DEVICE, batch_size=32)
    model = CNN(num_classes=len(class_names)).to(DEVICE)
    LEARNING_RATE = 0.001
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    training_losses, validation_losses, train_accuracies, validation_accuracies = epoch_loop(epochs, model, train_loader, test_loader, optimizer, criterion, DEVICE)
    
    # Plotting the training and validation losses
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(validation_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    main()


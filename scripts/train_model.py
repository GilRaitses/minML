import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Hyperparameters from config
        self.epochs = config.get("num_epochs", 20)
        self.learning_rate = config.get("learning_rate", 0.001)

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        self.model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Metrics
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

        print("Training complete.")
